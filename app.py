import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ----------------------------
# MongoDB Connection
# ----------------------------
MONGO_URI = "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/smart_energy?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["smart_energy"]
users_col = db["users"]
appliances_col = db["appliances"]
readings_col = db["energy_readings"]

# ----------------------------
# Streamlit Page & Colors
# ----------------------------
st.set_page_config(page_title="Smart Energy Dashboard", page_icon="⚡", layout="wide")

# Background Gradient + Black Buttons
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #007BFF, #FFC107, #FF0000);
    color: white;
}
div.stButton > button {
    background-color: black;
    color: white;
    height: 3em;
    width: 12em;
    border-radius: 8px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper Functions
# ----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email, username, password, address):
    if users_col.find_one({"email": email}):
        return "Email already exists."
    if users_col.find_one({"username": username}):
        return "Username already exists."
    users_col.insert_one({
        "name": name,
        "email": email,
        "username": username,
        "password": hash_password(password),
        "address": address,
        "funds": 0,
        "created_at": datetime.now()
    })
    return "User registered successfully!"

def login_user(username, password):
    user = users_col.find_one({"username": username})
    if user and user["password"] == hash_password(password):
        return user
    return None

def auto_simulate_energy(user_id):
    appliances = list(appliances_col.find({"user_id": user_id, "status": "on"}))
    for appliance in appliances:
        power = appliance.get("power_rating", 200)
        usage = round(np.random.uniform(0.1, 0.5) * (power/1000), 3)
        cost = round(usage * 100, 2)
        user = users_col.find_one({"_id": user_id})
        if user["funds"] >= cost:
            users_col.update_one({"_id": user_id}, {"$inc": {"funds": -cost}})
            readings_col.insert_one({
                "user_id": user_id,
                "appliance_id": appliance["_id"],
                "timestamp": datetime.now(),
                "power_consumption": usage,
                "charged_amount": cost
            })
        else:
            appliances_col.update_one({"_id": appliance["_id"]}, {"$set": {"status": "off"}})
            st.warning(f"Funds too low for {appliance['name']}. Appliance turned OFF.")

def predict_next_usage(user_id):
    appliances = list(appliances_col.find({"user_id": user_id}))
    predictions = {}
    for appliance in appliances:
        readings = list(readings_col.find({"appliance_id": appliance["_id"]}).sort("timestamp", 1))
        if len(readings) >= 5:  # Need at least 5 data points
            df = pd.DataFrame(readings)
            df["time_index"] = range(len(df))
            X = df[["time_index"]]
            y = df["power_consumption"]
            model = LinearRegression()
            model.fit(X, y)
            next_index = np.array([[len(df)]])
            pred = model.predict(next_index)[0]
            predictions[appliance["name"]] = round(pred, 3)
    return predictions

# ----------------------------
# Session State
# ----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# ----------------------------
# User Authentication
# ----------------------------
if st.session_state.user is None:
    st.title("Login / Register")
    option = st.selectbox("Choose Action", ["Login", "Register"])
    
    if option == "Register":
        with st.form("register_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            address = st.text_input("Address")
            submitted = st.form_submit_button("Register")
            if submitted:
                message = register_user(name, email, username, password, address)
                st.success(message)
    
    if option == "Login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user = login_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success(f"Welcome, {user['name']}!")
                else:
                    st.error("Invalid username or password.")

# ----------------------------
# Main Dashboard
# ----------------------------
if st.session_state.user:
    user_id = st.session_state.user["_id"]
    
    # Auto-refresh every 10 seconds
    count = st_autorefresh(interval=10000, limit=None)
    
    st.title(f"Dashboard - {st.session_state.user['name']}")
    
    # Display Current Funds
    current_funds = users_col.find_one({"_id": user_id})["funds"]
    st.metric("Current Balance (₦)", current_funds)
    
    # Alert for low funds
    if current_funds <= 500:
        st.error("⚠ Your account funds are low! Please fund to continue using appliances.")
    
    # Fund Account
    st.subheader("Fund Account")
    fund_amount = st.number_input("Enter amount to fund (₦)", min_value=500, step=500)
    if st.button("Add Funds"):
        users_col.update_one({"_id": user_id}, {"$inc": {"funds": fund_amount}})
        st.success(f"Funds added! New balance: ₦{users_col.find_one({'_id': user_id})['funds']}")
    
    # Add Appliance
    st.subheader("Add Appliances")
    appliance_name = st.selectbox("Select Appliance", ["Refrigerator", "Air Conditioner", "Washing Machine", "Television", "Electric Cooker"])
    power_rating = st.number_input("Power Rating (Watts)", min_value=50, max_value=2000, step=50)
    if st.button("Add Appliance"):
        appliances_col.insert_one({
            "user_id": user_id,
            "name": appliance_name,
            "power_rating": power_rating,
            "status": "off",
            "created_at": datetime.now()
        })
        st.success(f"{appliance_name} added!")
    
    # Turn On/Off Appliances
    st.subheader("Control Appliances")
    user_appliances = list(appliances_col.find({"user_id": user_id}))
    appliance_options = [a["name"] for a in user_appliances]
    selected_appliances = st.multiselect("Select Appliances", appliance_options)
    if st.button("Turn On Selected"):
        for appliance in user_appliances:
            if appliance["name"] in selected_appliances:
                appliances_col.update_one({"_id": appliance["_id"]}, {"$set": {"status": "on"}})
        st.success("Appliances turned ON")
    
    if st.button("Turn Off Selected"):
        for appliance in user_appliances:
            if appliance["name"] in selected_appliances:
                appliances_col.update_one({"_id": appliance["_id"]}, {"$set": {"status": "off"}})
        st.success("Appliances turned OFF")
    
    # Auto Energy Simulation
    auto_simulate_energy(user_id)
    
    # AI Energy Prediction
    st.subheader("AI Energy Predictions (Next Reading kWh)")
    predictions = predict_next_usage(user_id)
    if predictions:
        st.write(predictions)
    else:
        st.write("Not enough data for predictions yet.")
    
    # Reports
    st.subheader("Reports")
    readings = list(readings_col.find({"user_id": user_id}))
    if readings:
        df = pd.DataFrame(readings)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.merge(pd.DataFrame(user_appliances), left_on="appliance_id", right_on="_id", how="left")
        report_df = df[["name", "timestamp", "power_consumption", "charged_amount"]]
        st.dataframe(report_df)
        st.bar_chart(report_df.groupby("name")["power_consumption"].sum())
    else:
        st.write("No energy consumption records yet.")
    
    # Total Household Consumption
    st.subheader("Total Household Energy Consumption")
    total_df = pd.DataFrame()
    for appliance in user_appliances:
        readings = list(readings_col.find({"appliance_id": appliance["_id"]}))
        if readings:
            df_appl = pd.DataFrame(readings)
            df_appl["timestamp"] = pd.to_datetime(df_appl["timestamp"])
            df_appl = df_appl.groupby("timestamp")["power_consumption"].sum().reset_index()
            df_appl.rename(columns={"power_consumption": appliance["name"]}, inplace=True)
            if total_df.empty:
                total_df = df_appl
            else:
                total_df = pd.merge(total_df, df_appl, on="timestamp", how="outer")
    if not total_df.empty:
        total_df = total_df.fillna(0)
        total_df["Total Consumption"] = total_df[list(total_df.columns[1:])].sum(axis=1)
        st.line_chart(total_df[["timestamp", "Total Consumption"]].set_index("timestamp"))

