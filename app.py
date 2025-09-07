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
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #007BFF, #FFC107, #FF0000);
    color: green;
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

# ----------------------------
# Energy Simulation & Prediction
# ----------------------------
def simulate_energy(user_id):
    appliances = list(appliances_col.find({"user_id": user_id, "status": "on"}))
    user = users_col.find_one({"_id": user_id})
    for appliance in appliances:
        power = appliance.get("power_rating", 200)
        usage = round(np.random.uniform(0.05, 0.2) * (power/1000), 3)  # realistic usage kWh
        cost = round(usage * 100, 2)
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
        if len(readings) >= 5:
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
# Authentication
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
                if "successfully" in message:
                    st.success(message)
                else:
                    st.error(message)
    
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
    
    # Auto-refresh
    st_autorefresh(interval=15000, limit=None)  # 15s
    
    st.title(f"Dashboard - {st.session_state.user['name']}")
    
    # Display Current Funds
    current_funds = users_col.find_one({"_id": user_id})["funds"]
    st.metric("Current Balance (₦)", current_funds)
    
    if current_funds <= 500:
        st.error("⚠ Your account funds are low! Please fund to continue using appliances.")
    
    # Fund Account
