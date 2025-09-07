import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
from pymongo import MongoClient
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.text import MIMEText

# ----------------------------
# MongoDB Atlas Setup
# ----------------------------
MONGO_URI = "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/smart_energy?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["smart_energy"]
users_col = db["users"]
appliances_col = db["appliances"]
readings_col = db["energy_readings"]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Smart Energy Dashboard", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #007BFF, #FFC107, #FF0000); color: white;}
div.stButton > button { background-color: black; color: white; height: 3em; width: 12em; border-radius: 8px; font-size:16px;}
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

def send_email(to_email, subject, message):
    try:
        sender_email = "youremail@gmail.com"  # your email
        sender_password = "your_app_password"
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
    except Exception as e:
        st.error(f"Email failed: {e}")

# ----------------------------
# LSTM Functions for AI Prediction
# ----------------------------
def prepare_lstm_data(readings, seq_length=5):
    X, y = [], []
    for i in range(len(readings) - seq_length):
        X.append(readings[i:i+seq_length])
        y.append(readings[i+seq_length])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def train_lstm(readings):
    if len(readings) < 10:
        return None
    X, y = prepare_lstm_data(readings)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)
    return model

def predict_next_consumption(model, last_seq):
    return model.predict(last_seq.reshape((1, last_seq.shape[0], 1)), verbose=0)[0][0]

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
    st_autorefresh(interval=10000, limit=None)
    
    st.title(f"Dashboard - {st.session_state.user['name']}")
    current_funds = users_col.find_one({"_id": user_id})["funds"]
    st.metric("Current Balance (₦)", current_funds)
    
    if current_funds <= 500:
        st.error("⚠ Your account funds are low! Please fund to continue using appliances.")
        send_email(st.session_state.user["email"], "Low Funds Alert", 
                   f"Your account balance is below ₦500. Current balance: ₦{current_funds}")
    
    # Fund Account
    st.subheader("Fund Account")
    fund_amount = st.number_input("Enter amount to fund (₦)", min_value=500, step=500)
    if st.button("Add Funds"):
        users_col.update_one({"_id": user_id}, {"$inc": {"funds": fund_amount}})
        st.success(f"Funds added! New balance: ₦{users_col.find_one({'_id': user_id})['funds']}")
    
    # Add Appliances
    st.subheader("Add Appliances")
    appliance_name = st.selectbox("Select Appliance", ["Refrigerator","Air Conditioner","Washing Machine","Television","Electric Cooker"])
    power_rating = st.number_input("Power Rating (Watts)", min_value=50, max_value=2000, step=50)
    if st.button("Add Appliance"):
        appliances_col.insert_one({"user_id": user_id,"name": appliance_name,"power_rating": power_rating,"status": "off","created_at": datetime.now()})
        st.success(f"{appliance_name} added!")
    
    # Control Appliances
    st.subheader("Control Appliances")
    user_appliances = list(appliances_col.find({"user_id": user_id}))
    appliance_options = [a["name"] for a in user_appliances]
    selected_appliances = st.multiselect("Select Appliances", appliance_options)
    
    for appliance in user_appliances:
        if appliance["name"] in selected_appliances and appliance["status"] != "on":
            appliances_col.update_one({"_id": appliance["_id"]}, {"$set": {"status": "on"}})
        elif appliance["name"] not in selected_appliances and appliance["status"] != "off":
            appliances_col.update_one({"_id": appliance["_id"]}, {"$set": {"status": "off"}})
    
    # Energy Simulation
    st.subheader("Predicted Appliance Usage & Alerts")
    for appliance in user_appliances:
        readings_data = [r["power_consumption"] for r in readings_col.find({"appliance_id": appliance["_id"]})]
        model = train_lstm(readings_data)
        if model is not None and len(readings_data) >= 5:
            last_seq = np.array(readings_data[-5:])
            predicted = predict_next_consumption(model, last_seq)
            cost_predicted = predicted * 100
            st.write(f"{appliance['name']} predicted next usage: {predicted:.3f} kWh, predicted cost: ₦{cost_predicted:.2f}")
            if current_funds < cost_predicted:
                st.warning(f"⚠ Funds may run out if {appliance['name']} remains ON!")
                send_email(st.session_state.user["email"],
                           f"Predictive Alert: {appliance['name']}",
                           f"Predicted next energy cost for {appliance['name']}: ₦{cost_predicted:.2f}. Current balance: ₦{current_funds:.2f}. Please fund your account!")
