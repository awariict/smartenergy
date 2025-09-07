import streamlit as st
import numpy as np
import hashlib
from datetime import datetime
from pymongo import MongoClient
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.text import MIMEText
import pandas as pd

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
.stApp { background: linear-gradient(to right, #007BFF, #FFC107, #FF0000); color: green;}
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
        sender_email = "youremail@gmail.com"  # Replace
        sender_password = "your_app_password" # Replace
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
# Energy Logging
# ----------------------------
def log_energy_usage(user_id, appliance):
    start_time = appliance.get("start_time")
    if appliance["status"] == "on" and not start_time:
        appliances_col.update_one({"_id": appliance["_id"]}, {"$set": {"start_time": datetime.now()}})
    elif appliance["status"] == "off" and start_time:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        consumption = (appliance["power_rating"] / 1000) * duration
        cost = consumption * 100
        readings_col.insert_one({
            "user_id": user_id,
            "appliance": appliance["name"],
            "consumption": consumption,
            "cost": cost,
            "start_time": start_time,
            "end_time": end_time
        })
        appliances_col.update_one({"_id": appliance["_id"]}, {"$unset": {"start_time": ""}})

# ----------------------------
# Session State
# ----------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_option" not in st.session_state:
    st.session_state.auth_option = "Login"

# ----------------------------
# Authentication
# ----------------------------
if st.session_state.user is None:
    st.title("Login / Register")
    st.session_state.auth_option = st.selectbox("Choose Action", ["Login", "Register"], index=0)

    if st.session_state.auth_option == "Register":
        with st.form("register_form", clear_on_submit=True):
            name = st.text_input("Full Name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            username = st.text_input("Username", key="reg_username")
            password = st.text_input("Password", type="password", key="reg_password")
            address = st.text_input("Address", key="reg_address")
            register_clicked = st.form_submit_button(label="Register")
            if register_clicked:
                if not (name and email and username and password and address):
                    st.warning("Please fill in all fields.")
                else:
                    message = register_user(name, email, username, password, address)
                    if "successfully" in message:
                        st.success(message)
                    else:
                        st.error(message)

    elif st.session_state.auth_option == "Login":
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            login_clicked = st.form_submit_button(label="Login")
            if login_clicked:
                if not (username and password):
                    st.warning("Please enter username and password.")
                else:
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

    # Current Funds
    current_funds = users_col.find_one({"_id": user_id})["funds"]
    st.metric("Current Balance (₦)", current_funds)
    if current_funds <= 500:
        st.error("⚠ Your account funds are low!")
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
        appliances_col.insert_one({
            "user_id": user_id,
            "name": appliance_name,
            "power_rating": power_rating,
            "status": "off",
            "created_at": datetime.now()
        })
        st.success(f"{appliance_name} added!")

    # Control Appliances
    st.subheader("Control Appliances")
    user_appliances = list(appliances_col.find({"user_id": user_id}))
    appliance_options = [a["name"] for a in user_appliances]
    selected_appliances = st.multiselect("Select Appliances to Turn ON", appliance_options)

    for appliance in user_appliances:
        appliance_id = appliance["_id"]
        current_status = appliance.get("status", "off")

        if appliance["name"] in selected_appliances and current_status != "on":
            appliances_col.update_one({"_id": appliance_id}, {"$set": {"status": "on"}})
        elif appliance["name"] not in selected_appliances and current_status != "off":
            appliances_col.update_one({"_id": appliance_id}, {"$set": {"status": "off"}})

        # Log usage
        updated_appliance = appliances_col.find_one({"_id": appliance_id})
        log_energy_usage(user_id, updated_appliance)

    # Show appliance status
    for appliance in appliances_col.find({"user_id": user_id}):
        st.write(f"{appliance['name']}: {appliance.get('status','off').upper()}")

    # ----------------------------
    # Reports & Graphs
    # ----------------------------
    st.subheader("Energy Consumption Report")
    data = list(readings_col.find({"user_id": user_id}))
    if data:
        df = pd.DataFrame(data)
        df["appliance"] = df["appliance"].astype(str)

        # Line Chart
        st.line_chart(df.set_index("end_time")["consumption"])

        # Bar Chart (ranking)
        st.bar_chart(df.groupby("appliance")["consumption"].sum())

        # Appliance Ranking Table
        ranked = df.groupby("appliance")["consumption"].sum().sort_values(ascending=False)
        st.table(ranked)

        # Download Button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Report (CSV)", csv, "energy_report.csv", "text/csv")
    else:
        st.info("No usage data yet. Turn appliances ON/OFF to generate logs.")
