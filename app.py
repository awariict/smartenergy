import streamlit as st
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import OperationFailure
from bson.objectid import ObjectId
import os
import bcrypt
import threading
import time
import datetime
import random
import uuid
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

MONGO_URI = st.secrets.get("MONGO_URI", "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/smart_energy?retryWrites=true&w=majority")
DB_NAME = st.secrets.get("DB_NAME", "smart_energy")
PRICE_PER_KWH = float(os.environ.get("PRICE_PER_KWH", "150.0"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "8"))
BORROW_AMOUNT = float(os.environ.get("BORROW_AMOUNT", "500.0"))

@st.cache_resource(ttl=600)
def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.server_info()
    except Exception as e:
        st.error(f"Cannot connect to MongoDB: {e}")
        st.stop()
    db = client[DB_NAME]
    try:
        db.transactions.create_index([("timestamp", -1)], background=True)
    except:
        pass
    return db

db = get_db()
users_col = db.users
meters_col = db.meters
appliances_col = db.appliances
transactions_col = db.transactions

# ===== APPLIANCE FUNCTIONS =====
def get_appliance_consumption(meter_id, days_back=90):
    try:
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=days_back)
        transactions = list(transactions_col.find({
            "type": "deduction",
            "timestamp": {"$gte": cutoff_date},
            "metadata.appliance_id": {"$exists": True}
        }).sort("timestamp", 1))
        
        appliance_data = {}
        for t in transactions:
            app_id = t.get("metadata", {}).get("appliance_id")
            if app_id:
                if app_id not in appliance_data:
                    appliance_data[app_id] = {"cost": 0.0, "kwh": 0.0, "count": 0}
                appliance_data[app_id]["cost"] += t.get("amount", 0.0)
                appliance_data[app_id]["kwh"] += t.get("amount", 0.0) / PRICE_PER_KWH if PRICE_PER_KWH != 0 else 0
                appliance_data[app_id]["count"] += 1
        
        apps = list(appliances_col.find({"meter_id": meter_id}))
        for app in apps:
            app_id = app.get("appliance_id")
            if app_id not in appliance_data:
                appliance_data[app_id] = {"cost": 0.0, "kwh": 0.0, "count": 0}
            appliance_data[app_id].update({
                "name": app.get("type"),
                "location": app.get("location"),
                "power_w": app.get("power_rating_w"),
                "is_on": app.get("is_on"),
                "total_accum": app.get("total_accum_kwh", 0.0)
            })
        
        return appliance_data
    except:
        return {}

def rank_appliances_by_consumption(appliance_data):
    return sorted(appliance_data.items(), key=lambda x: x[1].get("kwh", 0), reverse=True)

def get_appliance_forecasts(meter_id, appliance_id, periods=7):
    try:
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=30)
        transactions = list(transactions_col.find({
            "type": "deduction",
            "timestamp": {"$gte": cutoff_date},
            "metadata.appliance_id": appliance_id
        }).sort("timestamp", 1))
        
        if len(transactions) < 5:
            return None
        
        data = []
        for t in transactions:
            energy = t.get("amount", 0.0) / PRICE_PER_KWH if PRICE_PER_KWH != 0 else 0
            ts = t["timestamp"]
            data.append({"timestamp": ts, "energy": energy, "hour": ts.hour, "day_of_week": ts.weekday()})
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if len(df) < 5:
            return None
        
        df['date_numeric'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 86400
        df['lag_1'] = df['energy'].shift(1)
        df['rolling_mean_3'] = df['energy'].rolling(3).mean()
        df = df.dropna()
        
        if len(df) < 3:
            return None
        
        X = df[['date_numeric', 'hour', 'day_of_week', 'lag_1', 'rolling_mean_3']].values
        y = df['energy'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_date = df['timestamp'].max()
        last_numeric = df['date_numeric'].max()
        last_energy = df['energy'].values[-1]
        
        forecasts = []
        for i in range(1, periods + 1):
            future_date = last_date + datetime.timedelta(days=i)
            X_future = np.array([[last_numeric + i, future_date.hour, future_date.weekday(), last_energy, last_energy]])
            pred = max(0, model.predict(X_future)[0])
            forecasts.append({"date": future_date, "forecast_kwh": pred})
        
        return forecasts
    except:
        return None

def get_savings_recommendations(appliance_data, avg_daily_cost):
    recommendations = []
    sorted_apps = sorted(appliance_data.items(), key=lambda x: x[1].get("power_w", 0), reverse=True)
    
    for app_id, data in sorted_apps:
        if data.get("is_on") and data.get("power_w", 0) > 100:
            daily_cost = (data.get("power_w", 0) / 1000.0) * 24 * PRICE_PER_KWH
            savings = daily_cost / avg_daily_cost * 100 if avg_daily_cost > 0 else 0
            recommendations.append({
                "appliance_id": app_id,
                "name": data.get("name"),
                "location": data.get("location"),
                "power_w": data.get("power_w"),
                "est_daily_cost": daily_cost,
                "savings_percent": savings,
                "is_on": data.get("is_on")
            })
    
    return sorted(recommendations, key=lambda x: x["est_daily_cost"], reverse=True)

def auto_turn_off_high_consumption(meter_id, appliance_data, threshold_w=500):
    turned_off = []
    for app_id, data in appliance_data.items():
        if data.get("power_w", 0) >= threshold_w and data.get("is_on"):
            app = appliances_col.find_one({"appliance_id": app_id})
            if app:
                appliances_col.update_one({"_id": app["_id"]}, {"$set": {"is_on": False, "manual_control": False}})
                turned_off.append({"name": data.get("name"), "power": data.get("power_w")})
    return turned_off

# ===== FORECASTING FUNCTIONS =====
def prepare_consumption_data(user_id, days_back=90):
    try:
        cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=days_back)
        transactions = list(transactions_col.find({
            "user_id": user_id,
            "type": "deduction",
            "timestamp": {"$gte": cutoff_date}
        }).sort("timestamp", 1))
        
        if len(transactions) < 10:
            return None
        
        data = []
        for t in transactions:
            energy_kwh = t.get("amount", 0.0) / PRICE_PER_KWH if PRICE_PER_KWH != 0 else 0
            ts = t["timestamp"]
            data.append({
                "timestamp": ts, "energy": energy_kwh, "cost": t.get("amount", 0.0),
                "hour": ts.hour, "day": ts.day, "month": ts.month, "day_of_week": ts.weekday()
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    except:
        return None

def calculate_consumption_stats(df):
    if df is None or len(df) == 0:
        return None
    
    daily_consumption = df.groupby(df['timestamp'].dt.date)['energy'].sum()
    daily_costs = df.groupby(df['timestamp'].dt.date)['cost'].sum()
    
    return {
        "total_energy_kwh": df['energy'].sum(),
        "total_cost": df['cost'].sum(),
        "avg_daily_kwh": daily_consumption.mean(),
        "avg_daily_cost": daily_costs.mean(),
        "max_daily_kwh": daily_consumption.max(),
        "min_daily_kwh": daily_consumption.min(),
        "std_daily_kwh": daily_consumption.std(),
        "peak_hour": df.groupby('hour')['energy'].mean().idxmax(),
        "off_peak_hour": df.groupby('hour')['energy'].mean().idxmin(),
    }

def forecast_arima_style(df, periods=30):
    try:
        if df is None or len(df) < 20:
            return None, None
        
        df_copy = df.copy()
        df_copy['date_numeric'] = (df_copy['timestamp'] - df_copy['timestamp'].min()).dt.total_seconds() / 86400
        df_copy['lag_1'] = df_copy['energy'].shift(1)
        df_copy['lag_7'] = df_copy['energy'].shift(7)
        df_copy['rolling_mean_7'] = df_copy['energy'].rolling(7).mean()
        df_copy['rolling_std_7'] = df_copy['energy'].rolling(7).std()
        df_copy = df_copy.dropna()
        
        feature_cols = ['date_numeric', 'hour', 'day_of_week', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
        X = df_copy[feature_cols].values
        y = df_copy['energy'].values
        
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        lr_model.fit(X, y)
        rf_model.fit(X, y)
        
        last_date = df_copy['timestamp'].max()
        last_numeric = df_copy['date_numeric'].max()
        last_energy = df_copy['energy'].values[-1]
        
        forecasts_lr, forecasts_rf, forecast_timestamps = [], [], []
        
        for i in range(1, periods + 1):
            future_date = last_date + datetime.timedelta(days=i)
            X_future = np.array([[last_numeric + i, future_date.hour, future_date.weekday(), last_energy, last_energy, last_energy, 0.1]])
            
            pred_lr = lr_model.predict(X_future)[0]
            pred_rf = rf_model.predict(X_future)[0]
            forecast = max(0, pred_lr * 0.4 + pred_rf * 0.6)
            
            forecasts_lr.append(pred_lr)
            forecasts_rf.append(pred_rf)
            forecast_timestamps.append(future_date)
        
        forecast_df = pd.DataFrame({
            'timestamp': forecast_timestamps,
            'ensemble_forecast': [max(0, l * 0.4 + r * 0.6) for l, r in zip(forecasts_lr, forecasts_rf)]
        })
        
        return forecast_df, {'lr': lr_model, 'rf': rf_model}
    except:
        return None, None

def get_customer_standing(user, consumption_stats, forecast_df):
    current_funds = user.get("funds", 0.0)
    borrowed = user.get("borrowed", 0.0)
    monthly_forecast_cost = forecast_df['ensemble_forecast'].sum() * PRICE_PER_KWH if forecast_df is not None and len(forecast_df) > 0 else 0.0
    days_until_depleted = current_funds / consumption_stats['avg_daily_cost'] if consumption_stats and consumption_stats['avg_daily_cost'] > 0 else 999
    
    return {
        "current_funds": current_funds,
        "borrowed_amount": borrowed,
        "net_position": current_funds - borrowed,
        "monthly_forecast_cost": monthly_forecast_cost,
        "days_until_depletion": max(0, days_until_depleted),
        "is_solvent": (current_funds - borrowed) >= 0,
        "status": "SOLVENT" if (current_funds - borrowed) >= 0 else "INSOLVENT"
    }

def generate_weekly_forecast(forecast_df):
    if forecast_df is None or len(forecast_df) == 0:
        return None
    
    forecast_df_copy = forecast_df.copy()
    forecast_df_copy['timestamp'] = pd.to_datetime(forecast_df_copy['timestamp'])
    forecast_df_copy['week'] = forecast_df_copy['timestamp'].dt.isocalendar().week
    
    weekly = forecast_df_copy.groupby('week').agg({'ensemble_forecast': 'sum', 'timestamp': 'first'}).reset_index()
    weekly.columns = ['week', 'total_kwh', 'week_start']
    weekly['cost'] = weekly['total_kwh'] * PRICE_PER_KWH
    return weekly

def generate_monthly_forecast(forecast_df):
    if forecast_df is None or len(forecast_df) == 0:
        return None
    
    forecast_df_copy = forecast_df.copy()
    forecast_df_copy['timestamp'] = pd.to_datetime(forecast_df_copy['timestamp'])
    forecast_df_copy['month'] = forecast_df_copy['timestamp'].dt.to_period('M')
    
    monthly = forecast_df_copy.groupby('month').agg({'ensemble_forecast': 'sum'}).reset_index()
    monthly.columns = ['month', 'total_kwh']
    monthly['cost'] = monthly['total_kwh'] * PRICE_PER_KWH
    return monthly

def get_early_warning_alerts(standing, consumption_stats):
    alerts = []
    if standing['days_until_depletion'] < 7 and standing['current_funds'] > 0:
        alerts.append({"type": "WARNING", "message": f"⚠️ Low Balance: {standing['days_until_depletion']:.1f} days remaining"})
    if standing['current_funds'] <= 0 and standing['borrowed_amount'] <= 0:
        alerts.append({"type": "CRITICAL", "message": "🔴 CRITICAL: No funds. Service may disconnect."})
    if standing['borrowed_amount'] > 0:
        alerts.append({"type": "DEBT", "message": f"💳 Debt: You owe ₦{standing['borrowed_amount']:,.2f}"})
    return alerts

def forecast_fund_depletion_date(user, consumption_stats):
    current_funds = user.get("funds", 0.0)
    if not consumption_stats or consumption_stats['avg_daily_cost'] <= 0 or current_funds <= 0:
        return None
    days_remaining = current_funds / consumption_stats['avg_daily_cost']
    return {"depletion_date": datetime.datetime.utcnow() + datetime.timedelta(days=days_remaining), "days_remaining": days_remaining}

def get_historical_daily_consumption(df):
    if df is None or len(df) == 0:
        return None
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy['date'] = df_copy['timestamp'].dt.date
    return df_copy.groupby('date').agg({'energy': 'sum', 'cost': 'sum'}).reset_index()

def get_hourly_consumption_pattern(df):
    if df is None or len(df) == 0:
        return None
    return df.groupby('hour').agg({'energy': 'mean', 'cost': 'mean'}).reset_index()

# ===== UTILITIES =====
def gen_meter_id():
    return f"MTR-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

def gen_appliance_id(meter_id):
    return f"APL-{meter_id}-{uuid.uuid4().hex[:6]}"

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    except:
        return False

def user_has_debt(user):
    return user.get("borrowed", 0.0) > 0.0

def user_can_borrow(user):
    return user.get("borrowed", 0.0) == 0.0 and user.get("funds", 0.0) == 0.0

def user_has_funded_before(user):
    return transactions_col.find_one({"user_id": user["_id"], "type": "fund"}) is not None

def debt_repay(user):
    to_pay = min(user.get("funds", 0.0), user.get("borrowed", 0.0))
    if to_pay > 0:
        users_col.update_one({"_id": user["_id"]}, {"$set": {"funds": user.get("funds", 0.0) - to_pay, "borrowed": user.get("borrowed", 0.0) - to_pay}})

def turn_off_all_appliances(meter_id):
    appliances_col.update_many({"meter_id": meter_id}, {"$set": {"is_on": False, "manual_control": False}})

def can_withdraw(user):
    try:
        last_withdraw = transactions_col.find_one({"user_id": user["_id"], "type": "withdraw"}, sort=[("timestamp", -1)])
    except:
        last_withdraw = None
    
    now = datetime.datetime.utcnow()
    if last_withdraw and isinstance(last_withdraw.get("timestamp"), datetime.datetime):
        if (now - last_withdraw["timestamp"]).days < 30:
            return False
    return not user_has_debt(user) and user.get("funds", 0.0) > 0.0

def max_withdraw_amount(user):
    def sum_amount(user_id, ttype):
        try:
            res = list(transactions_col.aggregate([
                {"$match": {"user_id": user_id, "type": ttype, "amount": {"$exists": True}}},
                {"$group": {"_id": None, "total": {"$sum": "$amount"}}}
            ]))
            return float(res[0]["total"]) if res else 0.0
        except:
            return 0.0
    
    fund_total = sum_amount(user["_id"], "fund")
    debt_repaid = sum_amount(user["_id"], "debt_repay")
    already_withdrawn = sum_amount(user["_id"], "withdraw")
    available = max(0.0, fund_total - debt_repaid - already_withdrawn)
    return min(0.3 * available, user.get("funds", 0.0))

def create_user(full_name, email, phone, username, password, address, role="user"):
    if users_col.find_one({"email": email}):
        return False, "Email already registered."
    if users_col.find_one({"phone": phone}):
        return False, "Phone already registered."
    if users_col.find_one({"username": username}):
        return False, "Username already taken."
    if users_col.find_one({"address": address}):
        return False, "Address already registered."
    
    pwd_hash = hash_password(password)
    user_doc = {
        "email": email, "username": username, "full_name": full_name, "phone": phone,
        "password_hash": pwd_hash, "address": address, "funds": 0.0, "borrowed": 0.0,
        "registered_at": datetime.datetime.utcnow(), "role": role,
    }
    res = users_col.insert_one(user_doc)
    
    meter_id = gen_meter_id()
    meters_col.insert_one({
        "meter_id": meter_id, "user_id": res.inserted_id, "address": address,
        "status": "active", "total_energy_kwh": 0.0, "created_at": datetime.datetime.utcnow(),
    })
    users_col.update_one({"_id": res.inserted_id}, {"$set": {"meter_id": meter_id}})
    
    DEFAULT_APPLIANCES = [
        {"type": "refrigerator", "location": "kitchen", "power_rating_w": 150},
        {"type": "tv", "location": "parlour", "power_rating_w": 120},
        {"type": "fan", "location": "bedroom", "power_rating_w": 60},
        {"type": "computers", "location": "bedroom", "power_rating_w": 200},
        {"type": "phones", "location": "bedroom", "power_rating_w": 10},
        {"type": "washing_machine", "location": "kitchen", "power_rating_w": 500},
        {"type": "cooking_gas", "location": "kitchen", "power_rating_w": 1000},
    ]
    
    for ap in DEFAULT_APPLIANCES:
        appliances_col.insert_one({
            "appliance_id": gen_appliance_id(meter_id),
            "meter_id": meter_id,
            "type": ap["type"],
            "location": ap["location"],
            "power_rating_w": ap["power_rating_w"],
            "is_on": False,
            "manual_control": False,
            "session": {"started_at": None, "accum_kwh_session": 0.0},
            "total_accum_kwh": 0.0,
            "created_at": datetime.datetime.utcnow(),
        })
    
    return True, "User registered successfully."

def authenticate_user(username, password):
    user = users_col.find_one({"username": username})
    if not user:
        return None
    return user if check_password(password, user.get("password_hash") or b"") else None

def safe_find_transactions(filter_query=None, sort_field="timestamp", limit=None):
    q = filter_query or {}
    try:
        cursor = transactions_col.find(q)
        if sort_field:
            cursor = cursor.sort(sort_field, -1)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    except:
        return []

# ===== SIMULATION =====
def _simulate_meter(user_id, stop_event):
    while not stop_event.is_set():
        user = users_col.find_one({"_id": user_id})
        if not user:
            break
        meter_id = user.get("meter_id")
        now = datetime.datetime.utcnow()
        if user.get("funds", 0.0) <= 0.0:
            turn_off_all_appliances(meter_id)
            time.sleep(POLL_INTERVAL)
            continue
        
        for app in appliances_col.find({"meter_id": meter_id}):
            if app.get("manual_control", False):
                is_on = app.get("is_on", False)
            else:
                tod = now.hour
                is_on = random.random() < (0.35 if app["type"] == "tv" and 18 <= tod <= 23 else 0.9 if app["type"] == "refrigerator" else 0.4 if app["type"] == "fan" and 9 <= tod <= 21 else 0.08)
            
            if is_on and user.get("funds", 0.0) > 0.0:
                power_w = app.get("power_rating_w", 100)
                kwh = (power_w / 1000.0) * (POLL_INTERVAL / 3600.0)
                cost = kwh * PRICE_PER_KWH
                
                updated = users_col.find_one_and_update(
                    {"_id": user_id, "funds": {"$gte": cost}},
                    {"$inc": {"funds": -cost}},
                    return_document=ReturnDocument.BEFORE
                )
                
                if updated:
                    appliances_col.update_one({"_id": app["_id"]}, {"$inc": {"total_accum_kwh": kwh}})
                    balance_after = updated.get("funds", 0.0) - cost
                    transactions_col.insert_one({
                        "user_id": user_id, "type": "deduction", "amount": float(cost),
                        "balance_after": float(balance_after), "metadata": {"appliance_id": app.get("appliance_id")},
                        "timestamp": now,
                    })
                    meters_col.update_one({"meter_id": meter_id}, {"$inc": {"total_energy_kwh": kwh}})
                else:
                    turn_off_all_appliances(meter_id)
        
        time.sleep(POLL_INTERVAL)

def start_simulation_session(user_id):
    if "sim_stop" in st.session_state:
        return
    stop_event = threading.Event()
    sim_thread = threading.Thread(target=_simulate_meter, args=(user_id, stop_event), daemon=True)
    st.session_state["sim_stop"] = stop_event
    sim_thread.start()

def stop_simulation_session():
    if "sim_stop" in st.session_state:
        st.session_state["sim_stop"].set()
        st.session_state.pop("sim_stop", None)

# ===== UI =====
st.set_page_config(page_title="Smart Energy System", layout="wide")
st.markdown("""<style>
.stApp { background: linear-gradient(to right, #007BFF, #FFC107, #FF0000); }
section[data-testid="stSidebar"] { background: black !important; }
.card { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 8px; }
.alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 10px 0; }
.alert-danger { background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; margin: 10px 0; }
.alert-info { background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 12px; margin: 10px 0; }
.alert-success { background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; margin: 10px 0; }
</style>""", unsafe_allow_html=True)

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "Login"

page = st.session_state["page"]

USER_SIDEBAR = [("Dashboard", "Dashboard"), ("Fund", "Fund"), ("Borrow", "Borrow"), ("Usage", "Usage"), ("🔌 Appliances", "Appliances"), ("📊 Forecast", "Forecast"), ("Meter Details", "Meter Details"), ("User Info", "User Info"), ("Billing", "Billing"), ("Logout", "Logout")]

ADMIN_SIDEBAR = [("Admin Info", "Admin Info"), ("Manage Users", "Manage Users"), ("Transaction Log", "Transaction Log"), ("Debtors", "Debtors"), ("Admin Funding", "Admin Funding"), ("Logout", "Logout")]

user = None
if st.session_state.get("user_id"):
    user = users_col.find_one({"_id": st.session_state["user_id"]})
    sidebar_items = ADMIN_SIDEBAR if user and user.get("role") == "admin" else USER_SIDEBAR
    st.sidebar.markdown("<div style='display:flex; flex-direction:column; align-items:center; gap:12px; margin-top:20px;'>", unsafe_allow_html=True)
    for label, pageval in sidebar_items:
        if st.sidebar.button(label, key=f"sidebar_{label}", use_container_width=True):
            st.session_state["page"] = pageval
            st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.title("Intelligent Smart Energy System")
st.write("Manage your simulated prepaid meter, appliances, and funds.")

# ===== PAGES =====
if page == "Register":
    st.header("Register")
    with st.form("register_form"):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone number")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        address = st.text_input("House address")
        role = st.selectbox("Account Type", ["user", "admin"])
        submitted = st.form_submit_button("Register")
    if submitted:
        ok, msg = create_user(full_name, email, phone, username, password, address, role)
        if ok:
            st.success(msg)
            st.session_state["page"] = "Login"
            st.rerun()
        else:
            st.error(msg)
    if st.button("Already have an account? Login"):
        st.session_state["page"] = "Login"
        st.rerun()

elif page == "Login":
    st.header("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        user = authenticate_user(username, password)
        if user:
            st.success("Login successful!")
            st.session_state["user_id"] = user["_id"]
            start_simulation_session(user["_id"])
            st.session_state["page"] = "Dashboard" if user.get("role") != "admin" else "Admin Info"
            st.rerun()
        else:
            st.error("Invalid credentials.")
    if st.button("Don't have an account? Register"):
        st.session_state["page"] = "Register"
        st.rerun()

elif user and user.get("role") != "admin":
    if page == "Dashboard":
        st.header(f"Dashboard - {user.get('full_name')}")
        st.write(f"Funds: ₦{user.get('funds',0):,.2f}")
        st.write(f"Meter: {user.get('meter_id')}")
        if user_has_debt(user):
            st.warning(f"You owe ₦{user.get('borrowed',0):,.2f}")
        
        apps = list(appliances_col.find({"meter_id": user.get("meter_id")}))
        st.subheader("Appliances")
        total_on = sum(1 for a in apps if a.get("is_on"))
        st.write(f"**On:** {total_on}/{len(apps)}")
        for a in apps:
            cols = st.columns([2, 1, 1, 1])
            cols[0].write(f"{a['type'].title()} ({a['location']})")
            cols[1].write(f"{a.get('total_accum_kwh',0):.4f} kWh")
            disable = user.get("funds",0.0) <= 0.0 or user_has_debt(user)
            new_state = cols[3].checkbox("On", value=a.get('is_on', False), key=f"toggle_{a.get('_id')}", disabled=disable)
            if new_state != a.get('is_on', False) and not disable:
                appliances_col.update_one({"_id": a.get("_id")}, {"$set": {"is_on": bool(new_state), "manual_control": True}})
                st.rerun()

    elif page == "Fund":
        st.header("Fund Account")
        with st.form("fund_form"):
            amount = st.number_input("Top-up (₦)", min_value=0.0, step=100.0)
            submitted = st.form_submit_button("Add funds")
        if submitted and amount > 0:
            users_col.update_one({"_id": user["_id"]}, {"$inc": {"funds": float(amount)}})
            updated = users_col.find_one({"_id": user["_id"]})
            transactions_col.insert_one({
                "user_id": user["_id"], "type": "fund", "amount": float(amount),
                "balance_after": float(updated.get("funds",0.0)), "metadata": {"reason": "top_up"},
                "timestamp": datetime.datetime.utcnow()
            })
            debt_repay(updated)
            st.success(f"Added ₦{amount:,.2f}")
            st.rerun()

    elif page == "Borrow":
        st.header("Borrow Funds")
        if user_can_borrow(user) and user_has_funded_before(user):
            if st.button("Confirm Borrow"):
                users_col.update_one({"_id": user["_id"]}, {"$set": {"funds": BORROW_AMOUNT, "borrowed": BORROW_AMOUNT}})
                transactions_col.insert_one({
                    "user_id": user["_id"], "type": "borrow", "amount": BORROW_AMOUNT,
                    "balance_after": BORROW_AMOUNT, "metadata": {"reason": "borrow"},
                    "timestamp": datetime.datetime.utcnow()
                })
                st.success(f"Borrowed ₦{BORROW_AMOUNT:,.2f}!")
                st.rerun()
        else:
            st.info("Cannot borrow at this time.")

    elif page == "Usage":
        st.header("Appliance Usage")
        meter = meters_col.find_one({"meter_id": user.get("meter_id")})
        st.write(f"Total Meter kWh: {meter.get('total_energy_kwh',0.0):.4f}")
        
        apps = list(appliances_col.find({"meter_id": user.get("meter_id")}))
        total = sum(a.get("total_accum_kwh", 0.0) for a in apps)
        st.write(f"**Total Consumption:** {total:.4f} kWh")

    elif page == "Appliances":
        st.header("🔌 APPLIANCE ANALYSIS")
        st.markdown("---")
        
        appliance_data = get_appliance_consumption(user.get("meter_id"))
        consumption_df = prepare_consumption_data(user["_id"], days_back=90)
        consumption_stats = calculate_consumption_stats(consumption_df) if consumption_df is not None else None
        
        if not appliance_data:
            st.info("No data yet.")
        else:
            st.subheader("⚡ HIGHEST CONSUMING APPLIANCES")
            ranked_apps = rank_appliances_by_consumption(appliance_data)
            
            rank_data = []
            for idx, (app_id, data) in enumerate(ranked_apps[:10]):
                rank_data.append({
                    "Rank": idx + 1,
                    "Appliance": f"{data.get('name', 'Unknown').title()} ({data.get('location')})",
                    "kWh": f"{data.get('kwh', 0):.4f}",
                    "Cost": f"₦{data.get('cost', 0):,.2f}",
                    "Power": f"{data.get('power_w', 0)}W",
                    "Status": "🟢 ON" if data.get('is_on') else "🔴 OFF"
                })
            
            rank_df = pd.DataFrame(rank_data)
            st.dataframe(rank_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.subheader("💡 SAVINGS RECOMMENDATIONS")
            if consumption_stats:
                recommendations = get_savings_recommendations(appliance_data, consumption_stats['avg_daily_cost'])
                if recommendations:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        for idx, rec in enumerate(recommendations[:5]):
                            st.markdown(f"<div class='alert-info'><strong>{idx+1}. {rec['name'].title()} ({rec['location']})</strong><br>Power: {rec['power_w']}W | Cost: ₦{rec['est_daily_cost']:,.2f}</div>", unsafe_allow_html=True)
                    with col2:
                        if st.button("🚀 AUTO-OFF"):
                            turned_off = auto_turn_off_high_consumption(user.get("meter_id"), appliance_data)
                            if turned_off:
                                for app in turned_off:
                                    st.markdown(f"<div class='alert-success'>✓ {app['name'].title()}</div>", unsafe_allow_html=True)
                                st.rerun()

            st.markdown("---")
            st.subheader("📊 APPLIANCE FORECASTS")
            ranked_top = rank_appliances_by_consumption(appliance_data)[:3]
            if ranked_top:
                for app_id, data in ranked_top:
                    with st.expander(f"📌 {data.get('name', 'Unknown').title()}"):
                        forecast = get_appliance_forecasts(user.get("meter_id"), app_id, periods=7)
                        if forecast:
                            forecast_table = []
                            for f in forecast:
                                forecast_table.append({
                                    "Date": f['date'].strftime("%Y-%m-%d"),
                                    "kWh": f"{f['forecast_kwh']:.4f}",
                                    "Cost": f"₦{f['forecast_kwh'] * PRICE_PER_KWH:,.2f}"
                                })
                            st.dataframe(pd.DataFrame(forecast_table), use_container_width=True, hide_index=True)
                            total_7day = sum(f['forecast_kwh'] for f in forecast)
                            st.metric("7-Day Total", f"{total_7day:.2f} kWh | ₦{total_7day * PRICE_PER_KWH:,.2f}")

    elif page == "Forecast":
        st.header("🔮 ENERGY FORECAST")
        st.markdown("---")
        
        consumption_df = prepare_consumption_data(user["_id"], days_back=90)
        
        if consumption_df is None or len(consumption_df) < 10:
            st.warning("⚠️ Need more data.")
        else:
            consumption_stats = calculate_consumption_stats(consumption_df)
            forecast_df, models = forecast_arima_style(consumption_df, periods=30)
            
            if forecast_df is not None:
                standing = get_customer_standing(user, consumption_stats, forecast_df)
                alerts = get_early_warning_alerts(standing, consumption_stats)
                
                st.subheader("📋 STANDING")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Balance", f"₦{standing['current_funds']:,.2f}")
                col2.metric("Borrowed", f"₦{standing['borrowed_amount']:,.2f}")
                col3.metric("Net", f"₦{standing['net_position']:,.2f}")
                col4.metric("Status", f"{'🟢' if standing['is_solvent'] else '🔴'} {standing['status']}")
                
                if alerts:
                    for alert in alerts:
                        if alert['type'] == 'CRITICAL':
                            st.markdown(f"<div class='alert-danger'>{alert['message']}</div>", unsafe_allow_html=True)
                        elif alert['type'] == 'WARNING':
                            st.markdown(f"<div class='alert-warning'>{alert['message']}</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("📊 STATISTICS")
                s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                s_col1.metric("Total kWh", f"{consumption_stats['total_energy_kwh']:.2f}")
                s_col2.metric("Total Cost", f"₦{consumption_stats['total_cost']:,.2f}")
                s_col3.metric("Avg Daily", f"{consumption_stats['avg_daily_kwh']:.2f} kWh")
                s_col4.metric("Peak Hour", f"{consumption_stats['peak_hour']:02d}:00")
                
                st.markdown("---")
                st.subheader("🔔 7-DAY FORECAST")
                early = forecast_df.head(7)
                early_cost = early['ensemble_forecast'].sum() * PRICE_PER_KWH
                
                e_col1, e_col2, e_col3 = st.columns(3)
                e_col1.metric("Total", f"{early['ensemble_forecast'].sum():.2f} kWh")
                e_col2.metric("Cost", f"₦{early_cost:,.2f}")
                e_col3.metric("Daily Avg", f"{early['ensemble_forecast'].mean():.2f} kWh")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[f"Day {i+1}" for i in range(7)], y=early['ensemble_forecast'].values, marker_color='#28a745'))
                fig.update_layout(title="7-Day Forecast", height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📅 WEEKLY FORECAST")
                weekly = generate_weekly_forecast(forecast_df)
                if weekly is not None:
                    fig_w = go.Figure()
                    fig_w.add_trace(go.Bar(x=[f"Week {i+1}" for i in range(len(weekly))], y=weekly['total_kwh'], marker_color='#FF6B6B'))
                    fig_w.update_layout(title="Weekly", height=400, template='plotly_white')
                    st.plotly_chart(fig_w, use_container_width=True)
                
                st.markdown("---")
                st.subheader("🗓️ MONTHLY FORECAST")
                monthly = generate_monthly_forecast(forecast_df)
                if monthly is not None:
                    fig_m = go.Figure()
                    fig_m.add_trace(go.Bar(x=[str(m) for m in monthly['month']], y=monthly['total_kwh'], marker_color='#4C72B0'))
                    fig_m.update_layout(title="Monthly", height=400, template='plotly_white')
                    st.plotly_chart(fig_m, use_container_width=True)

    elif page == "Meter Details":
        meter = meters_col.find_one({"meter_id": user.get("meter_id")})
        st.header("Meter Details")
        st.write(f"**Name:** {user.get('full_name')}")
        st.write(f"**Meter ID:** {meter.get('meter_id')}")
        st.write(f"**Total kWh:** {meter.get('total_energy_kwh',0.0):.4f}")

    elif page == "User Info":
        st.header("User Information")
        st.write(f"**Name:** {user.get('full_name')}")
        st.write(f"**Email:** {user.get('email')}")
        st.write(f"**Phone:** {user.get('phone')}")

    elif page == "Billing":
        st.header("Billing")
        transactions = safe_find_transactions({"user_id": user["_id"]}, limit=100)
        if transactions:
            rows = []
            for t in transactions:
                ts = t.get("timestamp")
                if isinstance(ts, datetime.datetime):
                    ts_str = ts.strftime('%Y-%m-%d %H:%M')
                else:
                    ts_str = str(ts)
                rows.append({
                    "Date": ts_str,
                    "Type": t.get("type"),
                    "Amount": f"₦{t.get('amount', 0.0):,.2f}",
                    "Balance": f"₦{t.get('balance_after', 0.0):,.2f}"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif page == "Logout":
        stop_simulation_session()
        st.session_state["user_id"] = None
        st.session_state["page"] = "Login"
        st.success("Logged out")
        st.rerun()

elif user and user.get("role") == "admin":
    if page == "Admin Info":
        st.header("Admin")
        st.write(f"**Name:** {user.get('full_name')}")
        st.write(f"**Email:** {user.get('email')}")

    elif page == "Manage Users":
        st.header("Manage Users")
        for u in users_col.find({"role": "user"}):
            cols = st.columns([3, 1])
            cols[0].write(f"{u.get('full_name')} ({u.get('username')})")
            if cols[1].button("Delete", key=f"del_{u.get('_id')}"):
                users_col.delete_one({"_id": u.get('_id')})
                st.success("Deleted")
                st.rerun()

    elif page == "Transaction Log":
        st.header("Transaction Log")
        txs = safe_find_transactions({}, limit=500)
        if txs:
            rows = []
            for t in txs:
                ts = t.get("timestamp")
                ts_str = ts.strftime('%Y-%m-%d %H:%M') if isinstance(ts, datetime.datetime) else str(ts)
                rows.append({
                    "Date": ts_str,
                    "Type": t.get("type"),
                    "Amount": f"₦{t.get('amount', 0.0):,.2f}"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif page == "Debtors":
        st.header("Debtors")
        for d in users_col.find({"role": "user", "borrowed": {"$gt": 0.0}}):
            st.write(f"**{d.get('full_name')}** - Owes: ₦{d.get('borrowed',0):,.2f}")

    elif page == "Admin Funding":
        st.header("Fund Users")
        for u in users_col.find({"role": "user"}):
            cols = st.columns([2, 1, 1])
            cols[0].write(f"{u.get('full_name')} - ₦{u.get('funds',0):,.2f}")
            amount = cols[1].number_input("Amount", min_value=0.0, step=100.0, key=f"amt_{u.get('_id')}")
            if cols[2].button("Fund", key=f"btn_{u.get('_id')}"):
                users_col.update_one({"_id": u["_id"]}, {"$inc": {"funds": amount}})
                st.success(f"Funded ₦{amount:,.2f}")
                st.rerun()

    elif page == "Logout":
        stop_simulation_session()
        st.session_state["user_id"] = None
        st.session_state["page"] = "Login"
        st.success("Logged out")
        st.rerun()

st.markdown("---")
st.write("Developed by: Happiness Sunday Eyeh.")
