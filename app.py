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
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ---------------------- Configuration ----------------------
MONGO_URI = st.secrets.get(
    "MONGO_URI",
    "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/smart_energy?retryWrites=true&w=majority"
)
DB_NAME = st.secrets.get("DB_NAME", "smart_energy")
PRICE_PER_KWH = float(os.environ.get("PRICE_PER_KWH", "150.0"))  # Naira per kWh
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "8"))  # seconds
BORROW_AMOUNT = float(os.environ.get("BORROW_AMOUNT", "500.0"))  # Naira allowed to borrow

# ---------------------- Database ----------------------
@st.cache_resource(ttl=600)
def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.server_info()  # Forces a call to check connection
    except Exception as e:
        st.error(f"Cannot connect to MongoDB: {e}")
        st.stop()
    db = client[DB_NAME]
    try:
        db.transactions.create_index([("timestamp", -1)], background=True)
    except Exception as e:
        st.warning(f"Could not create index on transactions.timestamp: {e}")
    return db

db = get_db()
users_col = db.users
meters_col = db.meters
appliances_col = db.appliances
transactions_col = db.transactions

# ---------------------- Professional Forecasting Functions ----------------------
def prepare_consumption_data(user_id, days_back=90):
    """Prepare comprehensive consumption data for forecasting."""
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
                "timestamp": ts,
                "energy": energy_kwh,
                "cost": t.get("amount", 0.0),
                "hour": ts.hour,
                "day": ts.day,
                "month": ts.month,
                "day_of_week": ts.weekday(),
                "week": ts.isocalendar()[1]
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        st.warning(f"Error preparing consumption data: {e}")
        return None

def calculate_consumption_stats(df):
    """Calculate comprehensive consumption statistics."""
    if df is None or len(df) == 0:
        return None
    
    stats_dict = {
        "total_energy_kwh": df['energy'].sum(),
        "total_cost": df['cost'].sum(),
        "avg_daily_kwh": df.groupby(df['timestamp'].dt.date)['energy'].sum().mean(),
        "avg_daily_cost": df.groupby(df['timestamp'].dt.date)['cost'].sum().mean(),
        "max_daily_kwh": df.groupby(df['timestamp'].dt.date)['energy'].sum().max(),
        "min_daily_kwh": df.groupby(df['timestamp'].dt.date)['energy'].sum().min(),
        "std_daily_kwh": df.groupby(df['timestamp'].dt.date)['energy'].sum().std(),
        "peak_hour": df.groupby('hour')['energy'].mean().idxmax(),
        "off_peak_hour": df.groupby('hour')['energy'].mean().idxmin(),
        "most_active_day": df['day_of_week'].mode()[0] if len(df) > 0 else 0,
    }
    return stats_dict

def forecast_arima_style(df, periods=30):
    """Advanced forecasting using ensemble approach."""
    try:
        if df is None or len(df) < 20:
            return None, None
        
        # Prepare features
        df_copy = df.copy()
        df_copy['date_numeric'] = (df_copy['timestamp'] - df_copy['timestamp'].min()).dt.total_seconds() / 86400
        
        # Calculate lagged features
        df_copy['lag_1'] = df_copy['energy'].shift(1)
        df_copy['lag_7'] = df_copy['energy'].shift(7)
        df_copy['rolling_mean_7'] = df_copy['energy'].rolling(7).mean()
        df_copy['rolling_std_7'] = df_copy['energy'].rolling(7).std()
        
        df_copy = df_copy.dropna()
        
        # Features
        feature_cols = ['date_numeric', 'hour', 'day_of_week', 'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
        X = df_copy[feature_cols].values
        y = df_copy['energy'].values
        
        # Train ensemble models
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        lr_model.fit(X, y)
        rf_model.fit(X, y)
        
        # Generate forecasts
        last_date = df_copy['timestamp'].max()
        last_numeric = df_copy['date_numeric'].max()
        last_energy = df_copy['energy'].values[-1]
        
        forecasts_lr = []
        forecasts_rf = []
        forecast_timestamps = []
        
        for i in range(1, periods + 1):
            future_date = last_date + datetime.timedelta(days=i)
            future_numeric = last_numeric + i
            future_hour = future_date.hour
            future_dow = future_date.weekday()
            
            X_future = np.array([[future_numeric, future_hour, future_dow, last_energy, last_energy, last_energy, 0.1]])
            
            pred_lr = lr_model.predict(X_future)[0]
            pred_rf = rf_model.predict(X_future)[0]
            
            # Ensemble average
            forecast = (pred_lr * 0.4 + pred_rf * 0.6)
            forecast = max(0, forecast)
            
            forecasts_lr.append(pred_lr)
            forecasts_rf.append(pred_rf)
            forecast_timestamps.append(future_date)
        
        forecast_df = pd.DataFrame({
            'timestamp': forecast_timestamps,
            'forecast': forecasts_lr,
            'forecast_rf': forecasts_rf,
            'ensemble_forecast': [(l * 0.4 + r * 0.6) for l, r in zip(forecasts_lr, forecasts_rf)]
        })
        
        return forecast_df, {'lr': lr_model, 'rf': rf_model}
    except Exception as e:
        st.warning(f"Forecasting error: {e}")
        return None, None

def get_customer_standing(user, consumption_stats, forecast_df):
    """Calculate comprehensive customer standing."""
    current_funds = user.get("funds", 0.0)
    borrowed = user.get("borrowed", 0.0)
    
    if forecast_df is not None and len(forecast_df) > 0:
        monthly_forecast_cost = forecast_df['ensemble_forecast'].sum() * PRICE_PER_KWH
    else:
        monthly_forecast_cost = 0.0
    
    if consumption_stats and consumption_stats['avg_daily_cost'] > 0:
        days_until_depleted = current_funds / consumption_stats['avg_daily_cost'] if consumption_stats['avg_daily_cost'] > 0 else 999
    else:
        days_until_depleted = 999
    
    standing = {
        "current_funds": current_funds,
        "borrowed_amount": borrowed,
        "net_position": current_funds - borrowed,
        "monthly_forecast_cost": monthly_forecast_cost,
        "days_until_depletion": max(0, days_until_depleted),
        "is_solvent": (current_funds - borrowed) >= 0,
        "status": "SOLVENT" if (current_funds - borrowed) >= 0 else "INSOLVENT"
    }
    
    return standing

def generate_weekly_forecast(forecast_df):
    """Generate weekly forecast from daily forecast."""
    if forecast_df is None or len(forecast_df) == 0:
        return None
    
    forecast_df_copy = forecast_df.copy()
    forecast_df_copy['timestamp'] = pd.to_datetime(forecast_df_copy['timestamp'])
    forecast_df_copy['week'] = forecast_df_copy['timestamp'].dt.isocalendar().week
    
    weekly = forecast_df_copy.groupby('week').agg({
        'ensemble_forecast': 'sum',
        'timestamp': 'first'
    }).reset_index()
    
    weekly.columns = ['week', 'total_kwh', 'week_start']
    weekly['cost'] = weekly['total_kwh'] * PRICE_PER_KWH
    
    return weekly

def generate_monthly_forecast(forecast_df):
    """Generate monthly forecast from daily forecast."""
    if forecast_df is None or len(forecast_df) == 0:
        return None
    
    forecast_df_copy = forecast_df.copy()
    forecast_df_copy['timestamp'] = pd.to_datetime(forecast_df_copy['timestamp'])
    forecast_df_copy['month'] = forecast_df_copy['timestamp'].dt.to_period('M')
    
    monthly = forecast_df_copy.groupby('month').agg({
        'ensemble_forecast': 'sum'
    }).reset_index()
    
    monthly.columns = ['month', 'total_kwh']
    monthly['cost'] = monthly['total_kwh'] * PRICE_PER_KWH
    
    return monthly

def get_early_warning_alerts(standing, consumption_stats):
    """Generate early warning alerts."""
    alerts = []
    
    if standing['days_until_depletion'] < 7 and standing['current_funds'] > 0:
        alerts.append({
            "type": "WARNING",
            "message": f"⚠️ Low Balance: Your funds will deplete in {standing['days_until_depletion']:.1f} days at current usage rate."
        })
    
    if standing['current_funds'] <= 0 and standing['borrowed_amount'] <= 0:
        alerts.append({
            "type": "CRITICAL",
            "message": "🔴 CRITICAL: No funds available. Services may be disconnected soon."
        })
    
    if standing['borrowed_amount'] > 0:
        alerts.append({
            "type": "DEBT",
            "message": f"💳 Outstanding Debt: You owe ₦{standing['borrowed_amount']:,.2f}"
        })
    
    if consumption_stats and consumption_stats['std_daily_kwh'] > consumption_stats['avg_daily_kwh'] * 0.5:
        alerts.append({
            "type": "INFO",
            "message": "📊 High Usage Variability: Your consumption patterns are highly variable. Monitor closely."
        })
    
    return alerts

def forecast_fund_depletion_date(user, consumption_stats):
    """Forecast exact date when funds will deplete."""
    current_funds = user.get("funds", 0.0)
    if not consumption_stats or consumption_stats['avg_daily_cost'] <= 0 or current_funds <= 0:
        return None
    
    days_remaining = current_funds / consumption_stats['avg_daily_cost']
    depletion_date = datetime.datetime.utcnow() + datetime.timedelta(days=days_remaining)
    
    return {
        "depletion_date": depletion_date,
        "days_remaining": days_remaining
    }

def get_historical_daily_consumption(df):
    """Get historical daily consumption."""
    if df is None or len(df) == 0:
        return None
    
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy['date'] = df_copy['timestamp'].dt.date
    
    daily = df_copy.groupby('date').agg({
        'energy': 'sum',
        'cost': 'sum'
    }).reset_index()
    
    return daily

def get_hourly_consumption_pattern(df):
    """Get hourly consumption patterns."""
    if df is None or len(df) == 0:
        return None
    
    hourly = df.groupby('hour').agg({
        'energy': 'mean',
        'cost': 'mean'
    }).reset_index()
    
    return hourly

# ---------------------- Utilities ----------------------
def gen_meter_id():
    return f"MTR-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

def gen_appliance_id(meter_id):
    return f"APL-{meter_id}-{uuid.uuid4().hex[:6]}"

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    except Exception:
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
        new_funds = user.get("funds", 0.0) - to_pay
        new_borrow = user.get("borrowed", 0.0) - to_pay
        users_col.update_one({"_id": user["_id"]}, {"$set": {"funds": new_funds, "borrowed": new_borrow}})
        transactions_col.insert_one({
            "user_id": user["_id"],
            "type": "debt_repay",
            "amount": float(to_pay),
            "balance_after": float(new_funds),
            "metadata": {"reason": "debt_repay"},
            "timestamp": datetime.datetime.utcnow()
        })

def turn_off_all_appliances(meter_id):
    appliances_col.update_many({"meter_id": meter_id}, {"$set": {"is_on": False, "manual_control": False}})

def can_withdraw(user):
    try:
        last_withdraw = transactions_col.find_one(
            {"user_id": user["_id"], "type": "withdraw"},
            sort=[("timestamp", -1)]
        )
    except OperationFailure:
        st.warning("Warning reading last withdraw: falling back to safe query.")
        last_withdraw = transactions_col.find_one(
            {"user_id": user["_id"], "type": "withdraw", "timestamp": {"$exists": True}},
            sort=[("timestamp", -1)]
        )
    now = datetime.datetime.utcnow()
    if last_withdraw and isinstance(last_withdraw.get("timestamp"), datetime.datetime):
        diff = (now - last_withdraw["timestamp"]).days
        if diff < 30:
            return False
    return not user_has_debt(user) and user.get("funds", 0.0) > 0.0

def max_withdraw_amount(user):
    def sum_amount(user_id, ttype):
        pipeline = [
            {"$match": {"user_id": user_id, "type": ttype, "amount": {"$exists": True}}},
            {"$group": {"_id": None, "total": {"$sum": "$amount"}}}
        ]
        try:
            res = list(transactions_col.aggregate(pipeline))
            return float(res[0]["total"]) if res else 0.0
        except OperationFailure:
            docs = transactions_col.find({"user_id": user_id, "type": ttype}).limit(10000)
            return float(sum(d.get("amount", 0.0) for d in docs))

    fund_total = sum_amount(user["_id"], "fund")
    debt_repaid = sum_amount(user["_id"], "debt_repay")
    already_withdrawn = sum_amount(user["_id"], "withdraw")
    available = max(0.0, fund_total - debt_repaid - already_withdrawn)
    max_allowed = 0.3 * available
    return min(max_allowed, user.get("funds", 0.0))

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
        "email": email,
        "username": username,
        "full_name": full_name,
        "phone": phone,
        "password_hash": pwd_hash,
        "address": address,
        "funds": 0.0,
        "borrowed": 0.0,
        "registered_at": datetime.datetime.utcnow(),
        "role": role,
    }
    res = users_col.insert_one(user_doc)
    meter_id = gen_meter_id()
    meter_doc = {
        "meter_id": meter_id,
        "user_id": res.inserted_id,
        "address": address,
        "status": "active",
        "total_energy_kwh": 0.0,
        "created_at": datetime.datetime.utcnow(),
    }
    meters_col.insert_one(meter_doc)
    users_col.update_one({"_id": res.inserted_id}, {"$set": {"meter_id": meter_id}})
    for ap in DEFAULT_APPLIANCES:
        app_doc = {
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
        }
        appliances_col.insert_one(app_doc)
    return True, "User registered successfully."

def authenticate_user(username, password):
    user = users_col.find_one({"username": username})
    if not user:
        return None
    if check_password(password, user.get("password_hash") or b""):
        return user
    return None

def safe_find_transactions(filter_query=None, sort_field="timestamp", limit=None, projection=None):
    q = filter_query or {}
    try:
        cursor = transactions_col.find(q, projection=projection)
        if sort_field:
            cursor = cursor.sort(sort_field, -1)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    except OperationFailure:
        st.warning("Database operation failed when reading transactions; returning a safe limited result.")
        safe_q = dict(q)
        safe_q.update({"timestamp": {"$exists": True}})
        try:
            cursor = transactions_col.find(safe_q, projection=projection).sort(sort_field, -1).limit(limit or 1000)
            return list(cursor)
        except Exception as e2:
            st.error(f"Failed to read transactions safely: {e2}")
            return []

# ---------------------- Simulation ----------------------
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
        appliances = list(appliances_col.find({"meter_id": meter_id}))
        for app in appliances:
            if app.get("manual_control", False):
                is_on = app.get("is_on", False)
            else:
                tod = now.hour
                base = 0.08
                if app["type"] == "tv":
                    base = 0.35 if 18 <= tod <= 23 else 0.08
                elif app["type"] == "refrigerator":
                    base = 0.9
                elif app["type"] == "fan":
                    base = 0.4 if 9 <= tod <= 21 else 0.15
                elif app["type"] == "computers":
                    base = 0.2 if 8 <= tod <= 22 else 0.05
                elif app["type"] == "phones":
                    base = 0.5
                elif app["type"] == "washing_machine":
                    base = 0.05
                elif app["type"] == "cooking_gas":
                    base = 0.12
                is_on = random.random() < base
            prev_on = app.get("is_on", False)
            if is_on and not prev_on:
                appliances_col.update_one({"_id": app["_id"]}, {"$set": {"is_on": True, "session.started_at": now}})
            elif not is_on and prev_on and not app.get("manual_control", False):
                appliances_col.update_one({"_id": app["_id"]}, {"$set": {"is_on": False}})
            if is_on and user.get("funds", 0.0) > 0.0:
                power_w = app.get("power_rating_w", 100)
                kwh = (power_w / 1000.0) * (POLL_INTERVAL / 3600.0)
                appliances_col.update_one({"_id": app["_id"]}, {"$inc": {"total_accum_kwh": kwh, "session.accum_kwh_session": kwh}})
                cost = kwh * PRICE_PER_KWH
                updated = users_col.find_one_and_update(
                    {"_id": user_id, "funds": {"$gte": cost}},
                    {"$inc": {"funds": -cost}},
                    return_document=ReturnDocument.BEFORE
                )
                if updated:
                    balance_after = (updated.get("funds", 0.0) - cost)
                    transactions_col.insert_one({
                        "user_id": user_id,
                        "type": "deduction",
                        "amount": float(cost),
                        "balance_after": float(balance_after),
                        "metadata": {"appliance_id": app.get("appliance_id")},
                        "timestamp": now,
                    })
                    meters_col.update_one({"meter_id": meter_id}, {"$inc": {"total_energy_kwh": kwh}})
                else:
                    appliances_col.update_one({"_id": app["_id"]}, {"$set": {"is_on": False}})
                    users_col.update_one({"_id": user_id}, {"$set": {"last_notification": "insufficient_funds"}})
                    turn_off_all_appliances(meter_id)
        for _ in range(POLL_INTERVAL):
            if stop_event.is_set():
                break
            time.sleep(1)

def start_simulation_session(user_id):
    if "sim_stop" in st.session_state and st.session_state.get("sim_stop"):
        return
    stop_event = threading.Event()
    sim_thread = threading.Thread(target=_simulate_meter, args=(user_id, stop_event), daemon=True)
    st.session_state["sim_stop"] = stop_event
    st.session_state["sim_thread"] = sim_thread
    sim_thread.start()

def stop_simulation_session():
    ev = st.session_state.get("sim_stop")
    if ev:
        ev.set()
    st.session_state.pop("sim_stop", None)
    st.session_state.pop("sim_thread", None)

# ---------------------- Default Appliances ----------------------
DEFAULT_APPLIANCES = [
    {"type": "refrigerator", "location": "kitchen", "power_rating_w": 150},
    {"type": "tv", "location": "parlour", "power_rating_w": 120},
    {"type": "fan", "location": "bedroom", "power_rating_w": 60},
    {"type": "computers", "location": "bedroom", "power_rating_w": 200},
    {"type": "phones", "location": "bedroom", "power_rating_w": 10},
    {"type": "washing_machine", "location": "kitchen", "power_rating_w": 500},
    {"type": "cooking_gas", "location": "kitchen", "power_rating_w": 1000},
]

# ---------------------- UI ----------------------
st.set_page_config(page_title="Smart Energy System", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #007BFF, #FFC107, #FF0000); }
section[data-testid="stSidebar"] { background: black !important; }
.sidebar-content { display: flex; flex-direction: column; align-items: center; gap: 12px; margin-top: 20px; }
[data-testid="stSidebar"] button[kind="secondary"], .sidebar-btn {
    width: 180px !important;
    min-width: 180px !important;
    max-width: 180px !important;
    height: 44px !important;
    min-height: 44px !important;
    max-height: 44px !important;
    background-color: white !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 auto 8px auto !important;
    font-size: 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:active,
[data-testid="stSidebar"] button[kind="secondary"]:focus {
    background: #007BFF !important;
    color: white !important;
}
.big-font { font-size:20px !important; }
.card { background: white;color: black !important; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 8px; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
.alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; border-radius: 4px; margin: 10px 0; }
.alert-danger { background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; border-radius: 4px; margin: 10px 0; }
.alert-info { background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 12px; border-radius: 4px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "Login"
page = st.session_state["page"]

USER_SIDEBAR = [
    ("Dashboard", "Dashboard"),
    ("Fund", "Fund"),
    ("Borrow", "Borrow"),
    ("Usage", "Usage"),
    ("📊 Forecast", "Forecast"),
    ("Meter Details", "Meter Details"),
    ("User Info", "User Info"),
    ("Billing", "Billing"),
    ("Logout", "Logout")
]
ADMIN_SIDEBAR = [
    ("Admin Info", "Admin Info"),
    ("Manage Users", "Manage Users"),
    ("Transaction Log", "Transaction Log"),
    ("Debtors", "Debtors"),
    ("Admin Funding", "Admin Funding"),
    ("Logout", "Logout")
]

user = None
if st.session_state.get("user_id"):
    user = users_col.find_one({"_id": st.session_state["user_id"]})
    sidebar_items = ADMIN_SIDEBAR if user and user.get("role") == "admin" else USER_SIDEBAR
    st.sidebar.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    for label, pageval in sidebar_items:
        if st.sidebar.button(label, key=f"sidebar_{label}"):
            st.session_state["page"] = pageval
            st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.title("Intelligent Smart Energy System")
st.write("Manage your simulated prepaid meter, appliances, and funds.")

# ---------------------- Register ----------------------
if page == "Register":
    st.header("Register")
    with st.form("register_form"):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone number")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        address = st.text_input("House address")
        role = st.selectbox("Account Type", ["user", "admin"], help="Select 'admin' to register as an admin")
        submitted = st.form_submit_button("Register")
    if submitted:
        ok, msg = create_user(full_name, email, phone, username, password, address, role)
        if ok:
            st.success(msg)
            st.info("You can now login using your username and password.")
            st.session_state["page"] = "Login"
            st.rerun()
        else:
            st.error(msg)
    if st.button("Already have an account? Login", key="login_from_register"):
        st.session_state["page"] = "Login"
        st.rerun()

# ---------------------- Login ----------------------
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
    if st.button("Don't have an account? Register", key="register_from_login"):
        st.session_state["page"] = "Register"
        st.rerun()

# ---------------------- USER VIEWS ----------------------
if user and user.get("role") != "admin":

    if page == "Dashboard":
        meter = meters_col.find_one({"meter_id": user.get("meter_id")})
        st.header("Dashboard")
        st.write(f"Welcome, **{user.get('full_name')}**")
        st.write(f"Funds available: ₦{user.get('funds',0):,.2f}")
        st.write(f"Meter: {user.get('meter_id')}")
        st.write(f"Address: {user.get('address')}")
        if user_has_debt(user):
            st.warning(f"You owe ₦{user.get('borrowed',0):,.2f}. Please fund your account to repay.")
        apps = list(appliances_col.find({"meter_id": user.get("meter_id")}))
        st.subheader("Appliances")
        total_on = sum(1 for a in apps if a.get("is_on"))
        counts_room = {}
        for a in apps:
            counts_room.setdefault(a.get("location","unknown"), 0)
            if a.get("is_on"):
                counts_room[a.get("location")] += 1
        st.markdown(f"**Total appliances on:** {total_on}")
        st.write("Appliances on by room:")
        for loc, cnt in counts_room.items():
            st.write(f"- {loc}: {cnt}")
        for a in apps:
            with st.container():
                st.markdown(f"<div class='card'><strong>{a['type'].title()} ({a['location']})</strong>", unsafe_allow_html=True)
                cols = st.columns([1,1,1,1])
                cols[0].write(f"ID: {a.get('appliance_id')}")
                cols[1].write(f"kWh total: {a.get('total_accum_kwh',0):.4f}")
                cols[2].write(f"Session kWh: {a.get('session',{}).get('accum_kwh_session',0):.4f}")
                toggle_key = f"toggle_{a.get('_id')}"
                disable_toggle = user.get("funds",0.0) <= 0.0 or user_has_debt(user)
                new_state = cols[3].checkbox("On", value=a.get('is_on', False), key=toggle_key, disabled=disable_toggle)
                if new_state != a.get('is_on', False) and not disable_toggle:
                    appliances_col.update_one({"_id": a.get("_id")}, {"$set": {"is_on": bool(new_state), "manual_control": True}})
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Fund":
        st.header("Fund Account")
        with st.form("fund_form"):
            amount = st.number_input("Top-up amount (₦)", min_value=0.0, value=0.0, step=100.0)
            submitted = st.form_submit_button("Add funds")
        if submitted and amount > 0:
            users_col.update_one({"_id": user["_id"]}, {"$inc": {"funds": float(amount)}})
            updated = users_col.find_one({"_id": user["_id"]})
            transactions_col.insert_one({
                "user_id": user["_id"],
                "type": "fund",
                "amount": float(amount),
                "balance_after": float(updated.get("funds",0.0)),
                "metadata": {"reason": "top_up"},
                "timestamp": datetime.datetime.utcnow()
            })
            debt_repay(updated)
            st.success(f"Added ₦{amount:,.2f} to your account.")
            st.session_state["page"] = "Dashboard"
            st.rerun()
        st.subheader("Withdraw funds to bank account")
        withdraw_allowed = can_withdraw(user)
        max_amount = max_withdraw_amount(user)
        if withdraw_allowed and max_amount > 0:
            with st.form("withdraw_form"):
                withdraw_amount = st.number_input("Amount to withdraw (₦)", min_value=0.0, max_value=max_amount, value=0.0, step=100.0)
                account_name = st.text_input("Account Name")
                account_number = st.text_input("Account Number")
                bank_name = st.text_input("Bank Name")
                withdraw_submit = st.form_submit_button("Withdraw")
            if withdraw_submit and withdraw_amount > 0:
                users_col.update_one({"_id": user["_id"]}, {"$inc": {"funds": -withdraw_amount}})
                updated = users_col.find_one({"_id": user["_id"]})
                transactions_col.insert_one({
                    "user_id": user["_id"],
                    "type": "withdraw",
                    "amount": withdraw_amount,
                    "balance_after": updated.get("funds",0.0),
                    "metadata": {
                        "account_name": account_name,
                        "account_number": account_number,
                        "bank_name": bank_name
                    },
                    "timestamp": datetime.datetime.utcnow()
                })
                st.success(f"Withdrawn ₦{withdraw_amount:,.2f} to {bank_name} account {account_number}. You can only withdraw once a month.")
                st.rerun()
        elif not withdraw_allowed:
            st.info("You may only withdraw once per month and cannot withdraw borrowed funds.")

    elif page == "Borrow":
        st.header("Borrow Funds")
        if user_can_borrow(user) and user_has_funded_before(user):
            if st.button("Confirm Borrow", key="confirm_borrow_btn"):
                users_col.update_one({"_id": user["_id"]}, {"$set": {"funds": BORROW_AMOUNT, "borrowed": BORROW_AMOUNT}})
                transactions_col.insert_one({
                    "user_id": user["_id"],
                    "type": "borrow",
                    "amount": BORROW_AMOUNT,
                    "balance_after": BORROW_AMOUNT,
                    "metadata": {"reason": "borrow"},
                    "timestamp": datetime.datetime.utcnow()
                })
                st.success(f"You borrowed ₦{BORROW_AMOUNT:,.2f}. Please pay back when you fund your account!")
                st.session_state["page"] = "Dashboard"
                st.rerun()
        elif user_can_borrow(user) and not user_has_funded_before(user):
            st.info("You must fund your account at least once before you can borrow.")
        elif user_has_debt(user):
            st.warning("You are already owing. Please fund your account to restore power.")
        else:
            st.info("You cannot borrow at this time.")

    elif page == "Usage":
        st.header("Appliance Usage & Live Energy Consumption")
        meter = meters_col.find_one({"meter_id": user.get("meter_id")})
        apps = list(appliances_col.find({"meter_id": user.get("meter_id")}))
        st.write(f"Total Meter kWh: {meter.get('total_energy_kwh',0.0):.4f}")
        st.subheader("Appliances")
        total_energy = 0.0
        for a in apps:
            total_energy += a.get("total_accum_kwh", 0.0)
            st.write(f"{a['type'].title()} ({a['location']}): {a.get('total_accum_kwh',0.0):.4f} kWh")
        st.write(f"**Total Appliance Energy Consumption:** {total_energy:.4f} kWh")
        # Live graph using last 30 deductions
        transactions = safe_find_transactions({"user_id": user["_id"], "type": "deduction"}, limit=30)
        if transactions:
            graph_data = []
            for t in transactions:
                graph_data.append({
                    "timestamp": t["timestamp"],
                    "energy": t.get("amount", 0.0) / PRICE_PER_KWH if PRICE_PER_KWH != 0 else 0
                })
            df = pd.DataFrame(graph_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            st.line_chart(df.set_index("timestamp")["energy"])
        else:
            st.info("No energy consumption data to plot yet.")

    elif page == "Forecast":
        st.header("🔮 Professional Energy Consumption Forecast & Analytics")
        
        # Prepare data
        consumption_df = prepare_consumption_data(user["_id"], days_back=90)
        
        if consumption_df is None or len(consumption_df) < 10:
            st.warning("⚠️ Insufficient data to generate forecasts. Please use the system for more than 10 transactions to enable detailed analytics.")
        else:
            # Calculate statistics
            consumption_stats = calculate_consumption_stats(consumption_df)
            
            # Generate forecasts
            forecast_df, models = forecast_arima_style(consumption_df, periods=30)
            
            if forecast_df is not None:
                # Get customer standing
                standing = get_customer_standing(user, consumption_stats, forecast_df)
                
                # Get depletion date
                depletion_info = forecast_fund_depletion_date(user, consumption_stats)
                
                # Generate alerts
                alerts = get_early_warning_alerts(standing, consumption_stats)
                
                # ===== SECTION 1: CUSTOMER STANDING =====
                st.divider()
                st.subheader("📋 CUSTOMER STANDING & FINANCIAL STATUS")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Balance", f"₦{standing['current_funds']:,.2f}", delta=None)
                with col2:
                    st.metric("Borrowed Amount", f"₦{standing['borrowed_amount']:,.2f}", delta=None)
                with col3:
                    net_color = "green" if standing['net_position'] >= 0 else "red"
                    st.metric("Net Position", f"₦{standing['net_position']:,.2f}", delta=None)
                with col4:
                    status_color = "🟢" if standing['is_solvent'] else "🔴"
                    st.metric("Status", f"{status_color} {standing['status']}", delta=None)
                
                # Display alerts
                if alerts:
                    st.subheader("⚠️ Alerts & Notifications")
                    for alert in alerts:
                        if alert['type'] == 'CRITICAL':
                            st.markdown(f"<div class='alert-danger'>{alert['message']}</div>", unsafe_allow_html=True)
                        elif alert['type'] == 'WARNING':
                            st.markdown(f"<div class='alert-warning'>{alert['message']}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='alert-info'>{alert['message']}</div>", unsafe_allow_html=True)
                
                # ===== SECTION 2: CONSUMPTION STATISTICS =====
                st.divider()
                st.subheader("📊 HISTORICAL CONSUMPTION STATISTICS (Last 90 Days)")
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Total Consumed (kWh)", f"{consumption_stats['total_energy_kwh']:.2f}")
                with stat_col2:
                    st.metric("Total Cost (₦)", f"₦{consumption_stats['total_cost']:,.2f}")
                with stat_col3:
                    st.metric("Avg Daily (kWh)", f"{consumption_stats['avg_daily_kwh']:.2f}")
                with stat_col4:
                    st.metric("Avg Daily Cost (₦)", f"₦{consumption_stats['avg_daily_cost']:,.2f}")
                
                stat_col5, stat_col6, stat_col7, stat_col8 = st.columns(4)
                with stat_col5:
                    st.metric("Max Daily (kWh)", f"{consumption_stats['max_daily_kwh']:.2f}")
                with stat_col6:
                    st.metric("Min Daily (kWh)", f"{consumption_stats['min_daily_kwh']:.2f}")
                with stat_col7:
                    peak_hour_name = f"{consumption_stats['peak_hour']:02d}:00"
                    st.metric("Peak Hour", peak_hour_name)
                with stat_col8:
                    off_peak_name = f"{consumption_stats['off_peak_hour']:02d}:00"
                    st.metric("Off-Peak Hour", off_peak_name)
                
                # ===== SECTION 3: HISTORICAL DAILY CONSUMPTION =====
                st.divider()
                st.subheader("📈 Historical Daily Consumption")
                
                daily_consumption = get_historical_daily_consumption(consumption_df)
                if daily_consumption is not None and len(daily_consumption) > 0:
                    fig_daily = go.Figure()
                    fig_daily.add_trace(go.Scatter(
                        x=daily_consumption['date'],
                        y=daily_consumption['energy'],
                        mode='lines+markers',
                        name='Daily Consumption (kWh)',
                        line=dict(color='#007BFF', width=2),
                        marker=dict(size=5)
                    ))
                    fig_daily.update_layout(
                        title="Daily Energy Consumption Trend",
                        xaxis_title="Date",
                        yaxis_title="Energy (kWh)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                # ===== SECTION 4: HOURLY CONSUMPTION PATTERN =====
                st.divider()
                st.subheader("⏰ Hourly Consumption Pattern")
                
                hourly_pattern = get_hourly_consumption_pattern(consumption_df)
                if hourly_pattern is not None and len(hourly_pattern) > 0:
                    fig_hourly = go.Figure()
                    fig_hourly.add_trace(go.Bar(
                        x=hourly_pattern['hour'],
                        y=hourly_pattern['energy'],
                        name='Avg Consumption (kWh)',
                        marker=dict(color='#FFC107')
                    ))
                    fig_hourly.update_layout(
                        title="Average Hourly Consumption Pattern",
                        xaxis_title="Hour of Day",
                        yaxis_title="Average Energy (kWh)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
                
                # ===== SECTION 5: EARLY FORECAST (7 DAYS) =====
                st.divider()
                st.subheader("🔔 Early Forecast (Next 7 Days)")
                
                early_forecast = forecast_df.head(7)
                if early_forecast is not None and len(early_forecast) > 0:
                    early_cost = early_forecast['ensemble_forecast'].sum() * PRICE_PER_KWH
                    early_col1, early_col2, early_col3 = st.columns(3)
                    with early_col1:
                        st.metric("7-Day Forecast (kWh)", f"{early_forecast['ensemble_forecast'].sum():.2f}")
                    with early_col2:
                        st.metric("7-Day Forecast Cost (₦)", f"₦{early_cost:,.2f}")
                    with early_col3:
                        avg_daily_forecast = early_forecast['ensemble_forecast'].mean()
                        st.metric("Daily Average (kWh)", f"{avg_daily_forecast:.2f}")
                    
                    fig_early = go.Figure()
                    fig_early.add_trace(go.Bar(
                        x=range(1, 8),
                        y=early_forecast['ensemble_forecast'].values,
                        name='Predicted Consumption',
                        marker=dict(color='#28a745')
                    ))
                    fig_early.update_layout(
                        title="7-Day Early Forecast",
                        xaxis_title="Day",
                        yaxis_title="Energy (kWh)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_early, use_container_width=True)
                    
                    # 7-day table
                    early_table = pd.DataFrame({
                        'Day': [f"Day {i+1}" for i in range(7)],
                        'Forecast (kWh)': [f"{v:.4f}" for v in early_forecast['ensemble_forecast'].values],
                        'Cost (₦)': [f"₦{v * PRICE_PER_KWH:,.2f}" for v in early_forecast['ensemble_forecast'].values]
                    })
                    st.dataframe(early_table, use_container_width=True)
                
                # ===== SECTION 6: WEEKLY FORECAST =====
                st.divider()
                st.subheader("📅 Weekly Forecast (Next 4-5 Weeks)")
                
                weekly_forecast = generate_weekly_forecast(forecast_df)
                if weekly_forecast is not None and len(weekly_forecast) > 0:
                    fig_weekly = go.Figure()
                    fig_weekly.add_trace(go.Bar(
                        x=[f"Week {i+1}" for i in range(len(weekly_forecast))],
                        y=weekly_forecast['total_kwh'],
                        name='Weekly Consumption (kWh)',
                        marker=dict(color='#FF6B6B')
                    ))
                    fig_weekly.update_layout(
                        title="Weekly Energy Forecast",
                        xaxis_title="Week",
                        yaxis_title="Energy (kWh)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_weekly, use_container_width=True)
                    
                    # Weekly table
                    weekly_table = pd.DataFrame({
                        'Week': [f"Week {i+1}" for i in range(len(weekly_forecast))],
                        'Total kWh': [f"{v:.2f}" for v in weekly_forecast['total_kwh']],
                        'Cost (₦)': [f"₦{v:,.2f}" for v in weekly_forecast['cost']]
                    })
                    st.dataframe(weekly_table, use_container_width=True)
                
                # ===== SECTION 7: MONTHLY FORECAST =====
                st.divider()
                st.subheader("🗓️ Monthly Forecast")
                
                monthly_forecast = generate_monthly_forecast(forecast_df)
                if monthly_forecast is not None and len(monthly_forecast) > 0:
                    fig_monthly = go.Figure()
                    fig_monthly.add_trace(go.Bar(
                        x=[str(m) for m in monthly_forecast['month']],
                        y=monthly_forecast['total_kwh'],
                        name='Monthly Consumption (kWh)',
                        marker=dict(color='#4C72B0')
                    ))
                    fig_monthly.update_layout(
                        title="Monthly Energy Forecast",
                        xaxis_title="Month",
                        yaxis_title="Energy (kWh)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)
                    
                    # Monthly table
                    monthly_table = pd.DataFrame({
                        'Month': [str(m) for m in monthly_forecast['month']],
                        'Total kWh': [f"{v:.2f}" for v in monthly_forecast['total_kwh']],
                        'Estimated Cost (₦)': [f"₦{v:,.2f}" for v in monthly_forecast['cost']]
                    })
                    st.dataframe(monthly_table, use_container_width=True)
                
                # ===== SECTION 8: FUND DEPLETION FORECAST =====
                st.divider()
                st.subheader("💰 Fund Depletion & Budget Analysis")
                
                if depletion_info:
                    days_rem = depletion_info['days_remaining']
                    depl_date = depletion_info['depletion_date']
                    
                    col_depl1, col_depl2, col_depl3 = st.columns(3)
                    with col_depl1:
                        st.metric("Days Until Depletion", f"{days_rem:.1f} days")
                    with col_depl2:
                        st.metric("Estimated Depletion Date", depl_date.strftime("%Y-%m-%d %H:%M"))
                    with col_depl3:
                        funds_needed = standing['monthly_forecast_cost'] - standing['current_funds']
                        if funds_needed > 0:
                            st.metric("Additional Funds Needed", f"₦{funds_needed:,.2f}")
                        else:
                            st.metric("Surplus After Month", f"₦{abs(funds_needed):,.2f}")
                
                # Budget recommendation
                st.subheader("💡 Budget Recommendations")
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    if standing['current_funds'] < standing['monthly_forecast_cost']:
                        shortfall = standing['monthly_forecast_cost'] - standing['current_funds']
                        st.warning(f"""
                        **Monthly Shortfall Alert 🚨**
                        
                        Your balance (₦{standing['current_funds']:,.2f}) is insufficient for the forecasted monthly cost (₦{standing['monthly_forecast_cost']:,.2f}).
                        
                        **You need ₦{shortfall:,.2f} more to maintain service throughout the month.**
                        """)
                    else:
                        surplus = standing['current_funds'] - standing['monthly_forecast_cost']
                        st.success(f"""
                        **Budget Status ✓**
                        
                        Your balance (₦{standing['current_funds']:,.2f}) is sufficient for the forecasted monthly cost (₦{standing['monthly_forecast_cost']:,.2f}).
                        
                        **Expected remaining balance: ₦{surplus:,.2f}**
                        """)
                
                with rec_col2:
                    daily_avg = consumption_stats['avg_daily_cost']
                    days_until_critical = standing['current_funds'] / daily_avg if daily_avg > 0 else 999
                    
                    if days_until_critical < 3:
                        st.error(f"""
                        **CRITICAL: Fund Immediately 🔴**
                        
                        At your current usage rate, your balance will deplete in **{days_until_critical:.1f} days**.
                        
                        Immediate action required to avoid service disconnection.
                        """)
                    elif days_until_critical < 7:
                        st.warning(f"""
                        **Low Balance Warning ⚠️**
                        
                        Consider topping up soon. Balance will last approximately **{days_until_critical:.1f} days**.
                        """)
                    else:
                        st.info(f"""
                        **Healthy Balance 💚**
                        
                        At current usage, balance will last approximately **{days_until_critical:.1f} days**.
                        """)
            else:
                st.error("❌ Unable to generate forecast. Please try again later.")

    elif page == "Meter Details":
        meter = meters_col.find_one({"meter_id": user.get("meter_id")})
        st.header("Meter Details")
        st.write(f"Meter Name: {user.get('full_name')}")
        st.write(f"Meter ID: {meter.get('meter_id')}")
        st.write(f"Address: {meter.get('address')}")
        st.write(f"Total Meter kWh: {meter.get('total_energy_kwh',0.0):.4f}")

    elif page == "User Info":
        st.header("User Info")
        st.write(f"Full Name: {user.get('full_name')}")
        st.write(f"Email: {user.get('email')}")
        st.write(f"Phone: {user.get('phone')}")
        st.write(f"Username: {user.get('username')}")
        st.write(f"Address: {user.get('address')}")
        st.info("You cannot change your details.")

    elif page == "Billing":
        meter = meters_col.find_one({"meter_id": user.get("meter_id")})
        st.header("Billing")
        st.write(f"Name: {user.get('full_name')}")
        st.write(f"Email: {user.get('email')}")
        st.write(f"Phone: {user.get('phone')}")
        st.write(f"Meter ID: {meter.get('meter_id')}")
        st.write(f"Address: {meter.get('address')}")
        st.write(f"Meter total kWh: {meter.get('total_energy_kwh',0.0):.4f}")
        transactions = safe_find_transactions({"user_id": user["_id"]}, limit=1000)
        st.subheader("Transactions")
        if transactions:
            rows = []
            for t in transactions:
                ts = t.get("timestamp")
                if not isinstance(ts, datetime.datetime):
                    try:
                        ts = pd.to_datetime(ts)
                    except Exception:
                        ts = None
                rows.append({
                    "timestamp": ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime.datetime) else "unknown",
                    "type": t.get("type"),
                    "amount": t.get("amount", 0.0),
                    "balance_after": t.get("balance_after", 0.0)
                })
            tx_df = pd.DataFrame(rows)
            st.dataframe(tx_df)
            csv = tx_df.to_csv(index=False)
            st.download_button("Print Transactions", data=csv, file_name="transactions.csv", mime="text/csv")
            if st.button("Withdraw Funds (Shortcut)", key="withdraw_from_billing"):
                st.session_state["page"] = "Fund"
                st.rerun()
        else:
            st.info("No transactions found.")

    elif page == "Logout":
        stop_simulation_session()
        st.session_state["user_id"] = None
        st.session_state["page"] = "Login"
        st.success("Logged out")
        st.rerun()

# ---------------------- ADMIN VIEWS ----------------------
if user and user.get("role") == "admin":

    if page == "Admin Info":
        st.header("Admin Info")
        st.write(f"Admin Name: {user.get('full_name')}")
        st.write(f"Email: {user.get('email')}")
        st.write(f"Phone: {user.get('phone')}")
        st.write(f"Username: {user.get
