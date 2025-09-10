# patched_smart_energy_app.py
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
        # Ensure an index on timestamp for fast / safe sorting (descending)
        db.transactions.create_index([("timestamp", -1)], background=True)
    except Exception as e:
        # Non-fatal: log to Streamlit so you can inspect
        st.warning(f"Could not create index on transactions.timestamp: {e}")
    return db

db = get_db()
users_col = db.users
meters_col = db.meters
appliances_col = db.appliances
transactions_col = db.transactions

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
    # Only if not already borrowed and funds are 0
    return user.get("borrowed", 0.0) == 0.0 and user.get("funds", 0.0) == 0.0

def user_has_funded_before(user):
    # safe find_one (no sort). This is lightweight.
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
    except OperationFailure as e:
        # fallback: only consider withdraws that have timestamp field
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
    # Use aggregation to compute totals server-side (efficient & safe).
    def sum_amount(user_id, ttype):
        pipeline = [
            {"$match": {"user_id": user_id, "type": ttype, "amount": {"$exists": True}}},
            {"$group": {"_id": None, "total": {"$sum": "$amount"}}}
        ]
        try:
            res = list(transactions_col.aggregate(pipeline))
            return float(res[0]["total"]) if res else 0.0
        except OperationFailure:
            # fallback to a limited client-side sum (in case aggregation fails)
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

# ---------------------- Helper: safe transaction reads ----------------------
def safe_find_transactions(filter_query=None, sort_field="timestamp", limit=None, projection=None):
    """
    Attempt to find transactions sorted by sort_field.
    If OperationFailure occurs (e.g. sort memory/disk issues), fallback to restricted query.
    """
    q = filter_query or {}
    try:
        cursor = transactions_col.find(q, projection=projection)
        if sort_field:
            cursor = cursor.sort(sort_field, -1)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    except OperationFailure as e:
        # log/show a warning and attempt a safer query: require timestamp exists and apply a reasonable limit
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
st.set_page_config(page_title="Smart Energy System", layout="centered")
# (your CSS and UI code unchanged below...)
# ... (keep the remainder of your UI code exactly as before) ...
# Replace direct calls to transactions_col.find(...).sort(...) with safe_find_transactions(...) where used:
# - In Billing: transactions = safe_find_transactions({"user_id": user["_id"]}, limit=1000)
# - In Admin Transaction Log: txs = safe_find_transactions({}, limit=2000)
# Also when constructing DataFrames, handle missing or non-datetime timestamps defensively:
#
# Example replacement snippet for Billing transactions usage:
#
# transactions = safe_find_transactions({"user_id": user["_id"]}, limit=1000)
# st.subheader("Transactions")
# if transactions:
#     rows = []
#     for t in transactions:
#         ts = t.get("timestamp")
#         if not isinstance(ts, datetime.datetime):
#             try:
#                 ts = pd.to_datetime(ts)
#             except Exception:
#                 ts = None
#         rows.append({
#             "timestamp": ts.strftime('%Y-%m-%d %H:%M:%S') if isinstance(ts, datetime.datetime) else "unknown",
#             "type": t.get("type"),
#             "amount": t.get("amount", 0.0),
#             "balance_after": t.get("balance_after", 0.0)
#         })
#     tx_df = pd.DataFrame(rows)
#     st.dataframe(tx_df)
#
# The rest of your UI remains the same - only swap those direct finds for safe_find_transactions and use aggregation for sums.

st.markdown("---")
st.write("Developed by: Happiness Sunday Eyeh.")
