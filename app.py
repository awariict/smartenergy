import streamlit as st
from pymongo import MongoClient, ReturnDocument
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
MONGO_URI = st.secrets.get("MONGO_URI", "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/smart_energy?retryWrites=true&w=majority")
DB_NAME = st.secrets.get("DB_NAME", "smart_energy")
PRICE_PER_KWH = float(os.environ.get("PRICE_PER_KWH", "150.0"))  # Naira per kWh
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "8"))  # seconds
BORROW_AMOUNT = float(os.environ.get("BORROW_AMOUNT", "500.0"))  # Naira allowed to borrow

# ---------------------- Database ----------------------
@st.cache_resource(ttl=600)
def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test connection immediately
    try:
        client.server_info()  # Forces a call to check connection
    except Exception as e:
        st.error(f"Cannot connect to MongoDB: {e}")
        st.stop()
    return client[DB_NAME]

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
    # Only allow borrowing if user has funded before
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
            "amount": to_pay,
            "balance_after": new_funds,
            "metadata": {"reason": "debt_repay"},
            "timestamp": datetime.datetime.utcnow()
        })

def turn_off_all_appliances(meter_id):
    appliances_col.update_many({"meter_id": meter_id}, {"$set": {"is_on": False, "manual_control": False}})

def can_withdraw(user):
    last_withdraw = transactions_col.find_one(
        {"user_id": user["_id"], "type": "withdraw"},
        sort=[("timestamp", -1)]
    )
    now = datetime.datetime.utcnow()
    if last_withdraw:
        diff = (now - last_withdraw["timestamp"]).days
        if diff < 30:
            return False
    return not user_has_debt(user) and user.get("funds", 0.0) > 0.0

def max_withdraw_amount(user):
    fund_total = sum(t["amount"] for t in transactions_col.find({"user_id": user["_id"], "type": "fund"}))
    debt_repaid = sum(t["amount"] for t in transactions_col.find({"user_id": user["_id"], "type": "debt_repay"}))
    already_withdrawn = sum(t["amount"] for t in transactions_col.find({"user_id": user["_id"], "type": "withdraw"}))
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
    return True, "Registration is successful."

def authenticate_user(username, password):
    user = users_col.find_one({"username": username})
    if not user:
        return None
    if check_password(password, user.get("password_hash") or b""):
        return user
    return None

# ---------------------- Simulation ----------------------
def _simulate_meter(user_id, stop_event):
    # Appliance prediction always runs unless funds == 0
    while not stop_event.is_set():
        user = users_col.find_one({"_id": user_id})
        if not user:
            break
        meter_id = user.get("meter_id")
        now = datetime.datetime.utcnow()
        # If funds = 0, turn off all and block
        if user.get("funds", 0.0) <= 0.0:
            turn_off_all_appliances(meter_id)
            time.sleep(POLL_INTERVAL)
            continue
        appliances = list(appliances_col.find({"meter_id": meter_id}))
        for app in appliances:
            # If manual_control True -> respect is_on. Otherwise, predict usage.
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
            # Only change state if not manual
            if is_on and not prev_on:
                appliances_col.update_one(
                    {"_id": app["_id"]},
                    {"$set": {"is_on": True, "session.started_at": now, "session.accum_kwh_session": 0}}
                )
            elif not is_on and prev_on and not app.get("manual_control", False):
                appliances_col.update_one({"_id": app["_id"]}, {"$set": {"is_on": False}})
            # If on, compute kWh for POLL_INTERVAL
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
    # Always start thread unless already running
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

# --- Sidebar CSS injection ---
st.set_page_config(page_title="Smart Energy System", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #007BFF, #FFC107, #FF0000); }
section[data-testid="stSidebar"] { background: black !important; }
.sidebar-content { display: flex; flex-direction: column; align-items: center; gap: 12px; margin-top: 20px; }
.sidebar-btn { width: 180px; height: 44px; background-color: white !important; color: black !important; font-weight: bold; border-radius: 8px; border: none; margin: 0 auto; display: block; }
.big-font { font-size:20px !important; }
.card { background: white;color: black !important; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 8px; }
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
    ("Meter Details", "Meter Details"),
    ("User Info", "User Info"),
    ("Billing", "Billing"),
    ("Logout", "Logout")
]
ADMIN_SIDEBAR = [
    ("Admin Info", "Admin Info"),
    ("Manage Users", "Manage Users"),
    ("Transaction Log", "Transaction Log"),
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

st.title("Eyeh Intelligent Smart Energy System")
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

    # (User views unchanged...)

    pass

# ---------------------- ADMIN VIEWS ----------------------
if user and user.get("role") == "admin":

    if page == "Admin Info":
        st.header("Admin Info")
        st.write(f"Admin Name: {user.get('full_name')}")
        st.write(f"Email: {user.get('email')}")
        st.write(f"Phone: {user.get('phone')}")
        st.write(f"Username: {user.get('username')}")
        st.info("Admin details cannot be deleted.")

    elif page == "Manage Users":
        st.header("Manage Users")
        users_list = list(users_col.find({"role": "user"}))
        for u in users_list:
            cols = st.columns([3,1])
            cols[0].write(f"{u.get('full_name')} ({u.get('username')}) - {u.get('email')}")
            if cols[1].button("Delete", key=f"del_{u.get('_id')}"):
                users_col.delete_one({"_id": u.get('_id')})
                meters_col.update_many({"user_id": u.get('_id')}, {"$set": {"status": "deleted"}})
                appliances_col.delete_many({"meter_id": u.get('meter_id')})
                st.success("User deleted.")
                st.rerun()

    elif page == "Transaction Log":
        st.header("Transaction Log (System-wide)")

        try:
            txs = list(transactions_col.find({}).sort("timestamp", -1))
        except Exception as e:
            st.error(f"Error fetching transactions: {e}")
            txs = list(transactions_col.find({}))  # fallback without sort

        if txs:
            tx_df = pd.DataFrame([{
                "user": (users_col.find_one({"_id": t.get("user_id")}) or {}).get("username", "deleted_user"),
                "timestamp": t.get('timestamp').strftime('%Y-%m-%d %H:%M:%S') if isinstance(t.get('timestamp'), datetime.datetime) else str(t.get('timestamp')),
                "type": t.get('type'),
                "amount": t.get('amount'),
                "balance_after": t.get('balance_after')
            } for t in txs])

            st.dataframe(tx_df)

            csv = tx_df.to_csv(index=False)
            st.download_button("Print All Transactions", data=csv, file_name="all_transactions.csv", mime="text/csv")
        else:
            st.info("No transactions found.")

    elif page == "Admin Funding":
        st.header("Admin Funding")
        username = st.text_input("Username of user to fund")
        amount = st.number_input("Amount to fund", min_value=0.0, step=100.0)
        if st.button("Fund User"):
            u = users_col.find_one({"username": username})
            if not u:
                st.error("User not found")
            else:
                users_col.update_one({"_id": u["_id"]}, {"$inc": {"funds": amount}})
                new_bal = u.get("funds", 0.0) + amount
                transactions_col.insert_one({
                    "user_id": u["_id"],
                    "type": "fund",
                    "amount": amount,
                    "balance_after": new_bal,
                    "metadata": {"admin_funded": True},
                    "timestamp": datetime.datetime.utcnow()
                })
                st.success(f"Funded {username} with {amount}. New balance: {new_bal}")

    elif page == "Logout":
        st.session_state["user_id"] = None
        stop_simulation_session()
        st.session_state["page"] = "Login"
        st.rerun()
