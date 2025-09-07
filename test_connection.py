from pymongo import MongoClient

MONGO_URI = "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/smart_energy?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)
db = client["smart_energy"]

# Test collections
users_col = db["users"]
appliances_col = db["appliances"]
readings_col = db["energy_readings"]

# Test inserting a document
try:
    users_col.insert_one({"test": "connection_successful"})
    print("✅ Connected successfully and test document inserted!")
except Exception as e:
    print("❌ Connection failed:", e)
