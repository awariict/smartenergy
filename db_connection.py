from pymongo import MongoClient

# Connect to MongoDB
MONGO_URI = "mongodb://localhost:27017/"  # Use your URI if different
client = MongoClient(MONGO_URI)

# Create / use database
db = client["smart_energy"]

# Collections
users_col = db["users"]
appliances_col = db["appliances"]
readings_col = db["energy_readings"]
