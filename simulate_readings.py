import random
from datetime import datetime
from db_connection import readings_col, appliances_col

# Get all appliances
appliances = list(appliances_col.find())

# Generate readings
for appliance in appliances:
    reading = {
        "appliance_id": appliance["_id"],
        "household_id": appliance["household_id"],
        "timestamp": datetime.now(),
        "power_consumption": round(random.uniform(0.1, 2.0), 2)  # kWh
    }
    readings_col.insert_one(reading)

print("Readings generated!")
