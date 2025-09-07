import time
import random
from datetime import datetime
from db_connection import readings_col, appliances_col

appliances = list(appliances_col.find())

while True:
    for appliance in appliances:
        reading = {
            "appliance_id": appliance["_id"],
            "household_id": appliance["household_id"],
            "timestamp": datetime.now(),
            "power_consumption": round(random.uniform(0.1, 2.0), 2)  # kWh
        }
        readings_col.insert_one(reading)
    print("New readings generated!")
    time.sleep(10)  # wait 10 seconds before next batch
