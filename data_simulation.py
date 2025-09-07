from db_connection import appliances_col

# Sample appliances
appliances = [
    {"name": "Refrigerator", "type": "Kitchen", "power_rating": 200, "household_id": "HH001"},
    {"name": "Air Conditioner", "type": "Living Room", "power_rating": 1500, "household_id": "HH001"},
    {"name": "Washing Machine", "type": "Laundry", "power_rating": 500, "household_id": "HH001"}
]

# Insert appliances into MongoDB
for appliance in appliances:
    appliances_col.insert_one(appliance)

print("Appliances added!")
