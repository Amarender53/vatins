from pymongo import MongoClient
import pandas as pd

# MongoDB connection setup
def get_mongo_connection():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["VATINS"]
    return db

def load_telegram_messages():
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    
    # Access the telegram_data database and messages collection
    db = client["telegram_data"]
    messages_collection = db["messages"]
    
    # Fetch all messages and convert to a DataFrame
    messages = list(messages_collection.find())
    df = pd.DataFrame(messages)
    return df

# Usage
df_messages = load_telegram_messages()

# Show the first few messages
print(df_messages.head())

def load_user_profiles():
    db = get_mongo_connection()
    pipeline = [
        {
            "$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "user_id",
                "as": "user_info"
            }
        },
        {
            "$unwind": {
                "path": "$user_info",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$project": {
                "user_id": 1,
                "common_categories": 1,
                "average_sentiment": 1,
                "top_emotions": 1,
                "top_keywords": 1,
                "geo_tags_detected": 1,
                "last_known_location": 1,
                "total_tipline_sent": 1,
                "active_days": 1,
                "first_tipline_number_date": 1,
                "last_tipline_number_date": 1,
                "reporter_risk_score": 1,
                "full_name": "$user_info.full_name",
                "phone_number": "$user_info.phone_number",
                "email": "$user_info.email"
            }
        }
    ]
    return pd.DataFrame(list(db.UserProfileSummary.aggregate(pipeline)))

def load_user_groups_messages(user_id):
    db = get_mongo_connection()
    
    groups = db.UserGroups.aggregate([
        {"$match": {"user_id": user_id}},
        {
            "$group": {
                "_id": "$user_id",
                "total_groups": {"$sum": 1},
                "active_groups": {"$sum": {"$cond": ["$is_active", 1, 0]}}
            }
        }
    ])
    groups = next(groups, {"total_groups": 0, "active_groups": 0})

    messages_count = db.UserMessages.count_documents({"user_id": user_id})

    categories = db.Tipline.aggregate([
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$tipline_category", "count": {"$sum": 1}}}
    ])
    category_counts = [{"tipline_category": item["_id"], "count": item["count"]} for item in categories]

    return groups, {"total_messages": messages_count}, category_counts

def load_user_messages(user_id):
    db = get_mongo_connection()
    messages = db.UserMessages.find({"user_id": user_id}).sort("timestamp", -1)
    return pd.DataFrame(messages)

def find_messages_by_user(user_id):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["telegram_data"]
    messages_collection = db["messages"]
    return list(messages_collection.find({"user_id": user_id}))

def find_messages_by_keyword(keyword):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["telegram_data"]
    messages_collection = db["messages"]
    return list(messages_collection.find({"text": {"$regex": keyword, "$options": "i"}}))

# Example usage
user_messages = find_messages_by_user("67f7b131942d60425929da89")
keyword_messages = find_messages_by_keyword("urgent")

print("Messages by user:", user_messages[:3])
print("Messages with keyword:", keyword_messages[:3])


