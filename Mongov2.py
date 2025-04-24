from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
import pandas as pd
from fpdf import FPDF
import os
import tempfile
import re
from datetime import datetime

app = FastAPI()

# ---------- MongoDB Connection ----------
def get_mongo_connection():
    client = MongoClient("mongodb://localhost:27017/")
    return client["VATINS"]

# ---------- Utility to Remove Emoji ----------
def remove_unicode_emoji(text):
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile("[" 
                               "\U0001F600-\U0001F64F"  # emoticons
                               "\U0001F300-\U0001F5FF"  # symbols & pictographs
                               "\U0001F680-\U0001F6FF"  # transport & map symbols
                               "\U0001F1E0-\U0001F1FF"  # flags
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# ---------- PDF Generator ----------
def generate_pdf_summary(user_data, category_counts, location_df, user_messages_df, group_data, message_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", '', 12)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="User Profile Summary", ln=True)

    # User Photo
    photo_path = os.path.join("E:/VATINS/photos", f"{user_data['user_id']}.jpg")
    if os.path.exists(photo_path):
        pdf.image(photo_path, x=150, y=10, w=40)
        pdf.ln(5)
    else:
        pdf.cell(0, 10, txt="(No photo available)", ln=True)

    # User Info Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Field", 1)
    pdf.cell(130, 10, "Value", 1, ln=True)
    pdf.set_font("Arial", '', 12)

    profile_fields = [
        ("User ID", user_data.get("user_id", "N/A")),
        ("Name", user_data.get("full_name", user_data.get("name", "N/A"))),
        ("Phone", user_data.get("phone_number", user_data.get("phonenumber", "N/A"))),
        ("Email", user_data.get("email", "N/A")),
        ("First Tipline", user_data.get("first_tipline_number_date", "N/A")),
        ("Last Tipline", user_data.get("last_tipline_number_date", "N/A")),
        ("Total Tiplines", user_data.get("total_tipline_sent", 0)),
        ("Active Days", user_data.get("active_days", 0)),
        ("Average Sentiment", user_data.get("average_sentiment", "N/A")),
        ("Risk Score", f"{float(user_data.get('reporter_risk_score') or 0):.2f}"),
        ("Total Groups Joined", group_data.get("total_groups", 0)),
        ("Active Groups", group_data.get("active_groups", 0)),
        ("Total Messages", message_data.get("total_messages", 0)),
    ]

    for label, value in profile_fields:
        pdf.cell(60, 10, str(label), 1)
        pdf.cell(130, 10, remove_unicode_emoji(str(value)), 1, ln=True)

    # Tipline Categories
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="\nTipline Category Counts:", ln=True)
    pdf.set_font("Arial", '', 12)
    if category_counts:
        for item in category_counts:
            pdf.cell(60, 10, item['tipline_category'], 1)
            pdf.cell(130, 10, str(item['count']), 1, ln=True)
    else:
        pdf.cell(0, 10, txt="No tipline data found.", ln=True)

    # Recent Messages
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="\nRecent Messages:", ln=True)
    pdf.set_font("Arial", '', 11)
    if not user_messages_df.empty:
        for idx, row in user_messages_df.head(5).iterrows():
            message = f"[{row['timestamp']}] Group: {row.get('group_name', 'Unknown')}\nMessage: {row.get('message_text', '')}"
            pdf.multi_cell(0, 10, txt=remove_unicode_emoji(message))
            pdf.ln(4)
    else:
        pdf.set_font("Arial", 'I', 11)
        pdf.cell(0, 10, txt="No recent messages found.", ln=True)

    # Location Info
    if not location_df.empty:
        place_name = location_df.iloc[0].get("place", "Unknown")
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="\nLast Known Location:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, txt=f"Place: {place_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(tempfile.gettempdir(), f"{user_data['user_id']}_profile_summary_{timestamp}.pdf")
    pdf.output(output_path)
    return output_path

# ---------- FastAPI Route ----------
@app.get("/generate-summary/{user_id}")
def get_summary(user_id: str):
    db = get_mongo_connection()

    user_data = db.UserProfileSummary.aggregate([
        {"$match": {"user_id": user_id}},
        {"$lookup": {
            "from": "users",
            "localField": "user_id",
            "foreignField": "user_id",
            "as": "user_info"
        }},
        {"$unwind": {"path": "$user_info", "preserveNullAndEmptyArrays": True}},
        {"$addFields": {
            "full_name": "$user_info.full_name",
            "phone_number": "$user_info.phone_number",
            "email": "$user_info.email"
        }}
    ])
    user_data = list(user_data)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user_data[0]

    # Groups data
    groups = db.UserGroups.aggregate([
        {"$match": {"user_id": user_id}},
        {"$group": {
            "_id": "$user_id",
            "total_groups": {"$sum": 1},
            "active_groups": {"$sum": {"$cond": ["$is_active", 1, 0]}}
        }}
    ])
    group_data = next(groups, {"total_groups": 0, "active_groups": 0})

    # Message data
    message_data = {"total_messages": db.UserMessages.count_documents({"user_id": user_id})}

    # Tipline Category Counts
    category_agg = db.Tipline.aggregate([
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$tipline_category", "count": {"$sum": 1}}}
    ])
    category_counts = [{"tipline_category": cat["_id"], "count": cat["count"]} for cat in category_agg]

    # Messages DataFrame
    messages = list(db.UserMessages.find({"user_id": user_id}).sort("timestamp", -1))
    user_messages_df = pd.DataFrame(messages)

    # Location DataFrame
    location_df = pd.DataFrame([user_data["last_known_location"]]) if user_data.get("last_known_location") else pd.DataFrame()

    # PDF Generation
    pdf_path = generate_pdf_summary(user_data, category_counts, location_df, user_messages_df, group_data, message_data)

    # Return as file
    response = FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))
    response.headers["Cache-Control"] = "no-cache"
    return response
