import streamlit as st
from pymongo import MongoClient
import pandas as pd
from fpdf import FPDF
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["telegram_data"]
users_col = db["users"]
messages_col = db["messages"]
groups_col = db["chats"]
user_groups_col = db["user_groups"]

st.set_page_config(page_title="Telegram User Dashboard", layout="wide")
st.title("📊 Telegram Insights Dashboard")

# Dropdown 1: Select User
user_data = list(users_col.find({}, {"user_id": 1, "first_name": 1, "last_name": 1, "username": 1}))
user_map = {
    f"{u.get('first_name', '')} {u.get('last_name', '')} (@{u.get('username', 'N/A')})": u["user_id"]
    for u in user_data if u.get("user_id")
}
selected_user = st.selectbox("👤 Select a User", list(user_map.keys()))
user_id = user_map.get(selected_user)

# Dropdown 2: Select Sentiment
all_sentiments = messages_col.distinct("sentiment")
selected_sentiment = st.selectbox("💬 Filter Messages by Sentiment", ["All"] + all_sentiments)

# Dropdown 3: Select Group
group_data = list(groups_col.find({}, {"chat_id": 1, "title": 1}))
group_map = {g.get("title", "Unknown Group"): g["chat_id"] for g in group_data if g.get("chat_id")}
selected_group = st.selectbox("👥 Select a Group", ["All"] + list(group_map.keys()))

# Fetch User Info
user_info = users_col.find_one({"user_id": user_id})

# Fetch Group Participation Info
user_groups = list(user_groups_col.find({"user_id": user_id}))
total_groups_count = len(user_groups)
active_groups = [g for g in user_groups if g.get("is_active")]
active_groups_count = len(active_groups)

# Fetch messages from date of join
messages = []
group_ids = []
joined_dates = {}

for g in user_groups:
    chat_id = g["chat_id"]
    joined_date = g.get("joined_date")
    if joined_date:
        group_ids.append(chat_id)
        joined_dates[chat_id] = joined_date

        msg_query = {
            "chat_id": chat_id,
            "user_id": user_id,
            "date": {"$gte": joined_date}
        }

        if selected_sentiment != "All":
            msg_query["sentiment"] = selected_sentiment
        if selected_group != "All" and group_map[selected_group] != chat_id:
            continue

        messages += list(messages_col.find(msg_query))

# Sort messages by date descending
messages = sorted(messages, key=lambda x: x.get("date"), reverse=True)
df = pd.DataFrame(messages)

# Count messages received (from others in same groups)
received_query = {
    "chat_id": {"$in": group_ids},
    "user_id": {"$ne": user_id},
}
received_messages_count = messages_col.count_documents(received_query)

# Get group sizes for display
group_sizes = {
    g["chat_id"]: groups_col.find_one({"chat_id": g["chat_id"]}).get("participants_count", 0)
    for g in user_groups
}

# Export Summary to PDF with complete dashboard
st.subheader("📄 Export Summary as PDF")

def export_pdf(user_info, messages, group_ids, active_groups, group_sizes, received_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # User Info
    pdf.cell(200, 10, f"User Summary - {user_info.get('first_name', '')} {user_info.get('last_name', '')}", ln=True)
    pdf.cell(200, 10, f"Username: @{user_info.get('username', 'N/A')}", ln=True)
    pdf.cell(200, 10, f"Phone: {user_info.get('phone', 'N/A')}", ln=True)
    pdf.cell(200, 10, f"Messages Sent: {len(messages)}", ln=True)
    pdf.cell(200, 10, f"Messages Received: {received_count}", ln=True)
    pdf.cell(200, 10, f"Groups Joined: {len(group_ids)}", ln=True)
    pdf.cell(200, 10, f"Active Groups: {len(active_groups)}", ln=True)
    pdf.cell(200, 10, f"Group Participants Total: {sum(group_sizes.values())}", ln=True)
    pdf.cell(200, 10, "Recent Messages:", ln=True)

    # Top 10 Messages
    top_messages = messages[:10]
    for m in top_messages:
        text = m.get('text', '')[:200].replace('\n', ' ')
        pdf.multi_cell(0, 10, f"{m.get('date')} - {text}")
    
    # Group Participation Details
    pdf.cell(200, 10, "Group Participation Summary:", ln=True)
    for chat_id, count in group_sizes.items():
        group_name = next((k for k, v in group_map.items() if v == chat_id), str(chat_id))
        pdf.cell(200, 10, f"{group_name}: {count} Participants", ln=True)

    # Top 10 Groups by Messages Sent
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$chat_id", "message_count": {"$sum": 1}}},
        {"$sort": {"message_count": -1}},
        {"$limit": 10}
    ]
    top_groups = list(messages_col.aggregate(pipeline))
    top_groups_df = pd.DataFrame(top_groups)
    
    if not top_groups_df.empty:
        top_groups_df.rename(columns={"_id": "chat_id"}, inplace=True)
        top_groups_df["Group Title"] = top_groups_df["chat_id"].apply(
            lambda cid: next((k for k, v in group_map.items() if v == cid), str(cid))
        )
        pdf.cell(200, 10, "Top 10 Groups by Messages Sent:", ln=True)
        for idx, row in top_groups_df.iterrows():
            pdf.cell(200, 10, f"{row['Group Title']}: {row['message_count']} messages", ln=True)

    # Generate Sentiment Distribution Chart
    if "sentiment" in df.columns:
        sentiment_counts = df["sentiment"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        fig1.savefig("sentiment_chart.png")
        pdf.cell(200, 10, "Sentiment Distribution Chart:", ln=True)
        pdf.image("sentiment_chart.png", x=10, y=pdf.get_y(), w=180)
        pdf.ln(50)  # Add space after image

    # Generate Word Cloud
    if "text" in df.columns:
        all_text = " ".join(df["text"].dropna().tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis("off")
        fig2.savefig("wordcloud.png")
        pdf.cell(200, 10, "Word Cloud of Messages:", ln=True)
        pdf.image("wordcloud.png", x=10, y=pdf.get_y(), w=180)
        pdf.ln(50)  # Add space after image

    buffer = BytesIO(pdf.output(dest='S'))
    buffer.seek(0)
    return buffer

pdf_data = export_pdf(user_info, messages, group_ids, active_groups, group_sizes, received_messages_count)
st.download_button("📥 Download PDF Report", data=pdf_data, file_name="telegram_user_summary.pdf", mime="application/pdf")
