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
user_groups_col = db["user_groups"]  # Assume this maps users to groups

st.set_page_config(page_title="Telegram User Dashboard", layout="wide")
st.title("📊 Telegram Insights Dashboard")

# Dropdown 1: Select User
user_data = list(users_col.find({}, {"user_id": 1, "first_name": 1, "last_name": 1, "username": 1}))
user_map = {
    f"{u.get('first_name', '')} {u.get('last_name', '')} (@{u.get('username', 'N/A')})": u["user_id"]
    for u in user_data if u.get("user_id")
}

selected_user = st.selectbox("Select a User", list(user_map.keys()))
user_id = user_map.get(selected_user)

# Dropdown 2: Select Sentiment
all_sentiments = messages_col.distinct("sentiment")
selected_sentiment = st.selectbox("Filter Messages by Sentiment", ["All"] + all_sentiments)

# Dropdown 3: Select Group
group_data = list(groups_col.find({}, {"chat_id": 1, "title": 1}))
group_map = {g.get("title", "Unknown Group"): g["chat_id"] for g in group_data if g.get("chat_id")}
selected_group = st.selectbox("Select a Group", ["All"] + list(group_map.keys()))

# Fetch User Info and Messages
user_info = users_col.find_one({"user_id": user_id})
message_query = {"user_id": user_id}
if selected_sentiment != "All":
    message_query["sentiment"] = selected_sentiment
if selected_group != "All":
    message_query["chat_id"] = group_map[selected_group]

messages = list(messages_col.find(message_query).sort("date", -1))
df = pd.DataFrame(messages)

# Group Participation
user_groups = list(user_groups_col.find({"user_id": user_id}))
group_ids = [g["chat_id"] for g in user_groups]
active_groups = [g for g in user_groups if g.get("is_active")]

# Group sizes
group_sizes = {
    g["chat_id"]: groups_col.find_one({"chat_id": g["chat_id"]}).get("participants_count", 0)
    for g in user_groups
}

# Display User Summary
st.subheader("👤 User Profile Summary")
col1, col2 = st.columns(2)
col1.metric("User ID", user_id)
col1.metric("Username", f"@{user_info.get('username', 'N/A')}")
col1.metric("Phone", user_info.get("phone", "N/A"))
col2.metric("Total Messages", len(messages))
col2.metric("Groups Joined", len(group_ids))
col2.metric("Active Groups", len(active_groups))

# Display Recent Messages
st.subheader("💬 Recent Messages")
for msg in df.head(5).to_dict(orient="records"):
    st.info(f"{msg.get('date')}\n\n{msg.get('text', '')}")

# Display Group Summary Table
st.subheader("📚 Group Participation Summary")
group_summary = pd.DataFrame.from_dict(group_sizes, orient="index", columns=["Participants"])
group_summary.index.name = "Chat ID"
st.dataframe(group_summary)

# Sentiment Pie Chart
if "sentiment" in df.columns:
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

# Keyword Cloud
if "text" in df.columns:
    st.subheader("☁️ Keyword Cloud")
    all_text = " ".join(df["text"].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis("off")
    st.pyplot(fig2)

# PDF Export
st.subheader("📄 Export Summary as PDF")
def export_pdf(user_info, messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"User Summary - {user_info.get('first_name', '')} {user_info.get('last_name', '')}", ln=True)
    pdf.cell(200, 10, f"Username: @{user_info.get('username', 'N/A')}", ln=True)
    pdf.cell(200, 10, f"Total Messages: {len(messages)}", ln=True)
    pdf.cell(200, 10, f"Groups Joined: {len(group_ids)}", ln=True)
    pdf.cell(200, 10, f"Active Groups: {len(active_groups)}", ln=True)
    pdf.cell(200, 10, f"Group Participants: {sum(group_sizes.values())}", ln=True)
    pdf.cell(200, 10, txt="Recent Messages:", ln=True)
    for m in messages[:5]:
        pdf.multi_cell(0, 10, f"{m.get('date')} - {m.get('text', '')[:100]}")
    buffer = BytesIO()
    pdf.output(dest='S').encode('latin-1')
    buffer.write(pdf.output(dest='S').encode('latin-1'))
    buffer.seek(0)
    return buffer

pdf_data = export_pdf(user_info, messages)
st.download_button("📅 Download PDF", data=pdf_data, file_name="user_summary.pdf", mime="application/pdf")