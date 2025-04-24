import streamlit as st
from pymongo import MongoClient
import pandas as pd
from fpdf import FPDF
from io import BytesIO

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["telegram_data"]
users_col = db["users"]
messages_col = db["messages"]
groups_col = db["UserGroups"]
chats_col = db["chats"]

st.set_page_config(layout="wide")
st.title("📊 Telegram User Full Profile Summary")

# Load user options
users = list(users_col.find({}, {"user_id": 1, "first_name": 1, "username": 1, "last_name": 1}))
options = {
    f"{u.get('first_name', '')} {u.get('last_name', '')} (@{u.get('username', '')})": u["user_id"]
    for u in users if "user_id" in u
}

user_key = st.selectbox("Select a User", list(options.keys()))
if user_key:
    user_id = options[user_key]
    user = users_col.find_one({"user_id": user_id})
    messages = list(messages_col.find({"user_id": user_id}))
    user_groups = list(groups_col.find({"user_id": user_id}))
    group_ids = [g["chat_id"] for g in user_groups]
    active_groups = [g for g in user_groups if g.get("is_active")]

    # Group participants info
    chat_details = list(chats_col.find({"chat_id": {"$in": group_ids}}))

    st.header("👤 User Summary")
    col1, col2 = st.columns(2)
    col1.markdown(f"**Name:** {user.get('first_name', '')} {user.get('last_name', '')}")
    col1.markdown(f"**Username:** @{user.get('username', 'N/A')}")
    col1.markdown(f"**Phone:** {user.get('phone', 'N/A')}")
    col2.metric("💬 Total Messages", len(messages))
    col2.metric("👥 Groups Joined", len(user_groups))
    col2.metric("✅ Active Groups", len(active_groups))

    # Group participants chart
    if chat_details:
        df_chats = pd.DataFrame(chat_details)
        df_chats = df_chats[["title", "participants_count"]]
        df_chats.columns = ["Group Name", "Participants"]
        st.markdown("### 👥 Participants in Each Group")
        st.dataframe(df_chats)

        st.bar_chart(df_chats.set_index("Group Name"))

    # PDF Generator
    def generate_pdf(user, messages, user_groups, chat_details):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, f"User Report: {user.get('first_name', '')} {user.get('last_name', '')}", ln=True)
        pdf.cell(0, 10, f"User ID: {user.get('user_id')}", ln=True)
        pdf.cell(0, 10, f"Username: @{user.get('username', 'N/A')}", ln=True)
        pdf.cell(0, 10, f"Phone: {user.get('phone', 'N/A')}", ln=True)
        pdf.cell(0, 10, f"Total Messages: {len(messages)}", ln=True)
        pdf.cell(0, 10, f"Groups Joined: {len(user_groups)}", ln=True)
        pdf.cell(0, 10, f"Active Groups: {len(active_groups)}", ln=True)

        pdf.cell(0, 10, "Participants in Groups:", ln=True)
        for c in chat_details:
            pdf.cell(0, 10, f"{c.get('title', '')} - {c.get('participants_count', 0)} members", ln=True)

        pdf.ln(5)
        pdf.cell(0, 10, "Recent Messages:", ln=True)
        for m in messages[:5]:
            pdf.multi_cell(0, 10, f"{m.get('date')} - {m.get('text', '')[:100]}")

        output = BytesIO()
        pdf.output(output, 'F')
        output.seek(0)
        return output

    pdf_data = generate_pdf(user, messages, user_groups, chat_details)

    st.download_button(
        label="📄 Download Full PDF Report",
        data=pdf_data,
        file_name=f"user_summary_{user_id}.pdf",
        mime="application/pdf"
    )
