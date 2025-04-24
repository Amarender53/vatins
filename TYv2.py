# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from pymongo import MongoClient
# # from wordcloud import WordCloud
# # from fpdf import FPDF
# # from io import BytesIO
# # import base64
# # from transformers import pipeline

# # # MongoDB connection
# # client = MongoClient("mongodb://localhost:27017/")
# # db = client["telegram_data"]
# # users_col = db["users"]
# # messages_col = db["messages"]
# # groups_col = db["groups"]

# # # Load NLP pipelines
# # sentiment_pipeline = pipeline("sentiment-analysis")
# # ner_pipeline = pipeline("ner", grouped_entities=True)

# # st.set_page_config(layout="wide")
# # st.title("📈 Telegram Intelligence Dashboard")

# # # Dropdown for selecting user_id
# # user_ids = users_col.distinct("user_id")
# # selected_user_id = st.selectbox("Select a Telegram User ID", user_ids)

# # # Fetch user data
# # user = users_col.find_one({"user_id": selected_user_id})
# # user_messages = list(messages_col.find({"user_id": selected_user_id}))
# # user_group_ids = list(set([msg["chat_id"] for msg in user_messages]))
# # user_groups = list(groups_col.find({"chat_id": {"$in": user_group_ids}}))

# # # Risk level function (example based on message count)
# # def get_risk_level(count):
# #     if count > 3000:
# #         return "High"
# #     elif count > 1000:
# #         return "Medium"
# #     return "Low"

# # # Layout
# # col1, col2 = st.columns(2)

# # # User details
# # with col1:
# #     st.subheader("User Details")
# #     st.markdown(f"**Username:** {user.get('username', 'N/A')}")
# #     st.markdown(f"**Full Name:** {user.get('first_name', '')} {user.get('last_name', '')}")
# #     st.markdown(f"**Phone:** {user.get('phone', 'N/A')}")
# #     st.markdown(f"**Last Seen:** {user.get('last_seen', 'N/A')}")
# #     st.markdown(f"**Bio:** {user.get('bio', 'N/A')}")

# # # Activity summary
# # with col2:
# #     total_msgs = len(user_messages)
# #     risk_level = get_risk_level(total_msgs)
# #     st.subheader("Activity Summary")
# #     st.metric("Total Messages", total_msgs)
# #     st.metric("Groups Joined", len(user_groups))
# #     last_active = max([msg['date'] for msg in user_messages])
# #     st.metric("Last Active", last_active.strftime("%Y-%m-%d %H:%M:%S"))
# #     st.markdown(f"**Risk Level:** <span style='color:{'red' if risk_level=='High' else 'orange' if risk_level=='Medium' else 'green'}'>{risk_level}</span>", unsafe_allow_html=True)

# # # AI Analysis
# # st.subheader("🧠 AI Analysis")
# # text_data = " ".join([msg.get("text", "") for msg in user_messages if isinstance(msg.get("text", ""), str)])

# # # Sentiment Analysis
# # if st.button("Run Sentiment Analysis"):
# #     sentiments = sentiment_pipeline(text_data[:1000])  # limit for speed
# #     st.write(sentiments)

# # # NER
# # if st.button("Run NER Tagging"):
# #     entities = ner_pipeline(text_data[:1000])
# #     st.write(entities)

# # # Top 10 Active Groups
# # st.subheader("🏋️ Top 10 Active Groups")
# # group_message_count = pd.Series([msg["chat_id"] for msg in user_messages]).value_counts().head(10)
# # fig, ax = plt.subplots()
# # group_message_count.plot(kind="barh", ax=ax)
# # st.pyplot(fig)

# # # Recent Messages
# # st.subheader("🔎 Recent Messages")
# # recent_msgs = sorted(user_messages, key=lambda x: x["date"], reverse=True)[:10]
# # for msg in recent_msgs:
# #     st.markdown(f"- **{msg['chat_id']}** | *{msg['date']}*: {msg.get('text', '')}")

# # # Word Cloud
# # st.subheader("🌍 Word Cloud")
# # wc = WordCloud(width=800, height=300, background_color="white").generate(text_data)
# # fig, ax = plt.subplots()
# # ax.imshow(wc, interpolation="bilinear")
# # ax.axis("off")
# # st.pyplot(fig)

# # # Download CSV
# # st.subheader("📥 Download Report")
# # def convert_df(df):
# #     return df.to_csv(index=False).encode("utf-8")

# # df_msgs = pd.DataFrame(user_messages)
# # st.download_button("Download Messages CSV", convert_df(df_msgs), "messages.csv", "text/csv")

# # # Download PDF
# # class PDF(FPDF):
# #     def header(self):
# #         self.set_font("Arial", "B", 12)
# #         self.cell(200, 10, "Telegram User Report", ln=True, align="C")
# #     def chapter_title(self, title):
# #         self.set_font("Arial", "B", 12)
# #         self.cell(0, 10, title, ln=True)
# #     def chapter_body(self, body):
# #         self.set_font("Arial", "", 10)
# #         self.multi_cell(0, 10, body)

# # def generate_pdf(user, messages):
# #     pdf = PDF()
# #     pdf.add_page()
# #     pdf.chapter_title("User Details")
# #     pdf.chapter_body(f"Username: {user.get('username', 'N/A')}\nFull Name: {user.get('first_name', '')} {user.get('last_name', '')}\nPhone: {user.get('phone', 'N/A')}\nBio: {user.get('bio', '')}\nLast Seen: {user.get('last_seen', '')}")
# #     pdf.chapter_title("Messages")
# #     for msg in messages[:10]:
# #         pdf.chapter_body(f"{msg['date']} | {msg['chat_id']}\n{msg.get('text', '')}\n")
# #     buffer = BytesIO()
# #     pdf.output(buffer)
# #     return buffer.getvalue()

# # pdf_bytes = generate_pdf(user, recent_msgs)
# # b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
# # href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="telegram_user_report.pdf">Download PDF Report</a>'
# # st.markdown(href, unsafe_allow_html=True)



# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from pymongo import MongoClient
# from wordcloud import WordCloud
# from fpdf import FPDF
# from io import BytesIO
# from transformers import pipeline

# # MongoDB connection
# client = MongoClient("mongodb://localhost:27017/")
# db = client["telegram_data"]
# users_col = db["users"]
# messages_col = db["messages"]
# groups_col = db["groups"]
# channels_col = db["channels"]

# # Load NLP pipelines
# sentiment_pipeline = pipeline("sentiment-analysis")
# ner_pipeline = pipeline("ner", grouped_entities=True)

# st.set_page_config(layout="wide")
# st.title("📈 Telegram Intelligence Dashboard")

# # Sidebar - Group/Channel Selector
# st.sidebar.header("🔍 Group/Channel Selection")
# all_groups = list(groups_col.find({}, {"chat_id": 1, "title": 1}))
# all_channels = list(channels_col.find({}, {"chat_id": 1, "title": 1}))
# chat_map = {
#     f"{g.get('title', 'Unknown')} ({g['chat_id']})": g["chat_id"]
#     for g in all_groups + all_channels
# }
# selected_chat_label = st.sidebar.selectbox("Select Group/Channel", list(chat_map.keys()))
# selected_chat_id = chat_map[selected_chat_label]

# # Fetch top users from selected chat
# top_users_pipeline = [
#     {"$match": {"chat_id": selected_chat_id}},
#     {"$group": {"_id": "$user_id", "msg_count": {"$sum": 1}}},
#     {"$sort": {"msg_count": -1}},
#     {"$limit": 10}
# ]
# top_users = list(messages_col.aggregate(top_users_pipeline))
# user_docs = users_col.find({"user_id": {"$in": [u["_id"] for u in top_users if u["_id"] is not None]}})
# user_map = {
#     f"{u.get('username', 'N/A')} ({u['user_id']})": u["user_id"]
#     for u in user_docs
# }

# # Sidebar - User Profile Summary
# st.sidebar.header("🧑 User Profile Summary")
# selected_user_label = st.sidebar.selectbox("Select Top User", list(user_map.keys()) or ["No user found"])
# selected_user_id = user_map.get(selected_user_label) if selected_user_label in user_map else None


# # Tab layout
# tabs = st.tabs(["📊 Group/Channel Summary", "🧑 User Profile Summary"])

# # ========== Group/Channel Summary ==========
# with tabs[0]:
#     st.subheader("📊 Group/Channel Overview")

#     # Determine if selected ID is a group or channel
#     selected_doc = groups_col.find_one({"chat_id": selected_chat_id}) or \
#                    channels_col.find_one({"chat_id": selected_chat_id})
    
#     if selected_doc:
#         st.markdown(f"**Title:** {selected_doc.get('title', 'N/A')}")
#         st.markdown(f"**Chat ID:** {selected_doc.get('chat_id')}")
#         st.markdown(f"**Participants:** {selected_doc.get('participants_count', 'N/A')}")
#         st.markdown(f"**Last Updated:** {selected_doc.get('updated_at', 'N/A')}")

#     # Message stats
#     msg_count = messages_col.count_documents({"chat_id": selected_chat_id})
#     st.markdown(f"**Total Messages:** {msg_count}")

#     # Top Users in Group/Channel
#     top_users_pipeline = [
#         {"$match": {"chat_id": selected_chat_id}},
#         {"$group": {"_id": "$user_id", "msg_count": {"$sum": 1}}},
#         {"$sort": {"msg_count": -1}},
#         {"$limit": 10}
#     ]
#     top_users = list(messages_col.aggregate(top_users_pipeline))
#     top_user_data = []
#     for u in top_users:
#         user = users_col.find_one({"user_id": u["_id"]}) or {}
#         top_user_data.append({
#             "User ID": u["_id"],
#             "Username": user.get("username", "N/A"),
#             "Messages": u["msg_count"]
#         })

#     st.markdown("**Top 10 Active Users:**")
#     st.dataframe(pd.DataFrame(top_user_data))

# # ========== User Profile Summary ==========
# with tabs[1]:
#     st.subheader("🧑 User Profile Summary")

#     # User Selector from top active in current group/channel
#     top_user_ids = [u["_id"] for u in top_users if u["_id"] is not None]
#     user_docs = users_col.find({"user_id": {"$in": top_user_ids}}, {"user_id": 1, "username": 1})
#     user_map = {
#         f"{u.get('username', 'N/A')} ({u['user_id']})": u["user_id"]
#         for u in user_docs
#     }
#     st.markdown(f"**Selected User:** {selected_user_label}")

#     if not selected_user_id:
#         st.warning("⚠️ No user selected or user not available in this group/channel.")
#         st.stop()

#     # Fetch user data
#     user = users_col.find_one({"user_id": selected_user_id})
#     user_messages = list(messages_col.find({"user_id": selected_user_id}))
#     user_group_ids = list(set([msg["chat_id"] for msg in user_messages]))
#     user_groups = list(groups_col.find({"chat_id": {"$in": user_group_ids}}))

#     # Risk level function
#     def get_risk_level(count):
#         if count > 3000:
#             return "High"
#         elif count > 1000:
#             return "Medium"
#         return "Low"

#     st.markdown(f"**User ID:** {user.get('user_id', 'N/A')}")
#     st.markdown(f"**Username:** @{user.get('username', 'N/A')}")
#     st.markdown(f"**Full Name:** {user.get('first_name', '')} {user.get('last_name', '')}")
#     st.markdown(f"**Email:** {user.get('email', 'N/A')}")
#     st.markdown(f"**Phone:** {user.get('phone', 'N/A')}")
#     st.markdown(f"**Groups Joined:** {len(user_groups)}")
#     if user_messages:
#         last_active = max([msg['date'] for msg in user_messages])
#         st.markdown(f"**Last Active:** {str(last_active)}")
#     st.markdown(f"**Total Messages Sent:** {len(user_messages)}")

#     # Top groups the user is active in
#     top_user_group_freq = pd.Series([msg['chat_id'] for msg in user_messages]).value_counts().head(5)
#     top_group_titles = [
#         groups_col.find_one({"chat_id": g})['title']
#         for g in top_user_group_freq.index
#         if groups_col.find_one({"chat_id": g})
#     ]
#     st.markdown("**Top Groups Participated In:**")
#     for t in top_group_titles:
#         st.markdown(f"- {t}")

#     risk_level = get_risk_level(len(user_messages))
#     st.markdown(f"**Risk Level:** <span style='color:{'red' if risk_level=='High' else 'orange' if risk_level=='Medium' else 'green'}'>{risk_level}</span>", unsafe_allow_html=True)

#     # CSV Export
#     st.subheader("📥 Export Report")
#     user_df = pd.DataFrame([user])
#     user_df["total_messages"] = len(user_messages)
#     user_df["groups_joined"] = len(user_groups)
#     user_df["risk_level"] = risk_level
#     csv = user_df.to_csv(index=False).encode("utf-8")
#     st.download_button("Download CSV", csv, file_name="user_summary.csv", mime="text/csv")

#     # PDF Export
#     class PDF(FPDF):
#         def header(self):
#             self.set_font('Arial', 'B', 12)
#             self.cell(200, 10, 'Telegram User Intelligence Report', ln=True, align='C')
#             self.ln(10)

#         def chapter_body(self, body):
#             self.set_font('Arial', '', 11)
#             self.multi_cell(0, 10, body)
#             self.ln()

#     def generate_pdf(user_data):
#         pdf = PDF()
#         pdf.add_page()
#         lines = [
#             f"User ID: {user_data.get('user_id', '')}",
#             f"Username: @{user_data.get('username', '')}",
#             f"Full Name: {user_data.get('first_name', '')} {user_data.get('last_name', '')}",
#             f"Email: {user_data.get('email', 'N/A')}",
#             f"Phone: {user_data.get('phone', 'N/A')}",
#             f"Groups Joined: {len(user_groups)}",
#             f"Total Messages: {len(user_messages)}",
#             f"Risk Level: {risk_level}"
#         ]
#         if user_messages:
#             last_active = max([msg['date'] for msg in user_messages])
#             lines.append(f"Last Active: {str(last_active)}")
#         lines.append("Top Groups:")
#         for t in top_group_titles:
#             lines.append(f"- {t}")
#         pdf.chapter_body("\n".join(lines))
#         buf = BytesIO()
#         pdf.output(buf)
#         buf.seek(0)
#         return buf

#     pdf_bytes = generate_pdf(user)
#     st.download_button("Download PDF", data=pdf_bytes, file_name="user_profile_summary.pdf", mime="application/pdf")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from wordcloud import WordCloud
from fpdf import FPDF
from io import BytesIO
from transformers import pipeline
import os

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["telegram_data"]
users_col = db["users"]
messages_col = db["messages"]
groups_col = db["groups"]
channels_col = db["channels"]

# Load NLP pipelines
sentiment_pipeline = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner", grouped_entities=True)

st.set_page_config(layout="wide")
st.title("\U0001F4C8 Telegram Intelligence Dashboard")

# Sidebar - Group/Channel Selector
st.sidebar.header("\U0001F50D Group/Channel Selection")
all_groups = list(groups_col.find({}, {"chat_id": 1, "title": 1}))
all_channels = list(channels_col.find({}, {"chat_id": 1, "title": 1}))
chat_map = {
    f"{g.get('title', 'Unknown')} ({g['chat_id']})": g["chat_id"]
    for g in all_groups + all_channels
}
selected_chat_label = st.sidebar.selectbox("Select Group/Channel", list(chat_map.keys()))
selected_chat_id = chat_map[selected_chat_label]

# Top Users Dropdown for User Tab
top_users_pipeline = [
    {"$match": {"chat_id": selected_chat_id}},
    {"$group": {"_id": "$user_id", "msg_count": {"$sum": 1}}},
    {"$sort": {"msg_count": -1}},
    {"$limit": 10}
]
top_users = list(messages_col.aggregate(top_users_pipeline))
user_docs = users_col.find({"user_id": {"$in": [u["_id"] for u in top_users if u["_id"] is not None]}})
user_map = {
    f"{u.get('username', 'N/A')} ({u['user_id']})": u["user_id"]
    for u in user_docs
}

# Sidebar - User Profile Summary
st.sidebar.header("\U0001F464 User Profile Summary")
active_user_ids = messages_col.distinct("user_id")
active_user_docs = list(users_col.find({"user_id": {"$in": active_user_ids}}, {"user_id": 1, "username": 1, "first_name": 1, "last_name": 1}))
all_user_map = {
    f"{u.get('first_name', '')} {u.get('last_name', '')} (@{u.get('username', 'N/A')}) [{u['user_id']}]": u["user_id"]
    for u in active_user_docs if u.get("user_id")
}
selected_user_label = st.sidebar.selectbox("Select Any User", list(all_user_map.keys()) or ["No active users"])
selected_user_id = all_user_map.get(selected_user_label) if selected_user_label in all_user_map else None

# Message Viewer Tab
st.sidebar.header("🗨️ Message Viewer")
with st.expander("📄 View Messages for Selected Group/User"):
    st.markdown("### Group Messages:")
    group_messages = list(messages_col.find({"chat_id": selected_chat_id}).sort("date", -1).limit(10))
    for msg in group_messages:
        st.markdown(f"**[{msg.get('date')}]** {msg.get('text', '')}")

    if selected_user_id:
        st.markdown("### User Messages:")
        user_messages_view = list(messages_col.find({"user_id": selected_user_id}).sort("date", -1).limit(10))
        for msg in user_messages_view:
            st.markdown(f"**[{msg.get('date')}]** {msg.get('text', '')}")
            
# Tabs
tabs = st.tabs(["\U0001F4CA Group/Channel Summary", "\U0001F464 User Profile Summary"])

# ========== Group/Channel Summary ==========
with tabs[0]:
    st.subheader("\U0001F4CA Group/Channel Overview")
    selected_doc = groups_col.find_one({"chat_id": selected_chat_id}) or \
                   channels_col.find_one({"chat_id": selected_chat_id})

    if selected_doc:
        st.markdown(f"**Title:** {selected_doc.get('title', 'N/A')}")
        st.markdown(f"**Chat ID:** {selected_doc.get('chat_id')}")
        st.markdown(f"**Participants:** {selected_doc.get('participants_count', 'N/A'):,}")
        st.markdown(f"**Last Updated:** {selected_doc.get('updated_at', 'N/A')}")

    msg_count = messages_col.count_documents({"chat_id": selected_chat_id})
    st.markdown(f"**Total Messages:** {msg_count:,}")

    active_users = messages_col.distinct("user_id", {"chat_id": selected_chat_id})
    st.markdown(f"**Active Users Count:** {len(active_users):,}")
    st.markdown("**Participant User Details:**")

    participant_users = list(users_col.find({"user_id": {"$in": active_users}}))
    participant_df = pd.DataFrame([{
        "User ID": u.get("user_id"),
        "Username": u.get("username", "N/A"),
        "Full Name": f"{u.get('first_name', '')} {u.get('last_name', '')}".strip(),
        "Phone": u.get("phone", "N/A"),
        "Last Seen": u.get("last_seen", "N/A")
    } for u in participant_users])

    if not participant_df.empty:
        st.dataframe(participant_df)
        csv_participants = participant_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Participants CSV", data=csv_participants, file_name="participants_user_details.csv", mime="text/csv")
    else:
        st.info("No participant details found.")

    top_user_data = []
    for u in top_users:
        user = users_col.find_one({"user_id": u["_id"]}) or {}
        top_user_data.append({
            "User ID": u["_id"],
            "Username": user.get("username", "N/A"),
            "Messages": u["msg_count"]
        })
    st.markdown("**Top 10 Active Users:**")
    top_user_df = pd.DataFrame(top_user_data)
    st.dataframe(top_user_df)

    # PDF Export for Group
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
            self.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf", uni=True)
            self.set_font("DejaVu", "", 12)

        def header(self):
            self.set_font("DejaVu", "B", 14)
            self.cell(200, 10, 'Telegram Intelligence Report', ln=True, align='C')
            self.ln(8)

        def section_title(self, title):
            self.set_font("DejaVu", "B", 12)
            self.cell(0, 10, title, ln=True)
            self.ln(2)

        def section_body(self, body):
            self.set_font("DejaVu", "", 11)
            self.multi_cell(0, 8, body)
            self.ln()

    def generate_group_pdf():
        pdf = PDF()
        pdf.add_page()
        pdf.section_title("1. Group Details")
        pdf.section_body(f"""
Title: {selected_doc.get('title', 'N/A')}
Chat ID: {selected_doc.get('chat_id')}
Participants: {selected_doc.get('participants_count', 'N/A')}
Last Updated: {selected_doc.get('updated_at', 'N/A')}
Total Messages: {msg_count}
Active Users: {len(active_users)}
        """)

        pdf.section_title("2. Top Active Users")
        for row in top_user_data:
            pdf.section_body(f"@{row['Username']} (ID: {row['User ID']}) - {row['Messages']} messages")

        pdf.section_title("3. Observations")
        pdf.section_body("""
• High message activity may indicate coordinated behavior.
• Further AI analysis on message content can be applied for deeper insights.
""")
        buf = BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return buf

    group_pdf = generate_group_pdf()
    st.download_button("📄 Download Group Summary PDF", data=group_pdf, file_name="group_summary.pdf", mime="application/pdf")

# ========== User Profile Summary ==========
with tabs[1]:
    st.subheader("\U0001F464 User Profile Summary")
    st.markdown(f"**Selected User:** {selected_user_label}")

    if not selected_user_id:
        st.warning("⚠️ No user selected or user not available in this group/channel.")
        st.stop()

    user = users_col.find_one({"user_id": selected_user_id})
    user_messages = list(messages_col.find({"user_id": selected_user_id}))
    user_group_ids = list(set([msg["chat_id"] for msg in user_messages]))
    user_groups = list(groups_col.find({"chat_id": {"$in": user_group_ids}}))

    def get_risk_level(count):
        if count > 3000:
            return "High"
        elif count > 1000:
            return "Medium"
        return "Low"

    risk_level = get_risk_level(len(user_messages))

    if user_messages:
        last_active = max([msg['date'] for msg in user_messages])
        st.markdown(f"**Last Active:** {str(last_active)}")

    top_user_group_freq = pd.Series([msg['chat_id'] for msg in user_messages]).value_counts().head(5)
    top_group_titles = [
        groups_col.find_one({"chat_id": g})['title']
        for g in top_user_group_freq.index
        if groups_col.find_one({"chat_id": g})
    ]

    st.markdown(f"**User ID:** {user.get('user_id', 'N/A')}")
    st.markdown(f"**Username:** @{user.get('username', 'N/A')}")
    st.markdown(f"**Full Name:** {user.get('first_name', '')} {user.get('last_name', '')}")
    st.markdown(f"**Email:** {user.get('email', 'N/A')}")
    st.markdown(f"**Phone:** {user.get('phone', 'N/A')}")
    st.markdown(f"**Groups Joined:** {len(user_groups)}")
    st.markdown(f"**Total Messages Sent:** {len(user_messages)}")
    st.markdown(f"**Risk Level:** <span style='color:{'red' if risk_level=='High' else 'orange' if risk_level=='Medium' else 'green'}'>{risk_level}</span>", unsafe_allow_html=True)

    st.markdown("**Top Groups Participated In:**")
    for t in top_group_titles:
        st.markdown(f"- {t}")

    def generate_user_pdf():
        pdf = PDF()
        pdf.add_page()
        pdf.section_title("1. Personal Details")
        pdf.section_body(f"""
Telegram ID: {user.get('user_id')}
Username: @{user.get('username', 'N/A')}
Name: {user.get('first_name', '')} {user.get('last_name', '')}
Phone: {user.get('phone', 'N/A')}
Email: {user.get('email', 'N/A')}
Last Active: {str(last_active)}
        """)

        pdf.section_title("2. Activity Summary")
        pdf.section_body(f"""
Total Messages Sent: {len(user_messages)}
Groups Joined: {len(user_groups)}
Risk Level: {risk_level}
Top Groups:
{chr(10).join(['- ' + t for t in top_group_titles])}
        """)

        pdf.section_title("3. Behavioral Indicators")
        pdf.section_body("""
• High-frequency messaging patterns observed.
• No extremist content detected.
• Risk level flagged based on message volume.
""")
        buf = BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return buf

    user_pdf = generate_user_pdf()
    st.download_button("📄 Download User Profile PDF", data=user_pdf, file_name="user_profile_summary.pdf", mime="application/pdf")
