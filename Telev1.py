# import streamlit as st
# import pandas as pd
# from pymongo import MongoClient
# from fpdf import FPDF
# import csv
# import io

# # MongoDB configuration
# MONGO_URI = "mongodb://localhost:27017/"
# DB_NAME = "telegram_data"

# # MongoDB connection
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# chats_collection = db["chats"]
# groups_collection = db["groups"]
# channels_collection = db["channels"]
# users_collection = db["users"]
# messages_collection = db["messages"]

# # Fetch data from MongoDB collections
# def fetch_chats():
#     return list(chats_collection.find())

# def fetch_groups():
#     return list(groups_collection.find())

# def fetch_channels():
#     return list(channels_collection.find())

# def fetch_users():
#     return list(users_collection.find())

# def fetch_messages():
#     return list(messages_collection.find())

# # Convert MongoDB data to DataFrame
# def create_dataframe(data, columns):
#     return pd.DataFrame(data, columns=columns)

# # Create PDF Summary
# def create_pdf_summary(chats_df, groups_df, channels_df, users_df, messages_df):
#     pdf = FPDF()
#     pdf.set_auto_page_break(auto=True, margin=15)
#     pdf.add_page()

#     # Title
#     pdf.set_font("Arial", size=14)
#     pdf.cell(200, 10, txt="Telegram Summary Report", ln=True, align="C")

#     # Chats Summary
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Chats Summary", ln=True)
#     pdf.set_font("Arial", size=10)
#     pdf.multi_cell(0, 10, txt=str(chats_df.to_string(index=False)))

#     # Groups Summary
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Groups Summary", ln=True)
#     pdf.set_font("Arial", size=10)
#     pdf.multi_cell(0, 10, txt=str(groups_df.to_string(index=False)))

#     # Channels Summary
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Channels Summary", ln=True)
#     pdf.set_font("Arial", size=10)
#     pdf.multi_cell(0, 10, txt=str(channels_df.to_string(index=False)))

#     # Users Summary
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Users Summary", ln=True)
#     pdf.set_font("Arial", size=10)
#     pdf.multi_cell(0, 10, txt=str(users_df.to_string(index=False)))

#     # Messages Summary
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt="Messages Summary", ln=True)
#     pdf.set_font("Arial", size=10)
#     pdf.multi_cell(0, 10, txt=str(messages_df.to_string(index=False)))

#     return pdf

# # Streamlit UI
# st.title('Telegram Data Dashboard')

# # Fetch data
# chats_data = fetch_chats()
# groups_data = fetch_groups()
# channels_data = fetch_channels()
# users_data = fetch_users()
# messages_data = fetch_messages()

# # Show data in tables
# st.header("Chats Data")
# chats_df = create_dataframe(chats_data, ["chat_id", "title", "type", "created_at", "updated_at"])
# st.write(chats_df)

# st.header("Groups Data")
# groups_df = create_dataframe(groups_data, ["chat_id", "title", "participants_count", "updated_at"])
# st.write(groups_df)

# st.header("Channels Data")
# channels_df = create_dataframe(channels_data, ["chat_id", "title", "participants_count", "updated_at"])
# st.write(channels_df)

# st.header("Users Data")
# users_df = create_dataframe(users_data, ["user_id", "first_name", "last_name", "username", "phone", "last_seen"])
# st.write(users_df)

# st.header("Messages Data")
# messages_df = create_dataframe(messages_data, ["message_id", "chat_id", "text", "date", "user_id", "metadata"])
# st.write(messages_df)

# # Export to CSV
# st.subheader("Download Data as CSV")
# csv_data = chats_df.to_csv(index=False)
# st.download_button("Download Chats Data as CSV", csv_data, file_name="chats_data.csv")

# csv_data = groups_df.to_csv(index=False)
# st.download_button("Download Groups Data as CSV", csv_data, file_name="groups_data.csv")

# csv_data = channels_df.to_csv(index=False)
# st.download_button("Download Channels Data as CSV", csv_data, file_name="channels_data.csv")

# csv_data = users_df.to_csv(index=False)
# st.download_button("Download Users Data as CSV", csv_data, file_name="users_data.csv")

# csv_data = messages_df.to_csv(index=False)
# st.download_button("Download Messages Data as CSV", csv_data, file_name="messages_data.csv")

# # Export to PDF
# st.subheader("Download Summary as PDF")

# # Create PDF
# pdf = create_pdf_summary(chats_df, groups_df, channels_df, users_df, messages_df)

# # Convert PDF to binary and allow download
# pdf_output = io.BytesIO()
# pdf.output(pdf_output)
# pdf_output.seek(0)

# st.download_button("Download PDF Summary", pdf_output, file_name="telegram_data_summary.pdf")


import streamlit as st
import pandas as pd
from pymongo import MongoClient
from fpdf import FPDF
import csv
import io

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "telegram_data"

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
chats_collection = db["chats"]
groups_collection = db["groups"]
channels_collection = db["channels"]
users_collection = db["users"]
messages_collection = db["messages"]

# Fetch data from MongoDB collections
def fetch_chats():
    return list(chats_collection.find())

def fetch_groups():
    return list(groups_collection.find())

def fetch_channels():
    return list(channels_collection.find())

def fetch_users():
    return list(users_collection.find())

def fetch_messages():
    return list(messages_collection.find())

# Convert MongoDB data to DataFrame
def create_dataframe(data, columns):
    return pd.DataFrame(data, columns=columns)

# Create PDF Summary with Unicode support
def create_pdf_summary(chats_df, groups_df, channels_df, users_df, messages_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add Unicode font (DejaVuSans)
    pdf.add_font('DejaVu', '', "E:\\VATINS\\fonts\\DejaVuSans.ttf", uni=True)
    pdf.set_font('DejaVu', '', 12)

    # Title
    pdf.cell(200, 10, txt="Telegram Summary Report", ln=True, align="C")

    # Chats Summary
    pdf.ln(10)
    pdf.set_font("DejaVu", '', 12)
    pdf.cell(200, 10, txt="Chats Summary", ln=True)
    pdf.set_font("DejaVu", '', 10)
    pdf.multi_cell(0, 10, txt=str(chats_df.to_string(index=False)))

    # Groups Summary
    pdf.ln(10)
    pdf.set_font("DejaVu", '', 12)
    pdf.cell(200, 10, txt="Groups Summary", ln=True)
    pdf.set_font("DejaVu", '', 10)
    pdf.multi_cell(0, 10, txt=str(groups_df.to_string(index=False)))

    # Channels Summary
    pdf.ln(10)
    pdf.set_font("DejaVu", '', 12)
    pdf.cell(200, 10, txt="Channels Summary", ln=True)
    pdf.set_font("DejaVu", '', 10)
    pdf.multi_cell(0, 10, txt=str(channels_df.to_string(index=False)))

    # Users Summary
    pdf.ln(10)
    pdf.set_font("DejaVu", '', 12)
    pdf.cell(200, 10, txt="Users Summary", ln=True)
    pdf.set_font("DejaVu", '', 10)
    pdf.multi_cell(0, 10, txt=str(users_df.to_string(index=False)))

    # Messages Summary
    pdf.ln(10)
    pdf.set_font("DejaVu", '', 12)
    pdf.cell(200, 10, txt="Messages Summary", ln=True)
    pdf.set_font("DejaVu", '', 10)
    pdf.multi_cell(0, 10, txt=str(messages_df.to_string(index=False)))

    return pdf

# Streamlit UI
st.title('Telegram Data Dashboard')

# Fetch data
chats_data = fetch_chats()
groups_data = fetch_groups()
channels_data = fetch_channels()
users_data = fetch_users()
messages_data = fetch_messages()

# Show data in tables
st.header("Chats Data")
chats_df = create_dataframe(chats_data, ["chat_id", "title", "type", "created_at", "updated_at"])
st.write(chats_df)

st.header("Groups Data")
groups_df = create_dataframe(groups_data, ["chat_id", "title", "participants_count", "updated_at"])
st.write(groups_df)

st.header("Channels Data")
channels_df = create_dataframe(channels_data, ["chat_id", "title", "participants_count", "updated_at"])
st.write(channels_df)

st.header("Users Data")
users_df = create_dataframe(users_data, ["user_id", "first_name", "last_name", "username", "phone", "last_seen"])
st.write(users_df)

st.header("Messages Data")
messages_df = create_dataframe(messages_data, ["message_id", "chat_id", "text", "date", "user_id", "metadata"])
st.write(messages_df)

# Export to CSV
st.subheader("Download Data as CSV")
csv_data = chats_df.to_csv(index=False)
st.download_button("Download Chats Data as CSV", csv_data, file_name="chats_data.csv")

csv_data = groups_df.to_csv(index=False)
st.download_button("Download Groups Data as CSV", csv_data, file_name="groups_data.csv")

csv_data = channels_df.to_csv(index=False)
st.download_button("Download Channels Data as CSV", csv_data, file_name="channels_data.csv")

csv_data = users_df.to_csv(index=False)
st.download_button("Download Users Data as CSV", csv_data, file_name="users_data.csv")

csv_data = messages_df.to_csv(index=False)
st.download_button("Download Messages Data as CSV", csv_data, file_name="messages_data.csv")

# Export to PDF
st.subheader("Download Summary as PDF")

# Create PDF
pdf = create_pdf_summary(chats_df, groups_df, channels_df, users_df, messages_df)

# Convert to bytes
pdf_bytes = pdf.output(dest='S').encode('latin1')  # Write to string, then encode
pdf_output = io.BytesIO(pdf_bytes)                 # Convert to BytesIO for Streamlit

# Show download button
st.download_button(
    label="📄 Download PDF Summary",
    data=pdf_output,
    file_name="telegram_data_summary.pdf",
    mime="application/pdf"
)



# Fetch message summary for a specific group/channel
def fetch_messages_summary(chat_id):
    messages = list(messages_collection.find({"chat_id": chat_id}))
    messages_summary = sorted(messages, key=lambda x: x['date'], reverse=True)[:10]
    return messages_summary

# Input for Chat ID
chat_id = st.number_input("Enter Chat ID", min_value=1)

if chat_id:
    messages_summary = fetch_messages_summary(chat_id)
    if messages_summary:
        message_data = pd.DataFrame(messages_summary, columns=["message_id", "text", "date", "user_id", "metadata"])
        st.header(f"Top 10 Messages for Chat ID {chat_id}")
        st.write(message_data)
    else:
        st.write("No messages found for this chat ID.")

