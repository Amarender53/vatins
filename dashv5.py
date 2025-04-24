########## Having Satellite Street View #############
import streamlit as st
import pandas as pd
import pymysql
import json
import pydeck as pdk
from io import BytesIO
from fpdf import FPDF

# Function to connect to the MySQL database and load profile summaries
def load_user_profiles():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='5368',
        database='VATINS',
        cursorclass=pymysql.cursors.DictCursor
    )
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM UserProfileSummary")
        rows = cursor.fetchall()
    return pd.DataFrame(rows)

# Load group stats and category counts
def load_user_groups_messages(user_id):
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='5368',
        database='VATINS',
        cursorclass=pymysql.cursors.DictCursor
    )
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) as total_groups,
                   SUM(CASE WHEN is_active THEN 1 ELSE 0 END) AS active_groups
            FROM UserGroups WHERE user_id = %s
        """, (user_id,))
        groups = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) as total_messages FROM UserMessages WHERE user_id = %s", (user_id,))
        messages = cursor.fetchone()

        cursor.execute("""
            SELECT tip_category, COUNT(*) as count
            FROM Tipline WHERE user_id = %s GROUP BY tip_category
        """, (user_id,))
        category_counts = cursor.fetchall()

        return groups, messages, category_counts

# Load all messages by a user
def load_user_messages(user_id):
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='5368',
        database='VATINS',
        cursorclass=pymysql.cursors.DictCursor
    )
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT message_id, message_text, group_name, timestamp 
            FROM UserMessages 
            WHERE user_id = %s 
            ORDER BY timestamp DESC
        """, (user_id,))
        messages = cursor.fetchall()
    return pd.DataFrame(messages)


# Decode JSON fields safely
def decode_json_field(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except:
        return []

# Sentiment emoji
def get_sentiment_color(sentiment):
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return "🟢 Positive"
    elif sentiment == 'negative':
        return "🔴 Negative"
    else:
        return "🟠 Neutral"

# Risk emoji
def get_risk_label(score):
    if score is None:
        return "⚪ Unknown"
    elif score >= 0.7:
        return f"🔴 High Risk ({score:.2f})"
    elif score >= 0.5:
        return f"🟠 Risk ({score:.2f})"
    elif score >= 0.25:
        return f"🟡 Medium ({score:.2f})"
    else:
        return f"🟢 Low ({score:.2f})"

# PDF Export
from fpdf import FPDF
from io import BytesIO

def generate_pdf_summary(user_data, category_counts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="User Profile Summary", ln=True, align='C')
    pdf.ln(10)

    keys_to_include = ['user_id', 'user_name', 'reporter_risk_score', 'total_tips_sent', 'active_days',
                       'first_tip_date', 'last_tip_date', 'average_sentiment', 'geo_tags_detected']
    
    for key in keys_to_include:
        label = key.replace("_", " ").title()
        value = user_data.get(key, 'N/A')
        pdf.cell(200, 10, txt=f"{label}: {value}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Tip Categories:", ln=True)

    pdf.set_font("Arial", size=11)
    for cat in category_counts:
        pdf.cell(200, 10, txt=f" - {cat['tip_category']}: {cat['count']}", ln=True)

    # Correct way to export PDF to BytesIO
    pdf_output = pdf.output(dest='S').encode('latin1')
    buffer = BytesIO(pdf_output)
    buffer.seek(0)
    return buffer


# App config
st.set_page_config(page_title="User Profiling Dashboard", layout="wide")
st.title("AI User Profiling Dashboard")

df = load_user_profiles()
if df.empty:
    st.warning("No profile data available.")
    st.stop()

user_ids = df["user_id"].unique()
selected_user = st.sidebar.selectbox("Select User", user_ids)
user_data = df[df["user_id"] == selected_user].iloc[0]

# Type conversions
user_data["reporter_risk_score"] = float(user_data["reporter_risk_score"] or 0)
user_data["total_tips_sent"] = int(user_data["total_tips_sent"] or 0)
user_data["active_days"] = int(user_data["active_days"] or 0)

common_categories = decode_json_field(user_data["common_categories"])
top_emotions = decode_json_field(user_data["top_emotions"])
top_keywords = decode_json_field(user_data["top_keywords"])
location = decode_json_field(user_data["last_known_location"])

group_data, message_data, category_counts = load_user_groups_messages(selected_user)
user_messages_df = load_user_messages(selected_user)

# Dashboard
col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total Tips", user_data["total_tips_sent"])
col2.metric("📅 Active Days", user_data["active_days"])
col3.metric("🔥 Risk Level", get_risk_label(user_data["reporter_risk_score"]))

col4, col5 = st.columns(2)
col4.markdown(f"**First Tip:** {user_data['first_tip_date']}")
col5.markdown(f"**Last Tip:** {user_data['last_tip_date']}")

st.markdown(f"**Average Sentiment:** {get_sentiment_color(user_data['average_sentiment'])}")
st.markdown(f"**Geo Tags Detected:** {'📍 Yes' if user_data['geo_tags_detected'] else '❌ No'}")

colg1, colg2, colg3 = st.columns(3)
colg1.metric("🧑‍🤝‍🧑 Groups Joined", int(group_data['total_groups'] or 0))
colg2.metric("📍 Active Groups", int(group_data['active_groups'] or 0))
colg3.metric("💬 Total Messages", int(message_data['total_messages'] or 0))

st.markdown("### Top Categories, Emotions & Keywords")
st.markdown(f"**Top Categories:** {', '.join(common_categories) if common_categories else 'None'}")
st.markdown(f"**Top Emotions:** {', '.join(top_emotions) if top_emotions else 'None'}")
st.markdown(f"**Top Keywords:** {', '.join(top_keywords) if top_keywords else 'None'}")

st.markdown("### Category-wise Tips Submitted")
for item in category_counts:
    st.markdown(f"- {item['tip_category']}: {item['count']} tips")

if user_data["geo_tags_detected"] and location:
    st.markdown("### 🌍 Last Known Location")
    st.pydeck_chart(pdk.Deck(
        map_style= 'mapbox://styles/mapbox/satellite-streets-v11',
        initial_view_state=pdk.ViewState(
            latitude=location["latitude"],
            longitude=location["longitude"],
            zoom=12,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=[{"lat": location["latitude"], "lon": location["longitude"]}],
                get_position='[lon, lat]',
                get_radius=200,
                get_color=[255, 0, 0],
                pickable=True,
            )
        ]
    ))
else:
    st.info("No geo-location data available.")

st.markdown("### User Messages")

if not user_messages_df.empty:
    for idx, row in user_messages_df.iterrows():
        with st.expander(f"🕒 {row['timestamp']} — 📌 Group: {row['group_name']}"):
            st.markdown(f"**Message ID:** {row['message_id']}")
            st.markdown(f"**Text:** {row['message_text']}")
else:
    st.info("No messages found for this user.")


# Downloads
st.markdown("### 📥 Download Summary")
colpdf, colcsv = st.columns(2)

with colpdf:
    pdf_buffer = generate_pdf_summary(user_data, category_counts)
    username = user_data.get("user_name", f"{selected_user}")
    pdf_filename = f"{username}_profile_summary.pdf"
    st.download_button("📄 Download PDF", data=pdf_buffer, file_name=pdf_filename, mime="application/pdf")

with colcsv:
    csv_data = pd.DataFrame([user_data]).to_csv(index=False)
    csv_filename = f"{username}_profile_summary.csv"
    st.download_button("📊 Download CSV", data=csv_data, file_name=csv_filename, mime="text/csv")
