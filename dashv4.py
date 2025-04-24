import streamlit as st
import pandas as pd
import pymysql
import json
import pydeck as pdk
from io import BytesIO
from fpdf import FPDF

# Function to connect to the MySQL database
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

def load_user_groups_messages(user_id):
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='5368',
        database='VATINS',
        cursorclass=pymysql.cursors.DictCursor
    )
    with connection.cursor() as cursor:
        # Total groups and active groups
        cursor.execute("""
            SELECT COUNT(*) as total_groups,
                   SUM(CASE WHEN is_active THEN 1 ELSE 0 END) AS active_groups
            FROM UserGroups WHERE user_id = %s
        """, (user_id,))
        groups = cursor.fetchone()

        # Total messages
        cursor.execute("SELECT COUNT(*) as total_messages FROM UserMessages WHERE user_id = %s", (user_id,))
        messages = cursor.fetchone()

        # Category-wise messages
        cursor.execute("""
            SELECT tip_category, COUNT(*) as count
            FROM Tipline WHERE user_id = %s GROUP BY tip_category
        """, (user_id,))
        category_counts = cursor.fetchall()

        return groups, messages, category_counts

# Helper: decode JSON
def decode_json_field(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except:
        return []

# Helper: sentiment color
def get_sentiment_color(sentiment):
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return "🟢 Positive"
    elif sentiment == 'negative':
        return "🔴 Negative"
    else:
        return "🟠 Neutral"

# Helper: risk level label
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

# PDF export
def generate_pdf_summary(user_data, category_counts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"User Summary: {user_data['user_id']}", ln=1)

    pdf.cell(200, 10, txt=f"Total Tips: {user_data['total_tips_sent']}", ln=1)
    pdf.cell(200, 10, txt=f"Active Days: {user_data['active_days']}", ln=1)
    pdf.cell(200, 10, txt=f"Risk Score: {user_data['reporter_risk_score']:.2f}", ln=1)
    pdf.cell(200, 10, txt=f"Average Sentiment: {user_data['average_sentiment']}", ln=1)

    pdf.cell(200, 10, txt=f"First Tip: {user_data['first_tip_date']}", ln=1)
    pdf.cell(200, 10, txt=f"Last Tip: {user_data['last_tip_date']}", ln=1)

    pdf.cell(200, 10, txt="Top Categories: " + ', '.join(decode_json_field(user_data['common_categories'])), ln=1)
    pdf.cell(200, 10, txt="Top Emotions: " + ', '.join(decode_json_field(user_data['top_emotions'])), ln=1)
    pdf.cell(200, 10, txt="Top Keywords: " + ', '.join(decode_json_field(user_data['top_keywords'])), ln=1)

    pdf.cell(200, 10, txt="Category-Wise Tips:", ln=1)
    for cat in category_counts:
        pdf.cell(200, 10, txt=f" - {cat['tip_category']}: {cat['count']}", ln=1)

    # Use dest='S' to return as string, then encode to bytes
    pdf_output = pdf.output(dest='S').encode('latin-1')  # FPDF uses latin-1 encoding
    return BytesIO(pdf_output)

# Page config
st.set_page_config(page_title="User Profiling Dashboard", layout="wide")
st.title("AI User Profiling Dashboard")

# Load and validate data
df = load_user_profiles()
if df.empty:
    st.warning("No profile data available.")
    st.stop()

user_ids = df["user_id"].unique()
selected_user = st.sidebar.selectbox("Select User", user_ids)
user_data = df[df["user_id"] == selected_user].iloc[0]

# Convert Decimal fields to Python native types
user_data["reporter_risk_score"] = float(user_data["reporter_risk_score"] or 0)
user_data["total_tips_sent"] = int(user_data["total_tips_sent"] or 0)
user_data["active_days"] = int(user_data["active_days"] or 0)

# Decode JSON fields
common_categories = decode_json_field(user_data["common_categories"])
top_emotions = decode_json_field(user_data["top_emotions"])
top_keywords = decode_json_field(user_data["top_keywords"])
location = decode_json_field(user_data["last_known_location"])

# Load group/message data
group_data, message_data, category_counts = load_user_groups_messages(selected_user)

# Convert group/message values to safe types
group_total = int(group_data["total_groups"] or 0)
group_active = int(group_data["active_groups"] or 0)
total_messages = int(message_data["total_messages"] or 0)

# Display user metrics
col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total Tips", user_data["total_tips_sent"])
col2.metric("📅 Active Days", user_data["active_days"])
col3.metric("🔥 Risk Level", get_risk_label(user_data["reporter_risk_score"]))

# Display group/message stats
col4, col5, col6 = st.columns(3)
col4.metric("👥 Groups Joined", group_total)
col5.metric("✅ Active Groups", group_active)
col6.metric("💬 Total Messages", total_messages)

# Timeline
col7, col8 = st.columns(2)
col7.markdown(f"**First Tip:** {user_data['first_tip_date']}")
col8.markdown(f"**Last Tip:** {user_data['last_tip_date']}")

# Sentiment & Geo
st.markdown(f"**Average Sentiment:** {get_sentiment_color(user_data['average_sentiment'])}")
st.markdown(f"**Geo Tags Detected:** {'📍 Yes' if user_data['geo_tags_detected'] else '❌ No'}")

# Top info
st.markdown("### Top Categories, Emotions & Keywords")
st.markdown(f"**Top Categories:** {', '.join(common_categories) if common_categories else 'None'}")
st.markdown(f"**Top Emotions:** {', '.join(top_emotions) if top_emotions else 'None'}")
st.markdown(f"**Top Keywords:** {', '.join(top_keywords) if top_keywords else 'None'}")

# Category-wise tips
st.markdown("### 🗂 Category-wise Tips Submitted")
for item in category_counts:
    st.markdown(f"- {item['tip_category']}: {item['count']} tips")

# Map
if user_data["geo_tags_detected"] and location:
    st.markdown("### 🌍 Last Known Location")
    st.pydeck_chart(pdk.Deck(
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

# Downloads
st.markdown("### 📥 Download Summary")
colpdf, colcsv = st.columns(2)

with colpdf:
    pdf_buffer = generate_pdf_summary(user_data, category_counts)
    st.download_button("📄 Download PDF", data=pdf_buffer, file_name="user_profile_summary.pdf", mime="application/pdf")

with colcsv:
    csv_data = pd.DataFrame([user_data]).to_csv(index=False)
    st.download_button("📊 Download CSV", data=csv_data, file_name="user_profile_summary.csv", mime="text/csv")
