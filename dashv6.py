######## MulTiplinele Maps Sattilite & HeatMap
import streamlit as st
import pandas as pd
import pymysql
import json
import pydeck as pdk
from io import BytesIO
from fpdf import FPDF
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------- DATABASE FUNCTIONS ----------
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
        cursor.execute("""
            SELECT COUNT(*) as total_groups,
                   SUM(CASE WHEN is_active THEN 1 ELSE 0 END) AS active_groups
            FROM UserGroups WHERE user_id = %s
        """, (user_id,))
        groups = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) as total_messages FROM UserMessages WHERE user_id = %s", (user_id,))
        messages = cursor.fetchone()

        cursor.execute("""
            SELECT Tipline_category, COUNT(*) as count
            FROM Tiplineline WHERE user_id = %s GROUP BY Tipline_category
        """, (user_id,))
        category_counts = cursor.fetchall()

        return groups, messages, category_counts

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

# ---------- UTILS ----------
def decode_json_field(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except:
        return []

def get_sentiment_color(sentiment):
    sentiment = sentiment.lower()
    if sentiment == 'positive':
        return "🟢 Positive"
    elif sentiment == 'negative':
        return "🔴 Negative"
    else:
        return "🟠 Neutral"

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

def extract_all_locations(df):
    locations = []
    for _, row in df.iterrows():
        loc = decode_json_field(row.get("last_known_location", {}))
        if isinstance(loc, dict) and "latitude" in loc and "longitude" in loc:
            lat = loc["latitude"]
            lon = loc["longitude"]
            place_name = get_place_name(lat, lon)
            locations.append({
                "lat": lat,
                "lon": lon,
                "user_id": row["user_id"],
                "risk": float(row.get("reporter_risk_score", 0)),
                "place": place_name
            })
    return pd.DataFrame(locations)

def generate_pdf_summary(user_data, category_counts, uploaded_image=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="User Profile Summary", ln=True, align='C')
    pdf.ln(10)

    keys_to_include = ['user_id', 'user_name', 'reporter_risk_score', 'total_Tiplines_sent', 'active_days',
                       'first_Tipline_date', 'last_Tipline_date', 'average_sentiment', 'geo_tags_detected']
    
    for key in keys_to_include:
        label = key.replace("_", " ").title()
        value = user_data.get(key, 'N/A')
        pdf.cell(200, 10, txt=f"{label}: {value}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Tipline Categories:", ln=True)

    pdf.set_font("Arial", size=11)
    for cat in category_counts:
        pdf.cell(200, 10, txt=f" - {cat['Tipline_category']}: {cat['count']}", ln=True)

    if uploaded_image:
        pdf.ln(10)
        image_path = "temp_map_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())
        pdf.image(image_path, x=10, y=pdf.get_y(), w=180)

    pdf_output = pdf.output(dest='S').encode('latin1')
    buffer = BytesIO(pdf_output)
    buffer.seek(0)
    return buffer

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="User Profiling Dashboard", layout="wide")
st.title("AI User Profiling Dashboard")

df = load_user_profiles()
if df.empty:
    st.warning("No profile data available.")
    st.stop()

user_ids = df["user_id"].unique()
selected_user = st.sidebar.selectbox("Select User", user_ids)
user_data = df[df["user_id"] == selected_user].iloc[0]

user_data["reporter_risk_score"] = float(user_data["reporter_risk_score"] or 0)
user_data["total_Tiplines_sent"] = int(user_data["total_Tiplines_sent"] or 0)
user_data["active_days"] = int(user_data["active_days"] or 0)

common_categories = decode_json_field(user_data["common_categories"])
top_emotions = decode_json_field(user_data["top_emotions"])
top_keywords = decode_json_field(user_data["top_keywords"])
location = decode_json_field(user_data["last_known_location"])

group_data, message_data, category_counts = load_user_groups_messages(selected_user)
user_messages_df = load_user_messages(selected_user)

col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total Tiplines", user_data["total_Tiplines_sent"])
col2.metric("📅 Active Days", user_data["active_days"])
col3.metric("🔥 Risk Level", get_risk_label(user_data["reporter_risk_score"]))

col4, col5 = st.columns(2)
col4.markdown(f"**First Tipline:** {user_data['first_Tipline_date']}")
col5.markdown(f"**Last Tipline:** {user_data['last_Tipline_date']}")

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

st.markdown("### Category-wise Tiplines Submitted")
for item in category_counts:
    st.markdown(f"- {item['Tipline_category']}: {item['count']} Tiplines")

@st.cache_data(show_spinner=False)
def get_place_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="streamlit-location")
        reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
        location = reverse((lat, lon), language="en")
        return location.address if location else "Unknown"
    except:
        return "Unknown"


if user_data["geo_tags_detected"] and location:
    st.markdown("### 🌍 Last Known Location")

    lat = location["latitude"]
    lon = location["longitude"]
    place_name = get_place_name(lat, lon)

    location_data = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "user_id": selected_user,
        "place": place_name
    }])

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=12,
            pitch=40,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=location_data,
                get_position='[lon, lat]',
                get_radius=300,
                get_fill_color=[255, 0, 0, 160],
                pickable=True,
            )
        ],
        toolTipline={"text": "User ID: {user_id}\nPlace: {place}"}
    ))

    st.markdown(f"**Place Name:** 📍 {place_name}")
else:
    st.info("No geo-location data available.")

# ---------- USER MESSAGES ----------
st.markdown("### User Messages")
if not user_messages_df.empty:
    for idx, row in user_messages_df.iterrows():
        with st.expander(f"🕒 {row['timestamp']} — 📌 Group: {row['group_name']}"):
            st.markdown(f"**Message ID:** {row['message_id']}")
            st.markdown(f"**Text:** {row['message_text']}")
else:
    st.info("No messages found for this user.")

# ---------- DOWNLOAD ----------
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
