######## User Photo with Multiplinele Maps Sattilite & HeatMap ##############
import streamlit as st
import pandas as pd
import pymysql
import re
import json
import pydeck as pdk
from io import BytesIO
from fpdf import FPDF
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from PIL import Image
import os
import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from fpdf import FPDF
from datetime import datetime
from pathlib import Path



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
            SELECT tipline_category, COUNT(*) as count
            FROM Tipline WHERE user_id = %s GROUP BY tipline_category
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

def load_user_profiles():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='5368',
        database='VATINS',
        cursorclass=pymysql.cursors.DictCursor
    )
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                ups.*, 
                u.full_name AS name, 
                u.phone_number AS phonenumber, 
                u.email AS email
            FROM 
                UserProfileSummary ups
            LEFT JOIN 
                users u ON ups.user_id = u.user_id
        """)
        rows = cursor.fetchall()
    return pd.DataFrame(rows)


def generate_satellite_map_html(location_df):
    view_state = pdk.ViewState(
        latitude=location_df["lat"].mean(),
        longitude=location_df["lon"].mean(),
        zoom=11,
        pitch=0,
        bearing=0
    )

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=location_df,
        get_position='[lon, lat]',
        get_color='[255, 0, 0, 160]',
        get_radius=200,
    )

    r = pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v12",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "Lat: {lat}\nLon: {lon}"}
    )

    return r.to_html(as_string=True)

def save_map_screenshot(html_content, output_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8') as tmp_file:
        tmp_file.write(html_content)
        tmp_file_path = tmp_file.name

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1200x900")
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("file://" + tmp_file_path)
    time.sleep(5)  # Wait for map tiles and elements to load
    driver.save_screenshot(output_path)
    driver.quit()
    os.remove(tmp_file_path)

def generate_pdf_summary(user_data, category_counts, location_df, user_messages_df, group_data, message_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", '', 12)

    # Section Title
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="User Profile Summary", ln=True)

    # User Photo
    photo_path = os.path.join("E:/VATINS/photos", f"{user_data['user_id']}.jpg")
    if os.path.exists(photo_path):
        pdf.image(photo_path, x=150, y=10, w=40)
        pdf.ln(5)

    # Table Headers
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Field", 1)
    pdf.cell(80, 10, "Value", 1, ln=True)
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
        ("Total Groups", group_data.get("total_groups", 0)),
        ("Active Groups", group_data.get("active_groups", 0)),
        ("Total Messages", message_data.get("total_messages", 0)),
    ]

    for label, value in profile_fields:
        pdf.cell(60, 10, str(label), 1)
        pdf.cell(80, 10, remove_unicode_emoji(str(value)), 1, ln=True)

    # Tipline Categories
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="\nTipline Category Counts:", ln=True)
    pdf.set_font("Arial", '', 12)
    for item in category_counts:
        pdf.cell(60, 10, item['tipline_category'], 1)
        pdf.cell(80, 10, str(item['count']), 1, ln=True)

    # Recent Messages
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="\nRecent Messages:", ln=True)
    pdf.set_font("Arial", '', 11)
    for idx, row in user_messages_df.head(5).iterrows():
        message = f"[{row['timestamp']}] Group: {row['group_name']}\nMessage: {row['message_text']}"
        pdf.multi_cell(0, 10, txt=remove_unicode_emoji(message))
        pdf.ln(4)

    # Place Name
    if not location_df.empty:
        place_name = location_df.iloc[0].get("place", "Unknown")
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="\nLast Known Location:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, txt=f"Place: {place_name}")

    output_path = os.path.join(tempfile.gettempdir(), f"{user_data['user_id']}_profile_summary.pdf")
    pdf.output(output_path)
    return output_path

def remove_unicode_emoji(text):
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

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

# ---------- UTILS ----------
def decode_json_field(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except:
        return []


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
user_data["total_tipline_sent"] = int(user_data["total_tipline_sent"] or 0)
user_data["active_days"] = int(user_data["active_days"] or 0)

common_categories = decode_json_field(user_data["common_categories"])
top_emotions = decode_json_field(user_data["top_emotions"])
top_keywords = decode_json_field(user_data["top_keywords"])
location = decode_json_field(user_data["last_known_location"])

group_data, message_data, category_counts = load_user_groups_messages(selected_user)
user_messages_df = load_user_messages(selected_user)

col1, col2, col3 = st.columns(3)
col1.metric("🧾 Total tipline", user_data["total_tipline_sent"])
col2.metric("📅 Active Days", user_data["active_days"])
col3.metric("🔥 Risk Level", get_risk_label(user_data["reporter_risk_score"]))

col4, col5 = st.columns(2)
col4.markdown(f"**First tipline:** {user_data.get('first_tipline_date', 'N/A')}")
col5.markdown(f"**Last tipline:** {user_data.get('last_tipline_date', 'N/A')}")

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

st.markdown("### Category-wise tipline Submitted")
for item in category_counts:
    st.markdown(f"- {item['tipline_category']}: {item['count']} tipline")

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

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=location_data,
        get_position='[lon, lat]',
        get_radius=300,
        get_fill_color=[255, 0, 0, 160],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=12,
        pitch=40,
    )

    tooltip = {
        "html": "<b>User ID:</b> {user_id}<br/><b>Place:</b> {place}<br/><b>Latitude:</b> {lat}<br/><b>Longitude:</b> {lon}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v11",
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    ))

    st.markdown(f"**Place Name:** 📍 {place_name}")
else:
    st.info("No geo-location data available.")

def generate_satellite_map_html(location_df):
    view_state = pdk.ViewState(
        latitude=location_df["lat"].mean(),
        longitude=location_df["lon"].mean(),
        zoom=11,
        pitch=0,
        bearing=0
    )

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=location_df,
        get_position='[lon, lat]',
        get_color='[255, 0, 0, 160]',
        get_radius=200,
    )

    r = pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-streets-v12",  # Satellite with labels
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "Lat: {lat}\nLon: {lon}"}
    )

    return r.to_html(as_string=True)

# ---------- USER MESSAGES ----------
st.markdown("### User Messages")
if not user_messages_df.empty:
    for idx, row in user_messages_df.iterrows():
        with st.expander(f"🕒 {row['timestamp']} — 📌 Group: {row['group_name']}"):
            st.markdown(f"**Message ID:** {row['message_id']}")
            st.markdown(f"**Text:** {row['message_text']}")
else:
    st.info("No messages found for this user.")

# ---------- DOWNLOAD SECTION ----------
colpdf, colcsv = st.columns(2)

# Build location_df for PDF
if user_data["geo_tags_detected"] and location:
    location_df = pd.DataFrame([{
        "lat": location["latitude"],
        "lon": location["longitude"],
        "user_id": selected_user,
        "place": get_place_name(location["latitude"], location["longitude"])
    }])
else:
    location_df = pd.DataFrame(columns=["lat", "lon", "user_id", "place"])


# PDF Button
with colpdf:
    pdf_buffer = generate_pdf_summary(
        user_data=user_data,
        category_counts=category_counts,
        location_df=location_df,
        user_messages_df=user_messages_df,
        group_data=group_data,
        message_data=message_data
    )
    
    if pdf_buffer and os.path.exists(pdf_buffer):
        with open(pdf_buffer, "rb") as f:
            st.download_button(
                label="📄 Download PDF Summary",
                data=f,
                file_name=f"{user_data['user_id']}_profile_summary.pdf",
                mime="application/pdf"
            )
    else:
        st.error("❌ PDF generation failed. Check console for errors.")



# CSV Button
with colcsv:
    csv_data = pd.DataFrame([user_data])  # Wrap Series in a list
    csv_bytes = csv_data.to_csv(index=False).encode('utf-8')
    csv_filename = f"{user_data['user_id']}_profile_summary.csv"

    st.download_button(
        label="📊 Download CSV",
        data=csv_bytes,
        file_name=csv_filename,
        mime="text/csv"
    )
