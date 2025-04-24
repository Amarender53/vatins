import streamlit as st
from pymongo import MongoClient
from transformers import pipeline
from fpdf import FPDF
from io import BytesIO
import textwrap
import unicodedata
import re
import os

# === MongoDB Connection ===
client = MongoClient("mongodb://localhost:27017/")
db = client["telegram_data"]
users_col = db["users"]
messages_col = db["messages"]

# === AI Models ===
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# === Font Paths ===
FONT_PATH = "E:/VATINS/fonts/DejaVuSans.ttf"
BOLD_FONT_PATH = FONT_PATH.replace(".ttf", "-Bold.ttf")

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("🧠 AI Intelligence Summary Report")

user_data = list(users_col.find({}, {"user_id": 1, "first_name": 1, "last_name": 1, "username": 1}))
user_map = {
    f"{u.get('first_name', '')} {u.get('last_name', '')} (@{u.get('username', 'N/A')})": u["user_id"]
    for u in user_data
}

selected_user = st.selectbox("Select a User", list(user_map.keys()))
user_id = user_map[selected_user]
user = users_col.find_one({"user_id": user_id})
messages = list(messages_col.find({"user_id": user_id}))
texts = [msg.get("text", "") for msg in messages if isinstance(msg.get("text", ""), str)]
full_text = " ".join(texts)[:8000]

# === Section Definitions ===
SECTIONS = {
    "User Details": lambda: [
        f"Name: {user.get('first_name', '')} {user.get('last_name', '')}",
        f"Username: @{user.get('username', 'N/A')}",
        f"Phone: {user.get('phone', 'N/A')}",
        f"Total Messages: {len(texts)}"
    ],
    "Interests": "Extract the user's interests from the following messages. Return 3 bullet points.",
    "Locations": "What locations (cities, regions, countries) are mentioned or inferred?",
    "Law Enforcement Concerns": "What activities may be of concern to law enforcement?",
    "Communication Patterns": "Summarize how this user communicates (tone, frequency, directness).",
    "Affiliations": "Mention any groups, organizations, or affiliations based on their messages.",
    "Financial Activities": "Are there any mentions of money transfers, crypto, or other financial talk?",
    "Digital Footprint": "What digital tools, apps, or platforms does this user mention?",
    "Ideological Indicators": "Any political, religious, or ideological positions indicated?",
    "Behavioral Red Flags": "Are there any warning signs or red flags in behavior?",
    "Use of Technology": "How does this user use technology (VPN, TOR, automation)?",
    "Cultural Context": "Identify the cultural or linguistic hints from the message set.",
    "Final Commentary": "Give a 3-5 line executive summary of this user."
}

# === LLM Section Summarization ===
def summarize_section(prompt, text):
    try:
        input_text = f"{prompt}\n{text[:1500]}"
        response = summarizer(input_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        # Split into points if possible
        points = re.split(r'\n|•|- ', response)
        points = [p.strip("•-•:. \n") for p in points if len(p.strip()) > 5]
        return points
    except:
        return ["[Error generating summary]"]

# === PDF Class ===
class IntelligencePDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        self.page_width = self.w - 2 * self.l_margin
        self.add_font("DejaVu", "", FONT_PATH, uni=True)
        if os.path.exists(BOLD_FONT_PATH):
            self.add_font("DejaVu", "B", BOLD_FONT_PATH, uni=True)
        self.set_font("DejaVu", "", 11)

    def clean(self, text):
        import unicodedata
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\u2600-\u26FF\u2700-\u27BF"
            "]+", flags=re.UNICODE)
        return unicodedata.normalize("NFKD", emoji_pattern.sub('', str(text)))

    def section(self, title, lines):  # ✅ THIS IS THE MISSING FUNCTION
        self.set_font("DejaVu", "B", 12)
        self.set_x(self.l_margin)
        self.cell(0, 10, title, ln=True)
        self.set_font("DejaVu", "", 11)

        for line in lines:
            clean = self.clean(line)
            wrapped = textwrap.wrap(clean, width=100)
            for i, subline in enumerate(wrapped):
                self.set_x(self.l_margin)
                if i == 0:
                    self.multi_cell(self.page_width, 6, f"• {subline}", align='L')
                else:
                    self.multi_cell(self.page_width, 6, f"  {subline}", align='L')
            self.ln(1)
        self.ln(2)

# === PDF Generation ===
def generate_pdf(sections_data):
    pdf = IntelligencePDF()
    pdf.add_page()
    for section, content in sections_data.items():
        pdf.section(section, content)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# === Run & Download ===
if st.button("📥 Generate PDF Report"):
    report_data = {}
    for section, content in SECTIONS.items():
        if callable(content):
            report_data[section] = content()
        else:
            report_data[section] = summarize_section(content, full_text)
    pdf_file = generate_pdf(report_data)
    st.download_button("📄 Download Intelligence Report", data=pdf_file, file_name="intelligence_report.pdf", mime="application/pdf")
