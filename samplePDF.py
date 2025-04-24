import streamlit as st
from PyPDF2 import PdfReader
from io import StringIO
import os
import openai  # or litellm
from dotenv import load_dotenv
import textwrap

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace with litellm config if needed

# ---------------- PDF Text Extraction ----------------
def extract_text_from_pdf(file):
    text_list = []
    try:
        reader = PdfReader(file)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except:
                st.error("Encrypted PDF. Unable to decrypt.")
                return []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_list.append(text.strip())
            else:
                text_list.append(f"[Page {page_num + 1}] No extractable text.")
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
    return text_list

# ---------------- GPT Summarization ----------------
def gpt_summarize(text, language="English"):
    chunks = []
    current = ""
    max_len = 3000  # chars

    for paragraph in textwrap.wrap(text, width=max_len, break_long_words=False):
        chunks.append(paragraph)

    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize this text in {language}:\n\n{chunk}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            summaries.append(response['choices'][0]['message']['content'].strip())
        except Exception as e:
            st.error(f"Summarization error: {e}")
            return "Error occurred during GPT summarization."

    final_summary = "\n\n".join(summaries)
    return final_summary

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📄 PDF Summarizer using GPT")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    with st.spinner("Extracting text from PDF..."):
        extracted_pages = extract_text_from_pdf(uploaded_file)

    full_text = "\n\n".join(extracted_pages)
    st.text_area("📃 Extracted Text", full_text, height=300)

    if full_text:
        if st.button("🧠 Summarize with GPT"):
            with st.spinner("Summarizing using GPT..."):
                summary = gpt_summarize(full_text)
            st.text_area("📝 Summary", summary, height=300)

            # Download both extracted + summarized
            summary_filename = uploaded_file.name.replace(".pdf", "_summary.txt")
            full_output = f"--- EXTRACTED TEXT ---\n\n{full_text}\n\n--- SUMMARY ---\n\n{summary}"
            st.download_button(
                label="📥 Download Summary",
                data=full_output,
                file_name=summary_filename,
                mime="text/plain"
            )
else:
    st.info("Please upload a PDF file to get started.")
