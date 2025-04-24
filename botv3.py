# from telegram import Update, InputFile
# from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
# from pymongo import MongoClient
# from fpdf import FPDF
# import os
# import asyncio
# import nest_asyncio

# # --- MongoDB Setup ---
# mongo_client = MongoClient("mongodb://localhost:27017/")
# db = mongo_client["telegram_data"]
# users_col = db["users"]
# messages_col = db["messages"]
# chats_col = db["chats"]

# # --- PDF Generation ---
# def generate_pdf(title: str, info: list[str], messages: list[dict], filename: str):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, txt=title, ln=True)

#     for line in info:
#         pdf.cell(200, 10, txt=line, ln=True)

#     pdf.cell(200, 10, txt="Recent Messages:", ln=True)
#     for msg in messages:
#         date = msg.get("date")
#         text = msg.get("text", "")[:100]
#         pdf.multi_cell(0, 10, f"{date} - {text}")

#     pdf.output(filename)

# # --- User Report Handler ---
# async def handle_user_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     try:
#         user_input = update.message.text.strip().replace("/report", "").strip()
#         user = None
#         telegram_id = None

#         # Try numeric user ID
#         try:
#             telegram_id = int(user_input)
#             user = users_col.find_one({"user_id": telegram_id})
#         except ValueError:
#             # Try username or name
#             user = users_col.find_one({
#                 "$or": [
#                     {"username": user_input},
#                     {"first_name": {"$regex": user_input, "$options": "i"}},
#                     {"last_name": {"$regex": user_input, "$options": "i"}}
#                 ]
#             })
#             if user:
#                 telegram_id = user.get("user_id")

#         if not user or not telegram_id:
#             await update.message.reply_text("❌ User not found in database.")
#             return

#         messages = list(messages_col.find({"user_id": telegram_id}).sort("date", -1).limit(20))
#         filename = f"user_{telegram_id}_report.pdf"

#         info = [
#             f"Username: {user.get('username', 'N/A')}",
#             f"Name: {user.get('first_name', '')} {user.get('last_name', '')}",
#             f"Total Messages: {len(messages)}"
#         ]
#         generate_pdf(f"User Report for Telegram ID: {telegram_id}", info, messages, filename)

#         await update.message.reply_document(InputFile(filename))
#         os.remove(filename)

#     except Exception as e:
#         await update.message.reply_text(f"❌ Error generating user report: {e}")


# # --- Group Report Handler ---
# async def group_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     try:
#         if len(context.args) != 1:
#             await update.message.reply_text("Usage: /group_report <chat_id>")
#             return

#         chat_id = int(context.args[0])
#         group = chats_col.find_one({"chat_id": chat_id})

#         if not group:
#             await update.message.reply_text("❌ Group not found.")
#             return

#         # Get total message count
#         total_messages_count = messages_col.count_documents({"chat_id": chat_id})

#         # Fetch recent messages
#         messages = list(messages_col.find({"chat_id": chat_id}).sort("date", -1).limit(20))
#         filename = f"group_{chat_id}_report.pdf"

#         info = [
#             f"Group Name: {group.get('title', 'N/A')}",
#             f"Type: {group.get('type', 'N/A')}",
#             f"Participants: {group.get('participants_count', 'N/A')}",
#             f"Total Messages: {total_messages_count}"
#         ]
#         generate_pdf(f"Group Report for Chat ID: {chat_id}", info, messages, filename)

#         await update.message.reply_document(InputFile(filename))
#         os.remove(filename)

#     except Exception as e:
#         await update.message.reply_text(f"❌ Error generating group report: {e}")

# # --- Start Command ---
# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     await update.message.reply_text(
#         "👋 Welcome! You can:\n"
#         "🔹 Send a username or Telegram ID for a user report.\n"
#         "🔹 Use /group_report <chat_id> for a group report."
#     )

# # --- Main Bot Entry ---
# async def main():
#     bot_token = "8141223725:AAFS4BAIRsSUEkVuGiRNoEOWrXAPzMwIMeo"  # Replace with your actual bot token
#     app = ApplicationBuilder().token(bot_token).build()

#     app.add_handler(CommandHandler("start", start))
#     app.add_handler(CommandHandler("group_report", group_report))
#     app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_input))

#     print("🤖 Bot is running...")
#     await app.run_polling()

# # --- Event Loop Handling ---
# if __name__ == "__main__":
#     nest_asyncio.apply()
#     asyncio.run(main())


import textwrap
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from pymongo import MongoClient
from fpdf import FPDF
import os
import asyncio
import nest_asyncio

# --- MongoDB Setup ---
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["telegram_data"]
users_col = db["users"]
messages_col = db["messages"]
chats_col = db["chats"]

# --- PDF Class with Unicode + Wrapping ---
class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = "E:/VATINS/fonts/DejaVuSans.ttf"
        if not os.path.exists(font_path):
            raise FileNotFoundError("DejaVuSans.ttf not found. Please ensure it's at 'E:/VATINS/fonts/DejaVuSans.ttf'")
        self.add_font('DejaVu', '', font_path, uni=True)
        self.set_font("DejaVu", '', 11)

    def header(self):
        self.set_font("DejaVu", 'B', 13)
        self.cell(0, 10, "Telegram Intelligence Report", ln=True, align='C')
        self.ln(5)

    def add_lines(self, lines):
        for line in lines:
            self.multi_cell(0, 8, txt=self.clean_text(line))
        self.ln()

    def clean_text(self, text):
        return ''.join(char for char in text if ord(char) < 65536)

# --- Generate PDF ---
def generate_pdf(title: str, info: list[str], messages: list[dict], filename: str):
    pdf = UnicodePDF()
    pdf.add_page()
    pdf.multi_cell(0, 10, title)
    pdf.ln(4)

    pdf.add_lines(info)
    pdf.multi_cell(0, 10, "Recent Messages:")
    pdf.ln(3)

    for msg in messages:
        date = str(msg.get("date", ""))
        text = msg.get("text", "")
        wrapped = textwrap.wrap(text.replace("\n", " "), width=100)
        cleaned_text = "\n".join(wrapped[:5])  # Avoid overflow
        try:
            pdf.multi_cell(0, 8, f"{date} - {pdf.clean_text(cleaned_text)}")
        except:
            pdf.multi_cell(0, 8, f"{date} - (unreadable text)")
        pdf.ln(1)

    pdf.output(filename)

# --- User Report Handler ---
async def handle_user_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip().replace("/report", "").strip()
        user = None
        telegram_id = None

        try:
            telegram_id = int(user_input)
            user = users_col.find_one({"user_id": telegram_id})
        except ValueError:
            user = users_col.find_one({
                "$or": [
                    {"username": user_input},
                    {"first_name": {"$regex": user_input, "$options": "i"}},
                    {"last_name": {"$regex": user_input, "$options": "i"}}
                ]
            })
            if user:
                telegram_id = user.get("user_id")

        if not user or not telegram_id:
            await update.message.reply_text("❌ User not found in database.")
            return

        messages = list(messages_col.find({"user_id": telegram_id}).sort("date", -1).limit(20))
        filename = f"user_{telegram_id}_report.pdf"

        info = [
            f"Username: @{user.get('username', 'N/A')}",
            f"Full Name: {user.get('first_name', '')} {user.get('last_name', '')}",
            f"Phone: {user.get('phone', 'N/A')}",
            f"Last Seen: {user.get('last_seen', 'N/A')}",
            f"Total Messages: {len(messages)}"
        ]
        generate_pdf(f"User Report for Telegram ID: {telegram_id}", info, messages, filename)

        await update.message.reply_document(InputFile(filename))
        os.remove(filename)

    except Exception as e:
        await update.message.reply_text(f"❌ Error generating user report: {e}")

# --- Group Report Handler ---
async def group_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 1:
            await update.message.reply_text("Usage: /group_report <chat_id>")
            return

        chat_id = int(context.args[0])
        group = chats_col.find_one({"chat_id": chat_id})

        if not group:
            await update.message.reply_text("❌ Group not found.")
            return

        messages = list(messages_col.find({"chat_id": chat_id}).sort("date", -1).limit(20))
        filename = f"group_{chat_id}_report.pdf"

        info = [
            f"Group Name: {group.get('title', 'N/A')}",
            f"Type: {group.get('type', 'N/A')}",
            f"Participants: {group.get('participants_count', 'N/A')}",
            f"Last Updated: {group.get('updated_at', 'N/A')}",
            f"Total Messages: {messages_col.count_documents({'chat_id': chat_id})}"
        ]
        generate_pdf(f"Group Report for Chat ID: {chat_id}", info, messages, filename)

        await update.message.reply_document(InputFile(filename))
        os.remove(filename)

    except Exception as e:
        await update.message.reply_text(f"❌ Error generating group report: {e}")

# --- Start Command ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome! You can:\n"
        "🔹 Send a username or Telegram ID for a user report.\n"
        "🔹 Use /group_report <chat_id> for a group report."
    )

# --- Main Bot ---
async def main():
    bot_token = "8141223725:AAFS4BAIRsSUEkVuGiRNoEOWrXAPzMwIMeo"  # Replace with your bot token
    app = ApplicationBuilder().token(bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("group_report", group_report))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_input))

    print("🤖 Bot is running...")
    await app.run_polling()

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
