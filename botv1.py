from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import pymongo
from fpdf import FPDF

# MongoDB setup
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["telegram_data"]
users_col = db["users"]
messages_col = db["messages"]

# PDF generation function
def generate_user_report(telegram_id: int, filename: str):
    user = users_col.find_one({"user_id": telegram_id})
    messages = list(messages_col.find({"user_id": telegram_id}))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"User Report for Telegram ID: {telegram_id}", ln=True)

    if user:
        pdf.cell(200, 10, txt=f"Username: {user.get('username', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Name: {user.get('first_name', '')} {user.get('last_name', '')}", ln=True)

    pdf.cell(200, 10, txt=f"Total Messages: {len(messages)}", ln=True)

    for msg in messages[:10]:  # limit for preview
        pdf.multi_cell(0, 10, f"{msg.get('date')} - {msg.get('text', '')[:100]}")

    pdf.output(filename)

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome! Send me a username or Telegram ID to get a user report.")

# Handle input (username or ID)
async def handle_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = update.message.text.strip().replace("/report", "").strip()

        # Try to parse as Telegram ID
        try:
            telegram_id = int(user_input)
            user = users_col.find_one({"user_id": telegram_id})
        except ValueError:
            # Try to search by username or name
            user = users_col.find_one({
                "$or": [
                    {"username": user_input},
                    {"first_name": {"$regex": user_input, "$options": "i"}},
                    {"last_name": {"$regex": user_input, "$options": "i"}}
                ]
            })
            if user:
                telegram_id = user["user_id"]
            else:
                telegram_id = None

        if not user or not telegram_id:
            await update.message.reply_text("User not found in database.")
            return

        filename = f"{telegram_id}_report.pdf"
        generate_user_report(telegram_id, filename)

        await update.message.reply_text("Here's the report:")
        await update.message.reply_document(InputFile(filename))

    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

# Main entry
async def main():
    bot_token = "YOUR_BOT_TOKEN"  # Replace with your actual bot token
    app = ApplicationBuilder().token(bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_input))

    print("Bot is running...")
    await app.run_polling()

if __name__ == "__main__":
    import asyncio

    try:
        asyncio.get_event_loop().run_until_complete(main())
    except RuntimeError as e:
        if "This event loop is already running" in str(e):
            print("Event loop already running. Using alternative method...")
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.ensure_future(main())
        else:
            raise
