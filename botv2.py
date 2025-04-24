from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import pymongo
from fpdf import FPDF
import os

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["telegram_data"]
chats_col = db["chats"]
messages_col = db["messages"]

def generate_group_report(chat_id: int, filename: str):
    group = chats_col.find_one({"chat_id": chat_id})
    messages = list(messages_col.find({"chat_id": chat_id}).sort("date", -1).limit(20))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=f"Group Report for Chat ID: {chat_id}", ln=True)

    if group:
        pdf.cell(200, 10, txt=f"Group Name: {group.get('title', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Type: {group.get('type', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Participants: {group.get('participants_count', 'N/A')}", ln=True)

    pdf.cell(200, 10, txt=f"Recent Messages:", ln=True)
    for msg in messages:
        text = msg.get('text', '') or ''
        date = msg.get('date')
        pdf.multi_cell(0, 10, f"{date} - {text[:100]}")

    pdf.output(filename)

# Command handler for /group_report
async def group_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = int(context.args[0])
        filename = f"group_{chat_id}_report.pdf"
        generate_group_report(chat_id, filename)

        await update.message.reply_text("Here is the group summary:")
        await update.message.reply_document(InputFile(filename))
        os.remove(filename)
    except Exception as e:
        await update.message.reply_text(f"Error generating group report: {e}")

# Bot setup
async def main():
    bot_token = "8141223725:AAFS4BAIRsSUEkVuGiRNoEOWrXAPzMwIMeo"
    app = ApplicationBuilder().token(bot_token).build()

    app.add_handler(CommandHandler("group_report", group_report))

    print("Bot is running...")
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
