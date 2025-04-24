import asyncio
from telethon import TelegramClient, events
from telethon.tl.types import Channel, Chat
from pymongo import MongoClient
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Telegram User API credentials
API_ID = 28464571
API_HASH = 'ceb559c6275cfe6fd2297547f0384da3'
PHONE = '+916300891714'

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "telegram_data"

class MongoDB:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.chats = self.db["chats"]
        self.groups = self.db["groups"]
        self.channels = self.db["channels"]  # New collection for channels
        self.users = self.db["users"]
        self.messages = self.db["messages"]
        self._create_indexes()
    
    def _create_indexes(self):
        self.chats.create_index("chat_id", unique=True)
        self.groups.create_index("chat_id", unique=True)
        self.channels.create_index("chat_id", unique=True)  # Index for channels
        self.users.create_index("user_id", unique=True)
        self.messages.create_index([("message_id", 1), ("chat_id", 1)], unique=True)
    
    async def save_chat(self, chat):
        try:
            chat_data = {
                "chat_id": chat.id,
                "title": chat.title,
                "type": "channel" if isinstance(chat, Channel) else "group",
                "created_at": chat.date,
                "updated_at": datetime.now()
            }
            self.chats.update_one({"chat_id": chat.id}, {"$set": chat_data}, upsert=True)
        except Exception as e:
            logger.error(f"Error saving chat: {e}")
    
    async def save_group(self, chat):
        try:
            group_data = {
                "chat_id": chat.id,
                "title": chat.title,
                "participants_count": getattr(chat, 'participants_count', 0),
                "updated_at": datetime.now()
            }
            self.groups.update_one({"chat_id": chat.id}, {"$set": group_data}, upsert=True)
        except Exception as e:
            logger.error(f"Error saving group: {e}")

    async def save_channel(self, chat):
        try:
            # Fetch the participants count for the channel
            participants_count = getattr(chat, 'participants_count', 0)
            
            channel_data = {
                "chat_id": chat.id,
                "title": chat.title,
                "participants_count": participants_count,
                "updated_at": datetime.now()
            }
            self.channels.update_one({"chat_id": chat.id}, {"$set": channel_data}, upsert=True)
        except Exception as e:
            logger.error(f"Error saving channel: {e}")
    
    async def save_user(self, user):
        try:
            user_data = {
                "user_id": user.id,
                "first_name": getattr(user, 'first_name', ''),
                "last_name": getattr(user, 'last_name', ''),
                "username": getattr(user, 'username', ''),
                "phone": getattr(user, 'phone', ''),
                "last_seen": datetime.now()
            }
            self.users.update_one({"user_id": user.id}, {"$set": user_data}, upsert=True)
        except Exception as e:
            logger.error(f"Error saving user: {e}")
    
    async def save_message(self, message, sender):
        try:
            message_data = {
                "message_id": message.id,
                "chat_id": message.chat_id,
                "text": message.text,
                "date": message.date,
                "user_id": sender.id if sender else None,
                "metadata": {
                    "views": getattr(message, 'views', None),
                    "forwards": getattr(message, 'forwards', None)
                }
            }
            self.messages.update_one(
                {"message_id": message.id, "chat_id": message.chat_id},
                {"$set": message_data},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error saving message: {e}")

async def main():
    mongo = MongoDB()
    client = TelegramClient('session_name', API_ID, API_HASH)

    await client.start(PHONE)
    logger.info("Client started successfully!")

    @client.on(events.NewMessage(chats=(Channel, Chat)))
    async def handler(event):
        try:
            chat = await event.get_chat()
            sender = await event.get_sender()

            # Save chat (channel or group data)
            await mongo.save_chat(chat)
            
            # Save group-specific data
            if isinstance(chat, Chat):
                await mongo.save_group(chat)

            # Save channel-specific data
            if isinstance(chat, Channel):
                await mongo.save_channel(chat)

            # Save user data
            if sender:
                await mongo.save_user(sender)

            # Save message data
            await mongo.save_message(event.message, sender)

            logger.info(f"Processed message from chat: {chat.title}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def fetch_historical_data():
        dialogs = await client.get_dialogs()
        for dialog in dialogs:
            if isinstance(dialog.entity, (Channel, Chat)):
                logger.info(f"Fetching history for: {dialog.name}")
                try:
                    # Save chat (channel or group data)
                    await mongo.save_chat(dialog.entity)

                    # Save group-specific data
                    if isinstance(dialog.entity, Chat):
                        participants = await client.get_participants(dialog.entity)
                        await mongo.save_group(dialog.entity)  # Save group data
                        for user in participants:
                            await mongo.save_user(user)

                    # Save channel-specific data
                    if isinstance(dialog.entity, Channel):
                        await mongo.save_channel(dialog.entity)

                    # Save messages from the dialog
                    async for message in client.iter_messages(dialog.entity, limit=1000):
                        sender = await message.get_sender()
                        if sender:
                            await mongo.save_user(sender)
                        await mongo.save_message(message, sender)

                except Exception as e:
                    logger.error(f"Error processing {dialog.name}: {e}")

    await fetch_historical_data()
    logger.info("Initial data fetch complete! Listening for new messages...")
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())
