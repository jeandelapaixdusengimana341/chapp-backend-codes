import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncpg
from openai import AsyncOpenAI

# 1. Resolve .env file relative to the script directory
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="Chapp Translation & Reels Engine")

# 2. Setup Local Media Directory for Video Reels
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# 3. Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    ai_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1", 
        api_key=OPENAI_API_KEY
    ) if OPENAI_API_KEY else None
except Exception:
    ai_client = None
    print("⚠️ OpenRouter Fallback Mode.")

@app.on_event("startup")
async def init_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is missing in .env file.")
        
    app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)
    
    async with app.state.db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                phone_number TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                is_paid BOOLEAN DEFAULT FALSE, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS language_preferences (
                reader_phone TEXT,
                sender_phone TEXT,
                target_lang TEXT NOT NULL,
                PRIMARY KEY (reader_phone, sender_phone)
            );
            CREATE TABLE IF NOT EXISTS messages (
                message_id SERIAL PRIMARY KEY,
                sender_phone TEXT NOT NULL,
                recipient_phone TEXT NOT NULL,
                original_text TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS videos (
                video_id SERIAL PRIMARY KEY,
                uploader_phone TEXT NOT NULL,
                video_url TEXT NOT NULL,          
                caption TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    print("💾 Database pool ready.")

@app.on_event("shutdown")
async def close_db():
    if hasattr(app.state, "db_pool"):
        await app.state.db_pool.close()

# --- CHAT WEBSOCKET ENGINE ---

class ChatManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, phone_number: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[phone_number] = websocket

    def disconnect(self, phone_number: str):
        self.active_connections.pop(phone_number, None)

    async def route_message(self, sender_phone: str, recipient_phone: str, raw_text: str):
        recipient_socket = self.active_connections.get(recipient_phone)
        target_lang = "EN-US"
        
        async with app.state.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT target_lang FROM language_preferences 
                WHERE reader_phone = $1 AND sender_phone = $2
            """, recipient_phone, sender_phone)
            if row:
                target_lang = row["target_lang"]

        translated_text = raw_text
        detected_source_lang = "UNKNOWN"

        if ai_client:
            try:
                response = await ai_client.chat.completions.create(
                    model="meta-llama/llama-3-8b-instruct:free",
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are Chapp's real-time translation assistant. Respond ONLY with a valid JSON object containing keys: 'translated_text' and 'detected_source_lang'."
                        },
                        {
                            "role": "user", 
                            "content": f"Target Language Code: {target_lang}. Message to translate: '{raw_text}'"
                        }
                    ]
                )
                
                content = response.choices[0].message.content
                ai_response = json.loads(content)
                translated_text = ai_response.get("translated_text", raw_text)
                detected_source_lang = ai_response.get("detected_source_lang", "UNKNOWN")
            except Exception as e:
                print(f"❌ OpenRouter translation error: {e}")

        async with app.state.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO messages (sender_phone, recipient_phone, original_text, source_lang)
                VALUES ($1, $2, $3, $4)
            """, sender_phone, recipient_phone, raw_text, detected_source_lang)

        if recipient_socket:
            payload = {
                "sender_phone": sender_phone,
                "original_text": raw_text,
                "translated_text": translated_text,
                "target_lang_used": target_lang,
                "was_translated": detected_source_lang.upper() != target_lang.upper()
            }
            await recipient_socket.send_text(json.dumps(payload))

manager = ChatManager()

# --- USER & AUTH ENDPOINTS ---

@app.post("/api/auth/register")
async def register_or_login_user(data: dict):
    phone_number = data.get("phone_number")
    username = data.get("username")
    if not phone_number or not username:
        raise HTTPException(status_code=400, detail="Missing phone_number or username.")
    
    async with app.state.db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO users (phone_number, username) VALUES ($1, $2)
            ON CONFLICT (phone_number) DO UPDATE SET username = EXCLUDED.username
        """, phone_number, username)
    return {"status": "success", "phone_number": phone_number, "username": username}

@app.post("/api/set-language")
async def update_sender_language_preference(data: dict):
    reader_phone = data.get("reader_phone")
    sender_phone = data.get("sender_phone")
    chosen_lang = data.get("target_lang", "EN-US").upper()
    
    async with app.state.db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO language_preferences (reader_phone, sender_phone, target_lang) VALUES ($1, $2, $3)
            ON CONFLICT (reader_phone, sender_phone) DO UPDATE SET target_lang = EXCLUDED.target_lang
        """, reader_phone, sender_phone, chosen_lang)
    return {"status": "success"}

@app.get("/api/history/{user_a}/{user_b}")
async def get_chat_history(user_a: str, user_b: str):
    async with app.state.db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT sender_phone, recipient_phone, original_text, source_lang, timestamp::text as timestamp 
            FROM messages 
            WHERE (sender_phone = $1 AND recipient_phone = $2) OR (sender_phone = $2 AND recipient_phone = $1)
            ORDER BY timestamp ASC
        """, user_a, user_b)
        return [dict(row) for row in rows]

@app.get("/api/chats/{phone_number}")
async def get_active_chat_list(phone_number: str):
    async with app.state.db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT DISTINCT contact_phone FROM (
                SELECT recipient_phone as contact_phone FROM messages WHERE sender_phone = $1
                UNION
                SELECT sender_phone as contact_phone FROM messages WHERE recipient_phone = $1
            ) DISTINCT_USERS WHERE contact_phone != $1
        """, phone_number)
        return [row["contact_phone"] for row in rows]

# --- REELS & VIDEO UPLOAD ENDPOINTS ---

# 1. Direct Multipart Video File Upload (Handles standard HTML/JS File Uploads)
@app.post("/api/videos/upload-file")
@app.post("/upload")  # Fallback route alias for common frontend upload paths
async def upload_video_file(
    uploader_phone: str = Form("Anonymous"),
    caption: str = Form(""),
    file: UploadFile = File(...)
):
    try:
        # Sanitize filename and save locally
        safe_filename = file.filename.replace(" ", "_")
        target_filepath = UPLOAD_DIR / safe_filename
        
        with open(target_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        public_video_url = f"/static/uploads/{safe_filename}"
        
        async with app.state.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO videos (uploader_phone, video_url, caption) VALUES ($1, $2, $3)
            """, uploader_phone, public_video_url, caption)
            
        return {
            "status": "success", 
            "video_url": public_video_url,
            "message": "Video uploaded successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video upload: {str(e)}")

# 2. JSON Payload Video Link Upload
@app.post("/api/videos/upload")
async def upload_new_video_json(data: dict):
    uploader_phone = data.get("uploader_phone")
    video_url = data.get("video_url") 
    caption = data.get("caption", "")
    
    if not uploader_phone or not video_url:
        raise HTTPException(status_code=400, detail="Missing uploader_phone or video_url.")
        
    async with app.state.db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO videos (uploader_phone, video_url, caption) VALUES ($1, $2, $3)
        """, uploader_phone, video_url, caption)
    return {"status": "success", "message": "Video URL stored successfully."}

# 3. Reels Feed Stream API (CONTACTS ONLY ENFORCEMENT)
@app.get("/api/videos/feed")
async def get_video_feed(viewer_phone: str = None):
    async with app.state.db_pool.acquire() as conn:
        # If no viewer_phone is passed, fallback to standard global feed
        if not viewer_phone:
            rows = await conn.fetch("""
                SELECT video_id, uploader_phone, video_url, caption, created_at::text as created_at
                FROM videos
                ORDER BY created_at DESC
            """)
            return {"reels": [dict(row) for row in rows]}

        # 1. Retrieve all users with whom viewer_phone has an active chat history
        contact_rows = await conn.fetch("""
            SELECT DISTINCT contact_phone FROM (
                SELECT recipient_phone as contact_phone FROM messages WHERE sender_phone = $1
                UNION
                SELECT sender_phone as contact_phone FROM messages WHERE recipient_phone = $1
            ) DISTINCT_USERS
        """, viewer_phone)
        
        allowed_phones = [row["contact_phone"] for row in contact_rows]
        
        # Always allow the user to view their own uploaded reels
        if viewer_phone not in allowed_phones:
            allowed_phones.append(viewer_phone)

        # 2. Query videos where the uploader is in the allowed contacts list
        rows = await conn.fetch("""
            SELECT video_id, uploader_phone, video_url, caption, created_at::text as created_at
            FROM videos
            WHERE uploader_phone = ANY($1::text[])
            ORDER BY created_at DESC
        """, allowed_phones)

        return {"reels": [dict(row) for row in rows]}

# --- WEBSOCKET LISTENER ---

@app.websocket("/ws/{phone_number}")
async def websocket_endpoint(websocket: WebSocket, phone_number: str):
    await manager.connect(phone_number, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            recipient_phone = message_data.get("recipient_phone")
            raw_text = message_data.get("text")
            
            if recipient_phone and raw_text:
                await manager.route_message(phone_number, recipient_phone, raw_text)
                
    except WebSocketDisconnect:
        manager.disconnect(phone_number)
    except Exception as e:
        print(f"⚠️ Error inside WebSocket loop: {e}")
        manager.disconnect(phone_number)