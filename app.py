import os
import io
import re
from typing import Optional
import hashlib
from datetime import datetime

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

from cryptography.fernet import Fernet, InvalidToken

import qrcode
import numpy as np
import cv2

from telegram import Update, Bot
from telegram.error import TelegramError

import asyncpg

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
FERNET_KEY = os.environ["FERNET_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
ALLOWED_USERNAMES = {
    u.strip().lower() for u in os.environ.get("ALLOWED_USERNAMES", "").split(",") if u.strip()
}
WEBHOOK_SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET_TOKEN", "").strip()

bot = Bot(token=TELEGRAM_TOKEN)
fernet = Fernet(FERNET_KEY)
app = FastAPI(title="QR Cipher Bot")

DB_POOL: asyncpg.Pool | None = None

HELP_TEXT = (
    "Привет! Я шифрую ФИО в QR и читаю зашифрованные QR-коды.\n\n"
    "Как пользоваться:\n"
    "• Отправьте текстом ФИО в формате «Фамилия Имя» или «Фамилия Имя Отчество» "
    "(допустимы кириллица/латиница и дефисы). Я зашифрую и пришлю QR.\n"
    "• Или пришлите фото QR-кода — я распознаю, расшифрую и верну исходный текст.\n\n"
)

NAME_RE = re.compile(
    r"^\s*"
    r"([A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*)\s+"
    r"([A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*)"
    r"(?:\s+([A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*))?"
    r"\s*$"
)

def _is_allowed_username(update: Update) -> bool:
    if not ALLOWED_USERNAMES:
        return True
    user = update.effective_user
    return bool(user and user.username and user.username.lower() in ALLOWED_USERNAMES)

def _normalize_fio(text: str) -> Optional[str]:
    m = NAME_RE.match(text or "")
    if not m:
        return None
    parts = [p for p in m.groups() if p]
    return " ".join(p.strip().title() for p in parts)

def _encrypt_text(text: str) -> str:
    token = fernet.encrypt(text.encode("utf-8"))
    return token.decode("utf-8")

def _decrypt_text(token: str) -> str:
    return fernet.decrypt(token.encode("utf-8")).decode("utf-8")

def _make_qr_png_bytes(payload: str) -> bytes:
    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

async def _download_photo_as_ndarray(file_id: str) -> np.ndarray:
    tg_file = await bot.get_file(file_id)
    data = await tg_file.download_as_bytearray()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение")
    return img

def _decode_qr(img_bgr: np.ndarray) -> Optional[str]:
    detector = cv2.QRCodeDetector()
    text, points, _ = detector.detectAndDecode(img_bgr)
    if text:
        return text
    texts, points, _ = detector.detectAndDecodeMulti(img_bgr)
    if texts:
        for t in texts:
            if t:
                return t
    return None

def _token_fingerprint(token_text: str) -> str:
    return hashlib.sha256(token_text.encode("utf-8")).hexdigest()

async def _init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)

async def _ensure_schema():
    assert DB_POOL is not None
    async with DB_POOL.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS used_tokens (
            token_hash TEXT PRIMARY KEY,
            used_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)

async def _is_token_used(token_text: str) -> bool:
    assert DB_POOL is not None
    fp = _token_fingerprint(token_text)
    async with DB_POOL.acquire() as conn:
        row = await conn.fetchrow("SELECT 1 FROM used_tokens WHERE token_hash = $1;", fp)
        return row is not None

async def _mark_token_used(token_text: str) -> bool:
    assert DB_POOL is not None
    fp = _token_fingerprint(token_text)
    async with DB_POOL.acquire() as conn:
        res = await conn.execute(
            "INSERT INTO used_tokens(token_hash, used_at) VALUES ($1, NOW()) ON CONFLICT DO NOTHING;",
            fp
        )
        return res.endswith(" 1")

async def _handle_start(update: Update):
    chat_id = update.effective_chat.id
    if _is_allowed_username(update):
        await bot.send_message(chat_id, HELP_TEXT)
    else:
        await bot.send_message(chat_id, "Этот бот доступен только для сотрудников.")

async def _handle_text(update: Update):
    chat_id = update.effective_chat.id
    raw = (update.effective_message.text or "").strip()
    if raw.startswith("/start"):
        await _handle_start(update)
        return

    fio = _normalize_fio(raw)
    if fio:
        token = _encrypt_text(fio)
        png = _make_qr_png_bytes(token)
        await bot.send_photo(
            chat_id=chat_id,
            photo=png,
            caption="Готово ✅\nЭто QR с зашифрованным ФИО."
        )
        return

    token_candidate = raw
    try:
        plain = _decrypt_text(token_candidate)
    except InvalidToken:
        await bot.send_message(chat_id, "Не удалось расшифровать. Проверьте ключ или содержимое.")
        return

    inserted = await _mark_token_used(token_candidate)
    if not inserted:
        await bot.send_message(chat_id, "Этот QR уже использован и больше недействителен.")
        return

    rnd = random.randint(18, 20)
    await bot.send_message(chat_id, f"Успех ✅\n\n{plain}\n\n{rnd} лет")

async def _handle_photo(update: Update):
    chat_id = update.effective_chat.id
    photos = update.effective_message.photo
    if not photos:
        await bot.send_message(chat_id, "Фото не найдено.")
        return

    file_id = photos[-1].file_id
    try:
        img = await _download_photo_as_ndarray(file_id)
        qr_text = _decode_qr(img)
        if not qr_text:
            await bot.send_message(chat_id, "QR-код не найден или не распознан 😕")
            return

        try:
            plain = _decrypt_text(qr_text)
        except InvalidToken:
            await bot.send_message(chat_id, "Не удалось расшифровать. Проверьте ключ или содержимое QR.")
            return

        inserted = await _mark_token_used(qr_text)
        if not inserted:
            await bot.send_message(chat_id, "Этот QR уже использован и больше недействителен.")
            return

        await bot.send_message(chat_id, f"Успех ✅\n\n{plain}")

    except Exception as e:
        await bot.send_message(chat_id, f"Ошибка при обработке фото: {e}")

@app.on_event("startup")
async def _on_startup():
    await _init_db_pool()
    await _ensure_schema()

@app.on_event("shutdown")
async def _on_shutdown():
    global DB_POOL
    if DB_POOL:
        await DB_POOL.close()
        DB_POOL = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)
):
    if WEBHOOK_SECRET_TOKEN and x_telegram_bot_api_secret_token != WEBHOOK_SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Bad secret token")

    payload = await request.json()

    try:
        update = Update.de_json(payload, bot)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Telegram update")

    msg = update.effective_message
    if msg is None:
        return {"status": "ok"}

    if msg.text and msg.text.strip().startswith("/start"):
        await _handle_start(update)
        return {"status": "ok"}

    if not _is_allowed_username(update):
        await bot.send_message(update.effective_chat.id, "Этот бот доступен только для сотрудников.")
        return JSONResponse({"status": "ignored"}, status_code=200)

    if msg.photo:
        await _handle_photo(update)
    elif msg.text:
        await _handle_text(update)
    else:
        await bot.send_message(msg.chat_id, "Отправьте ФИО (текст) или фото с QR-кодом.")

    return {"status": "ok"}
