import os
import io
import base64
from typing import Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cryptography.fernet import Fernet, InvalidToken

import qrcode
import numpy as np
import cv2

from telegram import Update, Bot
from telegram.error import TelegramError

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
FERNET_KEY = os.environ["FERNET_KEY"]
ALLOWED_USERNAMES = {
    u.strip().lower() for u in os.environ.get("ALLOWED_USERNAMES", "").split(",") if u.strip()
}

WEBHOOK_SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET_TOKEN", "").strip()

bot = Bot(token=TELEGRAM_TOKEN)
fernet = Fernet(FERNET_KEY)
app = FastAPI(title="QR Cipher Bot")

class TelegramUpdate(BaseModel):
    update_id: int

def _is_allowed_username(update: Update) -> bool:
    """Allow all if env empty, else require membership."""
    user = update.effective_user
    if not ALLOWED_USERNAMES:
        return True
    if user and user.username:
        return user.username.lower() in ALLOWED_USERNAMES
    return False

def _encrypt_text(text: str) -> str:
    token = fernet.encrypt(text.encode("utf-8"))
    return token.decode("utf-8")

def _decrypt_text(token: str) -> str:
    text = fernet.decrypt(token.encode("utf-8")).decode("utf-8")
    return text

def _make_qr_png_bytes(payload: str) -> bytes:
    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _download_photo_as_ndarray(file_id: str) -> np.ndarray:
    tg_file = bot.get_file(file_id)
    data = tg_file.download_as_bytearray()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
    return img

def _decode_qr(img_bgr: np.ndarray) -> Optional[str]:
    """Try single and multi QR decode with OpenCV."""
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

async def _handle_text(update: Update):
    chat_id = update.effective_chat.id
    text = (update.effective_message.text or "").strip()
    if not text:
        await bot.send_message(chat_id, "Отправьте ФИО (текст) или фото с QR-кодом.")
        return
    token = _encrypt_text(text)
    png = _make_qr_png_bytes(token)
    await bot.send_photo(
        chat_id=chat_id,
        photo=png,
        caption="Готово ✅\nВот QR с шифром.\n(Содержимое — зашифрованный токен.)"
    )

async def _handle_photo(update: Update):
    chat_id = update.effective_chat.id
    photos = update.effective_message.photo
    if not photos:
        await bot.send_message(chat_id, "Фото не найдено.")
        return
    file_id = photos[-1].file_id
    try:
        img = _download_photo_as_ndarray(file_id)
        qr_text = _decode_qr(img)
        if not qr_text:
            await bot.send_message(chat_id, "QR-код не найден или не распознан 😕")
            return
        try:
            plain = _decrypt_text(qr_text)
        except InvalidToken:
            await bot.send_message(chat_id, "Не удалось расшифровать. Проверьте ключ или содержимое QR.")
            return
        await bot.send_message(
            chat_id,
            f"Успех ✅\nРасшифрованный текст:\n\n{plain}"
        )
    except Exception as e:
        await bot.send_message(chat_id, f"Ошибка при обработке фото: {e}")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)
):
    if WEBHOOK_SECRET_TOKEN:
        if x_telegram_bot_api_secret_token != WEBHOOK_SECRET_TOKEN:
            raise HTTPException(status_code=401, detail="Bad secret token")

    payload = await request.json()
    try:
        update = Update.de_json(payload, bot)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Telegram update")

    if not _is_allowed_username(update):
        if update.effective_chat:
            await bot.send_message(update.effective_chat.id, "Доступ запрещён.")
        return JSONResponse({"status": "ignored"}, status_code=200)

    msg = update.effective_message
    if msg is None:
        return {"status": "ok"}

    if msg.photo:
        await _handle_photo(update)
    elif msg.text:
        await _handle_text(update)
    else:
        await bot.send_message(msg.chat_id, "Пожалуйста, отправьте текст (ФИО) или фото с QR-кодом.")

    return {"status": "ok"}
