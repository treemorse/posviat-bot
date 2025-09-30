import os
import io
import re
from typing import Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

from cryptography.fernet import Fernet, InvalidToken

import qrcode
import numpy as np
import cv2  # opencv-python-headless

from telegram import Update, Bot
from telegram.error import TelegramError

# --- env config ---
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
FERNET_KEY = os.environ["FERNET_KEY"]
ALLOWED_USERNAMES = {
    u.strip().lower() for u in os.environ.get("ALLOWED_USERNAMES", "").split(",") if u.strip()
}
WEBHOOK_SECRET_TOKEN = os.environ.get("WEBHOOK_SECRET_TOKEN", "").strip()

bot = Bot(token=TELEGRAM_TOKEN)
fernet = Fernet(FERNET_KEY)
app = FastAPI(title="QR Cipher Bot")

# --- help text (ru) ---
HELP_TEXT = (
    "Привет! Я шифрую ФИО в QR и читаю зашифрованные QR-коды.\n\n"
    "Как пользоваться:\n"
    "• Отправьте текстом ФИО в формате «Фамилия Имя» или «Фамилия Имя Отчество» "
    "(допустимы кириллица/латиница и дефисы). Я зашифрую и пришлю QR.\n"
    "• Или пришлите фото QR-кода — я распознаю, расшифрую и верну исходный текст.\n\n"
)

# --- fio валидация: 2 или 3 слова; буквы кириллицы/латиницы, дефисы и апостроф ---
NAME_RE = re.compile(
    r"^\s*"
    r"([A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*)\s+"          # фамилия
    r"([A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*)"             # имя
    r"(?:\s+([A-Za-zА-Яа-яЁё]+(?:[-'][A-Za-zА-Яа-яЁё]+)*))?"     # отчество (опц.)
    r"\s*$"
)

# --- utils ---
def _is_allowed_username(update: Update) -> bool:
    # если список пуст — разрешить всем
    if not ALLOWED_USERNAMES:
        return True
    user = update.effective_user
    return bool(user and user.username and user.username.lower() in ALLOWED_USERNAMES)

def _normalize_fio(text: str) -> Optional[str]:
    """
    проверяет формат «Фамилия Имя» или «Фамилия Имя Отчество» и нормализует (Title Case, один пробел).
    возвращает нормализованную строку или None, если формат не подходит.
    """
    m = NAME_RE.match(text or "")
    if not m:
        return None
    parts = [p for p in m.groups() if p]
    # title case бережно работает и с дефисами, например "иван-петров" -> "Иван-Петров"
    norm = " ".join(p.strip().title() for p in parts)
    return norm

def _encrypt_text(text: str) -> str:
    token = fernet.encrypt(text.encode("utf-8"))
    return token.decode("utf-8")

def _decrypt_text(token: str) -> str:
    return fernet.decrypt(token.encode("utf-8")).decode("utf-8")

def _make_qr_png_bytes(payload: str) -> bytes:
    # аккуратный чёрно-белый qr без лишних рамок
    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

async def _download_photo_as_ndarray(file_id: str) -> np.ndarray:
    """
    загружает фото через telegram api и декодирует в ndarray (bgr) для opencv.
    важно: методы асинхронные в ptb v21 (await get_file; await download_as_bytearray)
    """
    tg_file = await bot.get_file(file_id)
    data = await tg_file.download_as_bytearray()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение")
    return img

def _decode_qr(img_bgr: np.ndarray) -> Optional[str]:
    """
    пытается распознать одиночный и множественный qr через opencv.
    """
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

# --- handlers ---
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
    if not fio:
        await bot.send_message(
            chat_id,
            "Отправь ФИО в формате «Фамилия Имя» или «Фамилия Имя Отчество».\n"
            "Примеры: «Иванов Иван», «Смирнова Анна Сергеевна», «Санин-Петров Артём»."
        )
        return

    token = _encrypt_text(fio)
    png = _make_qr_png_bytes(token)
    await bot.send_photo(
        chat_id=chat_id,
        photo=png,
        caption="Готово ✅\nЭто QR с зашифрованным ФИО."
    )

async def _handle_photo(update: Update):
    chat_id = update.effective_chat.id
    photos = update.effective_message.photo
    if not photos:
        await bot.send_message(chat_id, "Фото не найдено.")
        return

    file_id = photos[-1].file_id  # берём самое большое превью
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
        await bot.send_message(chat_id, f"Успех ✅\n\n{plain}")
    except Exception as e:
        await bot.send_message(chat_id, f"Ошибка при обработке фото: {e}")

# --- routes ---
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None)
):
    # опциональная проверка секрета заголовка
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

    # /start должен отвечать явным сообщением и для разрешённых, и для запрещённых
    if msg.text and msg.text.strip().startswith("/start"):
        await _handle_start(update)
        return {"status": "ok"}

    # общий allow-лист для остальных сообщений
    if not _is_allowed_username(update):
        await bot.send_message(update.effective_chat.id, "Этот бот доступен только для сотрудников.")
        return JSONResponse({"status": "ignored"}, status_code=200)

    # маршрутизация
    if msg.photo:
        await _handle_photo(update)
    elif msg.text:
        await _handle_text(update)
    else:
        await bot.send_message(msg.chat_id, "Отправьте ФИО (текст) или фото с QR-кодом.")

    return {"status": "ok"}
