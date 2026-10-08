"""
Central configuration. Everything comes from environment variables / .env.
No secret is ever hard-coded here.
"""
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from the app folder first, then the project root (so both
# `app/.env` and `./.env` work).
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))


def _clean(value):
    value = (value or "").strip()
    return value or None


# ---------------------------------------------------------------- API keys
weather_api_key = _clean(os.getenv("WEATHER_API_KEY"))
groq_api_key = _clean(os.getenv("GROQ_API_KEY"))
tavily_api_key = _clean(os.getenv("TAVILY_API_KEY"))

# ------------------------------------------------------------ Groq models
# Configurable so you can swap a model without touching code if Groq
# retires one.
GROQ_TIPS_MODEL = os.getenv("GROQ_TIPS_MODEL", "openai/gpt-oss-120b")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)


# ----------------------------------------------------------- MongoDB Atlas
# Paste the connection string from Atlas (Connect -> Drivers) into .env as
# MONGODB_URI. It looks like:
#   mongodb+srv://<user>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_URI = _clean(os.getenv("MONGODB_URI"))
MONGODB_DB = _clean(os.getenv("MONGODB_DB")) or "cropcare"

# ------------------------------------------------------------------ Flask
SECRET_KEY = _clean(os.getenv("SECRET_KEY"))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
