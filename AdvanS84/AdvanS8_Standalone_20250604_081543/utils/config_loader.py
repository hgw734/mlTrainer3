import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "polygon_api_key": os.getenv("POLYGON_API_KEY", "DKYSsJRspRnuO2N5pp7dJpznTpQ6OF4d"),
    "telegram_token": os.getenv("TELEGRAM_TOKEN", "7208664258:AAHbuU0Q562fHapliu7GvbNz_TB3orPG_0A"),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", "5860539842"),
    "email_user": os.getenv("EMAIL_USER", "julian_212@yahoo.com"),
    "email_pass": os.getenv("EMAIL_PASS", "kldi lxzi zeil fdby"),
    "fred_api_key": os.getenv("FRED_API_KEY", "c2a2b890bd1ea280e5786eafabecafc5"),
    "run_universe": "elites_500_universe.json"
}