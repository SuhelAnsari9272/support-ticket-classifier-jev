import os
from dotenv import load_dotenv

load_dotenv()


JEV_MODEL    = os.environ.get("JEV_MODEL", "~typesafe/jev-latest")
JEV_BASE_URL = os.environ.get("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
JEV_API_KEY = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

VERIFY_SSL = os.environ.get("VERIFY_SSL", "true").strip().lower() not in ("false", "0", "no")

# print("JEV_API_KEY : ", JEV_API_KEY )