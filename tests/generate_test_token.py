import os
import jwt
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv(".env")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET is missing from .env")

# Generate a token valid for the load test
token = jwt.encode({"user_id": "load_test_user_1"}, JWT_SECRET, algorithm="HS256")
print(f"\nYour test token is ready. Run the following command before starting the load test:\n")
print(f"export TEST_TOKEN={token}\n")
