import os
import jwt
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET")

def access_token_generator():
    token = jwt.encode({"user_id": "some_id"}, JWT_SECRET, algorithm="HS256")
    return token

if __name__ == '__main__':
    access_token = access_token_generator()
    print('Access Token: ', access_token)