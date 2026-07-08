from flask import Flask, request, jsonify
from flask_socketio import disconnect
import jwt
from functools import wraps
from dotenv import load_dotenv
import logging
import os

load_dotenv()

# JWT Secret Key
JWT_SECRET = os.getenv("JWT_SECRET")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'text': 'Token is missing!'}), 403
        
        try:
            # Remove 'Bearer' prefix if present
            if 'Bearer' in token:
                token = token.split()[1]
            
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"verify_sub": False})
            current_user_id = data['user_id']
        except Exception as e:
            logging.error(f"Error decoding token: {e}")
            return {'text': 'Token is invalid!'}, 403
        
        # Pass current_user_id, Bearer token and maintain other args
        return f(current_user_id, token, *args, **kwargs)
    return decorated

def socket_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            # Get token from Authorization header (most reliable)
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                logging.error("No Authorization header found")
                disconnect()
                return False

            # Extract token (remove 'Bearer ' prefix)
            token = auth_header.split()[1] if 'Bearer' in auth_header else auth_header
            
            # Decode token
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            current_user_id = data['user_id']
            
            logging.info(f"Token decoded successfully for user: {current_user_id}")
            
            # Pass current_user_id AND original token to the function
            return f(current_user_id, token, *args, **kwargs)
            
        except Exception as e:
            logging.error(f"Socket auth error: {e}")
            disconnect()
            return False
    return decorated