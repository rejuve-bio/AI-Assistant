import socketio
import jwt
import os
import sys
import threading
import uuid
from dotenv import load_dotenv

# Load environment variables (for JWT_SECRET) from the parent directory
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    print("Error: JWT_SECRET not found in .env")
    sys.exit(1)

# Get username from command line argument, default to user1
username = sys.argv[1] if len(sys.argv) > 1 else "user1"

# Generate a valid token for this user
token = jwt.encode({"user_id": username}, JWT_SECRET, algorithm="HS256")

# Generate a unique conversation thread ID for this client session
thread_id = str(uuid.uuid4())

# Create standard python-socketio client
sio = socketio.Client()

@sio.event
def connect():
    print(f"\n[{username}] Connected to server successfully!")
    print(f"[{username}] Type your question and press Enter. (Type 'quit' to exit)")
    
@sio.event
def response(data):
    print(f"\n[{username}] Received response from Assistant: {data}")
    print(f"[{username}] > ", end="", flush=True)

@sio.event
def disconnect():
    print(f"\n[{username}] Disconnected from server.")

def input_thread():
    while True:
        try:
            question = input(f"[{username}] > ")
            if question.lower() == 'quit':
                sio.disconnect()
                break
            
            # Emit the 'question' event
            sio.emit('question', {'question': question, 'thread_id': thread_id})
            
        except (KeyboardInterrupt, EOFError):
            sio.disconnect()
            break

if __name__ == '__main__':
    print(f"Starting client for {username}...")
    try:
        # Note: socket.io-client passes custom headers this way
        sio.connect("http://localhost:8000", headers={"Authorization": f"Bearer {token}"})
        
        # Start a thread to read user input so it doesn't block the socketio event loop
        t = threading.Thread(target=input_thread)
        t.daemon = True
        t.start()
        
        sio.wait()
    except socketio.exceptions.ConnectionError as e:
        print(f"Connection failed: {e}")
        print("Make sure your FastAPI server is running on http://localhost:8000")
