import socketio
import time
import uuid
import json

# Socket.IO server URL
SERVER_URL = 'http://localhost:5012'

class AIAssistantClient:
    def __init__(self):
        # Use the working configuration from your debug test
        self.sio = socketio.Client(
            logger=False,  # Set to True for debugging
            engineio_logger=False
        )
        self.test_user_id = str(uuid.uuid4())[:8]
        self.setup_handlers()
        self.response_received = False
        self.latest_response = None

    def setup_handlers(self):
        """Setup all event handlers"""
        
        @self.sio.event
        def connect():
            print(f'[✔] Connected to AI Assistant Server')
            print(f'User ID: {self.test_user_id}')

        @self.sio.event
        def disconnect():
            print('[✖] Disconnected from server')

        @self.sio.on('message')
        def handle_message(data):
            print(f'Server message: {data}')

        @self.sio.on('status')
        def handle_status(data):
            print(f'[ℹ️] Status: {data}')

        @self.sio.on('processing_update')
        def handle_processing_update(data):
            status = data.get('status', 'unknown')
            message = data.get('message', 'Processing...')
            progress = data.get('progress', '')
            
            if progress:
                print(f'{message} ({progress}%)')
            else:
                print(f' {message}')

        @self.sio.on('query_response')
        def handle_query_response(data):
          
            response = data.get('response')
            if response:
                print(f'    Response: {response}')
            else:
                print(f'    Response: [NULL - Check server processing]')
            
            self.latest_response = data
            self.response_received = True

        @self.sio.on('error')
        def handle_error(data):
            print(f'Error: {data}')
            self.response_received = True

    def connect(self):
        """Connect to the server"""
        try:
            print(f'[🔌] Connecting to {SERVER_URL}...')
            self.sio.connect(SERVER_URL)
            return True
        except Exception as e:
            print(f'Connection failed: {e}')
            return False

    def disconnect(self):
        """Disconnect from server"""
        try:
            self.sio.disconnect()
        except:
            pass

    def join_room(self):
        """Join user-specific room"""
        print(f'Joining room for user {self.test_user_id}...')
        self.sio.emit('join_room', {'user_id': self.test_user_id, 'query':"what is gene FTO"})
        time.sleep(100)  # Give server time to process

    def send_query(self, query, context=None, graph=None, token=None):
        """Send a query to the AI assistant"""
        
        query_data = {
            'user_id': self.test_user_id,
            'query': query,
            'context': context or {'id': None, 'resource': None},
            'graph': graph
        }
        
        # Add token if provided
        if token:
            query_data['token'] = token
        
        print(f'\nSending query: "{query}"')
        
        # Reset response flag
        self.response_received = False
        self.latest_response = None
        
        # Send the query
        self.sio.emit('process_query', query_data)
        
        # Wait for response
        timeout = 120  # 30 seconds timeout
        elapsed = 0
        
        while not self.response_received and elapsed < timeout:
            time.sleep(0.5)
            elapsed += 0.5
        
        if not self.response_received:
            print(f'Timeout waiting for response after {timeout} seconds')
            return None
        
        return self.latest_response

    def test_ping(self):
        """Test ping/pong functionality"""
        print(f'\n[🏓] Testing ping...')
        
        @self.sio.on('pong')
        def handle_pong(data):
            print(f'[🏓] Pong received: {data}')
        
        self.sio.emit('ping', {'test': 'ping test'})
        time.sleep(2)

def main():
    print("🤖 AI Assistant WebSocket Client")
    print("=" * 50)
    
    client = AIAssistantClient()
    
    try:
        # Connect to server
        if not client.connect():
            return
        
        # Join user room
        client.join_room()
        
        # Test basic functionality
        client.test_ping()
      
    finally:
        print(f"\n[👋] Disconnecting...")
        client.disconnect()
        print("[🏁] Session ended")

if __name__ == '__main__':
    main()