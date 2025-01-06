import os
from app import create_app
from dotenv import load_dotenv

load_dotenv()
port = os.getenv('FLASK_PORT', 5003)
app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', port))
    app.run(host='0.0.0.0', port=port, debug=True)