import os
from app import create_app
from dotenv import load_dotenv

load_dotenv()
app = create_app()
port = int(os.getenv('FLASK_PORT', 5003))
debug=os.getenv('DEBUG', False)

if __name__ == '__main__':
    port = port
    app.run(host='0.0.0.0', port=port, debug=True)