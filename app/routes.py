from app.lib.auth import token_required
from flask import Blueprint, request, current_app
from dotenv import load_dotenv
import traceback


load_dotenv()
main_bp = Blueprint('main', __name__)

@main_bp.route('/query', methods=['POST'])
@token_required
def process_query(current_user_id, token):
    """
    This API accepts a query request and returns a response containing content 
    and an optional graph (if the query is graph-related), or just content for queries 
    without any graph-related information. It also supports Uploading files (e.g., PDF).
    """
    try:

        ai_assistant = current_app.config['ai_assistant']
        if request.is_json:
            data = request.get_json() 
            query = data.get('query', None)
            graph = data.get('graph', None)
            graph_id = data.get('graph_id', None)
            
            response = ai_assistant.assistant_response(query=query,user_id=current_user_id,graph=graph,graph_id=graph_id)
            return response

        elif 'file' in request.files:
            file = request.files.get('file')  
        
            # Ensure it's a valid PDF file
            if file and file.filename.lower().endswith('.pdf'):
                response = ai_assistant.assistant_response(query=None,user_id=current_user_id,graph=None,graph_id=None,file=file)
                return response
            else:
                return "Only PDF files are supported.", 400

    except Exception as e:
        traceback.print_exc()
        return f"Bad Response: {e}", 400

