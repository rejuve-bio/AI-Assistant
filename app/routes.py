from flask import Blueprint, request, current_app
from dotenv import load_dotenv
import traceback


load_dotenv()
main_bp = Blueprint('main', __name__)

@main_bp.route('/query', methods=['POST'])
def process_query():
    """
    This API accepts a query request and returns a response containing content 
    and an optional graph (if the query is graph-related), or just content for queries 
    without any graph-related information. It also supports Uploading files (e.g., PDF).
    """
    try:
        ai_assistant = current_app.config['ai_assistant']
        if request.is_json:
            data = request.get_json() 
            user = data.get('user', None)
            query = data.get('query', None)
            graph = data.get('graph', None)
            graph_id = data.get('graph_id', None)
            
            response = ai_assistant.assistant_response(query=query,user_id=user,graph=graph,graph_id=graph_id)
            return response

        elif request.form:
            data = request.form.to_dict()
            user = data.get('user', None)
            file = request.files['file']
        
            # Ensure it's a valid PDF file
            if file and file.filename.lower().endswith('.pdf'):
                response = ai_assistant.assistant_response(query=None,user_id=user,graph=None,graph_id=None,file=file)
                return response
            else:
                return "Only PDF files are supported.", 400

    except Exception as e:
        traceback.print_exc()
        return f"Bad Response: {e}", 400

