from app.lib.auth import token_required
from flask import Blueprint, request, current_app
from dotenv import load_dotenv
import traceback
import json

load_dotenv()
main_bp = Blueprint('main', __name__)

@main_bp.route('/query', methods=['POST'])
@token_required
def process_query(current_user_id, auth_token):
    """
    Notes:
    - `query`: Contains the user's question or prompt.
    - `file`: Used when a file (e.g., a PDF) is uploaded for processing.
    - `id`: Represents a graph ID and should be included if relevant to the query.(when Explaining a node is asked from a given graph)
    - `resource`: Identifies the type of resource associated with the `id`. Currently not in use but it may support other types (e.g., "Hypothesis") in the future.
    """

    try:
        ai_assistant = current_app.config['ai_assistant']
    
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            response = ai_assistant.assistant_response(
                            user_id=current_user_id,
                            file=file,
                            token=None,
                            query=None
                            )
            return response
        
        data = request.form
        if not data:
            return "Invalid request format.", 400

        # Extract fields from the input format
        query = data.get('query', None)
        context = json.loads(data.get('context', '{}'))  
        context_id = context.get('id', None)
        resource = context.get('resource', None)
        graph = data.get('graph',None)


        # Handle query processing
        response = ai_assistant.assistant_response(
            query=query,
            user_id=current_user_id,
            token=auth_token,
            graph_id=context_id,
            graph=graph
        )
        return response

    except Exception as e:
        traceback.print_exc()
        return f"Bad Response: {e}", 400

 