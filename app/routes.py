from flask import Blueprint, request, current_app
from dotenv import load_dotenv
import traceback

load_dotenv()
main_bp = Blueprint('main', __name__)

@main_bp.route('/query', methods=['POST'])
def process_query():
    '''
    This API accepts a query request and returns a response containing content 
    and an optional graph (if the query is graph-related), or just content for queries 
    without any graph-related information.
    '''
    try:
        data = request.json
        query = data.get('query', None)
        graph = data.get('graph', None)
        user = data.get('user', None)
        graph_id = data.get('graph_id',None)

        ai_assistant = current_app.config['ai_assistant']
        response = ai_assistant.assistant_response(query, graph, user, graph_id)
        return response
    except:
        traceback.print_exc()
        return "Bad Response", 400
  