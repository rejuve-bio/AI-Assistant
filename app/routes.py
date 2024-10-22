from flask import Blueprint, request, jsonify, current_app,Response
from app.services.annotation_service import query_knowledge_graph
from .services.ai_assistant import AIAssistantSystem
from .services.llm_models import get_llm_model
from .services.summarizer import Graph_Summarizer


main_bp = Blueprint('main', __name__)


@main_bp.route('/query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    config = current_app.config
    schema_text = open(config['schema_path'], 'r').read()

    llm = get_llm_model(config)    
    ai_system = AIAssistantSystem(llm, schema_text)

    json_query = ai_system.process_query_traversing(query)
    print("================== Json query Traversing ===============")
    print(json_query)
    kg_response = query_knowledge_graph(config['annotation_service_url'], json_query)
    if kg_response['status_code'] == 200:
        kg_responses = kg_response['response']
        print("Using the response from the first method.")
    else:
        print(f"Error from the first method: {kg_response['error']}, Status Code: {kg_response['status_code']} Attempting to generate query from the second method.")       
        json_query = ai_system.process_query(query)
        print("================== Generated Json query ===============")
        print(json_query)
        kg_response = query_knowledge_graph(config['annotation_service_url'], json_query)
        if kg_response['status_code'] == 200:
            kg_responses = kg_response['response']
            print("Using the response from the second method.")
        else:
            print(f"annotating user query failed {kg_response}")
            return Response("Empty_page", status=400)

    print("================== KG response ===============")
    print(kg_response)
    final_response = ai_system.process_kg_response(query, json_query, kg_responses)
    return jsonify(final_response)

@main_bp.route('/summarizer', methods=['POST'])
def summarize_graph():
    data = request.json
    graph = data.get('graph')

    if not graph:
        return jsonify({"error": "No Valid Graph is provided"}), 400
    try:
        config = current_app.config
        llm = get_llm_model(config)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    summary = Graph_Summarizer(llm)
    graph_summary = summary.open_ai_summarizer(graph)
    return jsonify(graph_summary)
  
