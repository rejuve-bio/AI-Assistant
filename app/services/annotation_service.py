import json
import requests


def query_knowledge_graph(kg_service_url, json_query):
    
    try:
        # Ensure the json_query has the correct structure
        if 'nodes' not in json_query or 'predicates' not in json_query:
            raise ValueError("Invalid JSON query structure")

        # Construct the payload in the format expected by the KG service
        payload = {
            "requests": json_query
        }
        
        print("Sending KG query:", json.dumps(payload, indent=2))  # Debug print
        response = requests.post(kg_service_url, json=payload)
        print("response from annotation", response)
        response.raise_for_status()
        return {
            "status_code": response.status_code,
            "response": response.json()
        }
    
    except requests.RequestException as e:
        # Capture the status code if the response is available
        status_code = e.response.status_code if e.response is not None else "No Status Code"
        print(f"Error querying knowledge graph: {e}")
        if e.response is not None:
            print(f"Response content: {e.response.text}")  # Print the response content for more detail
        return {
            "error": f"Failed to query knowledge graph: {str(e)}",
            "status_code": status_code
        }
    
    except ValueError as e:
        print(f"Error with JSON query structure: {e}")
        return {
            "error": str(e),
            "status_code": "Invalid JSON query structure"
        }