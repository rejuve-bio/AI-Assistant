
from typing import Dict, Any, Tuple, Optional, List, Union
from app.prompts.hypothesis_prompt import hypothesis_format_prompt,hypothesis_response
from app.storage.sql_redis_storage import RedisGraphManager
import logging
import os
import difflib
import requests


# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API endpoints from environment variables
HYPOTHESIS_CHAT_ENDPOINT = os.getenv('HYPOTHESIS_CHAT_ENDPOINT')
HYPOTHESIS_MAIN_ENDPOINT = os.getenv('HYPOTHESIS_MAIN_ENDPOINT')
HYPOTHESIS_DATA_API = os.getenv('HYPOTHESIS_DATA_API')


class HypothesisGeneration:
    """
    Handles generation and processing of hypotheses based on user queries.
    Interacts with the Hypothesis API to generate hypotheses and retrieve related information.
    """

    def __init__(self, llm) -> None:
        """
        Initialize the HypothesisGeneration class with an LLM instance.
        
        Args:
            llm: Language model instance for generating formatted queries and responses
        """
        self.llm = llm
        self.redis_graph_manager = RedisGraphManager()
        logger.info("HypothesisGeneration initialized with LLM")

    def _make_api_request(self, 
                         method: str, 
                         url: str, 
                         token: str, 
                         params: Optional[Dict] = None, 
                         headers: Optional[Dict] = None,
                         data:Optional[Dict] = None) -> Dict[str, Any]:
        """
        Helper method to make API requests with proper error handling.
        
        Args:
            method: HTTP method (GET, POST)
            url: API endpoint URL
            token: Authentication token
            params: Query parameters
            headers: Request headers
            data: Form data for POST requests
            
        Returns:
            Response as dictionary or error message
        """
        headers = {
            "Authorization": f"Bearer {token}"
        }
        try:
            logger.debug(f"Making {method} request to {url} with data {data} and params {params}")
            if data and method.upper() == "POST":
                response = requests.post(url, data=data, headers=headers)    
            elif method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, headers=headers)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
                
            response.raise_for_status()
            data = response.json()
            return data 
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": f"Request failed Please Try Again"}

    def get_enrich_id_genes_GO_terms(self, token: str, hypothesis_id: str, retrieved_keys: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Process the hypothesis_id to extract enrichment details using the retrieved_keys.
        Then query the appropriate endpoint to get graph/summary.
        
        Args:
            token: Authorization token
            hypothesis_id: The hypothesis ID to query
            retrieved_keys: Dictionary containing keys like "GO" or "Gene"
            
        Returns:
            Tuple of (summary, graph, go_term_used) or error message
        """
        logger.info(f"Processing hypothesis ID {hypothesis_id} with retrieved keys: {retrieved_keys}")
        
        # Step 1: Get hypothesis data
        hypothesis_data = self._make_api_request(
            "GET", 
            HYPOTHESIS_DATA_API, 
            token, 
            params={'id': hypothesis_id}
        )
        logger.info(f"Response from {HYPOTHESIS_DATA_API} are {hypothesis_data}")
        
        if "error" in hypothesis_data:
            logger.error(f"Failed to retrieve hypothesis data: {hypothesis_data['error']}")
            return hypothesis_data, {}, ""
        
        # Extract relevant information
        enrich_id = hypothesis_data.get("enrich_id")
        result = hypothesis_data.get("result", {})
        go_terms = result.get("GO_terms", [])
        causal_gene = result.get("causal_gene")
        
        logger.info(f"Found {len(go_terms)} GO terms for hypothesis ID {hypothesis_id}")
        
        # Initialize extracted result dictionary
        extracted = {"enrich id": enrich_id}

        # Handle causal gene if requested
        if "Gene" in retrieved_keys or "causal_gene" in retrieved_keys:
            if causal_gene:
                extracted["Gene"] = causal_gene
                logger.debug(f"Extracted causal gene: {causal_gene}")
            else:
                logger.debug("No causal gene found in hypothesis data")
        
        # Handle GO term selection
        go_term_name = retrieved_keys.get("GO")
        selected_go_term = None
        
        if go_term_name and go_terms:
            # Find closest matching GO term using string similarity
            go_names = [go["name"] for go in go_terms]
            closest_matches = difflib.get_close_matches(go_term_name, go_names, n=1, cutoff=0.6)
            
            if closest_matches:
                match = closest_matches[0]
                selected_go_term = next((go for go in go_terms if go["name"] == match), None)
                logger.info(f"Matched GO term '{go_term_name}' to '{match}'")
        
        # Default to first GO term if no match found
        if not selected_go_term and go_terms:
            selected_go_term = go_terms[0]
            logger.info(f"No matching GO term found, defaulting to: {selected_go_term['name']}")
        
        if not selected_go_term:
            logger.warning("No valid GO term found for hypothesis")
            return {"error": "No valid GO term found."}, {}, ""
        
        # Store selected GO term information
        extracted["GO"] = selected_go_term["name"]
        go_id = selected_go_term["id"]
        extracted["GO_id"] = go_id
        go_term_used = selected_go_term["name"]  # Store the GO term name for return

        # Get summary for the selected GO term
        logger.info(f"Fetching summary for enrich id {enrich_id} with GO ID {go_id}")
        summary_response = self._make_api_request(
            "POST", 
            HYPOTHESIS_DATA_API, 
            token, 
            params={"id": enrich_id, 
                    "go": go_id}
            )
        if "error" not in summary_response:
            logger.info(f"Successfully retrieved graph and summary {summary_response}")
            return summary_response["summary"], summary_response["graph"], go_term_used
        else:
            logger.error("Failed to fetch graph and summary")
            return {"error": "Failed to fetch graph and summary"}, {}, ""

    def get_by_hypothesis_id(self, token: str, hypothesis_id: str, query=None) -> Dict[str, Any]:
        """
        Retrieve hypothesis information by ID.
        
        Args:
            token: Authorization token
            query: User query
            hypothesis_id: Hypothesis ID to retrieve
            
        Returns:
            Dictionary containing hypothesis text or error message
        """
        logger.info(f"Retrieving hypothesis by ID: {hypothesis_id}")
        
        try:   
            if query: 
                data = {
                    "query": query,
                    "hypothesis_id": hypothesis_id}
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-www-form-urlencoded"
                    }
                response = requests.post(HYPOTHESIS_CHAT_ENDPOINT, data=data, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data
            else:
                cached_graph = self.redis_graph_manager.get_graph_by_id(hypothesis_id)
                if cached_graph and cached_graph.get("summary"):
                    logger.info(f"Cache hit for graph_id={graph_id} {cached_graph}")
                    return {"text": cached_graph["summary"]}

                data = {
                    "hypothesis_id": hypothesis_id
                }

                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-www-form-urlencoded"
                }
                try:
                    response = requests.post(HYPOTHESIS_CHAT_ENDPOINT, data=data, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    self.redis_graph_manager.create_graph(hypothesis_id=data['hypothesis_id'], graph_summary=data['summary'])
                    return data
                except Exception as e:
                    logger.error(f"Failed to retrieve hypothesis by ID: {response}")
                    return "NO summaries provided"
        except:
            return None

    def format_user_query(self, query: str) -> Dict[str, Any]:
        """
        Format user query using the LLM to extract relevant parameters.
        
        Args:
            query: Raw user query
            
        Returns:
            Formatted query parameters
        """
        logger.info(f"Formatting user query: {query}")
        
        try:
            prompt = hypothesis_format_prompt.format(question=query)
            response = self.llm.generate(prompt)
            
            if not response:
                logger.warning("LLM returned empty response for query formatting")
            else:
                logger.info(f"Successfully formatted query with {len(response)} parameters")
                
            return response
        except Exception as e:
            logger.error(f"Error formatting user query: {str(e)}")
            return {}

    def get_hypothesis(self, token: str, user_query: str) -> Union[Tuple[str, Dict[str, Any]], Dict[str, str]]:
        """
        Generate a hypothesis based on the user query.
        
        Args:
            token: Authorization token
            user_query: User's query string
            
        Returns:
            Tuple of (hypothesis_id, retrieved_keys) or error message dictionary
        """
        logger.info(f"Generating hypothesis for query: {user_query}")
        
        # Format the user query
        refactored_query = self.format_user_query(user_query)
        
        if not refactored_query:
            logger.warning("Failed to format user query")
            return {"text": "Sorry I can't help with your question. Please try again elaborating it."}
            
        try:
            # Separate payload parameters from retrieval keys
            payload = {}
            retrieved_keys = {}
            
            for key, value in refactored_query.items():
                if key in {"variant", "phenotype"}:
                    payload[key] = value
                else:
                    retrieved_keys[key] = value
                    
            logger.info(f"Sending request to endpoint {HYPOTHESIS_MAIN_ENDPOINT} Hypothesis request parameters: {payload}")
            
            # Make request to generate hypothesis
            response = self._make_api_request(
                "POST",
                HYPOTHESIS_MAIN_ENDPOINT,
                token,
                params=payload
            )
            
            logger.info(f"Response from {HYPOTHESIS_MAIN_ENDPOINT} endpoint are {response}")
            if "error" in response:
                logger.error(f"Failed to generate hypothesis: {response['error']}")
                return {"text": f"Sorry couldn't generate hypothesis for the given question {user_query}"}
                
            hypothesis_id = response.get("hypothesis_id")
            
            if not hypothesis_id:
                logger.error("No hypothesis ID returned from API")
                return {"text": "Failed to generate hypothesis - no ID returned"}
                
            logger.info(f"Successfully generated hypothesis with ID: {hypothesis_id}")
            return hypothesis_id, retrieved_keys
            
        except Exception as e:
            logger.error(f"Error generating hypothesis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"text": f"Sorry couldn't generate hypothesis for the given question {user_query}"}

    def generate_hypothesis(self, token: str, user_query: str) -> Dict[str, Any]:
        """
        Main method to generate a hypothesis response based on user query.
        
        Args:
            token: Authorization token
            user_query: User's query
            
        Returns:
            Formatted hypothesis response with resource information
        """
        logger.info(f"Processing complete hypothesis generation for: {user_query}")
        
        # Get hypothesis ID and retrieved keys
        result = self.get_hypothesis(token, user_query)
        
        # Check if we got an error instead of a tuple
        if isinstance(result, dict) and "text" in result:
            logger.warning(f"Failed to get hypothesis: {result['text']}")
            return result

        hypothesis_id, retrieved_keys = result
        
        # Get enriched data
        enriched_data, graph, go_term_used = self.get_enrich_id_genes_GO_terms(token, hypothesis_id, retrieved_keys)
        
        if "error" in enriched_data:
            logger.error(f"Failed to enrich hypothesis data: {enriched_data['error']}")
            return {"text": f"No hypothesis is found: {enriched_data['error']}"}
        
        # Generate final response
        logger.info("Generating final hypothesis response")
        logger.info(f"Using GO term: {go_term_used}")
        
        # Updated prompt to include the GO term used
        prompt = hypothesis_response.format(
            response=enriched_data,
            user_query=user_query, 
            graph=graph,
            go_term_used=go_term_used
        )
        response_text = self.llm.generate(prompt)
        # Store summary in Redis cache for 24 hours
        self.redis_graph_manager.create_graph(hypothesis_id=hypothesis_id, graph_summary=response_text)

        # Return in the new format with resource information
        return {
            "text": response_text,
            "resource": {
                "id": hypothesis_id,
                "type": "hypothesis"
            }
        }