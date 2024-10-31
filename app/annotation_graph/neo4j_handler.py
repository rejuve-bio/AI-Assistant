import logging
from typing import List
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Singleton class to manage Neo4j connection"""
    _instance = None
    _driver = None

    def __new__(cls, uri: str = None, username: str = None, password: str = None):
        if cls._instance is None:
            cls._instance = super(Neo4jConnection, cls).__new__(cls)
            if uri and username and password:
                cls._driver = GraphDatabase.driver(uri, auth=(username, password))
        return cls._instance

    @classmethod
    def get_driver(cls):
        if cls._driver is None:
            raise ConnectionError("Neo4j connection not initialized. Call with credentials first.")
        return cls._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def get_similar_property_values(self, label: str, 
                                    property_key: str, 
                                    search_value: str, 
                                    top_k: int = 10, 
                                    threshold: float = 0.3) -> List[str]:
        """
        Get distinct top k similar property values from Neo4j using Levenshtein similarity.
        
        Args:
            label (str): Node label to search (e.g., "gene")
            property_key (str): Property key to search (e.g., "gene_name")
            search_value (str): Value to search for
            threshold (float): Similarity threshold (0 to 1)
        
        Returns:
            List[str]: List of distinct similar property values sorted by similarity score
        """     
        query = f"""
        MATCH (n:{label})
        WITH DISTINCT n.{property_key} as value
        WHERE value IS NOT NULL
        WITH collect(value) as all_values
        UNWIND all_values as value
        WITH DISTINCT value, apoc.text.levenshteinSimilarity(
            LOWER(value), 
            LOWER($search_value)
        ) AS similarity
        WHERE similarity > $threshold
        RETURN value, similarity
        ORDER BY similarity DESC
        LIMIT {top_k}
        """
        
        try:
            driver = self.get_driver()
            with driver.session() as session:
                result = session.run(
                    query,
                    search_value=search_value,
                    threshold=threshold
                )
                similar_values = [(record["value"], record["similarity"]) 
                                for record in result]
            
            # Return just the values (without scores)
            return [(value, round(score, 2)) for value, score in similar_values]
        
        except Exception as e:
            logger.error(f"Error querying Neo4j: {str(e)}")
            return []