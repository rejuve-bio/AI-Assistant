import logging
from typing import List
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class SimilarityLookupError(RuntimeError):
    """The similarity search could not run.

    Distinct from "no similar values": returning empties for a failed lookup
    makes an outage indistinguishable from a gene that genuinely isn't there,
    and the user gets told it doesn't exist.
    """

class Neo4jConnection:
    _drivers = {}  # uri -> driver

    def __init__(self, uri: str = None, username: str = None, password: str = None):
        self.uri = uri
        if uri and username and password and uri not in Neo4jConnection._drivers:
            Neo4jConnection._drivers[uri] = GraphDatabase.driver(uri, auth=(username, password))
            logger.info(f"Neo4j driver created for {uri}")

    def get_driver(self):
        driver = Neo4jConnection._drivers.get(self.uri)
        if driver is None:
            raise ConnectionError(
                f"Neo4j connection not initialized for {self.uri!r}. Construct with credentials first."
            )
        return driver

    def close(self):
        driver = Neo4jConnection._drivers.pop(self.uri, None)
        if driver:
            driver.close()

    def get_similar_property_values_batch(self, label: str,
                                          property_key: str,
                                          search_values: List[str],
                                          top_k: int = 10,
                                          threshold: float = 0.7) -> dict:

        logger.info(f"Batch searching {len(search_values)} values in '{label}.{property_key}'.")

        query = f"""
        MATCH (n:{label})
        WITH DISTINCT n.{property_key} AS value
        WHERE value IS NOT NULL
        WITH collect(value) AS all_values
        UNWIND $search_values AS search_value
        UNWIND all_values AS value
        WITH search_value, value,
             // Jaro-Winkler, not Levenshtein: Levenshtein scores a transposition
             // as two edits, so "BRAC1" ranks BRCA1 below PRAC1, RAC1 and BRAT1
             // and it never reaches the candidate list at all.
             // Note this is a DISTANCE -- 0 is identical.
             1 - apoc.text.jaroWinklerDistance(LOWER(value), LOWER(search_value)) AS similarity
        WHERE similarity > $threshold
        ORDER BY search_value, similarity DESC
        WITH search_value, collect({{value: value, similarity: similarity}})[..{top_k}] AS top_matches
        RETURN search_value, top_matches
        """

        try:
            driver = self.get_driver()
            with driver.session() as session:
                result = session.run(query, search_values=search_values, threshold=threshold)
                batch = {}
                for record in result:
                    sv = record["search_value"]
                    batch[sv] = [(m["value"], round(m["similarity"], 2)) for m in record["top_matches"]]
                for sv in search_values:
                    if sv not in batch:
                        batch[sv] = []
                logger.info(f"Batch query returned results for {len(batch)} search values.")
                return batch

        except Exception as e:
            logger.error(f"Error in batch Neo4j query: {str(e)}")
            raise SimilarityLookupError(str(e)) from e

    def get_ids_for_property_values_batch(self, label: str,
                                          property_key: str,
                                          values: List[str]) -> dict:
        logger.info(f"Resolving database id for {len(values)} value(s) in '{label}.{property_key}'.")

        query = f"""
        MATCH (n:{label})
        WHERE n.{property_key} IN $values
        WITH n.{property_key} AS value, n.id AS id
        ORDER BY id
        WITH value, collect(id) AS ids
        RETURN value, ids[0] AS db_id, size(ids) AS candidate_count
        """

        try:
            driver = self.get_driver()
            with driver.session() as session:
                result = session.run(query, values=values)
                resolved = {}
                for record in result:
                    if record["candidate_count"] > 1:
                        logger.warning(
                            f"'{record['value']}' matches {record['candidate_count']} "
                            f"distinct '{label}.id' values in '{label}.{property_key}' — "
                            f"using {record['db_id']!r} (lowest id) deterministically."
                        )
                    resolved[record["value"]] = record["db_id"]
                for v in values:
                    resolved.setdefault(v, None)
                return resolved

        except Exception as e:
            logger.error(f"Error resolving ids for '{label}.{property_key}': {str(e)}")
            return {v: None for v in values}

    def get_similar_property_values(self, label: str,
                                    property_key: str,
                                    search_value: str,
                                    top_k: int = 10,
                                    threshold: float = 0.3):
 
        logger.info(f"Searching for similar values for '{search_value}' in label '{label}' with property key '{property_key}'.")

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
                logger.debug("Executing Neo4j query...")
                result = session.run(
                    query,
                    search_value=search_value,
                    threshold=threshold
                )
                similar_values = [(record["value"], round(record["similarity"], 2)) 
                                for record in result]
                logger.info(f"Found {len(similar_values)} similar values: {similar_values}.")

            return similar_values
        
        except Exception as e:
            logger.error(f"Error querying Neo4j: {str(e)}")
            return []