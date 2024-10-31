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
    