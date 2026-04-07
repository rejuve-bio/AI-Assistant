# test_response.py
import pytest
import os
import jwt
import io
import json
import logging
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load test environment variables (only if .env exists)
if os.path.exists('.env'):
    load_dotenv('.env')

# Generate token once at module level
JWT_SECRET = os.getenv("JWT_SECRET")
if JWT_SECRET is None:
    raise ValueError("JWT_SECRET environment variable is not set. Check your GitHub secrets or .env file.")

TEST_USER_ID = "some_id"
TEST_TOKEN = jwt.encode({"user_id": TEST_USER_ID}, JWT_SECRET, algorithm="HS256")

@pytest.fixture(scope="session", autouse=True)
def mock_external_services():
    """Mock MongoDB, Redis, Qdrant, SentenceTransformer, and logging"""
    mongo_client_mock = MagicMock()
    mongo_db_mock = MagicMock()
    mongo_collection_mock = MagicMock()
    
    mongo_client_mock.__getitem__.return_value = mongo_db_mock
    mongo_db_mock.__getitem__.return_value = mongo_collection_mock
    mongo_collection_mock.create_index = MagicMock()
    
    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    
    qdrant_mock = MagicMock()
    qdrant_collection_mock = MagicMock()
    qdrant_mock.get_collection.return_value = qdrant_collection_mock
    
    log_mock = MagicMock()
    log_mock.level = logging.INFO
    log_mock.emit = MagicMock()
    
    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = [0.0] * 384  # fake embedding

    # Patch the original SentenceTransformer class from the library
    with patch('pymongo.MongoClient', return_value=mongo_client_mock), \
         patch('redis.Redis', return_value=redis_mock), \
         patch('qdrant_client.QdrantClient', return_value=qdrant_mock), \
         patch('logging.handlers.TimedRotatingFileHandler', return_value=log_mock), \
         patch('sentence_transformers.SentenceTransformer', return_value=mock_model_instance):
        yield

@pytest.fixture
def client():
    """Create test client with test config"""
    from app import create_app
    app, socket = create_app()
    with app.test_client() as client:
        yield client

@pytest.fixture
def auth_headers():
    """Return headers with valid auth token"""
    return {
        'Authorization': f'Bearer {TEST_TOKEN}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

def test_health_check(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == "This is health check"

def test_query_endpoint(client, auth_headers):
    """Test endpoint with valid token"""
    response = client.post(
        '/query',
        headers=auth_headers,
        data={
            "query": "hello"
            }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response Data: {response.get_data(as_text=True)}")
    print(f"Response Headers: {dict(response.headers)}")
    assert response.status_code == 200

def test_query_responses(client, auth_headers):
    """Test various query types"""
    test_cases = [
        {"input": {"query": "explain gene FTO"}, "expected_key": "json_format"},
        {"input": {"query": "what is Rejuve Bio"}, "expected_key": "text"}
    ]

    for case in test_cases:
        response = client.post(
            '/query',
            headers=auth_headers,
            data={
                **case["input"],
                "context": json.dumps({"id": None, "resource": "annotation"})
            }
        )
        assert response.status_code == 200
        assert case["expected_key"] in response.json