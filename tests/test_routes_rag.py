
import pytest
import json
import os
from app import create_app

@pytest.fixture
def app():
    app = create_app()
    return app

def test_base_route(app):
    client = app.test_client()
    client = app.test_client()
    url = '/query'
    auth_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoic29tZV9pZCJ9.bLKE-JJSFBT9jJjd9_002lUDiikbvI2RoBkSJs0mTDk'
    query = "get me the summary of the pdf?"
    user = "4"

    response = client.post(url, json={'query': query,'user':user}, headers={'Authorization':auth_token})
    print("***",response.data)
    print("***",response.status_code)
    assert response.status_code == 200

