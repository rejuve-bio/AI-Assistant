
import pytest
import json
import os
from app import create_app
from helper.access_token_generator import access_token_generator

@pytest.fixture
def app():
    app = create_app()
    return app

def test_base_route(app):
    client = app.test_client()
    client = app.test_client()
    url = '/query'
    token = access_token_generator()
    query = "get me the summary of the pdf?"
    user = "4"

    response = client.post(url, json={'query': query,'user':user}, headers={'Authorization':token})
    print("***",response.data)
    print("***",response.status_code)
    assert response.status_code == 200

