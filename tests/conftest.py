import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_query():
    return {
        "query": "I'm running a marathon next month and need new shoes. What should I get?",
        "context": {
            "gender": "male",
            "age": 24,
            "location": "San Francisco, CA",
            "interests": ["fitness", "outdoor activities"]
        }
    }


@pytest.fixture
def minimal_query():
    return {"query": "best running shoes"}
