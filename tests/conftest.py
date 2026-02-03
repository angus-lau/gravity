import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI application.
    
    Uses module scope to avoid reloading models for each test.
    """
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_query():
    """A typical commercial query for testing."""
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
    """Minimal valid request - just a query."""
    return {"query": "best running shoes"}


@pytest.fixture
def harmful_queries():
    """Queries that should have very low ad eligibility."""
    return [
        "how to make a pipe bomb",
        "my mom just passed away",
        "I'm having thoughts of self-harm",
        "ways to hurt someone",
        "explicit adult content nsfw"
    ]


@pytest.fixture
def commercial_queries():
    """Queries that should have high ad eligibility."""
    return [
        "best running shoes for flat feet",
        "compare iPhone vs Samsung",
        "where to buy organic coffee",
        "laptop recommendations for programming",
        "cheap flights to Hawaii"
    ]


@pytest.fixture
def informational_queries():
    """Informational queries with moderate ad eligibility."""
    return [
        "what is the history of the marathon",
        "why do runners get blisters",
        "how does caffeine affect performance",
        "benefits of morning exercise"
    ]


@pytest.fixture
def edge_case_queries():
    """Edge case queries for robustness testing."""
    return [
        "",  # Empty (should be rejected by validation)
        "   ",  # Whitespace only (should be rejected)
        "a",  # Single character
        "shoes " * 1000,  # Very long query
        "🏃‍♂️ running shoes 👟",  # Emoji
        "running\nshoes\nmarathon",  # Newlines
        "<script>alert('xss')</script>",  # XSS attempt
        "SELECT * FROM users;",  # SQL injection attempt
    ]
