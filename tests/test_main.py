import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import os
import sqlite3
import tempfile

# Import your FastAPI app
from main import app, detect_key, transpose_chord, transpose_song

# Create test client
client = TestClient(app)


# Setup test database
@pytest.fixture
def test_db():
    # Use tempfile to create a temporary database file
    db_fd, db_path = tempfile.mkstemp()

    # Create test database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create necessary tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transpositions (
        id TEXT PRIMARY KEY,
        original_content TEXT NOT NULL,
        transposed_content TEXT NOT NULL,
        half_steps INTEGER NOT NULL,
        original_key TEXT,
        target_key TEXT,
        created_at TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS key_detections (
        id TEXT PRIMARY KEY,
        original_content TEXT NOT NULL,
        detected_key TEXT NOT NULL,
        confidence REAL NOT NULL,
        alternative_keys TEXT NOT NULL,
        created_at TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS api_logs (
        id TEXT PRIMARY KEY,
        api_name TEXT NOT NULL,
        request_data TEXT NOT NULL,
        response_data TEXT NOT NULL,
        created_at TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()

    # Yield the path to the test database
    yield db_path

    # Clean up after test
    os.close(db_fd)
    os.unlink(db_path)


# Test basic functions directly
def test_transpose_chord():
    assert transpose_chord('C', 2) == 'D'
    assert transpose_chord('G', 5) == 'C'
    assert transpose_chord('Am', 3) == 'Cm'


def test_transpose_song():
    input_song = "C G Am F"
    expected_output = "D A Bm G"
    assert transpose_song(input_song, 2) == expected_output


def test_detect_key():
    key, confidence, alternatives = detect_key("C F G C")
    assert isinstance(key, str)
    assert isinstance(confidence, float)
    assert isinstance(alternatives, list)


# Test API endpoints with mocked database
@patch('sqlite3.connect')
def test_transpose_endpoint(mock_connect, test_db):
    # Setup mock database connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    response = client.post(
        "/transpose/",
        json={"content": "C G Am F", "half_steps": 2}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["original_content"] == "C G Am F"
    assert data["transposed_content"] == "D A Bm G"
    assert data["half_steps"] == 2


@patch('sqlite3.connect')
def test_detect_key_endpoint(mock_connect, test_db):
    # Setup mock database connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    response = client.post(
        "/detect-key/",
        json={"content": "C G Am F"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["original_content"] == "C G Am F"
    assert isinstance(data["detected_key"], str)
    assert isinstance(data["confidence"], float)


@patch('openai.OpenAI')
@patch('sqlite3.connect')
def test_detect_key_ai_endpoint(mock_connect, mock_openai, test_db):
    # Setup mock database connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    # Mock the OpenAI client response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "detected_key": "C Major",
        "confidence": 0.95,
        "alternative_keys": ["A Minor", "F Major"],
        "analysis": "This is a classic C Major progression"
    })
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    # Test the endpoint
    response = client.post(
        "/detect-key-ai/",
        json={"content": "C G Am F", "use_ai": True}
    )

    assert response.status_code == 200