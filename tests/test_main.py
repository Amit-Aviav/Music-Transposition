import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import os
import sqlite3
import uuid
from datetime import datetime

# Import your FastAPI app
from main import app, detect_key, transpose_chord, transpose_song

# Create test client
client = TestClient(app)


# Setup test database
@pytest.fixture(autouse=True)
def setup_test_db():
    # Create test database
    conn = sqlite3.connect('test_transposition.db')
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

    # Use test database in app (patch sqlite3.connect)
    with patch('sqlite3.connect', return_value=sqlite3.connect('test_transposition.db')):
        yield

    # Clean up test database after tests
    if os.path.exists('test_transposition.db'):
        os.remove('test_transposition.db')


# Test for transpose_chord function
def test_transpose_chord():
    # Test major chords
    assert transpose_chord('C', 2) == 'D'
    assert transpose_chord('G', 5) == 'C'

    # Test minor chords
    assert transpose_chord('Am', 3) == 'Cm'

    # Test complex chords
    assert transpose_chord('Cmaj7', 4) == 'Emaj7'

    # Test sharps and flats
    assert transpose_chord('F#', 1) == 'G'
    assert transpose_chord('Bb', -2) == 'Ab'


# Test for transpose_song function
def test_transpose_song():
    # Test simple chord progression
    input_song = "C G Am F"
    expected_output = "D A Bm G"
    assert transpose_song(input_song, 2) == expected_output

    # Test multi-line chord progression
    input_song = "C G Am F\nF C G"
    expected_output = "D A Bm G\nG D A"
    assert transpose_song(input_song, 2) == expected_output


# Test for detect_key function
def test_detect_key():
    # Test simple C major progression
    key, confidence, alternatives = detect_key("C F G C")
    assert key == "C Major"
    assert confidence > 0.5
    assert "G Major" in alternatives or "A Minor" in alternatives

    # Test G major progression
    key, confidence, alternatives = detect_key("G D Em C")
    assert key == "G Major"
    assert confidence > 0.5


# Test for /transpose/ endpoint
def test_transpose_endpoint():
    response = client.post(
        "/transpose/",
        json={"content": "C G Am F", "half_steps": 2}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["original_content"] == "C G Am F"
    assert data["transposed_content"] == "D A Bm G"
    assert data["half_steps"] == 2


# Test for /detect-key/ endpoint
def test_detect_key_endpoint():
    response = client.post(
        "/detect-key/",
        json={"content": "C G Am F"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["original_content"] == "C G Am F"
    assert data["detected_key"] == "C Major"
    assert isinstance(data["confidence"], float)
    assert len(data["alternative_keys"]) > 0


# Test for history endpoint
def test_history_endpoint():
    # First add a transposition
    client.post(
        "/transpose/",
        json={"content": "C G Am F", "half_steps": 2}
    )

    # Then get history
    response = client.get("/history/")

    assert response.status_code == 200
    data = response.json()
    assert "transpositions" in data
    assert len(data["transpositions"]) >= 1
    assert data["transpositions"][0]["original_content"] == "C G Am F"


# Test for song structure endpoint
def test_song_structure_endpoint():
    response = client.post(
        "/song-structure/",
        json={"content": "C G Am F\nG C F G", "song_name": "Test Song", "artist": "Test Artist"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["song_name"] == "Test Song"
    assert data["artist"] == "Test Artist"
    assert len(data["structure"]) == 2


# Test for the new detect-key-ai endpoint
@patch('openai.OpenAI')
def test_detect_key_ai_endpoint(mock_openai):
    # Mock the OpenAI client response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "detected_key": "C Major",
        "confidence": 0.95,
        "alternative_keys": ["A Minor", "F Major"],
        "analysis": "This is a classic C Major progression"
    })
    mock_client.chat.completions.create.return_value = mock_response

    # Test the endpoint
    response = client.post(
        "/detect-key-ai/",
        json={"content": "C G Am F", "use_ai": True}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["detected_key"] == "C Major"
    assert "ai_analysis" in data
    assert data["ai_analysis"] == "This is a classic C Major progression"


# Test for database logging
@patch('openai.OpenAI')
def test_api_logs(mock_openai):
    # Setup mock response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "detected_key": "C Major",
        "confidence": 0.95,
        "alternative_keys": ["A Minor"],
        "analysis": "Test analysis"
    })
    mock_client.chat.completions.create.return_value = mock_response

    # Make API call
    client.post(
        "/detect-key-ai/",
        json={"content": "C G Am F", "use_ai": True}
    )

    # Check if log was created in database
    conn = sqlite3.connect('test_transposition.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM api_logs')
    logs = cursor.fetchall()
    conn.close()

    assert len(logs) >= 1
    # The first log should be for OpenAI
    assert "OpenAI" in logs[0][1]  # api_name field


# Test error handling
def test_error_handling():
    # Test with invalid input
    response = client.post(
        "/transpose/",
        json={"content": "", "half_steps": "invalid"}
    )
    assert response.status_code == 422  # Validation error

    # Test with non-existent ID
    response = client.get("/transposition/non-existent-id")
    assert response.status_code == 404