# Music Transposition API

A FastAPI-based service for music analysis, chord transposition, and song structure parsing.

## Features

- **Chord Transposition**: Transpose chord progressions up or down by any number of half steps
- **Key Detection**: Detect the musical key of chord progressions using both algorithmic and AI-powered methods
- **Chord Extraction**: Extract chord notations from text files
- **Song Structure Analysis**: Parse and organize songs into sections with chord progressions

## Tech Stack

- **FastAPI**: Modern, high-performance web framework for building APIs
- **SQLite**: Lightweight database for storing transpositions and analysis results
- **OpenAI API**: AI-powered musical analysis for enhanced key detection

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/music-transposition-api.git
   cd music-transposition-api
   ```

2. Install dependencies:
   ```
   pip install fastapi uvicorn pydantic python-multipart python-openai python-dotenv
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the API:
   ```
   uvicorn main:app --reload
   ```

5. Visit `http://localhost:8000/docs` to access the interactive API documentation.

## API Endpoints

### Transposition

- **POST /transpose/**: Transpose a chord progression by a specified number of half steps

### Key Detection

- **POST /detect-key/**: Detect the key of a chord progression using a basic algorithm
- **POST /detect-key-ai/**: Enhanced key detection using OpenAI API with fallback to the basic algorithm

### Chord Extraction

- **POST /extract-chords/**: Extract chords from a text file

### Song Structure

- **POST /song-structure/**: Analyze and structure a song into sections with chords

### History

- **GET /history/**: Get a history of all transpositions
- **GET /transposition/{id}**: Get a specific transposition by ID

## Examples

### Transposing a Chord Progression

```bash
curl -X POST "http://localhost:8000/transpose/" \
     -H "Content-Type: application/json" \
     -d '{"content": "C G Am F", "half_steps": 2}'
```

Response:
```json
{
  "id": "uuid",
  "original_content": "C G Am F",
  "transposed_content": "D A Bm G",
  "half_steps": 2,
  "original_key": "C Major",
  "target_key": "D Major",
  "created_at": "2025-04-25T12:34:56"
}
```

### Detecting Key with AI

```bash
curl -X POST "http://localhost:8000/detect-key-ai/" \
     -H "Content-Type: application/json" \
     -d '{"content": "C G Am F C F G C", "use_ai": true}'
```

## Database Schema

The application uses SQLite with the following tables:
- `transpositions`: Stores all chord transpositions
- `key_detections`: Stores results of basic key detection
- `ai_key_detections`: Stores results of AI-enhanced key detection
- `chord_extractions`: Stores extracted chords from text
- `song_structures`: Stores song structure analysis results
- `api_logs`: Logs all external API calls

## License

[MIT License](LICENSE)