# Music Transposition API🎵

A web-based API for analyzing songs, detecting keys, transposing chords, and parsing song structures.  
Built with FastAPI, Docker, and OpenAI for intelligent and scalable music analysis.

## Features
- 🎼 **Chord Transposition** – Shift chord progressions up/down by any number of semitones
- 🎹 **Key Detection** – Detect the musical key using rule-based and AI-powered methods
- 📝 **Chord Extraction** – Automatically extract chord notations from song lyrics
- 📊 **Song Structure Analysis** – Identify and organize song sections (e.g., verse, chorus) with chords
- 🧠 **AI Integration** – Leverage OpenAI for intelligent musical analysis with fallback logic

## Tech Stack
- **FastAPI** – Modern Python framework for building high-performance APIs
- **Docker** – Containerization for local and cloud deployment (Azure)
- **SQLite** – Lightweight relational database for storing analysis history
- **OpenAI API** – For enhanced key detection using AI
- **Uvicorn** – ASGI server for running the FastAPI app

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Amit-Aviav/Music-Transposition.git
   cd Music-Transposition
   ```

2. Create & activate venv:
   Windows (PowerShell):
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
   Note: This project uses the legacy OpenAI Python SDK (openai 0.27.x).

   macOS / Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Create & activate venv: 

Windows (PowerShell):
```
python -m venv venv
.\venv\Scripts\Activate.ps1
Note: This project uses the legacy OpenAI Python SDK (openai 0.27.x). 
```
macOS / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. (Optional) Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
If you don’t set OPENAI_API_KEY, the API should still run and non-AI endpoints will work.

5. Run the API:
   ```
   uvicorn main:app --reload
   ```

6. Visit `http://localhost:8000/docs` to access the interactive API documentation.

7. Run tests:
   ```
   pytest
   ```
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