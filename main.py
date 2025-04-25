# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import uuid
from datetime import datetime
import re
from fastapi import File, UploadFile

# Initialize FastAPI
app = FastAPI(title="Music Transposition API",
              description="API for transposing songs to different keys",
              version="1.0.0")


# Initialize Database
def init_db():
    conn = sqlite3.connect('transposition.db')
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()


class KeyDetectionRequest(BaseModel):
    content: str


class KeyDetectionResponse(BaseModel):
    id: str
    original_content: str
    detected_key: str
    confidence: float
    alternative_keys: List[str]
    created_at: str


class ChordInfo(BaseModel):
    chord: str
    position: int


class SectionInfo(BaseModel):
    name: str
    chords: List[ChordInfo]


class SongStructureRequest(BaseModel):
    content: str
    song_name: str
    artist: Optional[str] = None
    sections: Optional[List[str]] = None  # Optional section names like "Verse", "Chorus", etc.


class SongStructureResponse(BaseModel):
    id: str
    song_name: str
    artist: Optional[str] = None
    structure: List[SectionInfo]
    created_at: str


class ChordExtractionResponse(BaseModel):
    id: str
    original_content: str
    extracted_chords: str
    created_at: str


def detect_key(chord_progression):
    # Simple key detection based on chord frequency and common progressions
    # Note: This is a simplified approach - a real implementation would be more sophisticated

    # Define common chords in each key
    key_chords = {
        'C Major': ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim'],
        'G Major': ['G', 'Am', 'Bm', 'C', 'D', 'Em', 'F#dim'],
        'D Major': ['D', 'Em', 'F#m', 'G', 'A', 'Bm', 'C#dim'],
        'A Major': ['A', 'Bm', 'C#m', 'D', 'E', 'F#m', 'G#dim'],
        'E Major': ['E', 'F#m', 'G#m', 'A', 'B', 'C#m', 'D#dim'],
        'F Major': ['F', 'Gm', 'Am', 'Bb', 'C', 'Dm', 'Edim'],
        'Bb Major': ['Bb', 'Cm', 'Dm', 'Eb', 'F', 'Gm', 'Adim'],
        'A Minor': ['Am', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G'],
        'E Minor': ['Em', 'F#dim', 'G', 'Am', 'Bm', 'C', 'D'],
        'B Minor': ['Bm', 'C#dim', 'D', 'Em', 'F#m', 'G', 'A'],
    }

    # Parse the chord progression
    chords = []
    for line in chord_progression.split('\n'):
        chords.extend(line.strip().split())

    # Count matches for each key
    key_scores = {}
    for key, key_chord_list in key_chords.items():
        score = 0
        for chord in chords:
            # Basic chord name (without extensions)
            basic_chord = chord.replace('7', '').replace('maj', '').replace('min', 'm')

            # Check if chord is in this key
            if basic_chord in key_chord_list:
                score += 1

        key_scores[key] = score / len(chords) if chords else 0

    # Get the top keys
    sorted_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
    detected_key = sorted_keys[0][0]
    confidence = sorted_keys[0][1]
    alternative_keys = [k for k, s in sorted_keys[1:4]]

    return detected_key, confidence, alternative_keys


@app.post("/detect-key/", response_model=KeyDetectionResponse)
async def detect_song_key(request: KeyDetectionRequest):
    try:
        # Detect the key
        detected_key, confidence, alternative_keys = detect_key(request.content)

        # Generate a unique ID
        detection_id = str(uuid.uuid4())

        # Current timestamp
        timestamp = datetime.now().isoformat()

        # Store in database
        conn = sqlite3.connect('transposition.db')
        cursor = conn.cursor()
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
        INSERT INTO key_detections
        (id, original_content, detected_key, confidence, alternative_keys, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            detection_id,
            request.content,
            detected_key,
            confidence,
            ','.join(alternative_keys),
            timestamp
        ))
        conn.commit()
        conn.close()

        # Return the result
        return KeyDetectionResponse(
            id=detection_id,
            original_content=request.content,
            detected_key=detected_key,
            confidence=confidence,
            alternative_keys=alternative_keys,
            created_at=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Key detection failed: {str(e)}")


@app.post("/song-structure/", response_model=SongStructureResponse)
async def create_song_structure(request: SongStructureRequest):
    try:
        lines = request.content.strip().split('\n')
        structure = []

        # Create section names if not provided
        section_names = request.sections if request.sections else [f"Section {i + 1}" for i in range(len(lines))]

        # Ensure we have enough section names
        while len(section_names) < len(lines):
            section_names.append(f"Section {len(section_names) + 1}")

        # Process each line as a section
        for i, line in enumerate(lines):
            if i >= len(section_names):
                section_name = f"Section {i + 1}"
            else:
                section_name = section_names[i]

            chords = line.strip().split()
            chord_info_list = []

            for position, chord in enumerate(chords):
                chord_info_list.append(ChordInfo(chord=chord, position=position))

            structure.append(SectionInfo(name=section_name, chords=chord_info_list))

        # Generate a unique ID
        structure_id = str(uuid.uuid4())

        # Current timestamp
        timestamp = datetime.now().isoformat()

        # Store in database (serialize the structure to JSON for storage)
        import json
        structure_json = json.dumps(
            [{"name": s.name, "chords": [{"chord": c.chord, "position": c.position} for c in s.chords]} for s in
             structure])

        conn = sqlite3.connect('transposition.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS song_structures (
            id TEXT PRIMARY KEY,
            song_name TEXT NOT NULL,
            artist TEXT,
            structure TEXT NOT NULL,
            created_at TIMESTAMP
        )
        ''')
        cursor.execute('''
        INSERT INTO song_structures
        (id, song_name, artist, structure, created_at)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            structure_id,
            request.song_name,
            request.artist,
            structure_json,
            timestamp
        ))
        conn.commit()
        conn.close()

        # Return the result
        return SongStructureResponse(
            id=structure_id,
            song_name=request.song_name,
            artist=request.artist,
            structure=structure,
            created_at=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Creating song structure failed: {str(e)}")


@app.post("/extract-chords/", response_model=ChordExtractionResponse)
async def extract_chords_from_text(file: UploadFile = File(...)):
    try:
        # Read the file content
        content = await file.read()
        content_text = content.decode("utf-8")

        # Simple chord pattern - matches common chord patterns
        chord_pattern = r'\b([A-G][#b]?(?:m|maj|min|M|aug|dim|sus|add)?[0-9]?(?:/[A-G][#b]?)?)\b'

        # Find all chords in the text
        chords = re.findall(chord_pattern, content_text)

        # Join chords with spaces and format them in a readable way
        extracted_chords = " ".join(chords)

        # Format into lines of 4-6 chords for readability
        chord_lines = []
        chord_chunks = [chords[i:i + 4] for i in range(0, len(chords), 4)]
        for chunk in chord_chunks:
            chord_lines.append(" ".join(chunk))
        formatted_chords = "\n".join(chord_lines)

        # Generate a unique ID
        extraction_id = str(uuid.uuid4())

        # Current timestamp
        timestamp = datetime.now().isoformat()

        # Store in database (you would need to create a new table for this)
        conn = sqlite3.connect('transposition.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chord_extractions (
            id TEXT PRIMARY KEY,
            original_content TEXT NOT NULL,
            extracted_chords TEXT NOT NULL,
            created_at TIMESTAMP
        )
        ''')
        cursor.execute('''
        INSERT INTO chord_extractions
        (id, original_content, extracted_chords, created_at)
        VALUES (?, ?, ?, ?)
        ''', (
            extraction_id,
            content_text,
            formatted_chords,
            timestamp
        ))
        conn.commit()
        conn.close()

        # Return the result
        return ChordExtractionResponse(
            id=extraction_id,
            original_content=content_text,
            extracted_chords=formatted_chords,
            created_at=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chord extraction failed: {str(e)}")


# Models
class TranspositionRequest(BaseModel):
    content: str
    half_steps: int
    original_key: Optional[str] = None
    target_key: Optional[str] = None


class TranspositionResponse(BaseModel):
    id: str
    original_content: str
    transposed_content: str
    half_steps: int
    original_key: Optional[str] = None
    target_key: Optional[str] = None
    created_at: str


class TranspositionHistory(BaseModel):
    transpositions: List[TranspositionResponse]


# Initialize the database on startup
@app.on_event("startup")
async def startup_event():
    init_db()


# Custom transposition function
def transpose_chord(chord, half_steps):
    # Define the notes in order
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    flat_notes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    # Handle empty input
    if not chord:
        return chord

    # Extract the root note and the rest of the chord
    root = ""
    i = 0
    while i < len(chord) and (chord[i].isalpha() or chord[i] in ['#', 'b']):
        root += chord[i]
        i += 1

    # If we couldn't extract a valid root, return the original chord
    if not root:
        return chord

    # Handle flats and sharps
    if 'b' in root:
        # Use the flat notation
        base_note = root[0]
        if len(root) > 1 and root[1] == 'b':
            note_index = flat_notes.index(base_note)
            note_index = (note_index - 1) % 12
            root_note = flat_notes[note_index]
        else:
            note_index = flat_notes.index(root[0])
            root_note = flat_notes[note_index]

        # For multiple flats (bb)
        if root.count('b') > 1:
            for _ in range(root.count('b')):
                note_index = flat_notes.index(root_note)
                note_index = (note_index - 1) % 12
                root_note = flat_notes[note_index]

    elif '#' in root:
        # Use the sharp notation
        base_note = root[0]
        if len(root) > 1 and root[1] == '#':
            note_index = notes.index(base_note)
            note_index = (note_index + 1) % 12
            root_note = notes[note_index]
        else:
            note_index = notes.index(root[0])
            root_note = notes[note_index]

        # For multiple sharps (##)
        if root.count('#') > 1:
            for _ in range(root.count('#')):
                note_index = notes.index(root_note)
                note_index = (note_index + 1) % 12
                root_note = notes[note_index]

    else:
        # Simple case - just the note name
        try:
            note_index = notes.index(root)
            root_note = root
        except ValueError:
            # If we can't find the note, return the original
            return chord

    # Calculate the new note index
    if 'b' in root:
        note_index = flat_notes.index(root_note)
        new_index = (note_index + half_steps) % 12
        new_root = flat_notes[new_index]
    else:
        note_index = notes.index(root_note)
        new_index = (note_index + half_steps) % 12
        new_root = notes[new_index]

    # Preserve the rest of the chord (e.g., "m", "7", "maj7")
    chord_suffix = chord[len(root):]

    # Return the transposed chord
    return new_root + chord_suffix


def transpose_song(content, half_steps):
    lines = content.split('\n')
    transposed_lines = []

    for line in lines:
        # Split the line into words/chords
        elements = line.split()
        transposed_elements = []

        for element in elements:
            # Try to transpose each element as a chord
            try:
                transposed = transpose_chord(element, half_steps)
                transposed_elements.append(transposed)
            except Exception:
                # If it fails, keep the original element
                transposed_elements.append(element)

        # Join the elements back into a line
        transposed_lines.append(' '.join(transposed_elements))

    # Join the lines back into the full content
    return '\n'.join(transposed_lines)


# Endpoints
@app.post("/transpose/", response_model=TranspositionResponse)
async def transpose_music(request: TranspositionRequest):
    try:
        # Use our custom transposition function
        transposed_content = transpose_song(request.content, request.half_steps)

        # Generate a unique ID
        transposition_id = str(uuid.uuid4())

        # Current timestamp
        timestamp = datetime.now().isoformat()

        # Store in database
        conn = sqlite3.connect('transposition.db')
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO transpositions 
        (id, original_content, transposed_content, half_steps, original_key, target_key, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            transposition_id,
            request.content,
            transposed_content,
            request.half_steps,
            request.original_key,
            request.target_key,
            timestamp
        ))
        conn.commit()
        conn.close()

        # Return the result
        return TranspositionResponse(
            id=transposition_id,
            original_content=request.content,
            transposed_content=transposed_content,
            half_steps=request.half_steps,
            original_key=request.original_key,
            target_key=request.target_key,
            created_at=timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transposition failed: {str(e)}")


@app.get("/history/", response_model=TranspositionHistory)
async def get_transposition_history():
    try:
        conn = sqlite3.connect('transposition.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM transpositions ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()

        transpositions = []
        for row in rows:
            transpositions.append(TranspositionResponse(
                id=row[0],
                original_content=row[1],
                transposed_content=row[2],
                half_steps=row[3],
                original_key=row[4],
                target_key=row[5],
                created_at=row[6]
            ))

        return TranspositionHistory(transpositions=transpositions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.get("/transposition/{transposition_id}", response_model=TranspositionResponse)
async def get_transposition(transposition_id: str):
    try:
        conn = sqlite3.connect('transposition.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM transpositions WHERE id = ?', (transposition_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Transposition not found")

        return TranspositionResponse(
            id=row[0],
            original_content=row[1],
            transposed_content=row[2],
            half_steps=row[3],
            original_key=row[4],
            target_key=row[5],
            created_at=row[6]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve transposition: {str(e)}")
