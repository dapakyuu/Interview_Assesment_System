# Advanced Configuration

Konfigurasi advanced untuk customization dan optimization.

---

## ⚙️ Server Configuration

### FastAPI Settings (Python Server)

**Backend: `backend/Python/main.py`**

```python
import uvicorn
from app.server import app

# Server configuration
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",      # Listen on all interfaces
        port=7860,           # Port for Python server
        reload=False,        # Auto-reload (set True for development)
        log_level="info"     # info, debug, warning, error
    )
```

**Jupyter Notebook Server:**

```python
# Cell: Start Server (Port 8888 default for Jupyter)
import nest_asyncio
import uvicorn
from app.server import app

nest_asyncio.apply()  # Allow nested event loops

uvicorn.run(
    app,
    host="0.0.0.0",
    port=8888,
    log_level="info"
)
```

### CORS Configuration

**File: `backend/Python/app/server.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI Interview Assessment System",
    description="Automated interview analysis with cheating detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React dev server
        "http://localhost:5500",      # Live Server
        "https://your-frontend.vercel.app"  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

**For development (allow all origins):**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Only for development!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📁 File Storage Configuration

### Directory Structure

```
backend/Python/
├── uploads/              # Temporary video uploads
├── temp/                 # Temporary processing files
├── results/              # JSON results (session_id.json)
└── app/
    ├── config.py         # Configuration
    └── state.py          # Session state management
```

### Storage Configuration

**File: `backend/Python/app/config.py`**

```python
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for directory in [UPLOADS_DIR, TEMP_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File limits
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
MAX_DURATION = 3600  # 1 hour (seconds)

# Allowed video formats
ALLOWED_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.webm',
    '.flv', '.wmv', '.mpeg', '.mpg'
}

# Allowed video codecs
ALLOWED_CODECS = {'h264', 'h265', 'vp8', 'vp9', 'av1'}
```

### Session State Management

**File: `backend/Python/app/state.py`**

```python
from typing import Dict, Any
from datetime import datetime
import json

# In-memory session storage
sessions: Dict[str, Dict[str, Any]] = {}

def create_session(session_id: str, num_questions: int) -> None:
    """Create new session."""
    sessions[session_id] = {
        "session_id": session_id,
        "num_questions": num_questions,
        "status": "processing",
        "progress": 0,
        "current_step": "Initializing",
        "created_at": datetime.now().isoformat(),
        "results": None,
        "error": None
    }

def update_session(session_id: str, **kwargs) -> None:
    """Update session data."""
    if session_id in sessions:
        sessions[session_id].update(kwargs)

def get_session(session_id: str) -> Dict[str, Any]:
    """Get session data."""
    return sessions.get(session_id, {"status": "not_found"})

def save_results(session_id: str, results: Dict[str, Any]) -> None:
    """Save results to JSON file."""
    from app.config import RESULTS_DIR

    results_path = RESULTS_DIR / f"{session_id}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Update session
    update_session(session_id, status="completed", results=results)
```

### Google Drive Integration (Optional)

**Download videos from Google Drive:**

```python
import gdown
import os

def download_from_gdrive(url: str, output_path: str) -> str:
    """Download file from Google Drive URL."""
    try:
        # Extract file ID from URL
        if 'drive.google.com' in url:
            if '/d/' in url:
                file_id = url.split('/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            else:
                raise ValueError("Invalid Google Drive URL")

            # Download using gdown
            download_url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(download_url, output_path, quiet=False)

            return output_path
        else:
            raise ValueError("Not a Google Drive URL")

    except Exception as e:
        raise Exception(f"Google Drive download failed: {str(e)}")

# Usage in routes
from app.utils.gd_json_download import download_video_from_url

video_path = download_video_from_url(
    url="https://drive.google.com/file/d/FILE_ID/view",
    save_path=str(TEMP_DIR / f"{session_id}_video.mp4")
)
```

### Cleanup Strategy

```python
import os
import shutil
from datetime import datetime, timedelta

def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """Delete files older than max_age_hours."""
    now = datetime.now()

    for file in directory.iterdir():
        if file.is_file():
            file_age = now - datetime.fromtimestamp(file.stat().st_mtime)
            if file_age > timedelta(hours=max_age_hours):
                file.unlink()  # Delete file
                print(f"Deleted old file: {file.name}")

# Cleanup temp files after processing
def cleanup_session_files(session_id: str):
    """Delete temporary files for session."""
    from app.config import UPLOADS_DIR, TEMP_DIR

    # Delete uploads
    for file in UPLOADS_DIR.glob(f"{session_id}*"):
        file.unlink()

    # Delete temp files
    for file in TEMP_DIR.glob(f"{session_id}*"):
        file.unlink()

    print(f"✅ Cleaned up files for session {session_id}")
```

---

## 🎯 Processing Pipeline Configuration

### Complete Processing Pipeline

**System processes each video through 7 stages:**

```python
# Processing stages (sequential)
PIPELINE_STAGES = [
    "1. Audio Extraction",      # Extract audio using FFmpeg
    "2. Transcription",         # Whisper large-v3
    "3. Translation",           # DeepL EN↔ID
    "4. LLM Assessment",        # Llama 3.1-8B scoring
    "5. Cheating Detection",    # Visual + Audio analysis
    "6. Non-Verbal Analysis",   # Facial expressions, speech
    "7. Save Results"           # Save JSON + cleanup
]
```

### Audio Extraction Configuration

**Using FFmpeg to extract audio from video:**

```python
import subprocess
import os

def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video using FFmpeg."""
    # FFmpeg command
    command = [
        'ffmpeg',
        '-i', video_path,           # Input video
        '-vn',                      # No video
        '-acodec', 'pcm_s16le',     # Audio codec (WAV)
        '-ar', '16000',             # Sample rate (16kHz for Whisper)
        '-ac', '1',                 # Mono (1 channel)
        '-y',                       # Overwrite output
        output_path
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)

        # Verify file exists
        if not os.path.exists(output_path):
            raise FileNotFoundError("Audio extraction failed")

        return output_path

    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error: {e.stderr.decode()}")
    except FileNotFoundError:
        raise Exception("FFmpeg not found. Install: winget install FFmpeg")
```

**Audio settings:**

- **Format**: WAV (PCM 16-bit)
- **Sample Rate**: 16kHz (required for Whisper)
- **Channels**: Mono (1 channel)

### Transcription Configuration

```python
from faster_whisper import WhisperModel

# Whisper configuration
WHISPER_CONFIG = {
    "model_size": "large-v3",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "compute_type": "float16" if torch.cuda.is_available() else "int8",
    "beam_size": 10,            # Accuracy (5-10 recommended)
    "language": "en",           # Source language
    "vad_filter": True,         # Voice Activity Detection
    "vad_parameters": {
        "threshold": 0.3,
        "min_speech_duration_ms": 200,
        "min_silence_duration_ms": 1500
    }
}

# Initialize model (once at startup)
whisper_model = WhisperModel(
    WHISPER_CONFIG["model_size"],
    device=WHISPER_CONFIG["device"],
    compute_type=WHISPER_CONFIG["compute_type"]
)
```

### Cheating Detection Configuration

```python
# MediaPipe settings
CHEATING_CONFIG = {
    # Frame processing
    "frame_skip": 5,                     # Process every 5th frame
    "max_frames": 300,                   # Max frames to analyze
    "calibration_frames": 60,            # Frames for baseline

    # Face detection
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.6,
    "max_faces_allowed": 1,              # Detect multiple faces

    # Eye gaze thresholds
    "eye_ratio_right_limit": 0.6,        # Looking right
    "eye_ratio_left_limit": 1.6,         # Looking left

    # Head pose thresholds
    "head_turn_left_limit": 0.35,        # Head turned left
    "head_turn_right_limit": 0.65,       # Head turned right

    # Speaker diarization
    "max_speakers": 3,
    "segment_duration": 0.5,             # Seconds per segment
    "silhouette_threshold": 0.2          # Clustering quality
}
```

### Non-Verbal Analysis Configuration

```python
NON_VERBAL_CONFIG = {
    # Speech analysis
    "min_wpm_threshold": 60,             # Words per minute (too slow)
    "max_wpm_threshold": 200,            # Words per minute (too fast)
    "ideal_wpm": 130,                    # Optimal speaking rate

    # Facial expression
    "calibration_frames": 60,            # Establish baseline
    "smile_threshold": 0.02,             # Mouth width ratio
    "eyebrow_movement_threshold": 0.01,  # Movement detection

    # Eye contact
    "blink_rate_min": 10,                # Blinks per minute (min)
    "blink_rate_max": 30,                # Blinks per minute (max)
    "eye_closed_threshold": 0.02,        # EAR threshold

    # Scoring weights
    "weights": {
        "speech_pace": 0.4,
        "facial_expression": 0.3,
        "eye_movement": 0.3
    }
}
```

---

## 🚀 Performance Optimization

### GPU Configuration

```python
import torch

# Auto-detect best device
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Using device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

    # Memory management
    torch.cuda.empty_cache()
else:
    print("⚠️ No GPU detected, using CPU")
    print("For faster processing, consider using GPU")
```

**GPU Memory Optimization:**

```python
import gc

def cleanup_gpu_memory():
    """Clean up GPU memory after processing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU memory cleaned")

# Call after each video
result = process_video(video_path)
cleanup_gpu_memory()
```

### Processing Speed Optimization

**1. Reduce Whisper beam size:**

```python
# Default: beam_size=10 (best accuracy)
beam_size = 5  # Faster, still good (95% accuracy)
```

**2. Increase frame skip:**

```python
# Default: FRAME_SKIP=5
FRAME_SKIP = 10  # Process every 10th frame (2x faster)
MAX_FRAMES = 200  # Reduce from 300
```

**3. Use smaller Whisper model:**

```python
# large-v3: 98% accuracy, 45-60s
whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type)

# medium: 95% accuracy, 30-40s (← recommended for balance)
whisper_model = WhisperModel("medium", device=device, compute_type=compute_type)

# small: 90% accuracy, 20-30s
whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
```

**4. Disable iris tracking (MediaPipe):**

```python
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=False  # Disable iris (faster, less accurate)
)
```

### Batch Processing (Multiple Videos)

```python
def process_multiple_videos(video_paths: list) -> list:
    """Process multiple videos sequentially."""
    results = []

    for i, video_path in enumerate(video_paths, 1):
        print(f"\nProcessing {i}/{len(video_paths)}: {video_path}")

        try:
            result = process_single_video(video_path)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")
            results.append({"error": str(e)})
        finally:
            # Cleanup after each video
            cleanup_gpu_memory()

    return results
```

**⚠️ Note:** Sequential processing recommended for GPU to avoid OOM errors.

### Model Initialization Optimization

**Load models ONCE at startup (Singleton pattern):**

```python
class ModelManager:
    """Singleton for model management."""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelManager._initialized:
            self.load_models()
            ModelManager._initialized = True

    def load_models(self):
        """Load all models once."""
        from faster_whisper import WhisperModel
        from resemblyzer import VoiceEncoder
        from huggingface_hub import InferenceClient
        import mediapipe as mp
        import deepl

        print("🔄 Loading models...")

        # Whisper
        self.whisper = WhisperModel("large-v3", device=device, compute_type=compute_type)
        print("✅ Whisper loaded")

        # MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        print("✅ MediaPipe loaded")

        # Resemblyzer (CPU only)
        torch.set_num_threads(4)
        self.voice_encoder = VoiceEncoder(device='cpu')
        print("✅ Resemblyzer loaded")

        # HF Client
        self.hf_client = InferenceClient(api_key=os.getenv("HF_TOKEN"))
        print("✅ HF Client initialized")

        # DeepL
        self.translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
        print("✅ DeepL initialized")

        print("🎉 All models loaded!")

# Usage
models = ModelManager()  # Load once

# Use in processing
transcription = models.whisper.transcribe(audio_path)
face_results = models.face_mesh.process(frame)
```

---

## 📊 Logging Configuration

### Basic Logging Setup

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output
    ]
)

logger = logging.getLogger(__name__)
```

### Processing Step Logging

```python
def log_step(session_id: str, step: str, status: str = "started"):
    """Log processing step."""
    timestamp = datetime.now().isoformat()
    logger.info(f"[{session_id}] {step}: {status} at {timestamp}")

# Usage
log_step(session_id, "Audio Extraction", "started")
# ... processing ...
log_step(session_id, "Audio Extraction", "completed")
```

### Performance Logging

```python
import time
from functools import wraps

def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Starting {func.__name__}...")

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"✅ {func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"❌ {func.__name__} failed after {duration:.2f}s: {e}")
            raise

    return wrapper

# Usage
@log_execution_time
def transcribe_video(video_path):
    # ... transcription logic ...
    pass
```

### Session Progress Logging

```python
from app.state import update_session

def update_progress(session_id: str, progress: int, step: str):
    """Update session progress and log."""
    update_session(
        session_id,
        progress=progress,
        current_step=step
    )
    logger.info(f"[{session_id}] Progress: {progress}% - {step}")

# Usage in processing pipeline
update_progress(session_id, 10, "Extracting audio...")
# ...
update_progress(session_id, 30, "Transcribing video...")
# ...
update_progress(session_id, 100, "Processing complete")
```

---

## 🌐 Translation Configuration

### DeepL Settings

```python
TRANSLATION_CONFIG = {
    "provider": "deepl",
    "source_lang": "auto",  # Auto-detect
    "preserve_formatting": True,
    "formality": "default",  # default, more, less
    "cache_translations": True
}
```

### Fallback Translation

```python
# Use Google Translate as fallback
from googletrans import Translator

def translate_with_fallback(text, target_lang):
    try:
        # Try DeepL first
        result = deepl_translator.translate_text(
            text,
            target_lang=target_lang
        )
        return result.text
    except Exception as e:
        # Fallback to Google Translate
        logger.warning(f"DeepL failed, using Google Translate: {e}")
        translator = Translator()
        result = translator.translate(text, dest=target_lang)
        return result.text
```

---

## 🔄 Backup Configuration

### Automatic Backups

```python
BACKUP_CONFIG = {
    "enabled": True,
    "schedule": "daily",  # daily, weekly
    "retention_days": 30,
    "backup_path": "backups/",
    "include": [
        "results/",
        "transcriptions/",
        "logs/"
    ]
}
```

### Backup Script

```python
import shutil
from datetime import datetime

def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{BACKUP_CONFIG['backup_path']}/backup_{timestamp}"

    for folder in BACKUP_CONFIG['include']:
        shutil.copytree(folder, f"{backup_dir}/{folder}")

    logger.info(f"Backup created: {backup_dir}")
```

---

## 🔧 Development vs Production

### Development Configuration

**For local development/testing:**

```python
DEV_CONFIG = {
    # Server
    "host": "127.0.0.1",  # Localhost only
    "port": 8888,          # Jupyter default
    "reload": True,        # Auto-reload on code changes
    "log_level": "debug",  # Verbose logging

    # Processing
    "save_intermediate": True,  # Save temp files
    "cleanup_files": False,     # Keep files for inspection

    # Models
    "whisper_model": "small",   # Faster for testing
    "frame_skip": 10,           # Process fewer frames
    "max_frames": 100,

    # CORS
    "allow_all_origins": True   # Allow any frontend
}
```

### Production Configuration

**For deployment:**

```python
PROD_CONFIG = {
    # Server
    "host": "0.0.0.0",     # All interfaces
    "port": 7860,          # Python server port
    "reload": False,       # Disable auto-reload
    "log_level": "info",   # Less verbose

    # Processing
    "save_intermediate": False,  # Don't save temp files
    "cleanup_files": True,       # Auto-cleanup

    # Models
    "whisper_model": "large-v3",  # Best accuracy
    "frame_skip": 5,              # More frames
    "max_frames": 300,

    # CORS
    "allow_all_origins": False,   # Specific domains only
    "allowed_origins": [
        "https://your-frontend.vercel.app"
    ]
}
```

### Environment Detection

```python
import os

# Set via environment variable
ENV = os.getenv("ENVIRONMENT", "development")

if ENV == "production":
    CONFIG = PROD_CONFIG
    print("🚀 Running in PRODUCTION mode")
else:
    CONFIG = DEV_CONFIG
    print("🛠️ Running in DEVELOPMENT mode")

# Apply config
whisper_model = WhisperModel(
    CONFIG["whisper_model"],
    device=device,
    compute_type=compute_type
)
```

---

## 📦 Results Export

### JSON Export (Default)

**System saves results as JSON automatically:**

```python
import json
from app.config import RESULTS_DIR

def save_results(session_id: str, results: dict) -> str:
    """Save results to JSON file."""
    results_path = RESULTS_DIR / f"{session_id}.json"

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return str(results_path)

# Load results
def load_results(session_id: str) -> dict:
    """Load results from JSON file."""
    results_path = RESULTS_DIR / f"{session_id}.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Results not found for session {session_id}")

    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)
```

### CSV Export (Optional)

**Convert results to CSV for Excel:**

```python
import pandas as pd

def export_to_csv(results: dict, output_path: str):
    """Export results to CSV."""
    # Flatten results per question
    rows = []

    for i, question in enumerate(results.get("per_question_results", []), 1):
        row = {
            "Question_Number": i,
            "Question": question.get("question", ""),
            "Total_Score": question.get("total_score", 0),
            "Quality_Score": question.get("kualitas_jawaban", 0),
            "Coherence_Score": question.get("koherensi", 0),
            "Relevance_Score": question.get("relevansi", 0),
            "Transcription_EN": question.get("transkripsi_en", ""),
            "Transcription_ID": question.get("transkripsi_id", ""),
            "Duration_Seconds": question.get("durasi_detik", 0),
            "Cheating_Verdict": question.get("cheating_detection", {}).get("verdict", "N/A"),
            "Non_Verbal_Score": question.get("non_verbal_analysis", {}).get("total_score", 0)
        }
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ CSV exported to {output_path}")

# Usage
results = load_results(session_id)
export_to_csv(results, f"results/{session_id}.csv")
```

---

## ⚙️ Custom Configuration File

### Create `config.yaml`

```yaml
# AI Interview Assessment System - Configuration

server:
  host: 0.0.0.0
  port: 7860
  reload: false

models:
  whisper:
    model: large-v3
    device: auto # auto, cuda, cpu
    beam_size: 10
    language: en

  llm:
    model: meta-llama/Llama-3.1-8B-Instruct
    max_new_tokens: 500
    temperature: 0.3

processing:
  max_file_size_mb: 500
  max_duration_seconds: 3600

  cheating_detection:
    frame_skip: 5
    max_frames: 300
    min_detection_confidence: 0.6

  non_verbal:
    ideal_wpm: 130
    calibration_frames: 60

storage:
  cleanup_after_processing: true
  keep_results_days: 30

logging:
  level: INFO # DEBUG, INFO, WARNING, ERROR
  file: app.log
```

### Load Configuration

```python
import yaml
from pathlib import Path

def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

# Load config
CONFIG = load_config()

# Access values
WHISPER_MODEL = CONFIG['models']['whisper']['model']
BEAM_SIZE = CONFIG['models']['whisper']['beam_size']
SERVER_PORT = CONFIG['server']['port']

print(f"Loaded config: Whisper={WHISPER_MODEL}, Port={SERVER_PORT}")
```

---

## ✅ Configuration Checklist

Before deploying:

- [ ] API keys configured in `.env` (HF_TOKEN, DEEPL_API_KEY)
- [ ] Server port configured (7860 for Python, 8888 for Jupyter)
- [ ] CORS origins set correctly (production domains)
- [ ] GPU detected and configured (if available)
- [ ] Whisper model downloaded (large-v3 recommended)
- [ ] File storage directories created (uploads, temp, results)
- [ ] Logging configured and tested
- [ ] Cleanup strategy implemented
- [ ] Error handling added for all API calls
- [ ] Performance optimizations applied

---

## 📚 Additional Resources

- [Model Configuration](models.md) - Whisper, LLM, MediaPipe settings
- [API Keys Setup](api-keys.md) - HF Token, DeepL API key
- [API Endpoints](../api/endpoints.md) - Complete API reference
- [Troubleshooting](../troubleshooting/common-issues.md) - Common problems & solutions
- [Performance Guide](../troubleshooting/performance.md) - Optimization tips
