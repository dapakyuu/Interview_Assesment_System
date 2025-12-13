# Model Configuration

Panduan konfigurasi AI models yang digunakan dalam sistem.

---

## 🤖 Overview Model yang Digunakan

| Model                   | Purpose             | Size        | Hardware          | Processing Time     |
| ----------------------- | ------------------- | ----------- | ----------------- | ------------------- |
| faster-whisper large-v3 | Speech-to-Text      | ~3 GB       | GPU (recommended) | 45-90s per video    |
| Llama 3.1-8B-Instruct   | LLM Assessment      | API (Cloud) | Hugging Face API  | 15-30s per question |
| MediaPipe Face Mesh     | Facial Analysis     | ~20 MB      | CPU/GPU           | 30-90s per video    |
| Resemblyzer (GE2E)      | Speaker Diarization | ~50 MB      | CPU only          | 30-60s per video    |
| DeepL API               | Translation (EN↔ID) | API (Cloud) | DeepL API         | 2-5s per text       |

**Key Features:**

- **Whisper large-v3**: 98% transcription accuracy on clear audio
- **Llama 3.1-8B**: 3-dimensional scoring (quality, coherence, relevance) + logprobs confidence
- **MediaPipe**: 468 facial landmarks for expression & eye tracking
- **Resemblyzer**: Voice embeddings untuk multiple speaker detection
- **DeepL**: 98%+ translation quality (500k chars/month free)

---

## 🎙️ Whisper Configuration

### Model Selection

**System menggunakan `faster-whisper` (CTranslate2 optimized) bukan Transformers:**

Edit di `interview_assessment_system.ipynb` - Cell "Initialize Whisper Model":

```python
from faster_whisper import WhisperModel

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# Initialize model
whisper_model = WhisperModel(
    "large-v3",  # Best accuracy (recommended)
    # "medium",  # Balanced
    # "small",   # Fastest
    device=device,
    compute_type=compute_type,
    cpu_threads=4,
    num_workers=1
)

print(f"✅ Whisper Model loaded on {device}")
```

### Model Comparison

| Model    | Size    | Accuracy | Speed (GPU)         | Speed (CPU)         | Memory |
| -------- | ------- | -------- | ------------------- | ------------------- | ------ |
| large-v3 | ~3 GB   | ~98%     | 30-60s / 5min video | 2-4min / 5min video | 6-8 GB |
| medium   | ~1.5 GB | ~95%     | 20-40s / 5min video | 1-3min / 5min video | 3-4 GB |
| small    | ~500 MB | ~90%     | 10-20s / 5min video | 30-60s / 5min video | 1-2 GB |

**Recommendation:** Use `large-v3` untuk production (best accuracy)

### Transcription Parameters

```python
def transcribe_video(video_path, language="en"):
    # Extract audio first (via FFmpeg)
    audio_path = extract_audio(video_path)

    # Transcribe with optimized parameters
    segments, info = whisper_model.transcribe(
        audio_path,
        language=language,           # "en" or "id"
        beam_size=10,                # Higher = more accurate (5-10)
        best_of=10,                  # Sample multiple outputs
        temperature=0.0,             # Deterministic (0.0) vs creative (0.5+)
        vad_filter=True,             # Skip silence automatically
        vad_parameters={
            "threshold": 0.3,        # Voice activity threshold
            "min_speech_duration_ms": 200,
            "min_silence_duration_ms": 1500
        },
        initial_prompt="This is a professional interview in English.",  # Context
        word_timestamps=False,       # Set True for word-level timing
        condition_on_previous_text=True  # Use context from previous segments
    )

    # Extract segments
    transcription_segments = []
    for segment in segments:
        transcription_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "confidence": (
                sum(word.probability for word in segment.words) / len(segment.words)
                if segment.words else 0.0
            )
        })

    return transcription_segments
```

### Language-Specific Settings

```python
# English (default)
language = "en"
initial_prompt = "This is a professional interview in English."

# Indonesian
language = "id"
initial_prompt = "Ini adalah wawancara profesional dalam Bahasa Indonesia."

# Auto-detect (not recommended for consistency)
language = None
initial_prompt = None
```

### Dynamic Beam Size Adjustment

**System automatically adjusts beam_size based on audio duration:**

```python
def get_optimal_beam_size(duration_seconds):
    """Optimize beam_size based on video length."""
    if duration_seconds < 30:
        return 5   # Short video - fast processing
    elif duration_seconds < 120:
        return 7   # Medium video - balanced
    else:
        return 10  # Long video - best accuracy

# Usage
audio_duration = get_audio_duration(audio_path)
beam_size = get_optimal_beam_size(audio_duration)
```

---

## 🧠 LLM Configuration (Llama 3.1-8B-Instruct)

### API Setup (Hugging Face Inference API)

Edit `.env` file:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Get FREE API token: https://huggingface.co/settings/tokens (READ access sudah cukup)

**In notebook:**

```python
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

# Test connection
try:
    response = client.text_generation(
        model="meta-llama/Llama-3.1-8B-Instruct",
        prompt="Test",
        max_new_tokens=10
    )
    print("✅ Hugging Face API connected")
except Exception as e:
    print(f"❌ HF API Error: {e}")
```

### Generation Parameters

```python
# LLM assessment parameters
llm_config = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "max_new_tokens": 500,       # Max response length
    "temperature": 0.3,          # Low = deterministic, high = creative
    "top_p": 0.9,                # Nucleus sampling
    "return_full_text": False,   # Only return generated text
    "details": True              # Include logprobs for confidence
}
```

### Prompt Engineering (3-Dimensional Scoring)

**System uses structured JSON prompt for consistent scoring:**

```python
def generate_llm_assessment_prompt(transcription, question):
    """Generate assessment prompt for LLM."""
    prompt = f"""You are an expert HR interviewer. Analyze this interview answer and provide scores (0-100) for:

1. kualitas_jawaban: Answer quality, depth, examples, completeness
2. koherensi: Structure, clarity, logical flow
3. relevansi: Relevance to the question asked
4. analisis_llm: Brief analysis (2-3 sentences) in Indonesian

Question: {question}
Answer: {transcription}

Respond ONLY with valid JSON in this exact format:
{{
  "kualitas_jawaban": <score 0-100>,
  "koherensi": <score 0-100>,
  "relevansi": <score 0-100>,
  "analisis_llm": "<analysis in Indonesian>"
}}
"""
    return prompt
```

### Logprobs Confidence Extraction

**System extracts token-level confidence from LLM:**

```python
def extract_logprobs_confidence(response_details):
    """Extract confidence score from logprobs."""
    if not hasattr(response_details, 'top_tokens'):
        return 85.0  # Default confidence

    logprobs = []
    for token_info in response_details.top_tokens:
        if hasattr(token_info, 'logprob'):
            logprobs.append(token_info.logprob)

    if not logprobs:
        return 85.0

    # Convert logprobs to confidence using sigmoid
    import numpy as np
    avg_logprob = np.mean(logprobs)
    confidence = 1 / (1 + np.exp(-avg_logprob))
    return confidence * 100
```

### Batch Summary Generation

**For multiple videos, system generates aggregate summary:**

```python
def generate_batch_summary(all_assessments, candidate_name):
    """Generate comprehensive summary for all questions."""

    # Combine all answers and scores
    context = f"""Candidate: {candidate_name}

Analyze performance across {len(all_assessments)} interview questions:

"""

    for i, assessment in enumerate(all_assessments, 1):
        context += f"""Question {i}: {assessment['question']}
Score: {assessment['total_score']}/100
Analysis: {assessment['analisis_llm']}

"""

    prompt = context + """Provide a comprehensive 150-200 word summary in Indonesian covering:
1. Overall performance assessment
2. Key strengths demonstrated
3. Areas needing improvement
4. Specific recommendations

Write in paragraph format, professional tone."""

    response = client.text_generation(
        model="meta-llama/Llama-3.1-8B-Instruct",
        prompt=prompt,
        max_new_tokens=400,
        temperature=0.3
    )

    return response
```

### Fallback to Rule-Based Scoring

**System automatically falls back if LLM unavailable:**

```python
def rule_based_scoring(transcription, question):
    """Fallback scoring when LLM unavailable."""
    word_count = len(transcription.split())

    # Simple heuristics
    kualitas = min(100, word_count * 0.5)  # More words = better
    koherensi = 70  # Default moderate score
    relevansi = 75  # Assume relevant if answering

    return {
        "kualitas_jawaban": kualitas,
        "koherensi": koherensi,
        "relevansi": relevansi,
        "analisis_llm": "Assessment generated using rule-based fallback due to LLM API unavailability."
    }
```

### Alternative LLM Providers (Advanced)

```python
# Option 1: Local Llama via Ollama
import ollama

def llm_ollama(prompt):
    response = ollama.chat(
        model='llama3.1:8b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

# Option 2: OpenAI GPT-4
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

# Option 3: Anthropic Claude
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def llm_claude(prompt):
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Note:** Current system optimized for Hugging Face (FREE tier, no credit card needed)

---

## 👁️ MediaPipe Configuration

### Face Mesh Initialization (468 Landmarks)

**System uses Face Mesh untuk cheating detection dan non-verbal analysis:**

```python
import mediapipe as mp
import cv2

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,         # Video mode
    max_num_faces=2,                 # Detect up to 2 faces (for cheating)
    refine_landmarks=True,           # Include iris landmarks
    min_detection_confidence=0.6,    # Higher = stricter
    min_tracking_confidence=0.6
)

# Initialize Face Detection (for presence check)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,               # 0: short-range (<2m), 1: full-range
    min_detection_confidence=0.6
)

print("✅ MediaPipe initialized")
```

### Cheating Detection - Eye Gaze Tracking

**System tracks iris position untuk detect eyes off-screen:**

```python
# Landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = [33, 133]   # Inner, outer
RIGHT_EYE_CORNERS = [362, 263]

# Thresholds
EYE_RATIO_RIGHT_LIMIT = 0.6    # Looking right
EYE_RATIO_LEFT_LIMIT = 1.6     # Looking left

def calculate_eye_ratio(landmarks, eye_corners, iris_landmarks):
    """Calculate iris position ratio (0=left, 1=center, 2=right)."""
    # Get coordinates
    inner_corner = landmarks[eye_corners[0]]
    outer_corner = landmarks[eye_corners[1]]
    iris_center = landmarks[iris_landmarks[0]]

    # Calculate ratio
    eye_width = abs(outer_corner.x - inner_corner.x)
    iris_offset = iris_center.x - inner_corner.x
    ratio = iris_offset / eye_width if eye_width > 0 else 1.0

    return ratio

# Detect suspicious gaze
def is_eyes_off_screen(left_ratio, right_ratio):
    """Check if eyes looking away from screen."""
    return (
        left_ratio < EYE_RATIO_RIGHT_LIMIT or
        left_ratio > EYE_RATIO_LEFT_LIMIT or
        right_ratio < EYE_RATIO_RIGHT_LIMIT or
        right_ratio > EYE_RATIO_LEFT_LIMIT
    )
```

### Cheating Detection - Head Pose

**Track head orientation via nose position:**

```python
# Thresholds
HEAD_TURN_LEFT_LIMIT = 0.35
HEAD_TURN_RIGHT_LIMIT = 0.65

def calculate_nose_ratio(landmarks, frame_width):
    """Calculate nose position ratio (0=left, 0.5=center, 1=right)."""
    nose_tip = landmarks[1]  # Nose tip landmark
    nose_x = nose_tip.x
    return nose_x

def is_head_turned_away(nose_ratio):
    """Check if head turned significantly."""
    return (
        nose_ratio < HEAD_TURN_LEFT_LIMIT or
        nose_ratio > HEAD_TURN_RIGHT_LIMIT
    )
```

### Non-Verbal Analysis - Facial Expressions

**Track smile intensity and eyebrow movement:**

```python
# Key landmarks
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]

def calculate_smile_intensity(landmarks):
    """Measure smile based on mouth width."""
    mouth_left = landmarks[MOUTH_LEFT]
    mouth_right = landmarks[MOUTH_RIGHT]

    mouth_width = abs(mouth_right.x - mouth_left.x)
    return mouth_width  # Higher = more smile

def calculate_eyebrow_movement(landmarks, baseline_positions):
    """Measure eyebrow movement range (surprise, concern)."""
    current_positions = [
        landmarks[idx].y for idx in LEFT_EYEBROW + RIGHT_EYEBROW
    ]

    if baseline_positions is None:
        return 0.0, current_positions

    # Calculate deviation from baseline
    import numpy as np
    movement = np.std([
        abs(curr - base)
        for curr, base in zip(current_positions, baseline_positions)
    ])

    return movement, current_positions
```

### Non-Verbal Analysis - Eye Movement

**Track blink rate and eye contact percentage:**

```python
# Eye landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Thresholds
EYE_CLOSED_THRESHOLD = 0.02  # Ratio threshold

def calculate_eye_aspect_ratio(landmarks, eye_top, eye_bottom):
    """Calculate EAR (Eye Aspect Ratio) for blink detection."""
    top = landmarks[eye_top]
    bottom = landmarks[eye_bottom]

    eye_height = abs(top.y - bottom.y)
    return eye_height

def is_blinking(left_ear, right_ear):
    """Detect if eyes are closed (blink)."""
    return (
        left_ear < EYE_CLOSED_THRESHOLD and
        right_ear < EYE_CLOSED_THRESHOLD
    )
```

### Frame Processing Optimization

**System processes every Nth frame untuk efficiency:**

```python
# Configuration
FRAME_SKIP = 5       # Process every 5th frame
MAX_FRAMES = 300     # Maximum frames to analyze
CALIBRATION_FRAMES = 60  # Frames for baseline

def process_video_frames(video_path):
    """Process video with frame skipping."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    processed_count = 0
    results = []

    while cap.isOpened() and processed_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            results.append(result)
            processed_count += 1

        frame_count += 1

    cap.release()
    return results
```

### Lowering Thresholds for Difficult Videos

```python
# For low-quality videos or poor lighting
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.4,  # Lower from 0.6
    min_tracking_confidence=0.4,   # Lower from 0.6
    refine_landmarks=False         # Disable iris for speed
)

# Increase frame skip for very long videos
FRAME_SKIP = 10  # Process every 10th frame
MAX_FRAMES = 200  # Reduce max frames
```

---

## 🎤 Resemblyzer Configuration (Speaker Diarization)

### Initialization (CPU-Only)

**System force CPU mode untuk avoid cuDNN version mismatch:**

```python
from resemblyzer import VoiceEncoder
import torch

# Force CPU mode
torch.set_num_threads(4)  # Optimize CPU performance

# Initialize voice encoder
voice_encoder = VoiceEncoder(device='cpu')

print("✅ Resemblyzer Voice Encoder initialized (CPU mode)")
```

**Note:** Resemblyzer runs on CPU only to avoid CUDA/cuDNN compatibility issues. Performance masih cukup fast (~30-60s per video).

### Speaker Embedding Extraction

```python
import numpy as np
from pydub import AudioSegment

def extract_speaker_embeddings(audio_path, segment_duration=0.5):
    """Extract voice embeddings for diarization."""
    # Load audio
    audio = AudioSegment.from_wav(audio_path)

    # Convert to numpy array (16kHz mono required)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / np.max(np.abs(samples))  # Normalize

    # Split into segments
    segment_samples = int(segment_duration * audio.frame_rate)
    embeddings = []

    for i in range(0, len(samples), segment_samples):
        segment = samples[i:i + segment_samples]
        if len(segment) < segment_samples:
            break

        # Extract embedding
        embedding = voice_encoder.embed_utterance(segment)
        embeddings.append(embedding)

    return np.array(embeddings)
```

### Speaker Clustering (Agglomerative)

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def perform_speaker_diarization(embeddings, max_speakers=3):
    """Cluster embeddings to identify speakers."""
    if len(embeddings) < 2:
        return 1, None  # Only 1 speaker

    best_num_speakers = 1
    best_silhouette = -1

    # Try different number of clusters
    for n_clusters in range(2, min(max_speakers + 1, len(embeddings))):
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)

        # Calculate silhouette score (quality metric)
        try:
            silhouette = silhouette_score(embeddings, labels, metric='cosine')
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_num_speakers = n_clusters
        except:
            pass

    return best_num_speakers, best_silhouette
```

### Diarization Configuration

```python
# Configuration
DIARIZATION_CONFIG = {
    "segment_duration": 0.5,        # Seconds per segment
    "max_speakers": 3,              # Max expected speakers
    "silhouette_threshold": 0.2,    # Quality threshold
    "min_segments": 5               # Min segments required
}

def analyze_speaker_diarization(audio_path):
    """Complete diarization analysis."""
    # Extract embeddings
    embeddings = extract_speaker_embeddings(
        audio_path,
        segment_duration=DIARIZATION_CONFIG["segment_duration"]
    )

    if len(embeddings) < DIARIZATION_CONFIG["min_segments"]:
        return {
            "num_speakers_detected": 1,
            "silhouette_score": None,
            "confidence_score": 90.0,
            "verdict": "Safe",
            "details": "Audio too short for reliable diarization"
        }

    # Perform clustering
    num_speakers, silhouette = perform_speaker_diarization(
        embeddings,
        max_speakers=DIARIZATION_CONFIG["max_speakers"]
    )

    # Determine verdict
    if num_speakers == 1:
        verdict = "Safe"
        confidence = 90.0
        details = "Only 1 speaker detected (candidate only)"
    else:
        verdict = "High Risk"
        confidence = 85.0 if silhouette and silhouette > 0.2 else 75.0
        details = f"{num_speakers} speakers detected (potential assistance)"

    return {
        "num_speakers_detected": num_speakers,
        "silhouette_score": silhouette,
        "confidence_score": confidence,
        "verdict": verdict,
        "details": details
    }
```

### Error Handling

**System gracefully handles diarization failures:**

```python
def safe_speaker_diarization(audio_path):
    """Diarization with error handling."""
    try:
        return analyze_speaker_diarization(audio_path)
    except Exception as e:
        print(f"⚠️ Speaker diarization failed: {e}")
        # Return safe default
        return {
            "num_speakers_detected": 1,
            "silhouette_score": None,
            "confidence_score": 90.0,
            "verdict": "Safe",
            "details": "Diarization failed - defaulting to safe"
        }
```

---

## ⚡ Performance Optimization

### GPU vs CPU Processing

**System automatically detects dan memilih device terbaik:**

```python
import torch

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Using device: {device}")

# Verify CUDA (if available)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Performance Comparison (5-minute video):**

| Component        | GPU Time    | CPU Time    | Speedup         |
| ---------------- | ----------- | ----------- | --------------- |
| Whisper large-v3 | 45-60s      | 3-5 min     | 4-5x            |
| MediaPipe        | 30-45s      | 60-90s      | 2x              |
| LLM (HF API)     | 15-30s      | 15-30s      | Same (cloud)    |
| Resemblyzer      | 30-60s      | 30-60s      | Same (CPU only) |
| **Total**        | **2-3 min** | **5-8 min** | **2.5x**        |

### Memory Management

```python
import gc
import torch

def cleanup_memory():
    """Clear GPU/CPU memory after processing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Memory cleaned")

# Usage after each video
result = process_video(video_path)
cleanup_memory()
```

### Batch Processing Optimization

**For multiple videos:**

```python
def process_batch(video_paths):
    """Process multiple videos sequentially (safer for GPU memory)."""
    results = []
    for i, video_path in enumerate(video_paths, 1):
        print(f"Processing {i}/{len(video_paths)}: {video_path}")
        result = process_single_video(video_path)
        results.append(result)
        cleanup_memory()  # Clean after each video
    return results
```

### Model Loading Optimization

```python
# Load models ONCE at startup (not per video)
class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init_models()
        return cls._instance

    def init_models(self):
        """Initialize all models once."""
        from faster_whisper import WhisperModel
        self.whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        self.face_mesh = mp_face_mesh.FaceMesh()
        self.voice_encoder = VoiceEncoder(device='cpu')
        self.hf_client = InferenceClient(api_key=HF_TOKEN)
        print("✅ All models loaded")

# Usage
models = ModelManager()  # Load once
for video in videos:
    result = process_with_models(models, video)
```

## 🔄 Model Management

### Check Model Versions

```python
# Check faster-whisper version
import faster_whisper
print(f"faster-whisper: {faster_whisper.__version__}")

# Check other dependencies
import torch, mediapipe, resemblyzer, deepl
print(f"torch: {torch.__version__}")
print(f"mediapipe: {mediapipe.__version__}")

# HF client version
from huggingface_hub import __version__
print(f"huggingface-hub: {__version__}")
```

### Update Dependencies

```bash
# Update all packages
pip install --upgrade faster-whisper torch mediapipe resemblyzer deepl huggingface-hub

# Specific versions (from requirements.txt)
pip install faster-whisper==1.0.3
pip install torch==2.1.0
pip install mediapipe==0.10.8
```

### Model Cache Management

```python
import os
from pathlib import Path

# Check Whisper model cache
home = Path.home()
whisper_cache = home / ".cache" / "huggingface" / "hub"
if whisper_cache.exists():
    cache_size = sum(f.stat().st_size for f in whisper_cache.rglob('*') if f.is_file())
    print(f"Model cache: {cache_size / 1e9:.2f} GB")

# Pre-download models (for offline use)
from faster_whisper import WhisperModel
print("Downloading Whisper large-v3...")
WhisperModel("large-v3", device="cpu")  # Downloads if not cached
print("✅ Model cached")
```

## 📊 Model Monitoring

### Processing Time Tracking

```python
import time
from datetime import datetime

class PerformanceLogger:
    def __init__(self):
        self.timings = {}

    def start(self, step_name):
        """Start timing a step."""
        self.timings[step_name] = {"start": time.time()}

    def end(self, step_name):
        """End timing and print duration."""
        duration = time.time() - self.timings[step_name]["start"]
        print(f"⏱️ {step_name}: {duration:.2f}s")
        return duration

# Usage
logger = PerformanceLogger()

logger.start("audio_extraction")
# ... extract audio ...
logger.end("audio_extraction")

logger.start("transcription")
# ... transcribe ...
logger.end("transcription")
```

### Quality Metrics Tracking

```python
def track_quality_metrics(result):
    """Track model output quality."""
    metrics = {
        "transcription_confidence": result.get("transcription_confidence", 0.0),
        "llm_confidence": result.get("llm_confidence", 0.0),
        "face_detection_rate": result.get("face_detection_rate", 0.0)
    }

    print("\n📊 Quality Metrics:")
    for metric, value in metrics.items():
        status = "✅" if value > 0.7 else "⚠️"
        print(f"{status} {metric}: {value:.2f}")

    return metrics
```

### Resource Usage Monitoring

```python
import psutil

def log_resource_usage():
    """Monitor CPU, RAM, GPU usage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    print(f"\n🖥️ Resource Usage:")
    print(f"CPU: {cpu_percent}%")
    print(f"RAM: {memory.percent}% ({memory.used / 1e9:.1f} / {memory.total / 1e9:.1f} GB)")

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
```

## 🎯 Recommendations

### Production Configuration (Recommended)

```python
# Best accuracy for production
WHISPER_MODEL = "large-v3"
BEAM_SIZE = 10
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FRAME_SKIP = 5
MAX_FRAMES = 300

# Expected processing time (5-min video):
# GPU: 2-3 minutes
# CPU: 5-8 minutes
```

### Fast Processing Configuration

```python
# Faster processing, good accuracy
WHISPER_MODEL = "medium"
BEAM_SIZE = 5
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FRAME_SKIP = 10
MAX_FRAMES = 200

# Expected processing time (5-min video):
# GPU: 1-2 minutes
# CPU: 3-5 minutes
```

### Development/Testing Configuration

```python
# Quick testing with acceptable accuracy
WHISPER_MODEL = "small"
BEAM_SIZE = 5
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FRAME_SKIP = 15
MAX_FRAMES = 100

# Expected processing time (5-min video):
# GPU: <1 minute
# CPU: 2-3 minutes
```

## ⚙️ Troubleshooting

### Whisper Issues

**Problem: "CUDA out of memory"**

```python
# Solution 1: Use smaller model
whisper_model = WhisperModel("medium", device=device)

# Solution 2: Force CPU
whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")

# Solution 3: Use int8 quantization
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="int8")
```

**Problem: Poor transcription quality**

```python
# Increase beam size
beam_size = 15  # Higher = better (default: 10)

# Lower VAD threshold
vad_parameters = {"threshold": 0.2}  # More sensitive

# Add context
initial_prompt = "Interview about Python programming, algorithms, data structures."
```

### LLM (Hugging Face) Issues

**Problem: "401 Unauthorized"**

```bash
# Verify token
curl -H "Authorization: Bearer YOUR_HF_TOKEN" https://huggingface.co/api/whoami-v2
```

**Problem: "503 Service Unavailable"**

```python
# Wait for model warm-up
import time

for attempt in range(3):
    try:
        response = client.text_generation(prompt=prompt, model="meta-llama/Llama-3.1-8B-Instruct")
        break
    except Exception as e:
        if "loading" in str(e).lower():
            print(f"Waiting {(attempt+1)*10}s for model...")
            time.sleep((attempt + 1) * 10)
```

### MediaPipe Issues

**Problem: "No face detected"**

```python
# Lower confidence threshold
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3,  # Lower from 0.6
    min_tracking_confidence=0.3
)
```

**Problem: Slow processing**

```python
# Increase frame skip
FRAME_SKIP = 10  # Process every 10th frame

# Disable iris
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False)
```

### Resemblyzer Issues

**Problem: "cuDNN error"**

```python
# Force CPU (already implemented)
import torch
torch.set_num_threads(4)
voice_encoder = VoiceEncoder(device='cpu')
```

### DeepL Issues

**Problem: "456 Quota exceeded"**

```bash
# Check quota
curl https://api-free.deepl.com/v2/usage -H "Authorization: DeepL-Auth-Key YOUR_KEY"
```

### General Debugging

```python
# Enable logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test components separately
result = transcribe_video("test.mp4")  # Test Whisper only
result = detect_faces("test.mp4")      # Test MediaPipe only
```

---

## 📚 Additional Resources

- [Faster-Whisper Documentation](https://github.com/guillaumekln/faster-whisper)
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
- [MediaPipe Face Mesh Guide](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [Resemblyzer Repository](https://github.com/resemble-ai/Resemblyzer)
- [DeepL API Documentation](https://www.deepl.com/docs-api)

## ✅ Configuration Checklist

- [ ] Whisper model downloaded and tested (faster-whisper large-v3)
- [ ] Hugging Face API token configured in `.env`
- [ ] DeepL API key configured in `.env`
- [ ] MediaPipe initialized successfully
- [ ] Resemblyzer running in CPU mode
- [ ] Device (GPU/CPU) auto-detection working
- [ ] Performance logging enabled
- [ ] Error handling implemented

## 📖 Next Steps

- [API Keys Configuration](api-keys.md) - Setup DeepL and HF tokens
- [Advanced Configuration](advanced.md) - Fine-tune parameters
- [API Endpoints](../api/endpoints.md) - Test API with configured models
- [Troubleshooting Guide](../troubleshooting/common-issues.md) - Common problems & solutions
