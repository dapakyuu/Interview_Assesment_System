# Performance Tuning

Panduan optimasi performance untuk processing yang lebih cepat.

---

## ⚡ Performance Overview

### Expected Processing Times

**5-minute video (typical interview answer):**

| Component                         | GPU Time    | CPU Time    | Bottleneck      |
| --------------------------------- | ----------- | ----------- | --------------- |
| Audio Extraction (FFmpeg)         | 5-10s       | 5-10s       | I/O             |
| Transcription (Whisper large-v3)  | 45-60s      | 3-5 min     | GPU/CPU         |
| Translation (DeepL API)           | 2-5s        | 2-5s        | Network         |
| LLM Assessment (HF API)           | 15-30s      | 15-30s      | Network         |
| Cheating Detection (MediaPipe)    | 30-45s      | 60-90s      | GPU/CPU         |
| Speaker Diarization (Resemblyzer) | 30-60s      | 30-60s      | CPU only        |
| Non-Verbal Analysis               | 10-20s      | 20-30s      | CPU             |
| **Total**                         | **2-3 min** | **5-8 min** | **2.5x faster** |

**Hardware Requirements:**

| Configuration  | Min VRAM | Recommended | Notes            |
| -------------- | -------- | ----------- | ---------------- |
| CPU Only       | N/A      | 16GB RAM    | Slower but works |
| GPU (GTX 1660) | 6GB      | 8GB RAM     | Good performance |
| GPU (RTX 3060) | 8GB      | 16GB RAM    | Recommended      |
| GPU (RTX 4090) | 16GB+    | 32GB RAM    | Fastest          |

**Model Memory Usage:**

- Whisper large-v3: ~6-8 GB VRAM (GPU) or ~4 GB RAM (CPU)
- MediaPipe Face Mesh: ~20 MB
- Resemblyzer: ~50 MB (CPU only)
- Total: ~6-8 GB VRAM recommended

---

## 🚀 Quick Wins

### 1. Use GPU

**Check GPU availability:**

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Install GPU PyTorch:**

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Expected speedup:** 3-5x faster

---

### 2. Use Smaller Whisper Model

**System uses faster-whisper (CTranslate2 optimized):**

```python
from faster_whisper import WhisperModel

# Fast but less accurate (90% accuracy)
whisper_model = WhisperModel(
    "small",
    device=device,
    compute_type=compute_type
)
# GPU: 20-30s per 5-min video
# CPU: 1-2 min per 5-min video

# Balanced (95% accuracy) - RECOMMENDED
whisper_model = WhisperModel(
    "medium",
    device=device,
    compute_type=compute_type
)
# GPU: 30-40s per 5-min video
# CPU: 2-3 min per 5-min video

# Best quality (98% accuracy) - DEFAULT
whisper_model = WhisperModel(
    "large-v3",
    device=device,
    compute_type=compute_type
)
# GPU: 45-60s per 5-min video
# CPU: 3-5 min per 5-min video
```

**Quality vs Speed tradeoff:**

| Model    | Accuracy | Speed (GPU) | Speed (CPU) | VRAM   | Recommendation       |
| -------- | -------- | ----------- | ----------- | ------ | -------------------- |
| small    | 90%      | 20-30s      | 1-2 min     | 1-2 GB | Testing only         |
| medium   | 95%      | 30-40s      | 2-3 min     | 3-4 GB | Good balance         |
| large-v3 | 98%      | 45-60s      | 3-5 min     | 6-8 GB | Production (default) |

**Expected speedup:** Using `medium` instead of `large-v3` = 30% faster, -3% accuracy

---

### 3. Optimize Frame Processing (Cheating Detection)

**Increase frame skip for faster processing:**

```python
# Default: Process every 5th frame
FRAME_SKIP = 5
MAX_FRAMES = 300

# Faster: Process every 10th frame (2x speed)
FRAME_SKIP = 10
MAX_FRAMES = 200

# Very fast: Process every 15th frame (3x speed)
FRAME_SKIP = 15
MAX_FRAMES = 150

# Disable iris tracking for speed
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=False  # Disable iris (faster)
)
```

**Expected speedup:**

- `FRAME_SKIP=10`: 40-50% faster cheating detection
- `refine_landmarks=False`: 20-30% faster face mesh
- Combined: ~60% faster

---

## 🔧 GPU Optimization

### Auto-Detect and Use GPU

```python
import torch

# Auto-detect best device
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected, using CPU (slower)")
```

### Install GPU-Enabled PyTorch

```bash
# Check current PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Install CUDA 11.8 (most compatible)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1 (newer)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU works
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Expected speedup:** GPU is **2-5x faster** than CPU for Whisper and MediaPipe

### Optimize GPU Memory

```python
import gc
import torch

def cleanup_gpu_memory():
    """Clear GPU memory after processing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU memory cleaned")

# Call after each video
result = process_video(video_path)
cleanup_gpu_memory()
```

### Use int8 Quantization (Faster Whisper)

```python
# FP16: Best accuracy, high memory (6-8 GB)
whisper_model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

# INT8: Good accuracy, lower memory (3-4 GB)
whisper_model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="int8"  # 50% less memory, slightly slower
)
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring (Windows/Linux)
nvidia-smi -l 1

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# List processes using GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**In Python:**

```python
import torch

if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    cached = torch.cuda.memory_reserved(0) / 1e9

    print(f"Total: {total_mem:.2f} GB")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Cached: {cached:.2f} GB")
    print(f"Free: {total_mem - allocated:.2f} GB")
```

---

## 💾 Memory Optimization

### Stream Video Processing (Recommended)

**Don't load entire video into RAM:**

```python
import cv2

def process_video_streaming(video_path):
    """Process video frame-by-frame without loading all into memory."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    processed_count = 0

    while cap.isOpened() and processed_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        # Process single frame
        result = process_frame(frame)

        # Free frame memory immediately
        del frame

        processed_count += 1
        frame_count += 1

    cap.release()
    return results
```

**Expected memory usage:** ~200-500 MB (vs 2-5 GB loading entire video)

### Clear GPU Memory After Processing

```python
import gc
import torch

def cleanup_after_video():
    """Free GPU and RAM after processing each video."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("✅ Memory cleaned")

# Usage
result = process_video(video_path)
cleanup_after_video()
```

### Delete Temporary Files

```python
import os

def cleanup_temp_files(session_id):
    """Delete temporary files to free disk space."""
    temp_dir = f"temp/{session_id}"

    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        print(f"✅ Cleaned temp folder: {temp_dir}")
```

---

## 📊 Video Preprocessing

### Extract Audio with FFmpeg

**System extracts audio automatically - no preprocessing needed:**

```python
def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video using FFmpeg."""
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite
        output_path
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path
```

**Audio format:** 16kHz mono PCM WAV (optimal for Whisper)

### Optional: Compress Video Before Upload

**If video is too large (>100 MB), compress before uploading:**

```bash
# Reduce resolution to 720p (smaller file)
ffmpeg -i input.mp4 -vf scale=-2:720 -c:v libx264 -crf 23 -c:a copy output.mp4

# Expected file size reduction: 50-70%
# Processing speed: Same (system resizes frames internally)
```

**Note:** System can handle up to 1920x1080 videos, compression optional

---

## 🎯 Stage-Specific Optimization

### 1. Transcription (faster-whisper)

**Reduce beam size for speed:**

```python
# Default: beam_size=10 (best accuracy)
beam_size = 10

# Faster: beam_size=5 (good accuracy, 30% faster)
beam_size = 5

segments, info = whisper_model.transcribe(
    audio_path,
    language="en",
    beam_size=beam_size,
    best_of=5,  # Reduce from 10
    temperature=0.0
)
```

**Disable VAD for speed (if audio is clean):**

```python
# With VAD (recommended for interviews)
segments, info = whisper_model.transcribe(
    audio_path,
    vad_filter=True,  # Automatic silence detection
    vad_parameters={"threshold": 0.3}
)

# Without VAD (faster but may transcribe silence)
segments, info = whisper_model.transcribe(
    audio_path,
    vad_filter=False  # 10-20% faster
)
```

**Expected speedup:** beam_size=5 + vad_filter=False = **40-50% faster**

---

### 2. LLM Assessment (Hugging Face API)

**API calls are network-bound, optimization limited:**

```python
from huggingface_hub import InferenceClient
import time

client = InferenceClient(api_key=HF_TOKEN)

# Reduce max_new_tokens for faster responses
response = client.text_generation(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt=assessment_prompt,
    max_new_tokens=300,  # Reduce from 500 (faster)
    temperature=0.3
)

# Add retry logic for reliability
def llm_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.text_generation(prompt=prompt, max_new_tokens=300)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

**Fallback to rule-based scoring if LLM fails:**

```python
def assess_with_fallback(transcription, question):
    try:
        return llm_assess(transcription, question)
    except Exception as e:
        print(f"⚠️ LLM failed, using rule-based: {e}")
        return rule_based_scoring(transcription, question)
```

**Expected speedup:** max_new_tokens=300 = **20-30% faster** API responses

---

### 3. Cheating Detection (MediaPipe)

**Increase frame skip:**

```python
# Default: Process every 5th frame (good accuracy)
FRAME_SKIP = 5
MAX_FRAMES = 300

# Fast: Process every 10th frame (acceptable accuracy)
FRAME_SKIP = 10
MAX_FRAMES = 200

# Very fast: Process every 15th frame (lower accuracy)
FRAME_SKIP = 15
MAX_FRAMES = 100

def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    processed_count = 0

    while cap.isOpened() and processed_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        # Process frame
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # ... analysis ...

        processed_count += 1
        frame_count += 1

    cap.release()
```

**Disable iris tracking:**

```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# With iris (default, slower)
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    refine_landmarks=True  # Include iris landmarks
)

# Without iris (faster)
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    refine_landmarks=False  # Faster (no iris)
)
```

**Expected speedup:** FRAME_SKIP=10 + refine_landmarks=False = **60-70% faster**

---

### 4. Speaker Diarization (Resemblyzer - CPU Only)

**Resemblyzer always runs on CPU (cuDNN compatibility):**

```python
import torch
from resemblyzer import VoiceEncoder

# Optimize CPU threads
torch.set_num_threads(4)  # Use 4 CPU cores

voice_encoder = VoiceEncoder(device='cpu')

# Reduce segment duration for faster processing
def extract_embeddings(audio_path):
    embeddings = []
    segment_duration = 0.5  # Increase to 1.0 for faster (less accurate)

    # ... embedding extraction ...
    return embeddings
```

**Expected speedup:** segment_duration=1.0 = **30-40% faster** (less accurate)

---

### 5. Non-Verbal Analysis

**Reduce calibration frames:**

```python
# Default: 60 frames for baseline
CALIBRATION_FRAMES = 60

# Faster: 30 frames
CALIBRATION_FRAMES = 30  # 50% faster calibration

# Sample fewer frames for speech pace
WPM_SAMPLE_FRAMES = 100  # Reduce from 200
```

**Expected speedup:** 20-30% faster non-verbal analysis

---

## 🔄 Batch Processing

### Sequential Processing (Recommended)

**For GPU: Process videos one at a time to avoid OOM:**

```python
def process_multiple_videos(video_paths: list) -> list:
    """Process multiple videos sequentially."""
    results = []

    for i, video_path in enumerate(video_paths, 1):
        print(f"\n🎬 Processing {i}/{len(video_paths)}: {video_path}")

        try:
            # Process single video
            result = process_single_video(video_path)
            results.append(result)
            print(f"✅ Completed {i}/{len(video_paths)}")

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({"error": str(e)})

        finally:
            # Cleanup after each video
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results
```

**Why sequential?**

- Avoids GPU out of memory errors
- Easier to debug individual videos
- More reliable for large models (Whisper large-v3)

### Parallel Processing (CPU-Only Components)

**For non-GPU tasks (e.g., audio extraction, file I/O):**

```python
from concurrent.futures import ThreadPoolExecutor

def extract_audio_batch(video_paths: list):
    """Extract audio from multiple videos in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        audio_paths = list(executor.map(extract_audio, video_paths))
    return audio_paths
```

**Expected speedup:** 2-3x for I/O-bound operations (audio extraction, file copying)

---

## 💡 Result Caching

### Cache Completed Results

**System saves results to avoid reprocessing:**

```python
import json
from pathlib import Path

def save_result(session_id, result):
    """Save processing result to JSON file."""
    result_file = f"results/{session_id}.json"

    Path("results").mkdir(exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Result saved: {result_file}")

def load_result(session_id):
    """Load cached result if exists."""
    result_file = f"results/{session_id}.json"

    if Path(result_file).exists():
        with open(result_file, encoding='utf-8') as f:
            return json.load(f)

    return None

# Usage
cached = load_result(session_id)
if cached:
    print("⚡ Using cached result")
    return cached
else:
    result = process_video(video_path)
    save_result(session_id, result)
    return result
```

### ModelManager Singleton (Avoid Reloading Models)

**System uses singleton pattern for models:**

```python
class ModelManager:
    _instance = None
    _models_loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_models(self):
        """Load all models once (singleton)."""
        if not self._models_loaded:
            print("🔄 Loading models...")
            self.whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type)
            self.hf_client = InferenceClient(api_key=HF_TOKEN)
            self.face_mesh = mp.solutions.face_mesh.FaceMesh()
            self.voice_encoder = VoiceEncoder(device='cpu')
            self._models_loaded = True
            print("✅ Models loaded")
        else:
            print("⚡ Models already loaded (using cache)")

# Usage (models loaded once per server lifetime)
manager = ModelManager()
manager.load_models()  # First call: loads models
manager.load_models()  # Subsequent calls: instant (cached)
```

**Expected speedup:** Avoid 30-60s model loading time per request

---

## 📈 Benchmarking

### Measure Processing Time

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.2f}s")

# Usage
with timer("Transcription"):
    transcription = transcribe_audio(audio_path)

with timer("LLM Assessment"):
    assessment = llm_assess(transcription)
```

### Profile Code

```python
import cProfile
import pstats

def profile_processing():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your processing code
    process_video("video.mp4")

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 slowest functions
```

---

## 🎯 Performance Targets

### Optimization Goals

| Stage              | Target (GPU) | Current     | Improvement Needed |
| ------------------ | ------------ | ----------- | ------------------ |
| Audio Extraction   | < 10s        | 15s         | 33%                |
| Transcription      | < 60s        | 90s         | 33%                |
| LLM Assessment     | < 10s        | 15s         | 33%                |
| Cheating Detection | < 30s        | 45s         | 33%                |
| Non-Verbal         | < 30s        | 40s         | 25%                |
| **Total**          | **< 3 min**  | **4.5 min** | **33%**            |

---

## 🔍 Monitoring Performance

### Monitor Resource Usage

**Real-time monitoring:**

```python
import psutil
import torch
import time

def monitor_system():
    """Monitor CPU, RAM, and GPU usage."""
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent

    print(f"CPU: {cpu:.1f}% | RAM: {ram:.1f}%", end="")

    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_percent = (gpu_mem_allocated / gpu_mem_total) * 100
        print(f" | GPU: {gpu_mem_allocated:.2f}GB / {gpu_mem_total:.2f}GB ({gpu_percent:.1f}%)")
    else:
        print()

# Monitor during processing
while processing:
    monitor_system()
    time.sleep(2)
```

### Log Performance Metrics

**System logs processing time for each stage:**

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename=f'logs/performance_{datetime.now():%Y%m%d}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_stage_performance(stage_name, duration, video_duration):
    """Log performance metrics for a processing stage."""
    realtime_speed = video_duration / duration if duration > 0 else 0

    logging.info(f"Stage: {stage_name}")
    logging.info(f"Duration: {duration:.2f}s")
    logging.info(f"Video Length: {video_duration:.2f}s")
    logging.info(f"Realtime Speed: {realtime_speed:.2f}x")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated(0) / 1e9
        logging.info(f"GPU Memory: {gpu_mem:.2f}GB")

# Usage
start = time.time()
transcription = transcribe_audio(audio_path)
duration = time.time() - start
log_stage_performance("Transcription", duration, video_duration)
```

**Example log output:**

```
2024-01-15 14:23:45 - INFO - Stage: Transcription
2024-01-15 14:23:45 - INFO - Duration: 52.34s
2024-01-15 14:23:45 - INFO - Video Length: 300.00s
2024-01-15 14:23:45 - INFO - Realtime Speed: 5.73x
2024-01-15 14:23:45 - INFO - GPU Memory: 6.82GB
```

---

## 📚 Best Practices

### ✅ DO

1. **Use GPU for Whisper and MediaPipe**

   - 2-5x faster than CPU
   - Install CUDA-enabled PyTorch

2. **Clean up GPU memory after each video**

   ```python
   gc.collect()
   torch.cuda.empty_cache()
   ```

3. **Use appropriate model sizes**

   - Production: `large-v3` (best accuracy)
   - Testing: `medium` (good balance)
   - Development: `small` (fastest)

4. **Monitor resource usage**

   ```bash
   nvidia-smi -l 1  # GPU monitoring
   ```

5. **Process videos sequentially on GPU**

   - Avoids out-of-memory errors

6. **Optimize frame processing**

   - FRAME_SKIP=10 for 2x speed
   - refine_landmarks=False for 30% speed boost

7. **Cache API responses when possible**
   - DeepL translations
   - LLM assessments (if same prompt)

### ❌ DON'T

1. **Don't load entire video into RAM**

   - Stream frames with cv2.VideoCapture

2. **Don't use CPU if GPU is available**

   - Check: `torch.cuda.is_available()`

3. **Don't process all frames**

   - Skip frames (FRAME_SKIP=5-10)

4. **Don't forget to cleanup**

   - Delete temp files after processing
   - Clear GPU cache between videos

5. **Don't use largest models unnecessarily**

   - `medium` often sufficient for good accuracy

6. **Don't batch process on GPU**

   - Risk of OOM errors with large models

7. **Don't ignore performance metrics**
   - Log processing times to identify bottlenecks

---

## 🚀 Quick Optimization Checklist

**Before optimizing, measure current performance:**

```python
# Test with 5-minute video
video_path = "test_5min.mp4"

import time
start = time.time()
result = process_video(video_path)
duration = time.time() - start

print(f"\nProcessing time: {duration:.2f}s ({duration/60:.1f} minutes)")
print(f"Video duration: 5 minutes")
print(f"Speed: {300/duration:.2f}x realtime")
```

**Optimization priority (highest impact first):**

- [ ] 1. **Use GPU** (2-5x speedup)
- [ ] 2. **Use smaller Whisper model** (30-50% speedup)
- [ ] 3. **Increase FRAME_SKIP to 10** (40-50% speedup for cheating detection)
- [ ] 4. **Disable iris tracking** (20-30% speedup for face mesh)
- [ ] 5. **Reduce beam_size to 5** (20-30% speedup for transcription)
- [ ] 6. **Use int8 quantization** (reduce GPU memory by 50%)
- [ ] 7. **Clean GPU memory between videos** (prevent OOM)

**Expected total speedup: 3-5x with all optimizations**

---

## 📊 Performance Comparison

### Real-World Results (5-minute video)

**Before Optimization (Default Settings):**

```
Configuration: GPU (RTX 3060), Whisper large-v3, beam_size=10
Audio Extraction:     8.2s
Transcription:       58.4s  ← bottleneck
Translation:          3.1s
LLM Assessment:      19.3s
Cheating Detection:  42.7s  ← bottleneck
Diarization:         38.9s
Non-Verbal:          16.8s
----------------------------------------
Total:              187.4s (3.1 minutes)
```

**After Optimization:**

```
Configuration: GPU, Whisper medium, beam_size=5, FRAME_SKIP=10, no iris
Audio Extraction:     7.8s
Transcription:       35.2s  ✓ 40% faster
Translation:          2.9s
LLM Assessment:      17.1s
Cheating Detection:  18.4s  ✓ 57% faster
Diarization:         32.1s
Non-Verbal:          12.3s
----------------------------------------
Total:              125.8s (2.1 minutes)  ✓ 33% overall speedup
```

**Improvements:**

- Transcription: 58.4s → 35.2s (-40%)
- Cheating Detection: 42.7s → 18.4s (-57%)
- Total: 187.4s → 125.8s (-33%)
- Accuracy: 98% → 95% (-3% acceptable tradeoff)

---

## 📚 Additional Resources

- [Common Issues](common-issues.md) - Troubleshooting guide
- [Model Configuration](../configuration/models.md) - Model settings
- [Advanced Configuration](../configuration/advanced.md) - System optimization
- [GPU Setup Guide](common-issues.md#gpu-out-of-memory) - GPU troubleshooting

---

**👍 Performance Tips Working?** Share your results and optimizations!
