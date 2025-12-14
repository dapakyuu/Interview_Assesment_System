# Installation Guide

Panduan lengkap instalasi dan setup AI Interview Assessment System.

## Prerequisites

Sebelum memulai, pastikan Anda memiliki:

- [x] **Python 3.11.9** installed
- [x] **pip** (Python package manager)
- [x] **Git** (untuk cloning repository)
- [x] **10GB+ free disk space** (untuk models)
- [x] **Stable internet connection** (download models & dependencies)

### Optional but Recommended

- **NVIDIA GPU** dengan CUDA support (untuk processing 5-10x lebih cepat)
- **Virtual environment** tool (venv, conda)

---

## Step 1: Clone Repository

```bash
# Clone the repository
git clone <repo-url>

# Navigate to backend Python directory
cd Interview_Assesment_System-main/backend/Python
```

---

## Step 2: Create Virtual Environment

=== "Windows"

    ```powershell
    # Create virtual environment (Python 3.11)
    python -m venv .venv

    # Activate virtual environment
    .venv\Scripts\activate

    # Verify activation (should show (.venv) prefix in terminal)
    ```

=== "macOS/Linux"

    ```bash
    # Create virtual environment
    python3 -m venv .venv

    # Activate virtual environment
    source .venv/bin/activate

    # Verify activation (should show (.venv) prefix)
    ```

!!! tip "Virtual Environment"
Selalu gunakan virtual environment untuk menghindari konflik dependencies dengan project lain.

---

## Step 3: Install Dependencies

### Option A: Via Jupyter Notebook (Recommended)

```bash
# Install Jupyter first
pip install jupyter

# Launch notebook
jupyter notebook interview_assessment_system.ipynb
```

**Kemudian jalankan Cell 1 di notebook** yang berisi:

```python
# Cell 1: Install Safe Dependencies
!pip install --quiet ipywidgets jupyter
!pip install --quiet fastapi uvicorn nest-asyncio pyngrok python-multipart
!pip install --quiet tqdm imageio-ffmpeg deepl
!pip install --quiet silero-vad pydub soundfile scipy scikit-learn
!pip install --quiet huggingface-hub mediapipe torchcodec
!pip install --quiet gdown requests resemblyzer moviepy

# Important: Install specific versions
!pip install --quiet numpy==1.26.4
!pip install --quiet torch torchaudio
!pip install --quiet faster-whisper
```

### Option B: Via requirements.txt

```bash
# Install all dependencies at once
pip install -r requirements.txt
```

!!! warning "Compatibility"
**WAJIB install numpy==1.26.4** untuk menghindari konflik dengan faster-whisper dan torch.

---

## Step 4: Install FFmpeg

FFmpeg **REQUIRED** untuk audio extraction dan processing.

=== "Windows"

    **Download & Install:**

    1. **Download:** [FFmpeg Full Build](https://github.com/GyanD/codexffmpeg/releases/download/2025-11-27-git-61b034a47c/ffmpeg-2025-11-27-git-61b034a47c-full_build.zip)
    2. **Extract** file ZIP
    3. **Copy folder `bin`** dari hasil extract
    4. **Paste** folder `bin` ke `Interview_Assesment_System-main/backend/`

    Struktur akhir:
    ```
    Interview_Assesment_System-main/
    └── backend/
        ├── bin/           ← FFmpeg binaries (NEW)
        │   ├── ffmpeg.exe
        │   ├── ffprobe.exe
        │   └── ffplay.exe
        └── Python/
            └── ...
    ```

    **Verify Installation:**

    ```powershell
    # Navigate to backend/bin
    cd backend/bin

    # Test all binaries
    ffmpeg -version
    ffprobe -version
    ffplay -version
    ```

    !!! success "No PATH Setup Required"
        Dengan menaruh `bin` di folder `backend/`, tidak perlu setup System PATH!

=== "macOS"

    ```bash
    # Install via Homebrew
    brew install ffmpeg

    # Verify installation
    ffmpeg -version
    ```

=== "Linux (Ubuntu/Debian)"

    ```bash
    # Update package list
    sudo apt update

    # Install FFmpeg
    sudo apt install ffmpeg

    # Verify installation
    ffmpeg -version
    ```

---

## Step 5: Configure API Keys

### DeepL API (Translation EN↔ID)

**1. Sign up FREE:**

- Visit: [https://www.deepl.com/pro-api](https://www.deepl.com/pro-api)
- Pilih "Sign up for free"
- Verify email

**2. Get API Key:**

- Login → Account → API Keys
- Copy API key (format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx`)

**3. Configure di project:**

=== "Via .env File (Recommended)"

    ```bash
    # Rename env.example menjadi .env
    cp env.example .env

    # Edit .env file:
    DEEPL_API_KEY=your_api_key_here:fx
    ```

=== "Via Notebook (Alternative)"

    Edit **Cell 6** di ```interview_assessment_system.ipynb```:

    ```python
    # DeepL Configuration
    DEEPL_API_KEY = "YOUR_API_KEY_HERE:fx"
    translator = deepl.Translator(DEEPL_API_KEY)
    ```

**Free Tier Benefits:**

- 500,000 characters/month
- 98%+ translation quality
- EN ↔ ID bidirectional
- Automatic fallback jika quota habis

---

### Hugging Face API (LLM Assessment)

**1. Sign up FREE:**

- Visit: [https://huggingface.co/join](https://huggingface.co/join)
- Sign up dengan email atau GitHub

**2. Generate API Token:**

- Login → Settings → Access Tokens
- Click "New token"
- Token name: `interview-assessment`
- Role: **READ** (sudah cukup untuk inference)
- Click "Generate token"
- Copy token (starts with `hf_`)

**3. Configure di project:**

=== "Via .env File (Recommended)"

    ```bash
    # Edit .env file:
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    ```

=== "Via Notebook (Alternative)"

    Edit **Cell 7** di ```interview_assessment_system.ipynb```:

    ```python
    # Hugging Face Configuration
    HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
    client = InferenceClient(api_key=HF_TOKEN)
    ```

**Free Tier Benefits:**

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Unlimited requests (rate-limited ~30 req/min)
- No credit card required
- Logprobs support untuk confidence scoring
- Automatic fallback ke rule-based jika API gagal

---

## Step 6: Start Backend Server

### Option A: Via Jupyter Notebook (Recommended)

```bash
# Launch Jupyter Notebook
jupyter notebook interview_assessment_system.ipynb
```

**Execute cells in order:**

| Cell | Description                 | Time     | Note                       |
| ---- | --------------------------- | -------- | -------------------------- |
| 1    | Install dependencies        | 5-10 min | One-time only              |
| 2    | Import libraries            | 10 sec   | -                          |
| 3    | Setup directories           | 1 sec    | Creates folders            |
| 4    | Initialize Whisper Model    | 2-5 min  | ~3GB download (first time) |
| 5    | Initialize Voice Encoder    | 30 sec   | Resemblyzer                |
| 6    | Initialize DeepL Translator | 5 sec    | API test                   |
| 7    | Initialize Hugging Face LLM | 10 sec   | API test                   |
| 8-13 | Load Detection Functions    | 5 sec    | -                          |
| 14   | Define FastAPI App          | 1 sec    | API endpoints              |
| 15   | Start Server                | 5 sec    | Port 8888                  |

**Server akan running di:**

- `http://localhost:8888` (local)
- Atau `https://xxxx.ngrok.io` (jika ngrok enabled)

!!! success "Server Ready"
Jika melihat:
`    INFO:     Uvicorn running on http://0.0.0.0:8888
    INFO:     Application startup complete.
   `

    Server sudah siap menerima requests!

### Option B: Via Python Script

```bash
# Start server directly
python main.py

# Server starts on http://localhost:7860
```

!!! note "Port Differences" 
    - Jupyter Notebook: Port **8888** 
    - Python Script: Port **7860**

---

## Step 7: Configure Frontend API_BASE_URL

**PENTING:** Sebelum start frontend, ubah `API_BASE_URL` sesuai dengan backend server Anda.

### Update Upload.js

Navigate ke `frontend/Upload.js` dan ubah `API_BASE_URL`:

=== "Jupyter Notebook (Port 8888)"

    ```javascript
    // Line ~1-5 in Upload.js
    const API_BASE_URL = "http://localhost:8888";
    ```

=== "Python Script (Port 7860)"

    ```javascript
    // Line ~1-5 in Upload.js
    const API_BASE_URL = "http://localhost:7860";
    ```

=== "Production/Vercel"

    ```javascript
    // Line ~1-5 in Upload.js
    const API_BASE_URL = "https://your-backend-url.vercel.app";
    ```

### Update Halaman_dasboard.js

Navigate ke `frontend/Halaman_dasboard.js` dan ubah `BASE_URL`:

=== "Jupyter Notebook (Port 8888)"

    ```javascript
    // Line ~1-5 in Halaman_dasboard.js
    const API_BASE_URL = "http://localhost:8888";
    ```

=== "Python Script (Port 7860)"

    ```javascript
    // Line ~1-5 in Halaman_dasboard.js
    const API_BASE_URL = "http://localhost:7860";
    ```

=== "Production/Vercel"

    ```javascript
    // Line ~1-5 in Halaman_dasboard.js
    const API_BASE_URL = "https://your-backend-url.vercel.app";
    ```

!!! warning "Pastikan API_BASE_URL Match"
    `API_BASE_URL` di kedua file **HARUS sama** dengan port backend server yang Anda gunakan!

---

## Step 8: Start Frontend

=== "VS Code Live Server (Recommended)"

    1. Install extension: **Live Server** by Ritwick Dey
    2. Navigate ke folder ```frontend/```
    3. Right-click ```Upload.html```
    4. Select "Open with Live Server"
    5. Browser akan auto-open: ```http://127.0.0.1:5500/Upload.html```

=== "Python HTTP Server"

    ```bash
    # Navigate to frontend directory
    cd ../../frontend

    # Start simple HTTP server
    python -m http.server 5500
    ```

    Open browser: ```http://localhost:5500/Upload.html```

=== "Direct File (Not Recommended)"

    Bisa buka langsung ```Upload.html``` tapi akan ada **CORS errors**.

---

## Verification

### 1. Test Backend API

Visit `http://localhost:8888/docs` untuk melihat **FastAPI interactive documentation**.

Atau test dengan curl:

```bash
# Test health check
curl http://localhost:8888/

# Expected response:
# {"status":"ok","message":"FastAPI server is running"}
```

### 2. Test Frontend

1. Open `http://localhost:5500/Upload.html`
2. Input candidate name: "Test User"
3. Upload test video atau gunakan Google Drive URL
4. Monitor processing di Jupyter terminal
5. Check dashboard results

### 3. Verify Models Loaded

Di Jupyter Notebook, check output dari Cell 4-7:

```
✅ Whisper Model loaded: large-v3 (cuda/cpu)
✅ Voice Encoder loaded: Resemblyzer (cpu)
✅ DeepL Translator ready
✅ Hugging Face Client ready (Llama-3.1-8B-Instruct)
```

---

## GPU Setup (Optional)

Untuk processing **5-10x lebih cepat** dengan NVIDIA GPU:

### Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Install CUDA Toolkit (Jika belum ada)

**Windows:**

1. Download [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install dengan wizard (pilih Express installation)
3. Restart computer
4. Verify:

```bash
nvcc --version
```

**Linux:**

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### Install PyTorch with CUDA

```bash
# Uninstall CPU version
pip uninstall torch torchaudio

# Install GPU version (CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Performance Comparison

| Hardware           | Processing Speed | Memory       |
| ------------------ | ---------------- | ------------ |
| **CPU** (Intel i7) | 3-8 min/video    | 4-8 GB RAM   |
| **GPU** (RTX 3060) | 1-3 min/video    | 6-8 GB VRAM  |
| **GPU** (RTX 4090) | 30-90 sec/video  | 8-10 GB VRAM |

---

## Troubleshooting Installation

### ❌ Issue: pip install fails

**Symptoms:**

```
ERROR: Could not find a version that satisfies the requirement...
```

**Solutions:**

```bash
# Update pip
python -m pip install --upgrade pip

# Clear cache
pip cache purge

# Install specific version
pip install torch==2.0.1

# Install with no cache
pip install --no-cache-dir package_name
```

---

### ❌ Issue: Virtual environment not working

**Symptoms:**

```
'python' is not recognized as an internal or external command
```

**Solutions:**

```powershell
# Windows: Recreate environment
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv

# Activate with full path
.venv\Scripts\python.exe -m pip install --upgrade pip

# Run commands
.venv\Scripts\python.exe main.py
```

---

### ❌ Issue: FFmpeg not found

**Symptoms:**

```
FileNotFoundError: ffmpeg not found
RuntimeError: No audio backend is available
```

**Solutions:**

**1. Check folder bin exists:**

```powershell
# Navigate to backend
cd Interview_Assesment_System-main/backend

# Check bin folder
dir bin

# Should show:
# ffmpeg.exe, ffprobe.exe, ffplay.exe
```

**2. If folder bin missing:**

- Download [FFmpeg](https://github.com/GyanD/codexffmpeg/releases/download/2025-11-27-git-61b034a47c/ffmpeg-2025-11-27-git-61b034a47c-full_build.zip)
- Extract ZIP
- Copy folder `bin` ke `Interview_Assesment_System-main/backend/`

**3. Verify:**

```powershell
cd backend/bin
ffmpeg -version
ffprobe -version
ffplay -version
```


---

### ❌ Issue: CUDA Out of Memory

**Symptoms:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Force CPU mode (di notebook Cell 4)
device = "cpu"
compute_type = "int8"

whisper_model = WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8"
)
```

---

### ❌ Issue: Port Already in Use

**Symptoms:**

```
ERROR: [Errno 48] Address already in use
```

**Solutions:**

```bash
# Windows: Kill process
netstat -ano | findstr :8888
taskkill /PID <PID> /F

# macOS/Linux: Kill process
lsof -ti:8888 | xargs kill -9

# Or use different port
uvicorn.run(app, host="0.0.0.0", port=8889)
```

---

### ❌ Issue: DeepL API Error

**Symptoms:**

```
AuthorizationException: Invalid API key
```

**Solutions:**

1. Check API key format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx`
2. Pastikan ada `:fx` di akhir
3. Check quota: [DeepL Account](https://www.deepl.com/account/usage)
4. System akan auto-skip translation jika API gagal

---

### ❌ Issue: Hugging Face API Error

**Symptoms:**

```
HfHubHTTPError: 401 Client Error: Unauthorized
```

**Solutions:**

1. Check token format: starts with `hf_`
2. Verify token role: minimum **READ** access
3. Test token:

```python
from huggingface_hub import InferenceClient
client = InferenceClient(api_key="hf_your_token")
```

4. System akan fallback ke rule-based scoring jika LLM API gagal

---

### ❌ Issue: Jupyter Kernel Crash

**Symptoms:**

```
The kernel appears to have died. It will restart automatically.
```

**Solutions:**

```bash
# Reduce model size
whisper_model = WhisperModel("medium")  # Instead of large-v3

# Reduce frame processing
FRAME_SKIP = 10
MAX_FRAMES = 150

# Force garbage collection
import gc
gc.collect()
```

---

## System Requirements Summary

### Minimum Requirements

- **OS:** Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **CPU:** Intel i5 / AMD Ryzen 5 (4 cores)
- **RAM:** 8 GB
- **Storage:** 10 GB free space
- **Python:** 3.11.9

### Recommended Requirements

- **OS:** Windows 11, macOS 13+, Ubuntu 22.04+
- **CPU:** Intel i7 / AMD Ryzen 7 (8 cores)
- **RAM:** 16 GB
- **GPU:** NVIDIA RTX 3060+ (6GB+ VRAM)
- **Storage:** 20 GB SSD
- **Python:** 3.11.9

---

## Next Steps

✅ Installation complete! Sekarang pelajari cara menggunakan sistem:

[:octicons-arrow-right-24: Quick Start Guide](quickstart.md){ .md-button .md-button--primary }
[:octicons-arrow-right-24: Configuration Guide](../configuration/api-keys.md){ .md-button }
