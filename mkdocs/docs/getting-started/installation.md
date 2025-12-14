# Installation Guide

Panduan lengkap instalasi dan setup AI Interview Assessment System.

## Prerequisites

Sebelum memulai, pastikan Anda memiliki:

### Required

- [x] **Python 3.11.9** installed
- [x] **pip** (Python package manager)
- [x] **Git** (untuk cloning repository)
- [x] **10GB+ free disk space** (untuk models & processing)
- [x] **Stable internet connection** (download models ~3GB & dependencies)

### Optional but Recommended

- **NVIDIA GPU** dengan CUDA support (untuk processing 5-10x lebih cepat)
- **Virtual environment** tool (venv/conda)
- **SSD storage** (untuk I/O performance lebih baik)

---

## Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/Interview_Assesment_System.git

# Navigate to backend Python directory
cd Interview_Assesment_System/backend/Python
```

!!! tip "Repository Structure"
    Pastikan Anda berada di folder `backend/Python` sebelum melanjutkan instalasi dependencies.
    
    Jika download sebagai ZIP, extract terlebih dahulu kemudian masuk ke folder tersebut.

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

### Option A: Via requirements.txt (Recommended untuk Production)

```bash
# Install all dependencies at once
pip install -r requirements.txt

# Install resemblyzer manually jika terjadi error (Windows only, untuk menghindari dependency conflicts)
pip install resemblyzer --no-deps
```

!!! warning "Resemblyzer Installation"
    Pada Windows, resemblyzer kadang conflict dengan dependencies lain. Install manual dengan `--no-deps` jika mengalami error.

### Option B: Via Jupyter Notebook (Recommended untuk Development)

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

!!! warning "Compatibility"
    **WAJIB install numpy==1.26.4** untuk menghindari konflik dengan faster-whisper dan torch.
    
!!! info "Resemblyzer on Windows"
    Untuk Windows, install resemblyzer dengan flag `--no-deps` untuk menghindari konflik dependency.

---

## Step 4: Install FFmpeg

FFmpeg **REQUIRED** untuk audio extraction dan processing.

=== "Windows"

    **Download & Install:**

    1. **Download:** [FFmpeg Full Build](https://github.com/GyanD/codexffmpeg/releases/download/2025-11-27-git-61b034a47c/ffmpeg-2025-11-27-git-61b034a47c-full_build.zip)
    2. **Extract** file ZIP
    3. **Copy folder `bin`** dari hasil extract
    4. **Paste** folder `bin` ke `Interview_Assesment_System/backend/`

    Struktur akhir:
    ```
    Interview_Assesment_System/
    └── backend/
        ├── bin/           ← FFmpeg binaries (NEW)
        │   ├── ffmpeg.exe
        │   ├── ffprobe.exe
        │   └── ffplay.exe
        └── Python/
            ├── main.py
            └── requirements.txt
    ```

    **Verify Installation:**

    ```powershell
    # Navigate to backend/bin
    cd ../bin

    # Test all binaries
    .\ffmpeg.exe -version
    .\ffprobe.exe -version
    .\ffplay.exe -version
    ```

    !!! success "No PATH Setup Required"
        Dengan menaruh `bin` di folder `backend/`, tidak perlu setup System PATH!

=== "macOS"

    ```bash
    # Install via Homebrew
    brew install ffmpeg

    # Verify installation
    ffmpeg -version
    ffprobe -version
    ffplay -version
    ```

=== "Linux (Ubuntu/Debian)"

    ```bash
    # Update package list
    sudo apt update

    # Install FFmpeg
    sudo apt install ffmpeg

    # Verify installation
    ffmpeg -version
    ffprobe -version
    ffplay -version
    ```

!!! warning "Critical Component"
    Tanpa FFmpeg, audio extraction akan gagal dan proses analisis tidak bisa berjalan.

---

## Step 5: Configure API Keys

Sistem memerlukan 2 API keys (keduanya FREE):

### DeepL API (Translation Service)

**1. Sign up FREE:**

- Visit: [https://www.deepl.com/pro-api](https://www.deepl.com/pro-api)
- Click "Sign up for free"
- Fill registration form
- Verify email

**2. Get API Key:**

- Login → Account → API Keys
- Copy API key (looks like: `abc123de-f456-78gh-90ij-klmn12345678:fx`)

**3. Configure di project:**

Buat file `.env` di folder `backend/Python/` (copy dari `env.example`):

```bash
# File: backend/Python/.env
DEEPL_API_KEY=your_deepl_api_key_here:fx
HF_TOKEN=your_hf_token_here
```

!!! success "Free Tier Limits"
    - **500,000 characters/month** (gratis selamanya)
    - Cukup untuk ~200 video interview per bulan
    - Tidak perlu credit card

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

Tambahkan ke file `.env` yang sama:

```bash
# File: backend/Python/.env
DEEPL_API_KEY=your_deepl_api_key_here:fx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

!!! success "Free Tier Benefits"
    - Model: `meta-llama/Llama-3.1-8B-Instruct`
    - Unlimited requests (rate-limited ~30 req/min)
    - No credit card required
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

- Local: `http://localhost:8888` 
- Ngrok (optional): `https://xxxx.ngrok.io`

!!! success "Server Ready"
    Jika melihat:
    ```
    INFO:     Uvicorn running on http://0.0.0.0:8888
    INFO:     Application startup complete.
    ```
    Server sudah siap menerima requests!

### Option B: Via Python Script (main.py)

```bash
# Pastikan .env file sudah dikonfigurasi
# DEEPL_API_KEY dan HF_TOKEN harus ada di .env

# Start server directly
python main.py

# Server starts on http://localhost:7860
```

!!! note "Port Differences" 
    - **Jupyter Notebook**: Port `8888`
    - **Python Script (main.py)**: Port `7860`
    
    Pastikan `API_BASE_URL` di frontend sesuai dengan port yang digunakan!

---

## Step 7: Configure Frontend API_BASE_URL

**PENTING:** Sebelum start frontend, ubah `API_BASE_URL` sesuai dengan backend server Anda.

### Update File JavaScript Frontend

Navigate ke folder `frontend/` dan ubah 2 file berikut:

#### 1. Upload.js

Edit baris 1-5 di [frontend/Upload.js](../../../frontend/Upload.js):

=== "Jupyter Notebook (Port 8888)"

    ```javascript
    const API_BASE_URL = "http://localhost:8888";
    ```

=== "Python Script (Port 7860)"

    ```javascript
    const API_BASE_URL = "http://localhost:7860";
    ```

=== "Production/Vercel"

    ```javascript
    const API_BASE_URL = "https://your-backend-url.vercel.app";
    ```

#### 2. Halaman_dasboard.js

Edit baris 1-5 di [frontend/Halaman_dasboard.js](../../../frontend/Halaman_dasboard.js):

=== "Jupyter Notebook (Port 8888)"

    ```javascript
    const API_BASE_URL = "http://localhost:8888";
    ```

=== "Python Script (Port 7860)"

    ```javascript
    const API_BASE_URL = "http://localhost:7860";
    ```

=== "Production/Vercel"

    ```javascript
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

Setelah instalasi, lakukan verification berikut untuk memastikan semua komponen berfungsi:

### 1. Test Backend API

Buka browser dan visit `http://localhost:8888/docs` untuk melihat **FastAPI interactive documentation**.

Atau test dengan curl di terminal:

```bash
# Test health check
curl http://localhost:8888/

# Expected response:
# {"status":"ok","message":"FastAPI server is running"}
```

!!! success "API Ready"
    Jika melihat Swagger UI di `/docs`, backend API sudah siap digunakan!

### 2. Test Frontend

1. Open `http://localhost:5500/Upload.html` di browser
2. Input candidate name: `"Test User"`
3. Upload test video atau gunakan Google Drive URL
4. Monitor processing di Jupyter terminal atau console
5. Setelah selesai, check dashboard results di [Halaman_dasboard.html](http://localhost:5500/Halaman_dasboard.html)

### 3. Verify Models Loaded

Di Jupyter Notebook, check output dari Cell 4-7. Anda harus melihat:

```
✅ Whisper Model loaded: large-v3 (cuda/cpu)
✅ Voice Encoder loaded: Resemblyzer (cpu)
✅ DeepL Translator ready
✅ Hugging Face Client ready (Llama-3.1-8B-Instruct)
```

!!! tip "GPU Acceleration"
    Jika ada NVIDIA GPU, pastikan melihat `cuda` di output Cell 4, bukan `cpu`.

---

## GPU Setup (Optional)

Untuk processing **5-10x lebih cepat** dengan NVIDIA GPU, ikuti langkah berikut:

### Check GPU Availability

Jalankan di Python/Jupyter untuk cek GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
```

### Install CUDA Toolkit (Jika belum ada)

=== "Windows"

    1. Download [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
    2. Install dengan wizard (pilih Express installation)
    3. Restart computer
    4. Verify installation:
    
    ```powershell
    nvcc --version
    ```

=== "Linux (Ubuntu/Debian)"

    ```bash
    # Download CUDA installer
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    
    # Install CUDA
    sudo sh cuda_11.8.0_520.61.05_linux.run
    
    # Add to PATH
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
    source ~/.bashrc
    
    # Verify
    nvcc --version
    ```

### Install PyTorch with CUDA

```bash
# Uninstall CPU version first
pip uninstall torch torchaudio

# Install GPU-enabled version (CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

!!! success "Verification"
    Setelah install, re-run Cell 4 di notebook. Anda harus melihat:
    ```
    ✅ Whisper Model loaded: large-v3 (cuda)  # ← Bukan 'cpu'
    ```

### Performance Comparison

| Hardware           | Processing Speed | Memory       |
| ------------------ | ---------------- | ------------ |
| **CPU** (Intel i7) | 3-8 min/video    | 4-8 GB RAM   |
| **GPU** (RTX 3060) | 1-3 min/video    | 6-8 GB VRAM  |
| **GPU** (RTX 4090) | 30-90 sec/video  | 8-10 GB VRAM |

!!! tip "Recommended GPU"
    Minimum **6GB VRAM** (RTX 3060 atau lebih tinggi) untuk Whisper large-v3 dengan batch processing.

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
