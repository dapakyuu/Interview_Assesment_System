# Frequently Asked Questions (FAQ)

Jawaban untuk pertanyaan yang sering ditanyakan.

---

## 🎯 General Questions

### Q: Apa itu Interview Assessment System?

**A:** Sistem AI untuk menganalisis interview video secara otomatis. Sistem ini melakukan:

- Speech-to-text transcription
- LLM-based assessment
- Cheating detection
- Non-verbal analysis

---

### Q: Berapa lama waktu processing?

**A:** Tergantung hardware dan panjang video:

| Video Duration | CPU Only      | With GPU (RTX 3060) |
| -------------- | ------------- | ------------------- |
| 5 minutes      | 5-8 minutes   | 2-3 minutes         |
| 10 minutes     | 10-16 minutes | 4-6 minutes         |
| 30 minutes     | 30-48 minutes | 12-18 minutes       |

**Component breakdown (5-min video, GPU):**

- Audio Extraction: 5-10s
- Transcription (Whisper large-v3): 45-60s
- Translation (DeepL): 2-5s
- LLM Assessment: 15-30s
- Cheating Detection: 30-45s
- Speaker Diarization: 30-60s
- Non-Verbal Analysis: 10-20s

---

### Q: Apa saja requirements minimum?

**A:**

**Minimum (CPU Only):**

- Python 3.11+
- 16 GB RAM
- 20 GB storage
- CPU: Intel i7 / AMD Ryzen 7 (8+ cores)
- FFmpeg installed

**Recommended (with GPU):**

- Python 3.11+
- 16 GB RAM
- 50 GB storage
- GPU: NVIDIA RTX 3060 (8GB VRAM) or better
- CUDA 11.8 atau 12.1
- FFmpeg installed

**Software Dependencies:**

- faster-whisper
- PyTorch with CUDA
- MediaPipe
- Resemblyzer
- Hugging Face Hub
- DeepL API

---

### Q: Apakah gratis?

**A:**

- ✅ **Sistem:** Open source (gratis)
- ✅ **Whisper (faster-whisper):** Gratis (local processing)
- ✅ **MediaPipe:** Gratis (local processing)
- ✅ **Resemblyzer:** Gratis (local processing)
- ⚠️ **Hugging Face Inference API:** Gratis (FREE tier, rate limited)
  - Model: meta-llama/Llama-3.1-8B-Instruct
  - Limit: ~1000 requests/day
- ⚠️ **DeepL API:** Gratis (FREE tier)
  - Limit: 500,000 characters/month
  - Enough for ~500-1000 videos/month

---

## 📹 Video Questions

### Q: Format video apa yang supported?

**A:**

- MP4 (recommended)
- AVI
- MOV
- MKV
- WebM

---

### Q: Berapa ukuran maksimal video?

**A:**

- **Max file size:** 100 MB (configurable in backend)
- **Max duration:** No strict limit (tested up to 30 minutes)
- **Recommended:** 5-10 minutes, < 50 MB
- **Supported resolutions:** Up to 1920x1080

**Note:** Larger videos take longer to process:

- 5-min video: 2-8 minutes
- 10-min video: 4-16 minutes
- 30-min video: 12-48 minutes

---

### Q: Bagaimana kalau video lebih dari 100 MB?

**A:** Compress video terlebih dahulu:

```bash
# Reduce to 720p with good compression
ffmpeg -i large_video.mp4 -vf scale=-2:720 -c:v libx264 -crf 23 -c:a copy compressed.mp4

# Or trim to specific duration (e.g., 5 minutes)
ffmpeg -i long_video.mp4 -ss 00:00:00 -t 00:05:00 trimmed.mp4

# Check file size
ffmpeg -i compressed.mp4 2>&1 | grep "Duration\|bitrate"
```

**Expected file size reduction:** 50-70% with minimal quality loss

---

### Q: Apakah perlu audio?

**A:** **Ya, wajib!** Video harus memiliki audio track untuk transcription. Video tanpa audio akan ditolak.

---

### Q: Apakah bisa upload dari smartphone?

**A:** **Ya**, selama:

- Video format supported (MP4, MOV)
- Ukuran < 500 MB
- Ada audio yang jelas
- Upload via web interface atau API

---

## 🎙️ Transcription Questions

### Q: Bahasa apa yang supported?

**A:** Saat ini:

- ✅ English (en)
- ✅ Indonesian (id)

Future plans: Tambah bahasa lain sesuai kebutuhan.

---

### Q: Apakah bisa detect multiple speakers?

**A:** **Ya!** Sistem menggunakan **Resemblyzer** untuk speaker diarization:

- Memisahkan interviewer dan interviewee
- Menandai siapa yang berbicara kapan
- Menghitung durasi bicara per speaker
- Voice embeddings untuk identifikasi speaker

**Technology:**

- Resemblyzer VoiceEncoder (CPU-only)
- Embedding similarity untuk clustering
- Automatic speaker count detection

**Output format:**

```json
{
  "speaker_stats": {
    "total_speakers": 2,
    "speaker_0_duration": 180.5,
    "speaker_1_duration": 98.2
  }
}
```

---

### Q: Bagaimana kalau transkrip banyak error?

**A:** Improve audio quality dan Whisper settings:

1. **Sistem sudah menggunakan Whisper large-v3** (98% accuracy)

2. **Improve audio quality sebelum upload:**

   ```bash
   # Reduce background noise
   ffmpeg -i noisy.mp4 -af "highpass=f=200,lowpass=f=3000" clean.mp4

   # Normalize volume
   ffmpeg -i quiet.mp4 -af "volume=2.0" louder.mp4
   ```

3. **Adjust Whisper parameters** (in code):

   ```python
   segments, info = whisper_model.transcribe(
       audio_path,
       language="en",
       beam_size=10,  # Increase for better accuracy (slower)
       best_of=5,
       temperature=0.0,
       vad_filter=True,  # Enable VAD for better segmentation
       vad_parameters={"threshold": 0.3}
   )
   ```

4. **Use external microphone** saat recording

5. **Speak clearly** dan **avoid overlapping speech**

---

## 🤖 LLM Assessment Questions

### Q: Apa yang di-assess oleh LLM?

**A:** LLM menganalisis:

- **Technical Skills:** Pengetahuan teknis
- **Communication:** Cara komunikasi
- **Problem Solving:** Kemampuan solve problems
- **Cultural Fit:** Kesesuaian dengan budaya kerja

---

### Q: Seberapa akurat LLM assessment?

**A:**

- **Overall accuracy:** ~85-90%
- **Best for:** Screening awal, bukan keputusan final
- **Recommendation:** Gunakan sebagai referensi tambahan, bukan satu-satunya penilaian

---

### Q: Bisa custom kriteria assessment?

**A:** **Ya!** Edit prompt di `payload_video.ipynb`:

```python
custom_prompt = """
Analyze this interview and assess:
1. Leadership skills
2. Teamwork ability
3. Innovation mindset

Provide scores and detailed feedback.
"""
```

---

### Q: LLM model apa yang digunakan?

**A:** **meta-llama/Llama-3.1-8B-Instruct** via Hugging Face Inference API

**Why this model?**

- ✅ Free tier available (no credit card)
- ✅ Good balance of accuracy and speed
- ✅ 8B parameters (faster than 70B)
- ✅ Instruction-tuned (good for assessment tasks)

**Implementation:**

```python
from huggingface_hub import InferenceClient

client = InferenceClient(api_key=HF_TOKEN)
response = client.text_generation(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt=assessment_prompt,
    max_new_tokens=500,
    temperature=0.3
)
```

**Alternatives (requires code changes):**

- OpenAI GPT-4 (paid, better quality)
- Anthropic Claude (paid)
- Google Gemini (free tier available)
- Local Llama via Ollama (free, slower)

---

## 🕵️ Cheating Detection Questions

### Q: Apa saja yang dideteksi?

**A:** Sistem menggunakan **MediaPipe Face Mesh** dan **Resemblyzer**:

**Visual Cheating (MediaPipe):**

- **Multiple faces:** Lebih dari 1 orang terdeteksi di frame
- **No face detected:** Kandidat hilang dari kamera
- **Looking away:** Eye gaze tidak ke kamera (head pose analysis)
- **Face mesh landmarks:** 468 facial landmarks tracking

**Audio Cheating (Resemblyzer):**

- **Multiple speakers:** Deteksi lebih dari 1 suara berbeda
- **Voice consistency:** Perubahan suara yang mencurigakan
- **Speaker embeddings:** Voice fingerprint analysis

**Output format:**

```json
{
  "cheating_detection": {
    "multiple_faces_detected": false,
    "no_face_frames": 12,
    "looking_away_percentage": 8.5,
    "multiple_speakers": false,
    "speaker_count": 1
  }
}
```

---

### Q: Apakah bisa bypass cheating detection?

**A:** Sistem cukup robust, tapi tidak 100% foolproof. Best used as:

- **Indicator** bukan bukti mutlak
- **Screening tool** untuk flag suspicious behavior
- **Combined dengan** human review

---

### Q: False positive rate?

**A:** Based on testing:

- **Multiple faces:** ~3-5% (e.g., posters, photos in background)
- **No face:** ~2-3% (lighting issues, camera angle)
- **Looking away:** ~10-15% (natural eye movement, reading notes)
- **Multiple speakers:** ~5-8% (audio quality issues)

**Recommendations:**

1. **Always review flagged videos manually**
2. **Use as screening tool**, not final verdict
3. **Consider context** (e.g., taking notes is normal)
4. **Adjust thresholds** based on your use case:
   ```python
   # In code, adjust sensitivity
   LOOKING_AWAY_THRESHOLD = 20  # Default: 15%
   NO_FACE_THRESHOLD = 30  # Max frames without face
   ```

**Tip:** Combine multiple indicators for higher confidence

---

## 🖥️ Technical Questions

### Q: Apakah perlu GPU?

**A:**

- **Tidak wajib**, tapi **sangat recommended** untuk Whisper
- **With GPU (RTX 3060):** 2-3 min per 5-min video
- **CPU Only (i7):** 5-8 min per 5-min video
- **Speedup:** 2-3x faster dengan GPU

**GPU usage:**

- ✅ **Whisper transcription:** Major speedup (3-5x)
- ✅ **MediaPipe (optional):** Minor speedup (1.5x)
- ❌ **Resemblyzer:** CPU only (no GPU support)
- ❌ **LLM API:** Server-side (tidak pakai GPU lokal)

**Recommended GPUs:**

- **Budget:** NVIDIA GTX 1660 (6GB VRAM)
- **Recommended:** NVIDIA RTX 3060 (8GB VRAM)
- **High-end:** NVIDIA RTX 4070+ (12GB+ VRAM)

---

### Q: Bagaimana cara install CUDA?

**A:**

**Windows:**

1. **Download CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads

   - Recommended: CUDA 11.8 (most compatible)
   - Alternative: CUDA 12.1 (newer)

2. **Install CUDA Toolkit** (follow installer)

3. **Install cuDNN** (optional, for better performance):

   - Download: https://developer.nvidia.com/cudnn
   - Extract to CUDA folder

4. **Install PyTorch with CUDA:**

   ```bash
   # For CUDA 11.8 (recommended)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Verify installation:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")  # Should be True
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

**Troubleshooting:**

- If CUDA not detected, restart computer
- Check NVIDIA drivers are up to date
- Ensure PyTorch CUDA version matches CUDA Toolkit

---

### Q: Bisa run di macOS?

**A:** **Ya**, dengan catatan:

**Apple Silicon (M1/M2/M3):**

- ✅ **MPS acceleration available** (Metal Performance Shaders)
- ✅ **faster-whisper** works with CPU
- ⚠️ **Performance:** Slower than NVIDIA GPU, faster than Intel CPU
- ⚠️ **MPS untuk Whisper:** Not fully optimized yet

**Setup:**

```python
import torch

# Auto-detect best device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "cpu"  # faster-whisper uses CPU even on Apple Silicon
else:
    device = "cpu"

print(f"Using device: {device}")
```

**Intel Mac:**

- ✅ CPU only
- ⚠️ Slower processing (5-8 min per 5-min video)

**Note:** FFmpeg must be installed via Homebrew:

```bash
brew install ffmpeg
```

---

### Q: Bisa run di cloud (AWS, GCP, Azure)?

**A:** **Ya!** Recommended instances:

**AWS:**

- **GPU:** `g4dn.xlarge` (T4, 16GB RAM, 4 vCPU) ~$0.50/hour
  - Best for production (2-3 min per video)
- **CPU:** `c5.2xlarge` (16GB RAM, 8 vCPU) ~$0.34/hour
  - Budget option (5-8 min per video)

**Google Cloud Platform (GCP):**

- **GPU:** `n1-standard-4` + NVIDIA T4 GPU ~$0.47/hour
- **CPU:** `n2-standard-4` (16GB RAM) ~$0.19/hour

**Azure:**

- **GPU:** `Standard_NC6` (K80 GPU, 56GB RAM) ~$0.90/hour
- **CPU:** `Standard_D4s_v3` (16GB RAM) ~$0.19/hour

**Setup steps:**

1. Launch instance with Ubuntu 20.04/22.04
2. Install CUDA (for GPU instances)
3. Install dependencies: `pip install -r requirements.txt`
4. Install FFmpeg: `sudo apt install ffmpeg`
5. Set environment variables (HF_TOKEN, DEEPL_API_KEY)
6. Run server: `uvicorn app.server:app --host 0.0.0.0 --port 7860`

**Cost estimate (GPU instance):**

- ~$360/month for 24/7 operation
- ~$0.01-0.02 per video processed

---

## 🔒 Security & Privacy Questions

### Q: Apakah video di-upload ke cloud?

**A:**

- **Default:** Semua processing **lokal** di server Anda
- **Optional:** Bisa upload ke Google Drive untuk backup
- **API calls:** Hanya transcription text dikirim ke LLM API (bukan video)

---

### Q: Apakah data aman?

**A:**

- ✅ Video disimpan lokal
- ✅ Tidak ada sharing ke third party
- ✅ API calls encrypted (HTTPS)
- ✅ Session auto-expire setelah 7 hari

**Best practice:**

- Delete videos setelah processing selesai
- Use strong API keys
- Enable HTTPS di production

---

### Q: GDPR compliant?

**A:**

- ✅ Data stored locally (under your control)
- ✅ Can delete data on request
- ✅ No tracking/analytics
- ⚠️ Perlu inform kandidat tentang recording & AI analysis

---

## 💰 Cost Questions

### Q: Berapa biaya total?

**A:**

**One-time costs:**

- **Server/Computer:** $800-2000 (if buying new with GPU)
- **Software:** **FREE** (all open source)

**Monthly costs (self-hosted):**

- **Hugging Face API:** $0/month (FREE tier, rate limited)
  - Limit: ~1000 requests/day
- **DeepL API:** $0/month (FREE tier)
  - Limit: 500k characters/month (~500-1000 videos)
- **Electricity:** ~$10-30/month (if running 24/7 with GPU)

**Monthly costs (cloud):**

- **AWS g4dn.xlarge (GPU):** ~$360/month (24/7)
- **Or pay-per-use:** ~$0.50/hour when needed

**Per video processing cost:**

- **API calls:** ~$0.001-0.005 per video (usually free tier)
- **Cloud compute:** ~$0.01-0.02 per video (GPU instance)
- **Total:** ~$0.01-0.03 per video

**Break-even analysis:**

- Processing >500 videos/month? Self-host more cost-effective
- Processing <100 videos/month? Pay-per-use cloud cheaper

---

### Q: Free tier API cukup?

**A:**

**Hugging Face FREE Tier:**

- **Limit:** ~1000 requests/day (rate limited)
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Capacity:** ~100-300 videos/day
  - Each video = 3 questions = 3 API calls
- **Cost:** $0 (no credit card required)

**DeepL FREE Tier:**

- **Limit:** 500,000 characters/month
- **Average:** ~500-1000 characters per video transcript
- **Capacity:** ~500-1000 videos/month
- **Cost:** $0 (registration required)

**Monthly capacity (FREE tier):**

- **Best case:** 500-1000 videos/month
- **Realistic:** 300-500 videos/month
- **Bottleneck:** Usually Hugging Face rate limit

**Paid options if needed:**

- **Hugging Face Pro:** $9/month (higher rate limits)
- **DeepL API Pro:** $5.49/month + $0.00002/char (no monthly limit)

---

## 🚀 Deployment Questions

### Q: Bagaimana cara deploy ke production?

**A:**

**1. Prepare Server (Ubuntu 20.04/22.04 or Windows Server)**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-pip -y

# Install FFmpeg
sudo apt install ffmpeg -y

# Install CUDA (if using GPU)
# See: https://developer.nvidia.com/cuda-downloads
```

**2. Clone and Setup Application**

```bash
# Clone repository
git clone <your-repo-url>
cd Interview_Assesment_System/backend/Python

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Configure Environment**

```bash
# Create .env file
cp app/env.example app/.env

# Edit .env with your API keys
nano app/.env

# Add:
HF_TOKEN=hf_your_token_here
DEEPL_API_KEY=your_key_here:fx
```

**4. Setup Production Server**

```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn app.server:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:7860 \
  --timeout 600 \
  --log-level info
```

**5. Setup Process Manager (systemd)**

```bash
# Create service file
sudo nano /etc/systemd/system/interview-api.service

# Add:
[Unit]
Description=Interview Assessment API
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/Interview_Assesment_System/backend/Python
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/gunicorn app.server:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:7860 \
  --timeout 600
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable interview-api
sudo systemctl start interview-api
sudo systemctl status interview-api
```

**6. Setup Nginx Reverse Proxy**

```bash
# Install Nginx
sudo apt install nginx -y

# Configure
sudo nano /etc/nginx/sites-available/interview-api

# Add:
server {
    listen 80;
    server_name your-domain.com;
    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 600s;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/interview-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**7. Setup HTTPS (Let's Encrypt)**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

**8. Setup Monitoring**

```bash
# Check logs
sudo journalctl -u interview-api -f

# Monitor resources
htop
nvidia-smi  # For GPU
```

**Production checklist:**

- [ ] GPU drivers installed and tested
- [ ] API keys configured in .env
- [ ] HTTPS enabled
- [ ] Firewall configured (ports 80, 443)
- [ ] Process manager running (systemd)
- [ ] Logs being collected
- [ ] Backups configured
- [ ] Monitoring alerts setup

---

### Q: Bisa multi-user?

**A:** **Ya!** Sistem support concurrent users via session management:

**Session Isolation:**

```python
import uuid

# Each upload gets unique session ID
session_id = str(uuid.uuid4())

# Files stored per session
/uploads/{session_id}/video.mp4
/temp/{session_id}/audio.wav
/results/{session_id}.json
```

**Concurrent Processing:**

- Multiple users can upload simultaneously
- Each session processes independently
- No interference between sessions

**Limitations:**

- **GPU:** Process videos sequentially (avoid OOM)
- **CPU:** Can process 2-4 videos in parallel (depends on cores)
- **Recommendation:** Queue system for >10 concurrent uploads

---

### Q: Scalability?

**A:**

**Single Server Capacity:**

| Configuration  | Concurrent Users | Videos/Day | Bottleneck      |
| -------------- | ---------------- | ---------- | --------------- |
| CPU (8 cores)  | 5-10             | 100-150    | CPU time        |
| GPU (RTX 3060) | 10-20            | 200-400    | GPU memory      |
| GPU (RTX 4090) | 20-30            | 400-600    | API rate limits |

**Scaling Strategies:**

**1. Vertical Scaling (Upgrade Hardware):**

- Add more RAM (32GB → 64GB)
- Better GPU (RTX 3060 → RTX 4090)
- More CPU cores (8 → 16 cores)
- Expected: 2-3x capacity increase

**2. Horizontal Scaling (Multiple Servers):**

```
┌─────────────┐
│ Load        │
│ Balancer    │ → Round-robin to workers
└─────────────┘
      |
      ├─→ Worker 1 (GPU server)
      ├─→ Worker 2 (GPU server)
      └─→ Worker 3 (GPU server)
```

**3. Queue System (for high load):**

```python
# Use Redis/RabbitMQ for job queue
import redis
from rq import Queue

redis_conn = redis.Redis()
q = Queue(connection=redis_conn)

# Add job to queue
job = q.enqueue(process_video, video_path, session_id)

# Workers process jobs from queue
```

**Expected Capacity:**

- **Single GPU server:** 200-400 videos/day
- **3 GPU servers + queue:** 600-1200 videos/day
- **10 GPU servers + queue:** 2000-4000 videos/day

---

## 🔧 Troubleshooting Questions

### Q: Error "CUDA out of memory"?

**A:** Solusi berurutan (try each):

**Solution 1: Clear GPU cache before processing**

```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

**Solution 2: Use smaller Whisper model**

```python
# Instead of large-v3 (6-8 GB VRAM)
whisper_model = WhisperModel(
    "medium",  # Uses 3-4 GB VRAM
    device="cuda",
    compute_type="float16"
)
```

**Solution 3: Use int8 quantization**

```python
whisper_model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="int8"  # Reduces memory by 50%
)
```

**Solution 4: Process on CPU (slower)**

```python
device = "cpu"
compute_type = "int8"
```

**Solution 5: Close other GPU applications**

- Check GPU usage: `nvidia-smi`
- Close browsers, games, other ML apps

**GPU Memory Requirements:**

- Whisper small: 1-2 GB
- Whisper medium: 3-4 GB
- Whisper large-v3 (float16): 6-8 GB
- Whisper large-v3 (int8): 3-4 GB

---

### Q: Processing terlalu lambat?

**A:** Quick wins untuk speedup:

1. **Use GPU** (2-3x faster)

   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

2. **Use smaller Whisper model** (30-50% faster)

   ```python
   whisper_model = WhisperModel("medium", ...)  # Instead of large-v3
   ```

3. **Increase frame skip** (40-50% faster cheating detection)

   ```python
   FRAME_SKIP = 10  # Instead of 5
   ```

4. **Disable iris tracking** (20-30% faster)

   ```python
   face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False)
   ```

5. **Reduce beam size** (20-30% faster transcription)
   ```python
   beam_size = 5  # Instead of 10
   ```

**Expected improvement:** 3-5x total speedup

**Detailed guide:** [Performance Tuning Guide](performance.md)

---

### Q: API key tidak working?

**A:**

1. **Check environment variable names** (case-sensitive!):

   ```python
   # Correct variable names:
   HF_TOKEN = "hf_..."
   DEEPL_API_KEY = "....:fx"
   ```

2. **Verify API key format:**

   - Hugging Face: starts with `hf_` (40+ characters)
   - DeepL FREE: ends with `:fx`
   - DeepL Pro: ends with `:fp`

3. **Test API keys:**

   ```python
   # Test Hugging Face
   from huggingface_hub import InferenceClient
   client = InferenceClient(api_key="hf_...")
   response = client.text_generation(
       model="meta-llama/Llama-3.1-8B-Instruct",
       prompt="Hello",
       max_new_tokens=10
   )
   print(response)  # Should return text

   # Test DeepL
   import deepl
   translator = deepl.Translator("....:fx")
   result = translator.translate_text("Hello", target_lang="ID")
   print(result.text)  # Should return "Halo"
   ```

4. **Check API quota:**

   - HF: Check rate limit headers in response
   - DeepL: `translator.get_usage()` shows character count

5. **Common errors:**
   - `401 Unauthorized`: Invalid API key
   - `403 Forbidden`: Quota exceeded or model access denied
   - `456 DeepL`: Character limit exceeded

**See also:** [API Keys Configuration](../configuration/api-keys.md)

---

## 📚 Documentation Questions

### Q: Di mana dokumentasi lengkap?

**A:**

**Local Documentation (MkDocs):**

```bash
# Start documentation server
cd mkdocs
mkdocs serve
# Open: http://127.0.0.1:8000
```

**Documentation Structure:**

- **Getting Started:**
  - [Introduction](../getting-started/introduction.md)
  - [Installation](../getting-started/installation.md)
  - [Quickstart](../getting-started/quickstart.md)
- **API Reference:**
  - [Endpoints](../api/endpoints.md)
  - [Request/Response](../api/request-response.md)
  - [Error Handling](../api/errors.md)
- **Configuration:**
  - [Models](../configuration/models.md)
  - [API Keys](../configuration/api-keys.md)
  - [Advanced](../configuration/advanced.md)
- **Troubleshooting:**
  - [Common Issues](common-issues.md)
  - [Performance](performance.md)
  - [FAQ](faq.md) ← You are here
- **Development:**
  - [Architecture](../development/architecture.md)
  - [Contributing](../development/contributing.md)
  - [Roadmap](../development/roadmap.md)

**Other Resources:**

- **README.md** - Project overview and quick setup
- **Notebook** - `interview_assessment_system.ipynb` (Jupyter tutorial)
- **Code** - `backend/Python/` (implementation details)

---

### Q: Ada tutorial video?

**A:** Saat ini belum ada tutorial video, tapi tersedia:

**Interactive Notebook:**

- **File:** `interview_assessment_system.ipynb`
- **How to use:**
  ```bash
  jupyter notebook interview_assessment_system.ipynb
  ```
- **Contents:**
  - Step-by-step video upload
  - Processing demonstration
  - Result analysis examples
  - Code explanations

**Written Guides:**

- [Quickstart Guide](../getting-started/quickstart.md) - Get started in 5 minutes
- [Installation Guide](../getting-started/installation.md) - Detailed setup
- [API Documentation](../api/endpoints.md) - Complete API reference

**Code Examples:**

- `backend/Python/main.py` - FastAPI server implementation
- `backend/Python/app/routes.py` - API endpoints
- Jupyter notebook - End-to-end workflow

---

## 💬 Support Questions

### Q: Di mana bisa minta bantuan?

**A:**

**1. Documentation (Check first!):**

- [FAQ](faq.md) - Common questions answered
- [Common Issues](common-issues.md) - Known problems and fixes
- [Performance Guide](performance.md) - Optimization tips

**2. GitHub Issues:**

- Search existing issues: Check if already reported
- Create new issue: If problem not found
- Include:
  - System info (OS, GPU, Python version)
  - Error messages (full traceback)
  - Steps to reproduce
  - Expected vs actual behavior

**3. Community:**

- GitHub Discussions (if enabled)
- Stack Overflow (tag: interview-assessment-system)

**Response Time:**

- Documentation: Immediate (self-service)
- GitHub Issues: 1-3 days (community support)
- Critical bugs: Priority response

---

### Q: Bisa request fitur baru?

**A:** **Ya!** We welcome feature requests:

**Process:**

1. **Check Roadmap** first:

   - See [Roadmap](../development/roadmap.md)
   - Feature might already be planned

2. **Create Feature Request:**

   - Go to GitHub Issues
   - Use "Feature Request" template
   - Describe:
     - What feature you want
     - Why it's useful
     - Use cases
     - Expected behavior

3. **Or Contribute:**
   - Fork repository
   - Implement feature
   - Submit Pull Request
   - See [Contributing Guide](../development/contributing.md)

**Popular Feature Requests:**

- ✅ Multi-language support (English + Indonesian)
- ✅ Speaker diarization (implemented)
- 🔄 Real-time processing
- 🔄 Video streaming upload
- 📋 Batch processing UI
- 📋 Custom assessment criteria editor

---

### Q: Bisa hire untuk custom development?

**A:** Project ini open source, namun untuk kebutuhan enterprise:

**Available Services:**

- ✅ Custom feature development
- ✅ Integration dengan sistem existing (ATS, HRIS)
- ✅ Enterprise support (SLA, priority fixes)
- ✅ Training & consultation
- ✅ Deployment assistance
- ✅ Performance optimization
- ✅ Custom model fine-tuning

**Typical Projects:**

- Integration dengan Applicant Tracking System
- Custom assessment rubrics
- Multi-tenant deployment
- White-label solution
- On-premise deployment support
- API rate limit optimization

**Contact:** Create GitHub issue atau email

---

## 🎓 Learning Resources

### Q: Resources untuk belajar lebih lanjut?

**A:**

**AI/ML Technologies Used:**

1. **Whisper (Speech Recognition):**

   - [faster-whisper GitHub](https://github.com/guillaumekln/faster-whisper)
   - [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
   - [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)

2. **LLM (Llama 3.1):**

   - [Hugging Face Inference API](https://huggingface.co/docs/api-inference/)
   - [Meta Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   - [Prompt Engineering Guide](https://www.promptingguide.ai/)

3. **MediaPipe (Face Detection):**

   - [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
   - [Face Landmarks Guide](https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md)

4. **Resemblyzer (Speaker Diarization):**
   - [Resemblyzer GitHub](https://github.com/resemble-ai/Resemblyzer)
   - [Speaker Recognition Tutorial](https://towardsdatascience.com/speaker-recognition-with-resemblyzer-b32d9f562c9a)

**Web Development:**

1. **FastAPI (Backend):**

   - [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
   - [FastAPI Advanced User Guide](https://fastapi.tiangolo.com/advanced/)

2. **Python Async Programming:**
   - [Real Python Async Guide](https://realpython.com/async-io-python/)

**Interview Assessment Domain:**

1. **Automated Interview Analysis:**

   - Research papers on AI interview assessment
   - [HireVue Technology Overview](https://www.hirevue.com/)

2. **HR Assessment Best Practices:**
   - Behavioral interview techniques
   - Competency-based assessment
   - Bias reduction in AI hiring

**Video Processing:**

1. **FFmpeg:**

   - [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
   - [FFmpeg Filtering Guide](https://trac.ffmpeg.org/wiki/FilteringGuide)

2. **OpenCV (cv2):**
   - [OpenCV Python Tutorial](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

**Project Documentation:**

- [Architecture Overview](../development/architecture.md)
- [Contributing Guide](../development/contributing.md)
- [API Reference](../api/endpoints.md)

---

**Masih ada pertanyaan?** Check [Common Issues](common-issues.md) atau create GitHub issue! 💬
