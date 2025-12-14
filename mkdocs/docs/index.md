# 🎙️ AI Interview Assessment System

**Sistem AI untuk otomasi penilaian interview kandidat dengan speech-to-text transcription, cheating detection, dan analisis non-verbal mendalam.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Whisper](https://img.shields.io/badge/Whisper-large--v3-orange.svg)](https://github.com/openai/whisper)
[![Llama 3.1-8B](https://img.shields.io/badge/Llama_3.1--8B-LLM_Inference_API-red.svg)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face_Mesh-00C8FF.svg)](https://google.github.io/mediapipe/)
[![Resemblyzer](https://img.shields.io/badge/Resemblyzer-Speaker_Diarization-9C27B0.svg)](https://github.com/resemble-ai/Resemblyzer)
[![PyDub](https://img.shields.io/badge/PyDub-Audio_Processing-brightgreen.svg)](https://github.com/jiaaro/pydub)
[![DeepL](https://img.shields.io/badge/DeepL-Translation_API-0F2B46.svg)](https://www.deepl.com/docs-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue.svg)](https://www.mkdocs.org/)

---

## ✨ Key Features

<div class="grid cards" markdown>

-   :material-microphone:{ .lg .middle } **98% Accurate Transcription**

    ***

    Faster-Whisper large-v3 dengan confidence dari logprobs untuk hasil transcription berkualitas tinggi pada audio yang jernih.

    [:octicons-arrow-right-24: Learn more](getting-started/introduction.md)

-   :material-robot:{ .lg .middle } **LLM Assessment**

    ***

    Hugging Face Llama 3.1-8B-Instruct untuk semantic answer evaluation dengan confidence scoring dan logprobs analysis.

    [:octicons-arrow-right-24: Learn more](features/overview.md#llm-powered-assessment)

-   :material-shield-check:{ .lg .middle } **Cheating Detection**

    ***

    Multi-modal detection: Visual (MediaPipe face mesh 468 landmarks, eye/head tracking) + Audio (Resemblyzer speaker diarization).

    [:octicons-arrow-right-24: Learn more](features/overview.md#cheating-detection)

-   :material-account-eye:{ .lg .middle } **Non-Verbal Analysis**

    ***

    Scientific scoring untuk facial expressions, eye contact, blink rate, speech patterns dengan weighted confidence calculation.

    [:octicons-arrow-right-24: Learn more](features/overview.md#non-verbal-analysis)

-   :material-translate:{ .lg .middle } **Bilingual Support**

    ***

    English ↔ Indonesian translation via DeepL API dengan 98%+ accuracy (500k chars/month free tier).

    [:octicons-arrow-right-24: Learn more](configuration/api-keys.md#deepl-setup)

-   :material-chart-line:{ .lg .middle } **Dashboard Analytics**

    ***

    Interactive charts (Chart.js), aggregate reporting, comprehensive JSON results, dan PDF export capability.

    [:octicons-arrow-right-24: Learn more](getting-started/quickstart.md#dashboard-display)

-   :material-google-drive:{ .lg .middle } **Google Drive Support**

    ***

    Direct video download dari Google Drive URLs dengan auto file ID extraction menggunakan gdown library.

    [:octicons-arrow-right-24: Learn more](api/endpoints.md#upload-from-google-drive)

-   :material-lightning-bolt:{ .lg .middle } **Fast Processing**

    ***

    Background async processing: 1-3 min/video (GPU) atau 3-8 min/video (CPU) dengan auto-cleanup untuk hemat storage 99%+.

    [:octicons-arrow-right-24: Learn more](troubleshooting/performance.md)

</div>

---

## :material-rocket-launch: Quick Start

=== "Melalui Website Kami"

    **1. Kunjungi website kami di [https://interview-assesment-system.vercel.app/](https://interview-assesment-system.vercel.app/)**

    **2. Siapkan input berupa video atau JSON:**

    **Jika input berupa video:**

    - Masukkan nama kandidat
    - Pilih bahasa yang digunakan di video (Bahasa Inggris atau Bahasa Indonesia)
    - Pilih atau drag and drop file video ke container area
    - Masukkan pertanyaan sesuai dengan video
    - Jika ingin menghapus file video yang telah di-drop, tekan tombol hapus semua untuk menghapus semua video atau tekan tombol hapus pada masing-masing card preview video untuk menghapus secara spesifik

    **Jika input berupa JSON:**

    - Pastikan struktur input json seperti ini:

    ```json
    {
      "success": true,
      "data": {
        "candidate": {
          "name": "xxx"
        },
        "reviewChecklists": {
          "interviews": [
            {
              "positionId": 1,
              "question": "Question for video 1",
              "isVideoExist": true,
              "recordedVideoUrl": "your video 1 URL"
            },
            {
              "positionId": 2,
              "question": "Question for video 2",
              "isVideoExist": true,
              "recordedVideoUrl": "your video 2 URL"
            }
          ]
        }
      }
    }
    ```

    - Pilih bahasa yang digunakan di video (Bahasa Inggris atau Bahasa Indonesia)
    - Pilih atau drag and drop file JSON ke container area
    - Jika ingin menghapus file JSON yang telah dipilih atau di-drop, tekan tombol hapus

    **3. Klik tombol kirim** untuk mengirim file yang akan diproses oleh backend

    **4. Tunggu proses** hingga selesai sekitar 1-3 menit per video (Jika pakai GPU akan lebih cepat)

    **5. Setelah selesai**, halaman akan otomatis pindah ke halaman dashboard yang menampilkan semua hasil analisis mulai dari transkripsi, LLM assessment, cheating detection, dan analisis non verbal yang masing-masing dilengkapi dengan confidence score

    **6. Export hasil** (opsional): Download JSON atau laporan PDF

=== "Instalasi via Jupyter Notebook"

    ```bash
    # 1. Clone & setup
    git clone <repo>
    cd Interview_Assesment_System/backend/Python
    python -m venv .venv && .venv\Scripts\activate

    # 2. Buka jupyter notebook dan masukkan tokenmu di cell yang berisi
    # DEEPL_API_KEY = "YOUR_API_KEY_HERE:fx"
    # HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"

    # 3. Start server via jupyter notebook
    jupyter notebook interview_assessment_system.ipynb
    # jalankan cell sesuai OS yang kamu gunakan 
    # Run all cells → Server starts on http://localhost:8888

    # 4. Ubah API_BASE_URL di Upload.js dan Halaman_dasboard.js
    # API_BASE_URL = http://localhost:8888

    # 5. Open frontend
    # http://localhost:5500/Upload.html (via Live Server)
    ```

=== "Instalasi via Python Script"

    ```bash
    # 1. Clone & setup
    git clone <repo>
    cd Interview_Assesment_System/backend/Python
    python -m venv .venv && .venv\Scripts\activate

    # 2. Install (one command)
    pip install -r requirements.txt  # atau run Cell 1 di notebook

    # 3 install resemblyzer secara manual.
    pip install resemblyzer --no-deps

    # 4. Ganti env.example menjadi .env dan masukkan tokenmu disana.
    # DEEPL_API_KEY = "YOUR_API_KEY_HERE:fx"
    # HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"

    # 5. Pastikan sudah terdapat folder bin pada folder python
    python install_bin.py

    # 6. Start server
    python main.py
    # Server starts on http://localhost:7860

    # 7. Ubah API_BASE_URL di Upload.js dan Halaman_dasboard.js
    # API_BASE_URL = http://localhost:7860

    # 8. Open frontend
    # http://localhost:5500/Upload.html (via Live Server)
    ```

[:octicons-arrow-right-24: Detailed Installation Guide](getting-started/installation.md){ .md-button .md-button--primary }

---

## :material-speedometer: Performance Metrics

| Metric                      | Value                      |
| --------------------------- | -------------------------- |
| Transcription Accuracy      | ~98% (clear audio)         |
| Translation Quality         | ~98% (DeepL API)           |
| LLM Assessment Confidence   | 50-95% (logprobs-based)    |
| Cheating Detection Accuracy | ~92% (visual + audio)      |
| Non-Verbal Confidence       | 50-90% (video quality dep) |
| Processing Speed (GPU)      | 1-3 min/video              |
| Processing Speed (CPU)      | 3-8 min/video              |
| Storage Saved               | 99%+ (auto-cleanup)        |

[:octicons-arrow-right-24: Performance Details](troubleshooting/performance.md)

---

## :material-chart-box: System Architecture

```mermaid
graph TD
    A["Frontend Upload"] --> B{"Backend FastAPI"}
    B --> C["Google Drive Download"]
    B --> D["Whisper Transcription large-v3"]
    D --> E["DeepL Translation EN-ID"]
    E --> F["LLM Assessment Llama 3.1-8B"]
    F --> G["Cheating Detection Visual+Audio"]
    G --> H["Non-Verbal Analysis"]
    H --> I["Aggregate Reporting"]
    I --> J["Results JSON + Cleanup"]
    J --> K["Dashboard Display"]

    C --> D

    style A fill:#e1f5ff
    style B fill:#fff3e0
    style D fill:#e8f5e9
    style F fill:#f3e5f5
    style G fill:#ffe0b2
    style I fill:#e8f5e9
    style K fill:#fce4ec
```

[:octicons-arrow-right-24: Architecture Details](development/architecture.md)

---

## :material-tools: Technology Stack

=== "Backend"

    - **FastAPI** - Modern async web framework dengan CORS support
    - **faster-whisper** (large-v3) - CTranslate2 optimized speech-to-text
    - **Hugging Face Inference API** - LLM assessment (Llama 3.1-8B-Instruct)
    - **DeepL API** - Neural machine translation (500k chars/month free)
    - **MediaPipe** - Face mesh detection (468 landmarks)
    - **Resemblyzer** - Voice embeddings (GE2E speaker diarization)
    - **PyDub** - Audio processing & speech analysis
    - **PyTorch** - Deep learning framework

=== "Frontend"

    - **Vanilla JavaScript** - No framework overhead
    - **Chart.js** - Data visualization (radar charts)
    - **HTML5** - Drag & drop file upload
    - **CSS3** - Responsive design

=== "AI Models"

    - **Whisper large-v3** (~3GB) - OpenAI's SOTA speech recognition
    - **Llama 3.1-8B-Instruct** - Meta's LLM for semantic evaluation
    - **MediaPipe Face Mesh** - 468-point facial landmarks
    - **Resemblyzer GE2E** - Speaker embedding network (~50MB)

---

## :material-format-list-checks: What's Next?

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ***

    Complete step-by-step installation guide including FFmpeg setup dan virtual environment.

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **Features**

    ***

    Deep dive into semua fitur: transcription, cheating detection, non-verbal analysis, dan LLM assessment.

    [:octicons-arrow-right-24: Features Overview](features/overview.md)

-   :material-api:{ .lg .middle } **API Documentation**

    ***

    Complete API reference dengan request/response examples dan error handling.

    [:octicons-arrow-right-24: API Docs](api/endpoints.md)

-   :material-wrench:{ .lg .middle } **Configuration**

    ***

    Customize models, thresholds, API keys, dan advanced settings.

    [:octicons-arrow-right-24: Configuration](configuration/models.md)

</div>

---

## :material-help-circle: Support

<div class="grid cards" markdown>

-   :material-github:{ .lg .middle } **GitHub Issues**

    ***

    Report bugs atau request features.

    [:octicons-arrow-right-24: Open Issue](https://github.com/dapakyuu/Interview_Assesment_System/issues)

-   :material-book-open:{ .lg .middle } **Documentation**

    ***

    Browse comprehensive guides dan tutorials.

    [:octicons-arrow-right-24: View Docs](getting-started/introduction.md)

-   :material-frequently-asked-questions:{ .lg .middle } **FAQ**

    ***

    Common questions dan troubleshooting.

    [:octicons-arrow-right-24: View FAQ](troubleshooting/faq.md)

</div>

---

## :material-license: License

This project is licensed under the **MIT License** - see the [LICENSE](about/license.md) page for details.

---

**Built with ❤️ using FastAPI, Whisper, DeepL, and modern web technologies.**
