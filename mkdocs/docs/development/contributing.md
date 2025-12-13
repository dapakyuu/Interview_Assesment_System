# Contributing Guide

Panduan untuk berkontribusi ke Interview Assessment System.

---

## 🎉 Welcome Contributors!

Terima kasih atas minat Anda untuk berkontribusi! Kontribusi dalam bentuk apapun sangat dihargai:

- 🐛 Bug reports
- ✨ Feature requests
- 📝 Documentation improvements
- 💻 Code contributions
- 🧪 Testing

---

## 🚀 Quick Start

### 1. Fork & Clone Repository

```bash
# Fork repository via GitHub UI
# Then clone your fork
git clone https://github.com/dapakyuu/Interview_Assesment_System.git
cd Interview_Assesment_System
```

**Repository Structure:**

```
Interview_Assesment_System/
├── backend/Python/          # Backend API
├── frontend/                # Web interface
├── mkdocs/                  # Documentation
└── interview_assessment_system.ipynb  # Demo notebook
```

### 2. Setup Development Environment

**Backend Setup:**

```bash
# Navigate to backend
cd backend/Python

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FFmpeg (Windows)
winget install FFmpeg

# Install FFmpeg (Linux)
sudo apt install ffmpeg
```

**Environment Variables:**

```bash
# Copy environment template
cd app
cp env.example .env

# Edit .env with your API keys
# HF_TOKEN=hf_your_token_here
# DEEPL_API_KEY=your_key_here:fx
```

**Verify Setup:**

```python
# Test GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test imports
python -c "from faster_whisper import WhisperModel; print('✓ faster-whisper')"
python -c "import mediapipe; print('✓ MediaPipe')"
python -c "from resemblyzer import VoiceEncoder; print('✓ Resemblyzer')"
```

**Development Dependencies (Optional):**

```bash
# Install testing and formatting tools
pip install pytest pytest-cov black flake8 mypy
```

### 3. Create Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/bug-description
```

---

## 📋 Development Workflow

### Step 1: Make Changes

**Backend (Python):**

```bash
# Edit processing logic
code backend/Python/app/utils/transcription.py
code backend/Python/app/utils/llm_evaluator.py

# Edit API endpoints
code backend/Python/app/routes.py
code backend/Python/app/server.py

# Edit services
code backend/Python/app/services/whisper_service.py
code backend/Python/app/services/deepl_service.py
```

**Frontend (JavaScript):**

```bash
# Edit upload interface
code frontend/Upload.html
code frontend/Upload.js

# Edit dashboard
code frontend/Halaman_dasboard.html
code frontend/Halaman_dasboard.js
```

**Documentation:**

```bash
# Edit MkDocs documentation
cd mkdocs
code docs/api/endpoints.md
code docs/troubleshooting/common-issues.md

# Preview changes
mkdocs serve
# Open http://127.0.0.1:8000
```

### Step 2: Test Changes

**Manual Testing:**

```bash
# Run backend server
cd backend/Python
python main.py
# Server runs on http://localhost:7860

# Or with uvicorn
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload
```

**Test with Jupyter Notebook:**

```bash
# Run notebook for end-to-end test
jupyter notebook interview_assessment_system.ipynb
```

**Unit Tests (if available):**

```python
# Create test file: backend/Python/tests/test_transcription.py
import pytest
from app.utils.transcription import extract_audio, transcribe_audio
import os

def test_audio_extraction():
    """Test FFmpeg audio extraction."""
    video_path = "test_data/sample.mp4"
    audio_path = "test_data/output.wav"

    # Test extraction
    result = extract_audio(video_path, audio_path)

    assert os.path.exists(audio_path)
    assert os.path.getsize(audio_path) > 0

    # Cleanup
    os.remove(audio_path)

def test_transcription():
    """Test faster-whisper transcription."""
    audio_path = "test_data/sample.wav"

    result = transcribe_audio(audio_path, language="en")

    assert result is not None
    assert len(result) > 0
    assert isinstance(result, str)

# Run tests
# pytest tests/ -v
```

**API Testing:**

```bash
# Test upload endpoint
curl -X POST http://localhost:7860/upload \
  -F "files=@test_video.mp4" \
  -F "jumlah_pertanyaan=3" \
  -F "language=id"

# Test status endpoint
curl http://localhost:7860/status/{session_id}

# Test results endpoint
curl http://localhost:7860/results/{session_id}
```

### Step 3: Format Code

```bash
# Format Python code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

### Step 4: Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add speaker diarization feature"

# Or for bug fixes
git commit -m "fix: resolve audio extraction error"
```

**Commit Message Format:**

```
type: short description

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Add tests
- `chore`: Maintenance

### Step 5: Push & Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request via GitHub UI
```

---

## 🧪 Testing Guidelines

### Writing Tests

```python
# tests/test_transcription.py
import pytest
from pipeline.transcription import WhisperTranscriber

def test_transcription():
    """Test basic transcription"""
    transcriber = WhisperTranscriber()
    result = transcriber.transcribe("test_audio.wav", language="en")

    assert result is not None
    assert "text" in result
    assert len(result["text"]) > 0

def test_transcription_invalid_file():
    """Test error handling"""
    transcriber = WhisperTranscriber()

    with pytest.raises(FileNotFoundError):
        transcriber.transcribe("nonexistent.wav")
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pipeline --cov-report=html

# Run specific test
pytest tests/test_transcription.py::test_transcription -v

# Run tests matching pattern
pytest -k "transcription"
```

---

## 📝 Code Style

### Python Style Guide

Follow **PEP 8** with these specifics:

```python
# Good: Type hints, docstrings, clear naming
from typing import Dict, List
import os

def transcribe_audio(
    audio_path: str,
    language: str = "en",
    beam_size: int = 10
) -> str:
    """
    Transcribe audio file using faster-whisper.

    Args:
        audio_path: Path to WAV audio file (16kHz mono)
        language: Language code ('en' or 'id')
        beam_size: Beam size for decoding (higher = better quality, slower)

    Returns:
        str: Transcribed text

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If language not supported
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if language not in ["en", "id"]:
        raise ValueError(f"Unsupported language: {language}")

    # Load model (singleton pattern)
    from app.services.whisper_service import get_whisper_model
    model = get_whisper_model()

    # Transcribe
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True
    )

    # Combine segments
    full_text = " ".join([segment.text for segment in segments])
    return full_text.strip()

# Bad: No types, no docs, unclear
def transcribe(path,lang='en'):
    model=load_model()
    result=model.transcribe(path,language=lang)
    return result['text']
```

**Import Organization:**

```python
# Standard library
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import torch
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from faster_whisper import WhisperModel

# Local imports
from app.config import settings
from app.utils.transcription import extract_audio
from app.services.whisper_service import get_whisper_model
```

### JavaScript Style Guide

**Frontend Code (Upload.js, Halaman_dasboard.js):**

```javascript
// Good: Clear async/await, error handling, comments
async function uploadVideos(files, questionCount, language) {
  try {
    // Build form data
    const formData = new FormData();

    // Add all video files
    files.forEach((file, index) => {
      formData.append("files", file);
    });

    // Add metadata
    formData.append("jumlah_pertanyaan", questionCount);
    formData.append("language", language);

    // Upload to server
    const response = await fetch("http://localhost:7860/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.session_id;
  } catch (error) {
    console.error("Upload error:", error);
    throw error;
  }
}

// Poll for processing status
async function checkStatus(sessionId) {
  const response = await fetch(`http://localhost:7860/status/${sessionId}`);
  const data = await response.json();
  return data.status; // 'processing', 'completed', 'failed'
}

// Get results
async function getResults(sessionId) {
  const response = await fetch(`http://localhost:7860/results/${sessionId}`);
  if (!response.ok) {
    throw new Error("Results not found");
  }
  return await response.json();
}

// Bad: No error handling, unclear
function upload(files, count, lang) {
  var formData = new FormData();
  for (var i = 0; i < files.length; i++) formData.append("files", files[i]);
  formData.append("jumlah_pertanyaan", count);
  formData.append("language", lang);
  return fetch("http://localhost:7860/upload", {
    method: "POST",
    body: formData,
  });
}
```

### Formatting Tools

```bash
# Python: Black
black --line-length 88 .

# Python: isort (import sorting)
isort .

# JavaScript: Prettier
npx prettier --write "**/*.js"
```

---

## 📚 Documentation

### Docstring Format

```python
def process_video(
    video_path: str,
    session_id: str,
    language: str = "id",
    jumlah_pertanyaan: int = 3
) -> Dict[str, any]:
    """
    Process interview video through complete 7-stage pipeline.

    This function orchestrates the entire processing pipeline:
    1. Audio Extraction (FFmpeg)
    2. Transcription (faster-whisper)
    3. Translation (DeepL API)
    4. LLM Assessment (Llama 3.1 via HF API)
    5. Cheating Detection (MediaPipe + Resemblyzer)
    6. Non-Verbal Analysis (MediaPipe landmarks)
    7. Save Results (JSON)

    Args:
        video_path: Absolute path to video file (MP4, AVI, MOV, etc.)
        session_id: Unique session identifier (UUID)
        language: Target language for results ('en' or 'id')
        jumlah_pertanyaan: Number of questions/videos to process

    Returns:
        Dict[str, any]: Complete assessment results containing:
            - session_id: Session UUID
            - processing_time: Total processing time in seconds
            - results: List of per-question results with:
                - transkripsi_en: English transcription
                - transkripsi_id: Indonesian translation
                - kesimpulan_llm: LLM assessment summary
                - kualitas_jawaban: Answer quality score (1-10)
                - cheating_detection: Suspicious behavior flags
                - non_verbal_analysis: Facial expressions, gaze

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If language not in ['en', 'id']
        HTTPException: If API calls fail (DeepL, HF)
        ProcessingError: If any pipeline stage fails

    Example:
        >>> results = process_video(
        ...     video_path="uploads/abc-123/video_0.mp4",
        ...     session_id="abc-123-def-456",
        ...     language="id",
        ...     jumlah_pertanyaan=3
        ... )
        >>> print(results["processing_time"])
        185.4
        >>> print(results["results"][0]["kualitas_jawaban"])
        8

    Notes:
        - Processing time: GPU 2-3 min, CPU 5-8 min per 5-min video
        - Requires HF_TOKEN and DEEPL_API_KEY environment variables
        - Results saved to results/{session_id}.json
    """
    pass
```

### Comment Best Practices

```python
# Good: Explain WHY, not WHAT
# Use frame skip to reduce processing time while maintaining accuracy
frame_skip = 5

# Bad: Obvious comment
# Set frame skip to 5
frame_skip = 5
```

---

## 🔍 Code Review Process

### What Reviewers Look For

1. **Correctness:** Does it work as intended?
2. **Tests:** Are there adequate tests?
3. **Documentation:** Is it well documented?
4. **Style:** Does it follow style guidelines?
5. **Performance:** Are there any bottlenecks?
6. **Security:** Any security concerns?

### Responding to Reviews

```bash
# Make requested changes
git add .
git commit -m "refactor: address review comments"

# Push changes
git push origin feature/your-feature-name
```

---

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:

1. Upload video with ...
2. Select language ...
3. Error occurs at ...

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Screenshots**
If applicable, add screenshots.

**Environment:**

- OS: [e.g., Windows 11]
- Python version: [e.g., 3.11.9]
- GPU: [e.g., NVIDIA RTX 3070]

**Additional context**

- Error logs
- Video details (format, size, duration)
```

---

## ✨ Feature Requests

### Feature Request Template

```markdown
**Feature description**
Clear description of the feature.

**Use case**
Why is this feature needed?

**Proposed solution**
How should this be implemented?

**Alternatives considered**
Other approaches you've thought about.

**Additional context**
Mockups, examples, references.
```

---

## 🎯 Areas to Contribute

### High Priority

- ⭐ **Performance Optimization**

  - Optimize frame skipping algorithm
  - Reduce GPU memory usage
  - Faster model loading (caching)
  - Batch processing improvements

- ⭐ **Multi-language Support**

  - Add more languages beyond EN/ID
  - Auto language detection
  - Better translation quality

- ⭐ **Testing & Reliability**

  - Add unit tests for utils/
  - Integration tests for API endpoints
  - Test coverage for error handling
  - Edge case testing

- ⭐ **UI/UX Improvements**
  - Better progress indicators during processing
  - More detailed error messages
  - Responsive dashboard design
  - Export results to PDF/Excel

### Good First Issues

- 📝 **Documentation**

  - Fix typos in docs
  - Add more code examples
  - Translate docs to English
  - Add FAQ entries

- 🎨 **Frontend Improvements**

  - Fix CSS styling issues
  - Add loading animations
  - Improve mobile responsiveness
  - Add dark mode

- 🐛 **Bug Fixes**

  - Fix file upload validation
  - Handle edge cases in processing
  - Fix memory leaks
  - Improve error messages

- ✅ **Testing**
  - Write tests for existing functions
  - Add integration tests
  - Create test fixtures
  - Add CI/CD workflows

### Advanced Contributions

- 🚀 **Architecture Improvements**

  - Implement job queue (Redis + RQ/Celery)
  - Add PostgreSQL database
  - Microservices separation
  - WebSocket for real-time updates

- 🔒 **Security & Auth**

  - Add API key authentication
  - Implement rate limiting
  - Add user management
  - Session encryption

- 📊 **Advanced Features**

  - Batch video processing UI
  - Advanced analytics dashboard
  - Custom assessment criteria editor
  - Video streaming upload

- 🤖 **ML Improvements**
  - Fine-tune Whisper for Indonesian
  - Custom LLM prompts
  - Better cheating detection algorithms
  - Emotion recognition

**Current Roadmap:** See [roadmap.md](roadmap.md) for detailed plans

---

## 💻 Development Tips

### Run Backend with Hot Reload

```bash
# Navigate to backend
cd backend/Python

# Run with auto-reload (development)
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload

# Or run main.py directly
python main.py
```

### Debug with Jupyter Notebook

```python
# In interview_assessment_system.ipynb
%load_ext autoreload
%autoreload 2

# Import modules - will auto-reload on changes
from app.utils.transcription import transcribe_audio
from app.utils.llm_evaluator import assess_answer

# Test individual functions
result = transcribe_audio("test.wav", language="en")
print(result)
```

### Enable Debug Logging

```python
# Add to app/config.py or main.py
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Show all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Processing frame {frame_idx}")
logger.info(f"Transcription completed: {len(text)} chars")
logger.warning("Low audio quality detected")
logger.error(f"Failed to load model: {error}")
```

### Monitor GPU Usage During Development

```bash
# Watch GPU in real-time
watch -n 1 nvidia-smi

# Or in Python
import torch

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
```

### Test API Endpoints Quickly

```bash
# Install httpie for better API testing
pip install httpie

# Test upload (httpie syntax)
http -f POST localhost:7860/upload \
  files@test.mp4 \
  jumlah_pertanyaan=3 \
  language=id

# Test with curl
curl -X POST http://localhost:7860/upload \
  -F "files=@test_video.mp4" \
  -F "jumlah_pertanyaan=3" \
  -F "language=id"
```

### Use VS Code Debugger

**Create `.vscode/launch.json`:**

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "7860",
        "--reload"
      ],
      "jinja": true,
      "justMyCode": false,
      "cwd": "${workspaceFolder}/backend/Python"
    }
  ]
}
```

**Set breakpoints in VS Code and press F5 to debug**

---

## 🔧 Pre-commit Hooks

### Setup Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install
```

### `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
```

---

## 📊 Performance Benchmarking

### Benchmark Your Changes

```python
import time

def benchmark(func, *args, iterations=10):
    """Benchmark function performance"""
    times = []
    for _ in range(iterations):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end - start)

    avg = sum(times) / len(times)
    print(f"{func.__name__}: {avg:.2f}s avg")
    return avg

# Usage
benchmark(transcribe_audio, "test.wav")
```

---

## 🏆 Recognition

Contributors will be:

- ✨ Listed in CONTRIBUTORS.md
- 📝 Mentioned in release notes
- 🎖️ Given credit in documentation

---

## 📞 Getting Help

**Questions or Issues?**

1. **Check Documentation First:**

   - [Installation Guide](../getting-started/installation.md)
   - [Quickstart Guide](../getting-started/quickstart.md)
   - [Common Issues](../troubleshooting/common-issues.md)
   - [FAQ](../troubleshooting/faq.md)
   - [API Reference](../api/endpoints.md)

2. **Search Existing Issues:**

   - GitHub Issues: Search for similar problems
   - Check closed issues for solutions

3. **Create New Issue:**

   - Use appropriate template (bug/feature)
   - Include system info (OS, Python version, GPU)
   - Provide error logs and reproduction steps
   - Add screenshots if helpful

4. **Join Community:**
   - GitHub Discussions (if enabled)
   - Open issue for general questions
   - Tag with `question` label

**Useful Resources:**

- **Development:**

  - [Architecture Overview](architecture.md)
  - [Contributing Guide](contributing.md) (this page)
  - [Roadmap](roadmap.md)

- **Configuration:**

  - [Model Configuration](../configuration/models.md)
  - [API Keys Setup](../configuration/api-keys.md)
  - [Advanced Settings](../configuration/advanced.md)

- **Troubleshooting:**
  - [Performance Tuning](../troubleshooting/performance.md)
  - [Common Issues](../troubleshooting/common-issues.md)
  - [FAQ](../troubleshooting/faq.md)

**Response Times:**

- Documentation: Immediate (self-service)
- GitHub Issues: 1-3 business days
- Critical bugs: Priority response
- Feature requests: Reviewed weekly

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the project's license.

---

**Thank you for contributing!** 🎉 Your efforts help make this project better for everyone!
