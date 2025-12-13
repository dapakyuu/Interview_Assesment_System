# Request & Response Format

Detail format request dan response untuk setiap endpoint.

---

## 📤 Upload Request Details

### Video Upload Format

**Supported Video Formats:**

- MP4 (H.264, H.265) - **Recommended**
- WebM (VP8, VP9, Opus)
- AVI
- MOV (QuickTime)

**Video Requirements:**

| Property      | Requirement | Recommended         |
| ------------- | ----------- | ------------------- |
| Max File Size | 500 MB      | < 100 MB            |
| Max Duration  | Unlimited   | 5-15 minutes        |
| Resolution    | Any         | 720p - 1080p        |
| Codec         | H.264/H.265 | H.264               |
| Audio         | Required    | Mono/Stereo, 16kHz+ |
| Frame Rate    | 15-60 fps   | 25-30 fps           |

**Language Codes:**

- `en` - English
- `id` - Indonesian (Bahasa Indonesia)

---

## 📊 Response Format

### Standard Success Response (Upload)

```json
{
  "success": true,
  "session_id": "abc123def456789...",
  "uploaded_videos": 3,
  "message": "Videos uploaded successfully. Processing started in background.",
  "status_url": "/status/abc123def456789..."
}
```

### Standard Error Response

```json
{
  "detail": "No videos uploaded"
}
```

```json
{
  "detail": "Number of questions (2) does not match number of videos (3)"
}
```

```json
{
  "detail": "Error processing videos: FFmpeg not found"
}
```

---

## 🔍 Detailed Response Schemas

### Transcription Response

Transcription data termasuk dalam response `/results/{session_id}` di field `content[].result`:

```json
{
  "transkripsi_en": "I once led a team in a website development project for a company. The challenge I faced was coordinating the team and resolving technical issues that arose during the development process.",
  "transkripsi_id": "Saya pernah memimpin tim dalam proyek pembangunan website untuk sebuah perusahaan. Tantangan yang saya hadapi adalah koordinasi tim dan menyelesaikan masalah teknis yang muncul selama proses pengembangan.",
  "metadata": {
    "video_file": "video_1.mp4",
    "audio_file": "audio_1.wav",
    "word_count": 141,
    "duration": 45.184,
    "avg_confidence": 98.66879999999999,
    "min_confidence": 92.1875,
    "max_confidence": 99.902344,
    "language": "en",
    "translation_used": true,
    "model": "faster-whisper large-v3",
    "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
    "deepl_used": true,
    "hf_api_used": true
  }
}
```

**Key Features:**

- **Bilingual Output**: `transkripsi_en` (English) dan `transkripsi_id` (Indonesian)
- **High Accuracy**: Avg confidence 98%+ dengan Whisper large-v3
- **DeepL Translation**: 98%+ accuracy untuk EN ↔ ID translation
- **Word Timestamps**: Available dalam internal processing (tidak di-export ke JSON final)

---

### LLM Assessment Response

LLM assessment ada di 2 level: **per-question** (`content[].result.penilaian`) dan **aggregate** (`llm_results`):

**Per-Question Assessment:**

```json
{
  "penilaian": {
    "kualitas_jawaban": 68,
    "koherensi": 75,
    "relevansi": 69,
    "total_score": 70.66666666666667,
    "analisis_llm": "Kandidat memberikan gambaran yang cukup jelas tentang pengalaman memimpin tim dalam proyek pembangunan website untuk perusahaan. Tantangan yang disebutkan terkait dengan koordinasi tim dan penyelesaian masalah teknis menunjukkan pemahaman dasar tentang kompleksitas manajemen proyek. Namun, jawaban bisa lebih kuat jika kandidat memberikan contoh spesifik tentang bagaimana mereka mengatasi tantangan tersebut dan hasil akhir dari proyek.",
    "logprobs_confidence": 91.8,
    "final_score_llm": 93.97142857142857
  }
}
```

**Aggregate LLM Results:**

```json
{
  "llm_results": {
    "kesimpulan_llm": "Secara keseluruhan, kandidat menunjukkan kinerja yang cukup baik dalam wawancara. Pada pertanyaan pertama mengenai kemampuan kepemimpinan, kandidat memberikan jawaban yang koheren dan relevan dengan skor total 70.67...",
    "rata_rata_confidence_score": 90.03333333333333,
    "avg_total_llm": 70.11111111111111,
    "final_score_llm": 89.80952380952381,
    "avg_logprobs_confidence": 90.03333333333333,
    "summary_logprobs_confidence": "Confidence score logprobs dari LLM untuk semua pertanyaan: rata-rata = 90.03%, min = 85.7%, max = 93.4%",
    "reused_single_analysis": false
  }
}
```

**Scoring Dimensions:**

- `kualitas_jawaban` (0-100): Answer quality
- `koherensi` (0-100): Coherence and structure
- `relevansi` (0-100): Relevance to question
- `total_score`: Average of 3 dimensions
- `final_score_llm`: Boosted with logprobs confidence

**Model:** Meta-Llama/Llama-3.1-8B-Instruct via Hugging Face API

---

### Cheating Detection Response

Cheating detection ada di 2 level: **per-question** (`content[].result.cheating_detection`) dan **aggregate** (`aggregate_cheating_detection`):

**Per-Question Detection:**

```json
{
  "cheating_detection": {
    "visual": {
      "total_frames_analyzed": 709,
      "suspicious_frames": 65,
      "suspicious_percentage": 9.169535545023155,
      "face_detected_percentage": 98.45134874469268,
      "confidence_score": 90.78946445497685,
      "verdict": "Medium Risk",
      "issues_detected": [
        "Eyes off-screen: 45 frames (6.35%)",
        "Head turned away: 20 frames (2.82%)"
      ]
    },
    "audio": {
      "num_speakers_detected": 1,
      "silhouette_score": null,
      "confidence_score": 90.0,
      "verdict": "Safe",
      "details": "Only 1 speaker detected (candidate only)"
    },
    "final_verdict": "Medium Risk",
    "overall_confidence_score": 90.39473222748843,
    "cheating_score": 9.169535545023155
  }
}
```

**Aggregate Detection:**

```json
{
  "aggregate_cheating_detection": {
    "avg_cheating_score": 6.1133333333333335,
    "avg_overall_confidence": 89.77999999999999,
    "verdict_distribution": {
      "Safe": 1,
      "Medium Risk": 1,
      "High Risk": 1
    },
    "questions_with_issues": [
      {
        "question_number": 2,
        "question": "Ceritakan pengalaman Anda dalam menangani konflik dalam tim...",
        "verdict": "Medium Risk",
        "cheating_score": 9.17,
        "issues": [
          "Visual: Eyes off-screen (45 frames), Head turned away (20 frames)",
          "Audio: 1 speaker(s) detected (Safe)"
        ]
      }
    ],
    "all_indicators": {
      "visual_issues": {
        "total_suspicious_frames": 136,
        "avg_suspicious_percentage": 6.11,
        "common_issues": ["Eyes off-screen", "Head turned away"]
      },
      "audio_issues": {
        "avg_num_speakers": 1.0,
        "questions_with_multiple_speakers": 0
      }
    },
    "risk_level": "MEDIUM",
    "summary": "Analyzed 3 question(s). Average cheating score: 6.11%. Verdict distribution: Safe=1, Medium Risk=1, High Risk=1.",
    "final_aggregate_verdict": "Medium Risk"
  }
}
```

**Detection Methods:**

- **Visual Analysis**: MediaPipe Face Mesh (468 landmarks)
  - Eye gaze tracking
  - Head pose estimation
  - Face detection percentage
- **Audio Analysis**: Resemblyzer (GE2E speaker diarization)
  - Speaker count detection
  - Voice consistency check

**Verdict Thresholds:**

- `Safe`: < 5% suspicious frames
- `Medium Risk`: 5-20% suspicious frames
- `High Risk`: ≥ 20% suspicious frames

---

### Non-Verbal Analysis Response

Non-verbal analysis ada di 2 level: **per-question** (`content[].result.non_verbal_analysis`) dan **aggregate** (`aggregate_non_verbal_analysis`):

**Per-Question Analysis:**

```json
{
  "non_verbal_analysis": {
    "speech_analysis": {
      "speaking_ratio": 0.58,
      "speaking_interpretation": "Fairly active",
      "speech_rate_wpm": 149.83,
      "speech_interpretation": "Ideal pace",
      "num_pauses": 12,
      "pause_interpretation": "Fluent",
      "confidence": 91.3
    },
    "facial_expression_analysis": {
      "smile_intensity": 0.17,
      "smile_interpretation": "Neutral",
      "eyebrow_movement": 0.02,
      "eyebrow_interpretation": "Controlled",
      "confidence": 78.4
    },
    "eye_movement_analysis": {
      "blink_rate_per_min": 22.9,
      "blink_interpretation": "Normal",
      "eye_contact_percentage": 86.5,
      "eye_contact_interpretation": "Very good",
      "confidence": 89.1
    },
    "non_verbal_confidence_score": 86.26666666666667,
    "performance_status": "EXCELLENT"
  }
}
```

**Aggregate Analysis:**

```json
{
  "aggregate_non_verbal_analysis": {
    "overall_confidence_score": 77.2,
    "overall_performance_status": "GOOD",
    "summary": "speaking ratio 0.58 (fairly active), speech rate ~150 wpm (ideal), eye contact 85.2% (very good), blink rate 22.3/min (normal), smile intensity 0.18 (neutral), eyebrow movement 0.03 (controlled), pauses 13 (fluent)"
  }
}
```

**Analysis Components:**

1. **Speech Patterns:**

   - Speaking ratio (time speaking vs silence)
   - Speech rate (words per minute)
   - Pause count and duration
   - **Ideal**: 0.5-0.7 ratio, 130-160 WPM, < 20 pauses

2. **Facial Expressions:**

   - Smile intensity (0-1 scale)
   - Eyebrow movement (standard deviation)
   - **Ideal**: 0.1-0.3 smile, 0.02-0.05 eyebrow

3. **Eye Movement:**
   - Blink rate (per minute)
   - Eye contact percentage
   - **Ideal**: 15-25 blinks/min, > 80% eye contact

**Performance Status:**

- `EXCELLENT`: ≥ 85% confidence
- `GOOD`: 70-85% confidence
- `FAIR`: 50-70% confidence
- `POOR`: < 50% confidence

**Technology:** MediaPipe Face Mesh (468 facial landmarks)

---

## ⏱️ Response Times

**Average Processing Times:**

| Stage                     | CPU Only         | With GPU        |
| ------------------------- | ---------------- | --------------- |
| Video Upload              | < 5 seconds      | < 5 seconds     |
| Google Drive Download     | 10-30 seconds    | 10-30 seconds   |
| Audio Extraction (FFmpeg) | 5-10 seconds     | 5-10 seconds    |
| Transcription (Whisper)   | 2-4 minutes      | 30-60 seconds   |
| Translation (DeepL)       | 2-5 seconds      | 2-5 seconds     |
| LLM Assessment (Llama)    | 15-30 seconds    | 8-15 seconds    |
| Cheating Detection        | 45-90 seconds    | 20-40 seconds   |
| Non-Verbal Analysis       | 30-60 seconds    | 15-30 seconds   |
| JSON Save & Cleanup       | 2-5 seconds      | 2-5 seconds     |
| **Total (5 min video)**   | **5-8 minutes**  | **2-3 minutes** |
| **Total (10 min video)**  | **8-12 minutes** | **3-5 minutes** |

**Note:** Times vary based on video length, resolution, and CPU/GPU specs. GPU recommended for production.

---

## 📏 Response Size Limits

| Endpoint                | Max Response Size | Typical Size |
| ----------------------- | ----------------- | ------------ |
| `/status/{session_id}`  | 5 KB              | 1-2 KB       |
| `/results/{session_id}` | 5 MB              | 50-200 KB    |
| Upload Response         | 5 KB              | 1 KB         |

**Typical `/results` Size by Video Count:**

- 1 video: ~20-30 KB
- 3 videos: ~50-100 KB
- 5 videos: ~100-200 KB

---

## 🔄 Session Management

Sistem menggunakan **session-based processing** dengan unique session ID:

**Session ID Format:**

```text
35050a4f73d34d8f8818fb7a2bc3f5cd
```

- UUID v4 format (32 hex characters)
- Generated saat upload
- Digunakan untuk track status dan retrieve results
- Saved di `backend/Python/results/{session_id}.json`

**Session Lifecycle:**

1. **Upload** → Generate session_id
2. **Processing** → Track via `/status/{session_id}`
3. **Completed** → Retrieve via `/results/{session_id}`
4. **Storage** → JSON file persisted permanently

---

## 📝 Best Practices

### 1. Polling Status

```python
import time
import requests

def wait_for_completion(session_id, port=8888):
    """Poll status endpoint until processing completes."""
    url = f"http://localhost:{port}/status/{session_id}"

    while True:
        response = requests.get(url)
        data = response.json()
        status = data["status"]

        print(f"Status: {status} - {data.get('message', '')}")

        if status == "completed":
            return True, data.get("redirect")
        elif status == "error":
            return False, data.get("error")

        time.sleep(5)  # Poll every 5 seconds

# Usage
success, info = wait_for_completion("abc123def456...")
if success:
    print(f"✅ Completed! Redirect: {info}")
else:
    print(f"❌ Error: {info}")
```

### 2. Error Handling

```python
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout

try:
    # Upload videos
    response = requests.post(
        "http://localhost:8888/upload_json",
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    result = response.json()

    if result.get("success"):
        print(f"Session ID: {result['session_id']}")

except HTTPError as e:
    # 400, 404, 500 errors
    print(f"HTTP {e.response.status_code}: {e.response.json()['detail']}")

except ConnectionError:
    print("Cannot connect to server. Is it running on port 8888?")

except Timeout:
    print("Request timed out. Server might be busy.")

except Exception as e:
    print(f"Unexpected error: {e}")
```

### 3. Complete Workflow

```python
import requests
import time
import json

def process_interview(candidate_name, videos_data, language="en"):
    """Complete interview processing workflow."""

    # Step 1: Upload
    payload = {
        "candidate_name": candidate_name,
        "interviews": videos_data,
        "language": language
    }

    upload_response = requests.post(
        "http://localhost:8888/upload_json",
        json=payload
    )
    upload_response.raise_for_status()
    session_id = upload_response.json()["session_id"]

    print(f"📤 Uploaded. Session: {session_id}")

    # Step 2: Poll status
    while True:
        status_response = requests.get(
            f"http://localhost:8888/status/{session_id}"
        )
        status_data = status_response.json()

        if status_data["status"] == "completed":
            break
        elif status_data["status"] == "error":
            raise Exception(status_data.get("error"))

        print(f"⏳ {status_data.get('message', 'Processing...')}")
        time.sleep(5)

    # Step 3: Get results
    results_response = requests.get(
        f"http://localhost:8888/results/{session_id}"
    )
    results = results_response.json()

    # Step 4: Save results
    with open(f"results_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Completed! Results saved.")
    return results

# Usage
videos = [
    {
        "positionId": 1,
        "question": "Tell me about yourself",
        "isVideoExist": True,
        "recordedVideoUrl": "https://drive.google.com/file/d/FILE_ID/view"
    }
]

results = process_interview("John Doe", videos, "en")
print(f"LLM Score: {results['llm_results']['final_score_llm']:.1f}")
```

---

## 📚 See Also

- [API Endpoints](endpoints.md)
- [Error Codes](errors.md)
- [Quickstart Guide](../getting-started/quickstart.md)
