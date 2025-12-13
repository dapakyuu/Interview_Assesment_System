# API Endpoints Reference

Complete API documentation untuk AI Interview Assessment System. Semua endpoints accessible via FastAPI backend running on **port 8888** (Jupyter Notebook) atau **port 7860** (Python script).

---

## Base URL

**Jupyter Notebook:**

```http
http://localhost:8888
```

**Python Script:**

```http
http://localhost:7860
```

**Interactive Documentation:**

```http
http://localhost:8888/docs  # Swagger UI
http://localhost:8888/redoc # ReDoc
```

---

## 📤 Upload & Processing

### POST /upload

Upload multiple interview videos dengan multipart/form-data untuk local file upload.

**Endpoint:**

```http
POST /upload
Content-Type: multipart/form-data
```

**Form Data Parameters:**

| Field            | Type     | Required | Description                     |
| ---------------- | -------- | -------- | ------------------------------- |
| `candidate_name` | String   | ✅       | Nama kandidat                   |
| `videos`         | File[]   | ✅       | 1-5 video files                 |
| `question`       | String[] | ✅       | Question per video (same order) |
| `language`       | String   | ✅       | `"en"` atau `"id"`              |

**Supported Video Formats:**

- MP4 (H.264, H.265)
- WebM (VP8, VP9, Opus)
- AVI
- MOV (QuickTime)

**File Size Limits:**

- Recommended: < 100MB per video
- Maximum: 500MB per video

**Example Request (cURL):**

```bash
curl -X POST http://localhost:8888/upload \
  -F "candidate_name=John Doe" \
  -F "videos=@video1.mp4" \
  -F "videos=@video2.mp4" \
  -F "videos=@video3.mp4" \
  -F "question=Tell me about yourself" \
  -F "question=Why this position?" \
  -F "question=Your biggest achievement?" \
  -F "language=en"
```

**Example Request (Python):**

```python
import requests

url = "http://localhost:8888/upload"

files = [
("videos", ("video1.mp4", open("video1.mp4", "rb"), "video/mp4")),
("videos", ("video2.mp4", open("video2.mp4", "rb"), "video/mp4")),
("videos", ("video3.mp4", open("video3.mp4", "rb"), "video/mp4"))
]

data = {
"candidate_name": "John Doe",
"question": [
"Tell me about yourself",
"Why this position?",
"Your biggest achievement?"
],
"language": "en"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Example Request (JavaScript/Frontend):**

```javascript
const formData = new FormData();
formData.append("candidate_name", "John Doe");
formData.append("language", "en");

// Add videos
const videoFiles = document.getElementById("videoInput").files;
for (let i = 0; i < videoFiles.length; i++) {
  formData.append("videos", videoFiles[i]);
}

// Add questions
const questions = ["Q1", "Q2", "Q3"];
questions.forEach((q) => formData.append("question", q));

fetch("http://localhost:8888/upload", {
  method: "POST",
  body: formData,
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

**Response (200 OK):**

```json
{
  "success": true,
  "session_id": "abc123def456...",
  "uploaded_videos": 3,
  "message": "Videos uploaded successfully. Processing started in background.",
  "status_url": "/status/abc123def456..."
}
```

**Response (400 Bad Request):**

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

**Response (500 Internal Server Error):**

```json
{
  "detail": "Error processing videos: [error message]"
}
```

---

### POST /upload_json

Upload videos via Google Drive URLs menggunakan JSON payload. **Recommended untuk integration dengan external systems.**

**Endpoint:**

```http
POST /upload_json
Content-Type: application/json
```

**JSON Body:**

| Field                           | Type    | Required | Description                |
| ------------------------------- | ------- | -------- | -------------------------- |
| `candidate_name`                | String  | ✅       | Nama kandidat              |
| `interviews`                    | Array   | ✅       | Array of interview objects |
| `interviews[].positionId`       | Integer | ✅       | Question ID/position       |
| `interviews[].question`         | String  | ✅       | Interview question text    |
| `interviews[].isVideoExist`     | Boolean | ✅       | `true` if video available  |
| `interviews[].recordedVideoUrl` | String  | ✅       | Google Drive share URL     |
| `language`                      | String  | ✅       | `"en"` atau `"id"`         |

**Supported Google Drive URL Formats:**

```bash
# Format 1: Standard sharing link
https://drive.google.com/file/d/FILE_ID/view

# Format 2: Open link
https://drive.google.com/open?id=FILE_ID

# Format 3: UC link
https://drive.google.com/uc?id=FILE_ID

# Format 4: Direct file ID
FILE_ID (33+ characters)
```

**Example Request (cURL):**

```bash
curl -X POST http://localhost:8888/upload_json \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_name": "Jane Smith",
    "interviews": [
      {
        "positionId": 1,
        "question": "Tell me about your Python experience",
        "isVideoExist": true,
        "recordedVideoUrl": "https://drive.google.com/file/d/1ABC123XYZ456/view"
      },
      {
        "positionId": 2,
        "question": "Explain async programming",
        "isVideoExist": true,
        "recordedVideoUrl": "https://drive.google.com/file/d/2DEF789UVW012/view"
      }
    ],
    "language": "en"
  }'
```

**Example Request (Python):**

```python
import requests
import json

url = "http://localhost:8888/upload_json"

payload = {
"candidate_name": "Jane Smith",
"interviews": [
{
"positionId": 1,
"question": "Tell me about your Python experience",
"isVideoExist": True,
"recordedVideoUrl": "https://drive.google.com/file/d/1ABC123XYZ456/view"
},
{
"positionId": 2,
"question": "Explain async programming",
"isVideoExist": True,
"recordedVideoUrl": "https://drive.google.com/file/d/2DEF789UVW012/view"
}
],
"language": "en"
}

headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

**Example Request (JavaScript):**

```javascript
const payload = {
  candidate_name: "Jane Smith",
  interviews: [
    {
      positionId: 1,
      question: "Tell me about your Python experience",
      isVideoExist: true,
      recordedVideoUrl: "https://drive.google.com/file/d/1ABC123XYZ456/view",
    },
    {
      positionId: 2,
      question: "Explain async programming",
      isVideoExist: true,
      recordedVideoUrl: "https://drive.google.com/file/d/2DEF789UVW012/view",
    },
  ],
  language: "en",
};

fetch("http://localhost:8888/upload_json", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

**Response (200 OK):**

```json
{
  "success": true,
  "session_id": "xyz789abc123...",
  "uploaded_videos": 2,
  "message": "Videos are being downloaded from Google Drive and processed in background",
  "status_url": "/status/xyz789abc123..."
}
```

**Response (400 Bad Request):**

```json
{
  "detail": "No interviews provided"
}
```

```json
{
  "detail": "Failed to extract file ID from URL: [url]"
}
```

**Response (500 Internal Server Error):**

```json
{
  "detail": "Google Drive download failed: [error message]"
}
```

!!! tip "Google Drive Setup"
Pastikan video di Google Drive di-set sebagai **"Anyone with the link can view"** agar dapat di-download oleh sistem.

---

## 📊 Status & Results

### GET /status/{session_id}

Real-time polling untuk check processing status. Frontend biasanya poll setiap **5 seconds**.

**Endpoint:**

```http
GET /status/{session_id}
```

**Path Parameters:**

| Parameter    | Type   | Description                       |
| ------------ | ------ | --------------------------------- |
| `session_id` | String | UUID session dari upload response |

**Example Request:**

```bash
curl http://localhost:8888/status/abc123def456...
```

**Example Request (Python):**

```python
import requests
import time

session_id = "abc123def456..."
url = f"http://localhost:8888/status/{session_id}"

while True:
response = requests.get(url)
data = response.json()

    print(f"Status: {data['status']} - {data.get('message', '')}")

    if data['status'] == 'completed':
        print(f"Redirect to: {data['redirect']}")
        break
    elif data['status'] == 'error':
        print(f"Error: {data.get('error')}")
        break

    time.sleep(5)  # Poll every 5 seconds

```

**Example Request (JavaScript):**

```javascript
const sessionId = "abc123def456...";
const pollStatus = async () => {
  const response = await fetch(`http://localhost:8888/status/${sessionId}`);
  const data = await response.json();

  console.log(`Status: ${data.status}`);

  if (data.status === "completed") {
    window.location.href = data.redirect;
  } else if (data.status === "processing") {
    setTimeout(pollStatus, 5000); // Poll again in 5s
  }
};

pollStatus();
```

**Response - Processing (200 OK):**

```json
{
  "status": "processing",
  "progress": "2/3",
  "message": "Transcribing video 2/3...",
  "current_video": 2,
  "total_videos": 3
}
```

**Progress Messages:**

| Message                                    | Stage              | Notes                   |
| ------------------------------------------ | ------------------ | ----------------------- |
| `"Downloading video from Google Drive..."` | Download           | Google Drive only       |
| `"Extracting audio from video..."`         | Audio Extraction   | FFmpeg                  |
| `"Transcribing video X/Y..."`              | Transcription      | Whisper large-v3        |
| `"Translating transcription..."`           | Translation        | DeepL API               |
| `"Performing LLM assessment..."`           | LLM                | Llama 3.1-8B            |
| `"Detecting cheating indicators..."`       | Cheating Detection | MediaPipe + Resemblyzer |
| `"Analyzing non-verbal cues..."`           | Non-Verbal         | Facial/Eye/Speech       |
| `"Saving results..."`                      | Saving             | JSON write              |

**Response - Completed (200 OK):**

```json
{
  "status": "completed",
  "redirect": "halaman_dasboard.html?session=abc123def456...",
  "result": {
    "success": true,
    "successful_videos": 3,
    "total_videos": 3,
    "session_id": "abc123def456..."
  }
}
```

**Response - Error (200 OK):**

```json
{
  "status": "error",
  "error": "FFmpeg not found. Please install FFmpeg.",
  "session_id": "abc123def456..."
}
```

**Response - Not Found (404):**

```json
{
  "detail": "Session not found"
}
```

---

### GET /results/{session_id}

Retrieve complete assessment results dalam format JSON.

**Endpoint:**

```http
GET /results/{session_id}
```

**Path Parameters:**

| Parameter    | Type   | Description     |
| ------------ | ------ | --------------- |
| `session_id` | String | UUID session ID |

**Example Request:**

```bash
curl http://localhost:8888/results/abc123def456...
```

**Example Request (Python):**

```python
import requests
import json

session_id = "abc123def456..."
url = f"http://localhost:8888/results/{session_id}"

response = requests.get(url)
results = response.json()

# Save to file

with open(f'results_{session_id}.json', 'w', encoding='utf-8') as f:
json.dump(results, f, indent=2, ensure_ascii=False)

# Extract key metrics

llm_score = results['llm_results']['final_score_llm']
cheating_verdict = results['aggregate_cheating_detection']['final_aggregate_verdict']
nonverbal_score = results['aggregate_non_verbal_analysis']['overall_confidence_score']

print(f"LLM Score: {llm_score}")
print(f"Cheating Verdict: {cheating_verdict}")
print(f"Non-Verbal Score: {nonverbal_score}")
```

**Response (200 OK):**

```json
{
  "success": true,
  "name": "Raifal Bagus",
  "session": "35050a4f73d34d8f8818fb7a2bc3f5cd",
  "llm_results": {
    "kesimpulan_llm": "Secara keseluruhan, kandidat menunjukkan kinerja yang cukup baik dalam wawancara. Pada pertanyaan pertama mengenai kemampuan kepemimpinan, kandidat memberikan jawaban yang koheren dan relevan dengan skor total 70.67. Kandidat menjelaskan pengalaman memimpin tim proyek pembangunan website dengan baik, meskipun ada beberapa area yang bisa diperbaiki dalam hal kedalaman analisis. Pada pertanyaan kedua tentang penanganan konflik tim, kandidat memberikan respons yang sangat baik dengan skor total 86.33. Kandidat menunjukkan pemahaman yang kuat tentang pentingnya komunikasi terbuka dan pendekatan empatik dalam menyelesaikan konflik. Namun, pada pertanyaan ketiga mengenai kelemahan diri, kandidat memberikan jawaban yang kurang memuaskan dengan skor total 53.33. Jawaban kandidat terlalu singkat dan tidak memberikan contoh konkret atau rencana pengembangan yang jelas. Dari segi non-verbal, kandidat menunjukkan performa yang baik dengan skor kepercayaan diri keseluruhan 77.2% dan status performa \"GOOD\". Kandidat memiliki rasio berbicara yang cukup aktif (0.58), kecepatan bicara yang ideal (sekitar 150 kata per menit), dan kontak mata yang sangat baik (85.2%). Namun, ada beberapa indikasi potensi kecurangan yang perlu diperhatikan, terutama pada pertanyaan kedua di mana terdeteksi risiko menengah dengan 9.17% frame mencurigakan, termasuk mata yang melihat ke luar layar dan kepala yang berpaling. Secara keseluruhan, kandidat menunjukkan potensi yang baik tetapi masih ada ruang untuk perbaikan, terutama dalam memberikan jawaban yang lebih mendalam dan menghindari perilaku yang dapat menimbulkan kecurigaan selama wawancara.",
    "rata_rata_confidence_score": 90.03333333333333,
    "avg_total_llm": 70.11111111111111,
    "final_score_llm": 89.80952380952381,
    "avg_logprobs_confidence": 90.03333333333333,
    "summary_logprobs_confidence": "Confidence score logprobs dari LLM untuk semua pertanyaan: rata-rata = 90.03%, min = 85.7%, max = 93.4%",
    "reused_single_analysis": false
  },
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
        "question": "Ceritakan pengalaman Anda dalam menangani konflik dalam tim. Apa langkah-langkah yang Anda ambil?",
        "verdict": "Medium Risk",
        "cheating_score": 9.17,
        "issues": [
          "Visual: Eyes off-screen (45 frames), Head turned away (20 frames)",
          "Audio: 1 speaker(s) detected (Safe)"
        ]
      },
      {
        "question_number": 3,
        "question": "Apa kelemahan terbesar Anda dan bagaimana Anda mengatasinya?",
        "verdict": "High Risk",
        "cheating_score": 12.16,
        "issues": [
          "Visual: Eyes off-screen (71 frames), Head turned away (15 frames)",
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
    "summary": "Analyzed 3 question(s). Average cheating score: 6.11%. Verdict distribution: Safe=1, Medium Risk=1, High Risk=1. 2 question(s) flagged with potential issues. Common visual indicators: Eyes off-screen, Head turned away. Average speakers detected: 1.0.",
    "final_aggregate_verdict": "Medium Risk"
  },
  "aggregate_non_verbal_analysis": {
    "overall_confidence_score": 77.2,
    "overall_performance_status": "GOOD",
    "summary": "speaking ratio 0.58 (fairly active), speech rate ~150 wpm (ideal), eye contact 85.2% (very good), blink rate 22.3/min (normal), smile intensity 0.18 (neutral), eyebrow movement 0.03 (controlled), pauses 13 (fluent)"
  },
  "content": [
    {
      "id": 1,
      "question": "Ceritakan tentang pengalaman Anda memimpin sebuah proyek atau tim. Apa tantangan yang Anda hadapi?",
      "result": {
        "penilaian": {
          "kualitas_jawaban": 68,
          "koherensi": 75,
          "relevansi": 69,
          "total_score": 70.66666666666667,
          "analisis_llm": "Kandidat memberikan gambaran yang cukup jelas tentang pengalaman memimpin tim dalam proyek pembangunan website untuk perusahaan. Tantangan yang disebutkan terkait dengan koordinasi tim dan penyelesaian masalah teknis menunjukkan pemahaman dasar tentang kompleksitas manajemen proyek. Namun, jawaban bisa lebih kuat jika kandidat memberikan contoh spesifik tentang bagaimana mereka mengatasi tantangan tersebut dan hasil akhir dari proyek. Penjelasan tentang strategi koordinasi atau penyelesaian masalah teknis yang konkret akan meningkatkan kualitas jawaban.",
          "logprobs_confidence": 91.8,
          "final_score_llm": 93.97142857142857
        },
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
        },
        "cheating_detection": {
          "visual": {
            "total_frames_analyzed": 709,
            "suspicious_frames": 0,
            "suspicious_percentage": 0.0,
            "face_detected_percentage": 100.0,
            "confidence_score": 100.0,
            "verdict": "Safe",
            "issues_detected": []
          },
          "audio": {
            "num_speakers_detected": 1,
            "silhouette_score": null,
            "confidence_score": 90.0,
            "verdict": "Safe",
            "details": "Only 1 speaker detected (candidate only)"
          },
          "final_verdict": "Safe",
          "overall_confidence_score": 95.0,
          "cheating_score": 0.0
        },
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
    },
    {
      "id": 2,
      "question": "Ceritakan pengalaman Anda dalam menangani konflik dalam tim. Apa langkah-langkah yang Anda ambil?",
      "result": {
        "penilaian": {
          "kualitas_jawaban": 88,
          "koherensi": 85,
          "relevansi": 86,
          "total_score": 86.33333333333333,
          "analisis_llm": "Kandidat memberikan jawaban yang sangat baik dan terstruktur. Mereka menunjukkan pemahaman yang kuat tentang pentingnya komunikasi terbuka dan pendekatan yang empatik dalam menyelesaikan konflik. Langkah-langkah yang dijelaskan - mendengarkan kedua belah pihak, mencari solusi bersama, dan memastikan semua pihak merasa didengarkan - menunjukkan kematangan dalam manajemen konflik. Jawaban ini koheren, relevan dengan pertanyaan, dan mencerminkan pengalaman praktis yang berharga dalam dinamika tim.",
          "logprobs_confidence": 85.7,
          "final_score_llm": 86.01428571428571
        },
        "non_verbal_analysis": {
          "speech_analysis": {
            "speaking_ratio": 0.59,
            "speaking_interpretation": "Fairly active",
            "speech_rate_wpm": 150.52,
            "speech_interpretation": "Ideal pace",
            "num_pauses": 14,
            "pause_interpretation": "Fluent",
            "confidence": 91.5
          },
          "facial_expression_analysis": {
            "smile_intensity": 0.18,
            "smile_interpretation": "Neutral",
            "eyebrow_movement": 0.03,
            "eyebrow_interpretation": "Controlled",
            "confidence": 78.6
          },
          "eye_movement_analysis": {
            "blink_rate_per_min": 21.4,
            "blink_interpretation": "Normal",
            "eye_contact_percentage": 84.7,
            "eye_contact_interpretation": "Very good",
            "confidence": 88.2
          },
          "non_verbal_confidence_score": 86.1,
          "performance_status": "EXCELLENT"
        },
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
        },
        "transkripsi_en": "In my previous job, there was a conflict between two team members regarding the division of tasks. I took steps to listen to both sides, understand their concerns, and then we discussed together to find a solution that was fair to both parties. I made sure that everyone felt heard and appreciated.",
        "transkripsi_id": "Di pekerjaan saya sebelumnya, ada konflik antara dua anggota tim terkait pembagian tugas. Saya mengambil langkah untuk mendengarkan kedua belah pihak, memahami kekhawatiran mereka, dan kemudian kami berdiskusi bersama untuk menemukan solusi yang adil bagi kedua belah pihak. Saya memastikan bahwa semua orang merasa didengarkan dan dihargai.",
        "metadata": {
          "video_file": "video_2.mp4",
          "audio_file": "audio_2.wav",
          "word_count": 215,
          "duration": 56.96,
          "avg_confidence": 98.89353488372093,
          "min_confidence": 93.652344,
          "max_confidence": 99.902344,
          "language": "en",
          "translation_used": true,
          "model": "faster-whisper large-v3",
          "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
          "deepl_used": true,
          "hf_api_used": true
        }
      }
    },
    {
      "id": 3,
      "question": "Apa kelemahan terbesar Anda dan bagaimana Anda mengatasinya?",
      "result": {
        "penilaian": {
          "kualitas_jawaban": 50,
          "koherensi": 55,
          "relevansi": 55,
          "total_score": 53.333333333333336,
          "analisis_llm": "Jawaban kandidat sangat singkat dan kurang mendalam. Meskipun mereka mengidentifikasi 'kurang percaya diri' sebagai kelemahan, tidak ada penjelasan konkret tentang bagaimana mereka mengatasinya atau langkah-langkah spesifik yang telah diambil untuk perbaikan. Jawaban ini terasa lebih seperti pernyataan umum daripada refleksi diri yang bermakna. Kandidat seharusnya memberikan contoh situasi di mana kelemahan ini muncul dan strategi yang telah mereka terapkan untuk mengembangkan kepercayaan diri mereka.",
          "logprobs_confidence": 93.4,
          "final_score_llm": 89.6
        },
        "non_verbal_analysis": {
          "speech_analysis": {
            "speaking_ratio": 0.56,
            "speaking_interpretation": "Fairly active",
            "speech_rate_wpm": 149.56,
            "speech_interpretation": "Ideal pace",
            "num_pauses": 13,
            "pause_interpretation": "Fluent",
            "confidence": 91.1
          },
          "facial_expression_analysis": {
            "smile_intensity": 0.19,
            "smile_interpretation": "Neutral",
            "eyebrow_movement": 0.02,
            "eyebrow_interpretation": "Controlled",
            "confidence": 78.3
          },
          "eye_movement_analysis": {
            "blink_rate_per_min": 22.6,
            "blink_interpretation": "Normal",
            "eye_contact_percentage": 84.5,
            "eye_contact_interpretation": "Very good",
            "confidence": 88.4
          },
          "non_verbal_confidence_score": 85.93333333333334,
          "performance_status": "EXCELLENT"
        },
        "cheating_detection": {
          "visual": {
            "total_frames_analyzed": 709,
            "suspicious_frames": 86,
            "suspicious_percentage": 12.12976473842055,
            "face_detected_percentage": 96.89422424824119,
            "confidence_score": 87.87023526157946,
            "verdict": "High Risk",
            "issues_detected": [
              "Eyes off-screen: 71 frames (10.01%)",
              "Head turned away: 15 frames (2.12%)"
            ]
          },
          "audio": {
            "num_speakers_detected": 1,
            "silhouette_score": null,
            "confidence_score": 90.0,
            "verdict": "Safe",
            "details": "Only 1 speaker detected (candidate only)"
          },
          "final_verdict": "High Risk",
          "overall_confidence_score": 88.93511763078973,
          "cheating_score": 12.12976473842055
        },
        "transkripsi_en": "My biggest weakness is that I am less confident in facing new challenges, but I am working to overcome it.",
        "transkripsi_id": "Kelemahan terbesar saya adalah kurang percaya diri dalam menghadapi tantangan baru, tetapi saya sedang berusaha untuk mengatasinya.",
        "metadata": {
          "video_file": "video_3.mp4",
          "audio_file": "audio_3.wav",
          "word_count": 82,
          "duration": 21.12,
          "avg_confidence": 98.71707317073171,
          "min_confidence": 95.410156,
          "max_confidence": 99.853516,
          "language": "en",
          "translation_used": true,
          "model": "faster-whisper large-v3",
          "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
          "deepl_used": true,
          "hf_api_used": true
        }
      }
    }
  ],
  "metadata": {
    "total_videos": 3,
    "successful_videos": 3,
    "failed_videos": 0,
    "total_duration_seconds": 123.264,
    "processed_at": "2025-01-27T04:17:02.698779+00:00",
    "model": "faster-whisper large-v3",
    "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
    "deepl_used": true,
    "hf_api_used": true
  }
}
```

**Response (404 Not Found):**

```json
{
  "detail": "Results not found for session: abc123def456..."
}
```

---

### GET / (Health Check)

Simple health check endpoint untuk verify server is running.

**Endpoint:**

```http
GET /
```

**Example Request:**

```bash
curl http://localhost:8888/
```

**Response (200 OK):**

```json
{
  "status": "ok",
  "message": "FastAPI server is running",
  "version": "1.0.0",
  "endpoints": [
    "POST /upload",
    "POST /upload_json",
    "GET /status/{session_id}",
    "GET /results/{session_id}"
  ]
}
```

---

## � Complete Workflow Example

### Scenario: 3-Video Interview Assessment

**Step 1: Upload Videos**

```python
import requests
import time
import json

# Upload via Google Drive URLs

url = "http://localhost:8888/upload_json"
payload = {
"candidate_name": "Jane Smith",
"interviews": [
{
"positionId": 1,
"question": "Tell me about your Python experience",
"isVideoExist": True,
"recordedVideoUrl": "https://drive.google.com/file/d/FILE_ID_1/view"
},
{
"positionId": 2,
"question": "Explain async vs sync programming",
"isVideoExist": True,
"recordedVideoUrl": "https://drive.google.com/file/d/FILE_ID_2/view"
},
{
"positionId": 3,
"question": "How would you optimize a slow query?",
"isVideoExist": True,
"recordedVideoUrl": "https://drive.google.com/file/d/FILE_ID_3/view"
}
],
"language": "en"
}

response = requests.post(url, json=payload)
data = response.json()
session_id = data['session_id']
print(f"Session ID: {session_id}")
```

**Step 2: Poll Status**

```python

# Poll every 5 seconds until completed

status_url = f"http://localhost:8888/status/{session_id}"

while True:
response = requests.get(status_url)
status_data = response.json()

    print(f"[{time.strftime('%H:%M:%S')}] {status_data['status']}: {status_data.get('message', '')}")

    if status_data['status'] == 'completed':
        print(f"✅ Processing completed!")
        print(f"Redirect: {status_data['redirect']}")
        break
    elif status_data['status'] == 'error':
        print(f"❌ Error: {status_data.get('error')}")
        break

    time.sleep(5)

```

**Step 3: Retrieve Results**

```python

# Get complete results

results_url = f"http://localhost:8888/results/{session_id}"
response = requests.get(results_url)
results = response.json()

# Save to file

with open(f'assessment_{session_id}.json', 'w', encoding='utf-8') as f:
json.dump(results, f, indent=2, ensure_ascii=False)

# Extract key metrics

print("\n=== ASSESSMENT SUMMARY ===")
print(f"Candidate: {results['name']}")
print(f"Total Videos: {results['metadata']['total_videos']}")

print(f"\n📊 LLM Assessment:")
print(f" Final Score: {results['llm_results']['final_score_llm']:.1f}/100")
print(f" Confidence: {results['llm_results']['avg_logprobs_confidence']:.1f}%")

print(f"\n🛡️ Cheating Detection:")
print(f" Verdict: {results['aggregate_cheating_detection']['final_aggregate_verdict']}")
print(f" Score: {results['aggregate_cheating_detection']['avg_cheating_score']:.1f}%")

print(f"\n😊 Non-Verbal Analysis:")
print(f" Status: {results['aggregate_non_verbal_analysis']['overall_performance_status']}")
print(f" Confidence: {results['aggregate_non_verbal_analysis']['overall_confidence_score']:.1f}%")

# Per-question details

print(f"\n📝 Per-Question Scores:")
for item in results['content']:
llm_score = item['result']['penilaian']['total']
cheating_score = item['result']['cheating_detection']['cheating_score']
print(f" Q{item['id']}: LLM={llm_score:.1f}, Cheating={cheating_score:.1f}%")
```

**Expected Output:**

```text
Session ID: abc123def456...

[10:30:00] processing: Downloading video from Google Drive...
[10:30:15] processing: Extracting audio from video...
[10:30:25] processing: Transcribing video 1/3...
[10:31:20] processing: Performing LLM assessment...
[10:31:45] processing: Detecting cheating indicators...
[10:32:50] processing: Transcribing video 2/3...
[10:33:40] processing: Performing LLM assessment...
[10:34:05] processing: Detecting cheating indicators...
[10:35:10] processing: Transcribing video 3/3...
[10:36:00] processing: Performing LLM assessment...
[10:36:25] processing: Detecting cheating indicators...
[10:37:30] processing: Saving results...
[10:37:35] completed: Processing completed successfully
✅ Processing completed!
Redirect: halaman_dasboard.html?session=abc123def456...

=== ASSESSMENT SUMMARY ===
Candidate: Jane Smith
Total Videos: 3

📊 LLM Assessment:
 Final Score: 85.5/100
 Confidence: 89.2%

🛡️ Cheating Detection:
 Verdict: Safe
 Score: 2.3%

😊 Non-Verbal Analysis:
 Status: GOOD
 Confidence: 77.5%

📝 Per-Question Scores:
 Q1: LLM=87.5, Cheating=1.8%
 Q2: LLM=91.2, Cheating=2.1%
 Q3: LLM=78.9, Cheating=3.0%
```

## 📋 Response Schema Details

### Session ID Format

UUID v4 format (32 hex characters):

```text
abc123def456789012345678901234567890
```

### Status Values

| Status         | Description                      | Next Action         |
| -------------- | -------------------------------- | ------------------- |
| `"processing"` | Video is being processed         | Continue polling    |
| `"completed"`  | Processing finished successfully | Fetch results       |
| `"error"`      | An error occurred                | Check error message |

### Verdict Values

**Cheating Detection:**

| Verdict         | Cheating Score | Description                     |
| --------------- | -------------- | ------------------------------- |
| `"Safe"`        | < 5%           | No suspicious activity          |
| `"Medium Risk"` | 5-20%          | Some suspicious frames          |
| `"High Risk"`   | ≥ 20%          | Significant suspicious activity |

**Non-Verbal Performance:**

| Status        | Confidence | Description             |
| ------------- | ---------- | ----------------------- |
| `"EXCELLENT"` | ≥ 85%      | Outstanding performance |
| `"GOOD"`      | 70-85%     | Good performance        |
| `"FAIR"`      | 50-70%     | Acceptable performance  |
| `"POOR"`      | < 50%      | Needs improvement       |

---

## 🚨 Error Handling

### Common Error Responses

**400 Bad Request:**

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Examples:**

- `"No videos uploaded"`
- `"Number of questions does not match number of videos"`
- `"Invalid language code. Use 'en' or 'id'"`
- `"Failed to extract file ID from URL"`

**404 Not Found:**

```json
{
  "detail": "Session not found"
}
```

**500 Internal Server Error:**

```json
{
  "detail": "Error processing videos: [detailed error message]"
}
```

**Examples:**

- `"FFmpeg not found. Please install FFmpeg"`
- `"CUDA out of memory"`
- `"DeepL API key invalid"`
- `"Hugging Face API rate limit exceeded"`

### Error Recovery

**Retry Strategy:**

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries():
"""Create requests session with automatic retries."""
session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# Usage

session = create_session_with_retries()
response = session.post(url, json=payload)
```

**Timeout Handling:**

```python
import requests
from requests.exceptions import Timeout, ConnectionError

try:
response = requests.post(
url,
json=payload,
timeout=30 # 30 second timeout
)
response.raise_for_status()
data = response.json()

except Timeout:
print("Request timed out. Server might be busy.")

except ConnectionError:
print("Could not connect to server. Is it running?")

except requests.exceptions.HTTPError as e:
print(f"HTTP error: {e}")
print(f"Response: {e.response.text}")
```

---

## 🔐 CORS Configuration

Server has CORS enabled untuk allow frontend requests:

**Allowed Origins:**

```python
allow_origins=["*"]  # All origins allowed (development)
```

**Allowed Methods:**

```python
allow_methods=["GET", "POST", "PUT", "DELETE"]
```

**Allowed Headers:**

```python
allow_headers=["*"]
```

**Production Recommendation:**

```python

# In production, specify exact origins

app.add_middleware(
CORSMiddleware,
allow_origins=[
"http://localhost:5500",
"https://yourdomain.com"
],
allow_credentials=True,
allow_methods=["GET", "POST"],
allow_headers=["Content-Type"]
)
```

---

## 📊 Performance Considerations

### Request Timeouts

Recommended timeout values:

| Endpoint       | Timeout | Reason                         |
| -------------- | ------- | ------------------------------ |
| `/upload`      | 60s     | Large file upload              |
| `/upload_json` | 30s     | JSON processing + Google Drive |
| `/status`      | 10s     | Quick status check             |
| `/results`     | 30s     | Large JSON response            |

### Rate Limiting

**Current Implementation:**

- No rate limiting (development mode)
- All requests processed

**Production Recommendation:**

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/upload")
@limiter.limit("10/hour") # 10 uploads per hour per IP
async def upload(...):
...

@app.get("/status/{session_id}")
@limiter.limit("60/minute") # Status polling
async def status(...):
...
```

### Response Size

**Typical Response Sizes:**

| Endpoint   | Size      | Notes                    |
| ---------- | --------- | ------------------------ |
| `/upload`  | < 1 KB    | Small JSON               |
| `/status`  | < 1 KB    | Small JSON               |
| `/results` | 50-200 KB | Large JSON with all data |

**Compression:**

Server automatically compresses responses > 1KB using gzip.

---

## 🔧 Development Tools

### Interactive API Documentation

**Swagger UI:**

```http
http://localhost:8888/docs
```

**Features:**

- Try endpoints directly in browser
- See request/response schemas
- Test authentication
- Download OpenAPI spec

**ReDoc:**

```http
http://localhost:8888/redoc
```

**Features:**

- Clean documentation interface
- Search functionality
- Code samples
- API reference

### OpenAPI Schema

Download OpenAPI 3.0 schema:

```bash
curl http://localhost:8888/openapi.json > openapi.json
```

Use for:

- Generate client SDKs
- Import into Postman
- API testing tools
- Documentation generators

---

## 📚 Related Documentation

**Next Steps:**

- [Request/Response Examples](request-response.md) - More detailed examples
- [Error Reference](errors.md) - Complete error codes and solutions
- [Configuration](../configuration/api-keys.md) - Setup API keys
- [Troubleshooting](../troubleshooting/common-issues.md) - Fix common issues

**Integration Guides:**

- [Quick Start](../getting-started/quickstart.md) - Your first assessment
- [Features Overview](../features/overview.md) - Understand all features
- [Configuration Guide](../configuration/models.md) - Customize settings
