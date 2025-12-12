# app/utils/non_verbal.py

import os
import cv2
import math
import time
import numpy as np
import subprocess
import mediapipe as mp

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# ============================================================
# CONFIG
# ============================================================

FRAME_SKIP = 5
MAX_FRAMES = 300
EARLY_EXIT_THRESHOLD = 30
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
CALIBRATION_FRAMES = 60
USE_CALIBRATION = True

# ============================================================
# STATISTICAL BASELINES (OPTIMIZED)
# ============================================================

STATS = {
    "blink_rate_per_minute": {"mean": 17, "sd": 10, "reliability": 0.88},
    "eye_contact_percentage": {"mean": 65, "sd": 20, "reliability": 0.84},
    "average_smile_intensity": {"mean": 0.18, "sd": 0.14, "reliability": 0.78},
    "eyebrow_movement_range": {"mean": 0.025, "sd": 0.018, "reliability": 0.75},
    "head_movement_intensity": {"mean": 0.5, "sd": 0.30, "reliability": 0.82},
    "speaking_ratio": {"mean": 0.58, "sd": 0.22, "reliability": 0.90},
    "speech_rate_wpm": {"mean": 145, "sd": 30, "reliability": 0.92},
}

# Weighted scoring per metric
WEIGHTS = {
    "speech_rate_wpm": 0.26,
    "speaking_ratio": 0.24,
    "blink_rate_per_minute": 0.18,
    "eye_contact_percentage": 0.16,
    "head_movement_intensity": 0.10,
    "average_smile_intensity": 0.04,
    "eyebrow_movement_range": 0.02,
}

# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio_fixed(video_path, audio_output_path="temp_audio.wav"):
    """Ekstrak audio menggunakan FFmpeg dengan optimasi"""
    try:
        print(f"   ⏳ Mengekstrak audio dari {video_path}...")

        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # Turunkan dari 44100 ke 16000 (cukup untuk speech)
            '-ac', '1',      # Mono, bukan stereo
            '-y',
            audio_output_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if os.path.exists(audio_output_path):
            print(f"   ✅ Audio berhasil diekstrak: {audio_output_path}")
            return audio_output_path
        else:
            raise Exception("Audio extraction failed")

    except Exception as e:
        print(f"   ❌ Error ekstraksi audio: {str(e)}")
        return None
    
# ============================================================
# SPEECH TEMPO ANALYSIS
# ============================================================

def analyze_speech_tempo(audio_path):
    """Speech analysis dengan error handling"""
    try:
        audio = AudioSegment.from_file(audio_path)

        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=500,
            silence_thresh=-40
        )

        total_speaking_time = sum([(end - start) for start, end in nonsilent_ranges]) / 1000
        total_duration = len(audio) / 1000
        num_pauses = len(nonsilent_ranges) - 1

        estimated_words = total_speaking_time * 2.5
        speech_rate = (estimated_words / total_speaking_time) * 60 if total_speaking_time > 0 else 0

        return {
            "total_duration_seconds": round(total_duration, 2),
            "speaking_time_seconds": round(total_speaking_time, 2),
            "silence_time_seconds": round(total_duration - total_speaking_time, 2),
            "number_of_pauses": num_pauses,
            "speech_rate_wpm": round(speech_rate, 2),
            "speaking_ratio": round(total_speaking_time / total_duration, 2) if total_duration > 0 else 0
        }
    except Exception as e:
        print(f"   ⚠️ Speech analysis error: {e}")
        return {
            "total_duration_seconds": 0,
            "speaking_time_seconds": 0,
            "silence_time_seconds": 0,
            "number_of_pauses": 0,
            "speech_rate_wpm": 0,
            "speaking_ratio": 0
        }
        
# ============================================================
# FACIAL EXPRESSION ANALYSIS (WITH CALIBRATION)
# ============================================================

def analyze_facial_expressions(video_path):
    """OPTIMIZED: Frame skipping, early exit, simplified tracking + CALIBRATION"""
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        refine_landmarks=False  # ⚡ CRITICAL: Matikan iris tracking
    )

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   📹 Video: {total_frames} frames @ {fps} FPS")
    print(f"   ⚡ Processing every {FRAME_SKIP} frames (max {MAX_FRAMES} frames)")

    expression_data = {
        "smile_intensity": [],
        "eyebrow_movement": [],
        "head_pose": []
    }

    # 🎯 CALIBRATION: Simpan data awal untuk baseline
    calibration_data = {
        "smile_intensity": [],
        "eyebrow_movement": []
    }

    frame_count = 0
    processed_count = 0
    no_face_count = 0
    is_calibration_phase = USE_CALIBRATION

    while cap.isOpened() and processed_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        if no_face_count >= EARLY_EXIT_THRESHOLD:
            print(f"   ⚠️ No face detected for {EARLY_EXIT_THRESHOLD} consecutive frames, stopping...")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            no_face_count = 0
            landmarks = results.multi_face_landmarks[0]

            left_mouth = landmarks.landmark[61]
            right_mouth = landmarks.landmark[291]
            smile_width = abs(right_mouth.x - left_mouth.x)

            left_eyebrow = landmarks.landmark[70]
            right_eyebrow = landmarks.landmark[300]
            eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2

            nose_tip = landmarks.landmark[1]

            # 🎯 CALIBRATION PHASE: Kumpulkan baseline data
            if is_calibration_phase and processed_count < CALIBRATION_FRAMES:
                calibration_data["smile_intensity"].append(smile_width)
                calibration_data["eyebrow_movement"].append(eyebrow_height)

                if processed_count == CALIBRATION_FRAMES - 1:
                    print(f"   ✅ Calibration complete using {CALIBRATION_FRAMES} frames")
                    is_calibration_phase = False

            # Simpan data normal
            expression_data["smile_intensity"].append(smile_width)
            expression_data["eyebrow_movement"].append(eyebrow_height)
            expression_data["head_pose"].append({
                "x": nose_tip.x,
                "y": nose_tip.y,
                "z": nose_tip.z
            })

            processed_count += 1
        else:
            no_face_count += 1

        if processed_count % 20 == 0 and processed_count > 0:
            print(f"   ... processed {processed_count} frames")

    cap.release()
    face_mesh.close()

    if len(expression_data["smile_intensity"]) == 0:
        print("   ⚠️ No face detected in entire video")
        return {
            "average_smile_intensity": 0,
            "smile_variation": 0,
            "eyebrow_movement_range": 0,
            "total_frames_analyzed": frame_count,
            "face_detected_percentage": 0,
            "calibration_applied": False
        }

    # 🎯 APPLY CALIBRATION: Normalize berdasarkan baseline
    baseline_smile = np.mean(calibration_data["smile_intensity"]) if calibration_data["smile_intensity"] else 0
    baseline_eyebrow = np.mean(calibration_data["eyebrow_movement"]) if calibration_data["eyebrow_movement"] else 0

    calibration_applied = USE_CALIBRATION and len(calibration_data["smile_intensity"]) > 0

    if calibration_applied:
        # Normalize: subtract baseline untuk mengukur perubahan dari neutral state
        calibrated_smiles = [abs(s - baseline_smile) for s in expression_data["smile_intensity"]]
        calibrated_eyebrows = [abs(e - baseline_eyebrow) for e in expression_data["eyebrow_movement"]]

        print(f"   🎯 Calibration baseline - Smile: {baseline_smile:.4f}, Eyebrow: {baseline_eyebrow:.4f}")

        return {
            "average_smile_intensity": round(np.mean(calibrated_smiles), 4),
            "smile_variation": round(np.std(calibrated_smiles), 4),
            "eyebrow_movement_range": round(np.std(calibrated_eyebrows), 4),
            "baseline_smile_intensity": round(baseline_smile, 4),
            "baseline_eyebrow_position": round(baseline_eyebrow, 4),
            "total_frames_analyzed": frame_count,
            "face_detected_percentage": round(len(expression_data["smile_intensity"]) / (frame_count / FRAME_SKIP) * 100, 2),
            "calibration_applied": True
        }
    else:
        return {
            "average_smile_intensity": round(np.mean(expression_data["smile_intensity"]), 4),
            "smile_variation": round(np.std(expression_data["smile_intensity"]), 4),
            "eyebrow_movement_range": round(np.std(expression_data["eyebrow_movement"]), 4),
            "total_frames_analyzed": frame_count,
            "face_detected_percentage": round(len(expression_data["smile_intensity"]) / (frame_count / FRAME_SKIP) * 100, 2),
            "calibration_applied": False
        }

# ============================================================
# EYE MOVEMENT ANALYSIS
# ============================================================

def analyze_eye_movement(video_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True  # Penting untuk deteksi iris
    )

    cap = cv2.VideoCapture(video_path)

    eye_data = {
        "gaze_positions": [],
        "blink_count": 0,
        "eye_contact_percentage": 0
    }

    prev_eye_closed = False
    frame_count = 0
    direct_gaze_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Eye landmarks (mata kiri: 33, 133; mata kanan: 362, 263)
            left_eye_top = landmarks.landmark[159]
            left_eye_bottom = landmarks.landmark[145]
            right_eye_top = landmarks.landmark[386]
            right_eye_bottom = landmarks.landmark[374]

            # Deteksi kedipan (Eye Aspect Ratio)
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            avg_eye_height = (left_eye_height + right_eye_height) / 2

            # Threshold untuk mata tertutup
            eye_closed = avg_eye_height < 0.01

            if eye_closed and not prev_eye_closed:
                eye_data["blink_count"] += 1

            prev_eye_closed = eye_closed

            # Iris tracking untuk gaze direction
            # Iris center landmarks: 468-473
            if len(landmarks.landmark) > 473:
                left_iris = landmarks.landmark[468]
                right_iris = landmarks.landmark[473]

                # Simpan posisi gaze
                gaze_x = (left_iris.x + right_iris.x) / 2
                gaze_y = (left_iris.y + right_iris.y) / 2
                eye_data["gaze_positions"].append({"x": gaze_x, "y": gaze_y})

                # Deteksi eye contact (gaze ke tengah frame)
                if 0.4 < gaze_x < 0.6 and 0.3 < gaze_y < 0.7:
                    direct_gaze_count += 1

    cap.release()

    if frame_count > 0:
        eye_data["eye_contact_percentage"] = round((direct_gaze_count / frame_count) * 100, 2)
        eye_data["blink_rate_per_minute"] = round((eye_data["blink_count"] / frame_count) * (30 * 60), 2)

    return {
        "total_blinks": eye_data["blink_count"],
        "blink_rate_per_minute": eye_data.get("blink_rate_per_minute", 0),
        "eye_contact_percentage": eye_data["eye_contact_percentage"],
        "gaze_stability": round(np.std([g["x"] for g in eye_data["gaze_positions"]]), 4) if eye_data["gaze_positions"] else 0
    }

# ============================================================
# Z-SCORE + CONFIDENCE SCORING
# ============================================================

def score_conf(metric, value):
    if metric not in STATS:
        return 0, 0, 0

    mean = STATS[metric]["mean"]
    sd = STATS[metric]["sd"]
    reliability = STATS[metric]["reliability"]

    z = (value - mean) / sd
    base = math.exp(-(z**2) / 2)

    adjusted = base * reliability
    uncertainty = (1 - reliability) * 100

    return z, adjusted, uncertainty

# ============================================================
# INTERPRETATION ENGINE
# ============================================================

def interpret_non_verbal_analysis(analysis_json):
    """Interpretasi hasil analisis non-verbal dalam format sederhana"""
    interpretations = {}

    # Analisis bicara
    speech = analysis_json.get("speech_analysis", {})
    if speech:
        speaking_ratio = speech.get("speaking_ratio", 0) or speech.get("avg_speaking_ratio", 0)
        pauses = speech.get("number_of_pauses", 0) or speech.get("avg_pauses", 0)
        rate = speech.get("speech_rate_wpm", 0) or speech.get("avg_speech_rate", 0)

        if speaking_ratio > 0.65:
            speaking_label = "very active"
        elif speaking_ratio > 0.5:
            speaking_label = "fairly active"
        else:
            speaking_label = "least active"

        if pauses > 40:
            pause_label = "frequent pauses"
        elif pauses > 25:
            pause_label = "normal"
        else:
            pause_label = "fluent"

        if 135 <= rate <= 165:
            rate_label = "ideal"
        elif rate > 165:
            rate_label = "fast"
        else:
            rate_label = "slow"

        interpretations["speech_analysis"] = (
            f"speaking ratio {speaking_ratio:.2f} ({speaking_label}), "
            f"pauses {pauses} ({pause_label}), "
            f"speech rate {rate} wpm ({rate_label})"
        )

    # Analisis ekspresi wajah
    facial = analysis_json.get("facial_expression_analysis", {})
    if facial:
        smile_intensity = facial.get("average_smile_intensity", 0) or facial.get("avg_smile_intensity", 0)
        eyebrow_range = facial.get("eyebrow_movement_range", 0) or facial.get("avg_eyebrow_movement_range", 0)

        if eyebrow_range > 0.035:
            eyebrow_label = "expressive"
        elif eyebrow_range > 0.018:
            eyebrow_label = "natural"
        else:
            eyebrow_label = "controlled"

        if smile_intensity > 0.25:
            smile_label = "positive"
        elif smile_intensity > 0.12:
            smile_label = "friendly"
        else:
            smile_label = "neutral"

        interpretations["facial_expression_analysis"] = (
            f"smile intensity = {smile_intensity:.2f} ({smile_label}), "
            f"eyebrow movement = {eyebrow_range:.3f} ({eyebrow_label})"
        )

    # Analisis gerakan mata
    eye = analysis_json.get("eye_movement_analysis", {})
    if eye:
        blink_rate = eye.get("blink_rate_per_minute", 0) or eye.get("avg_blink_rate", 0)
        eye_contact = eye.get("eye_contact_percentage", 0) or eye.get("avg_eye_contact", 0)

        if eye_contact > 75:
            contact_label = "very good"
        elif eye_contact > 55:
            contact_label = "good"
        else:
            contact_label = "needs improvement"

        if blink_rate > 25:
            blink_label = "high"
        elif blink_rate > 10:
            blink_label = "normal"
        else:
            blink_label = "low"

        interpretations["eye_movement_analysis"] = (
            f"eye contact = {eye_contact}% ({contact_label}), "
            f"blink rate = {blink_rate} ({blink_label})"
        )

    return interpretations

# ============================================================
# SCIENTIFIC CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence_scientific(analysis_json):
    """Hitung confidence score dengan scientific rigor"""
    confidence_per_metric = {}
    uncertainty_per_metric = {}
    total_conf = 0.0
    total_uncertainty = 0.0

    for metric in WEIGHTS.keys():
        value = None
        if metric in analysis_json.get("speech_analysis", {}):
            value = analysis_json["speech_analysis"].get(metric)
        elif metric in analysis_json.get("facial_expression_analysis", {}):
            value = analysis_json["facial_expression_analysis"].get(metric)
        elif metric in analysis_json.get("eye_movement_analysis", {}):
            value = analysis_json["eye_movement_analysis"].get(metric)
        elif metric in analysis_json.get("head_movement_analysis", {}):
            value = analysis_json["head_movement_analysis"].get(metric)

        if value is not None:
            _, conf, uncertainty = score_conf(metric, value)
            confidence_per_metric[metric] = round(conf * 100, 2)
            uncertainty_per_metric[metric] = round(uncertainty, 2)
            total_conf += conf * WEIGHTS[metric]
            total_uncertainty += uncertainty * WEIGHTS[metric]

    raw_score = total_conf * 100
    scaled_score = 50 + (raw_score * 0.50)

    total_confidence_percent = round(scaled_score, 2)
    total_uncertainty_percent = round(total_uncertainty, 2)

    lower_bound = round(max(0, total_confidence_percent - total_uncertainty_percent), 2)
    upper_bound = round(min(100, total_confidence_percent + total_uncertainty_percent), 2)

    if total_confidence_percent >= 80:
        confidence_level = "High"
        interpretation = "Model prediksi sangat reliable"
    elif total_confidence_percent >= 70:
        confidence_level = "Good"
        interpretation = "Model prediksi reliable untuk decision-making"
    elif total_confidence_percent >= 60:
        confidence_level = "Moderate"
        interpretation = "Model prediksi cukup reliable, pertimbangkan faktor tambahan"
    elif total_confidence_percent >= 50:
        confidence_level = "Fair"
        interpretation = "Model prediksi perlu dukungan data tambahan"
    else:
        confidence_level = "Low"
        interpretation = "Confidence rendah, perlukan verifikasi manual"

    return {
        "confidence_per_metric": confidence_per_metric,
        "uncertainty_per_metric": uncertainty_per_metric,
        "total_confidence_score": total_confidence_percent,
        "confidence_interval": {
            "lower": lower_bound,
            "upper": upper_bound,
            "margin_of_error": total_uncertainty_percent
        },
        "confidence_level": confidence_level,
        "interpretation": interpretation,
        "reliability_notes": f"Confidence interval: [{lower_bound}% - {upper_bound}%] dengan margin of error ±{total_uncertainty_percent}%"
    }

# ============================================================
# PERFORMANCE LEVEL
# ============================================================

def get_performance_level(avg_confidence):
    """Tentukan level performa berdasarkan confidence score"""
    if avg_confidence >= 80:
        return "Very High"
    elif avg_confidence >= 70:
        return "Good"
    elif avg_confidence >= 60:
        return "Average"
    elif avg_confidence >= 50:
        return "Below Average"
    else:
        return "Need Improvement"
    
# ============================================================
# FINAL RECOMMENDATION
# ============================================================

def get_recommendation(avg_confidence, confidence_interval, interpretations):
    """Generate rekomendasi berdasarkan analisis dengan transparency"""
    performance_level = get_performance_level(avg_confidence)
    lower = confidence_interval["lower"]
    upper = confidence_interval["upper"]

    if avg_confidence >= 75 and lower >= 68:
        return f"RECOMMEND - Performa non-verbal {performance_level.lower()} dengan high confidence (CI: {lower}-{upper}%)"
    elif avg_confidence >= 65 and lower >= 55:
        return f"CONSIDER - Performa non-verbal {performance_level.lower()} dengan moderate confidence (CI: {lower}-{upper}%)"
    elif avg_confidence >= 55:
        return f"REVIEW - Performa non-verbal {performance_level.lower()}, memerlukan evaluasi tambahan (CI: {lower}-{upper}%)"
    else:
        return f"NOT RECOMMEND - Performa non-verbal {performance_level.lower()} dengan low confidence (CI: {lower}-{upper}%)"

# ============================================================
# MAIN: FULL NON-VERBAL ANALYSIS PIPELINE
# ============================================================

def analyze_interview_video_with_confidence(video_path, audio_path=None):
    """Analisis video interview dengan optimasi penuh + scientific confidence scoring"""
    start_time = time.time()
    print("🎬 Memulai analisis interview (OPTIMIZED + SCIENTIFIC)...")
    
    # ✅ Track if we created temp file
    temp_audio_created = False
    
    if audio_path is None:
        print("📤 Mengekstrak audio dari video...")
        filename = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = f"{filename}_temp.wav"
        temp_audio_created = True  # ✅ Mark that we created it
        audio_path = extract_audio_fixed(video_path, audio_path)
        if not audio_path:
            return {
                'analysis': {},
                'confidence_score': 0,
                'confidence_level': 'Failed',
                'confidence_components': {},
                'interpretations': {},
                'processing_time_seconds': 0
            }

    print("\n📊 Analyzing speech...")
    speech_analysis = analyze_speech_tempo(audio_path)

    print("\n😊 Analyzing facial expressions...")
    facial_analysis = analyze_facial_expressions(video_path)

    print("\n👁️ Analyzing eye movement...")
    eye_analysis = analyze_eye_movement(video_path)

    analysis_result = {
        "speech_analysis": speech_analysis,
        "facial_expression_analysis": facial_analysis,
        "eye_movement_analysis": eye_analysis,
    }

    conf_result = calculate_confidence_scientific(analysis_result)
    interpretations = interpret_non_verbal_analysis(analysis_result)

    elapsed = time.time() - start_time

    print(f'\n✅ Non-Verbal Analysis Complete in {elapsed:.1f}s')
    print(f'   Confidence: {conf_result["total_confidence_score"]}% ({conf_result["confidence_level"]})')

    # ✅ CLEANUP: Delete temp audio file if we created it
    if temp_audio_created and audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024) if os.path.exists(audio_path) else 0
            print(f'   🗑️  Temp audio deleted: {os.path.basename(audio_path)} ({file_size_mb:.2f} MB freed)')
        except Exception as e:
            print(f'   ⚠️  Failed to delete temp audio: {str(e)}')

    return {
        'analysis': analysis_result,
        'confidence_score': conf_result['total_confidence_score'],
        'confidence_level': conf_result['confidence_level'],
        'confidence_components': conf_result['confidence_per_metric'],  # ✅ FIXED: Use confidence_per_metric instead of confidence_components
        'confidence_interval': conf_result['confidence_interval'],
        'interpretations': interpretations,
        'processing_time_seconds': round(elapsed, 2)
    }


# ============================================================
# BATCH SUMMARY
# ============================================================

def summarize_non_verbal_batch(assessment_results):
    """Ringkasan batch dengan scientific rigor dan transparency"""
    speaking_ratios, pauses, speech_rates = [], [], []
    smiles, eyebrows, eye_contacts, blink_rates = [], [], [], []
    confidence_scores = []
    all_intervals = []

    for item in assessment_results:
        nv = item["result"]["non_verbal_analysis"]

        sp = nv["speech_analysis"]
        speaking_ratios.append(sp["speaking_ratio"])
        pauses.append(sp["number_of_pauses"])
        speech_rates.append(sp["speech_rate_wpm"])

        fc = nv["facial_expression_analysis"]
        smiles.append(fc["average_smile_intensity"])
        eyebrows.append(fc["eyebrow_movement_range"])

        ey = nv["eye_movement_analysis"]
        eye_contacts.append(ey["eye_contact_percentage"])
        blink_rates.append(ey["blink_rate_per_minute"])

        conf_result = calculate_confidence_scientific(nv)
        confidence_scores.append(conf_result["total_confidence_score"])
        all_intervals.append(conf_result["confidence_interval"])

    avg_confidence = round(np.mean(confidence_scores), 2) if confidence_scores else 0
    std_confidence = round(np.std(confidence_scores), 2) if confidence_scores else 0
    max_confidence = round(max(confidence_scores), 2) if confidence_scores else 0
    min_confidence = round(min(confidence_scores), 2) if confidence_scores else 0

    avg_lower = round(np.mean([ci["lower"] for ci in all_intervals]), 2)
    avg_upper = round(np.mean([ci["upper"] for ci in all_intervals]), 2)
    avg_margin = round(np.mean([ci["margin_of_error"] for ci in all_intervals]), 2)

    if avg_confidence >= 80:
        confidence_level = "High"
    elif avg_confidence >= 70:
        confidence_level = "Good"
    elif avg_confidence >= 60:
        confidence_level = "Moderate"
    elif avg_confidence >= 50:
        confidence_level = "Fair"
    else:
        confidence_level = "Low"

    aggregated_data = {
        "speech_analysis": {
            "avg_speaking_ratio": round(np.mean(speaking_ratios), 3),
            "avg_pauses": round(np.mean(pauses), 2),
            "avg_speech_rate": round(np.mean(speech_rates), 2)
        },
        "facial_expression_analysis": {
            "avg_smile_intensity": round(np.mean(smiles), 4),
            "avg_eyebrow_movement_range": round(np.mean(eyebrows), 4)
        },
        "eye_movement_analysis": {
            "avg_eye_contact": round(np.mean(eye_contacts), 2),
            "avg_blink_rate": round(np.mean(blink_rates), 2)
        }
    }

    interpretations = interpret_non_verbal_analysis(aggregated_data)
    summary_text = " ".join([
        interpretations.get("speech_analysis", ""),
        interpretations.get("facial_expression_analysis", ""),
        interpretations.get("eye_movement_analysis", "")
    ])

    poor_performance_count = sum(1 for score in confidence_scores if score < 60)
    poor_performance_percentage = round((poor_performance_count / len(confidence_scores) * 100), 2) if confidence_scores else 0

    performance_level = get_performance_level(avg_confidence)

    confidence_interval = {
        "lower": avg_lower,
        "upper": avg_upper,
        "margin_of_error": avg_margin
    }

    recommendation = get_recommendation(avg_confidence, confidence_interval, interpretations)

    return {
        "overall_performance_status": performance_level,
        "overall_confidence_score": avg_confidence,
        "summary": summary_text,
    }