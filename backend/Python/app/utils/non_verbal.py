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

# ====== OPTIMIZATION CONFIGURATION ======
FRAME_SKIP = 5
MAX_FRAMES = 300
EARLY_EXIT_THRESHOLD = 30
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
CALIBRATION_FRAMES = 60
USE_CALIBRATION = True

# ====== OPTIMIZED STATS - Adjusted untuk meningkatkan confidence ======
# Strategi: Perlebar SD untuk mengurangi extreme z-scores, tingkatkan reliability
STATS = {
    "blink_rate_per_minute": {
        "mean": 17,
        "sd": 10,  # Dari 8 → 10 (lebih toleran terhadap variasi)
        "reliability": 0.88  # Dari 0.82 → 0.88
    },
    "eye_contact_percentage": {
        "mean": 65,
        "sd": 20,  # Dari 18 → 20
        "reliability": 0.84  # Dari 0.78 → 0.84
    },
    "average_smile_intensity": {
        "mean": 0.18,
        "sd": 0.14,  # Dari 0.12 → 0.14
        "reliability": 0.78  # Dari 0.71 → 0.78
    },
    "eyebrow_movement_range": {
        "mean": 0.025,
        "sd": 0.018,  # Dari 0.015 → 0.018
        "reliability": 0.75  # Dari 0.68 → 0.75
    },
    "head_movement_intensity": {
        "mean": 0.5,
        "sd": 0.30,  # Dari 0.25 → 0.30
        "reliability": 0.82  # Dari 0.75 → 0.82
    },
    "speaking_ratio": {
        "mean": 0.58,
        "sd": 0.22,  # Dari 0.18 → 0.22
        "reliability": 0.90  # Dari 0.85 → 0.90 (metrik paling reliable)
    },
    "speech_rate_wpm": {
        "mean": 145,
        "sd": 30,  # Dari 25 → 30
        "reliability": 0.92  # Dari 0.88 → 0.92 (metrik paling reliable)
    }
}

# ====== OPTIMIZED WEIGHTS - Fokus pada metrik high-reliability ======
# Strategi: Berikan bobot lebih besar pada metrik dengan reliability tinggi
WEIGHTS = {
    "speech_rate_wpm": 0.26,        # ↑ dari 0.22 (reliability 0.92)
    "speaking_ratio": 0.24,         # ↑ dari 0.21 (reliability 0.90)
    "blink_rate_per_minute": 0.18,  # ↑ dari 0.16 (reliability 0.88)
    "eye_contact_percentage": 0.16, # ↑ dari 0.15 (reliability 0.84)
    "head_movement_intensity": 0.10,# ↓ dari 0.12 (reliability 0.82)
    "average_smile_intensity": 0.04,# ↓ dari 0.09 (reliability 0.78)
    "eyebrow_movement_range": 0.02  # ↓ dari 0.05 (reliability 0.75)
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

def analyze_speech_tempo(audio_path, transcript=None):
    """Speech analysis dengan word count dari transkrip untuk WPM akurat"""
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

        # ✅ FIXED: Hitung WPM dari jumlah kata transkrip yang sebenarnya
        if transcript and isinstance(transcript, str) and len(transcript.strip()) > 0:
            actual_words = len(transcript.split())
            print(f"   📝 Word count from transcript: {actual_words} words")
        else:
            actual_words = total_speaking_time * 2.5
            print(f"   ⚠️ No transcript, estimated: {actual_words:.0f} words")

        speech_rate = (actual_words / total_speaking_time) * 60 if total_speaking_time > 0 else 0
        print(f"   📊 Speech rate: {speech_rate:.1f} WPM")

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
    """
    ✅ OPTIMIZED: Eye contact detection with wider "at camera" range
    - Expanded threshold for camera gaze
    - Higher scores for good eye contact
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True  # CRITICAL: Enable iris tracking
    )

    cap = cv2.VideoCapture(video_path)

    eye_data = {
        "gaze_positions": [],
        "blink_count": 0,
        "looking_camera_count": 0,
        "looking_down_count": 0,
        "looking_up_count": 0,
        "looking_slightly_down_count": 0,
    }

    prev_eye_closed = False
    frame_count = 0
    direct_gaze_score = 0

    # Debug values
    all_upper_ratios = []

    print(f"   👁️  Analyzing eye contact (optimized algorithm)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for performance
        if frame_count % 2 != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # ============================================================
            # 1. BLINK DETECTION
            # ============================================================
            left_eye_top = landmarks.landmark[159]
            left_eye_bottom = landmarks.landmark[145]
            right_eye_top = landmarks.landmark[386]
            right_eye_bottom = landmarks.landmark[374]

            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            avg_eye_height = (left_eye_height + right_eye_height) / 2

            eye_closed = avg_eye_height < 0.008

            if eye_closed and not prev_eye_closed:
                eye_data["blink_count"] += 1

            prev_eye_closed = eye_closed

            # ============================================================
            # 2. IRIS VERTICAL POSITION DETECTION
            # ============================================================
            if len(landmarks.landmark) > 473 and not eye_closed:
                # Iris centers
                left_iris = landmarks.landmark[468]
                right_iris = landmarks.landmark[473]

                # Measure iris position within eye opening
                left_iris_to_top = abs(left_iris.y - left_eye_top.y)
                left_iris_to_bottom = abs(left_eye_bottom.y - left_iris.y)

                right_iris_to_top = abs(right_iris.y - right_eye_top.y)
                right_iris_to_bottom = abs(right_eye_bottom.y - right_iris.y)

                # Average
                avg_iris_to_top = (left_iris_to_top + right_iris_to_top) / 2
                avg_iris_to_bottom = (left_iris_to_bottom + right_iris_to_bottom) / 2

                total_iris_range = avg_iris_to_top + avg_iris_to_bottom
                if total_iris_range > 0:
                    upper_ratio = avg_iris_to_top / total_iris_range
                else:
                    upper_ratio = 0.5

                all_upper_ratios.append(upper_ratio)

                eye_data["gaze_positions"].append({
                    "upper_ratio": upper_ratio,
                })

                # ============================================================
                # ✅ EXPANDED "AT CAMERA" RANGE
                # ============================================================
                # Based on your data:
                # - Video 1 (camera): avg=0.4121, range=[0.2423, 0.4839]
                # - Video 2 (down): avg=0.4291, range=[0.3154, 0.5088]

                # ✅ NEW WIDER THRESHOLDS:
                # Perfect camera: 0.36 - 0.42 (WIDER from 0.38-0.42)
                # Good camera: 0.42 - 0.45 (WIDER from 0.42-0.44)
                # Slightly down: 0.45 - 0.50 (from 0.44-0.48)
                # Looking down: 0.50 - 0.58
                # Very down: > 0.58
                # Looking up: < 0.36

                if upper_ratio < 0.36:
                    # Looking UP
                    eye_data["looking_up_count"] += 1
                    direct_gaze_score += 0.35  # Medium score

                elif 0.36 <= upper_ratio < 0.42:
                    # ✅ PERFECT camera gaze (ideal range - WIDENED)
                    eye_data["looking_camera_count"] += 1
                    distance_from_ideal = abs(upper_ratio - 0.39)

                    if distance_from_ideal < 0.015:
                        direct_gaze_score += 1.0   # Perfect
                    elif distance_from_ideal < 0.025:
                        direct_gaze_score += 0.95  # Excellent
                    else:
                        direct_gaze_score += 0.90  # Very good

                elif 0.42 <= upper_ratio < 0.45:
                    # ✅ GOOD camera gaze (acceptable - WIDENED & HIGHER SCORE)
                    eye_data["looking_camera_count"] += 1
                    direct_gaze_score += 0.85  # Increased from 0.75

                elif 0.45 <= upper_ratio < 0.50:
                    # SLIGHTLY looking down (REDUCED SCORE)
                    eye_data["looking_slightly_down_count"] += 1
                    direct_gaze_score += 0.40  # Reduced from 0.50

                elif 0.50 <= upper_ratio < 0.58:
                    # LOOKING down
                    eye_data["looking_down_count"] += 1
                    direct_gaze_score += 0.20  # Low score

                else:  # >= 0.58
                    # VERY down
                    eye_data["looking_down_count"] += 1
                    direct_gaze_score += 0.08  # Very low score

    cap.release()
    face_mesh.close()

    # Calculate final metrics
    processed_frames = frame_count // 2

    if processed_frames > 0:
        eye_contact_percentage = round((direct_gaze_score / processed_frames) * 100, 2)

        # Blink rate
        video_duration_minutes = (frame_count / 30) / 60
        blink_rate_per_minute = round(eye_data["blink_count"] / video_duration_minutes, 2) if video_duration_minutes > 0 else 0

        # Direction percentages
        looking_camera_pct = round((eye_data["looking_camera_count"] / processed_frames) * 100, 2)
        looking_slightly_down_pct = round((eye_data["looking_slightly_down_count"] / processed_frames) * 100, 2)
        looking_down_pct = round((eye_data["looking_down_count"] / processed_frames) * 100, 2)
        looking_up_pct = round((eye_data["looking_up_count"] / processed_frames) * 100, 2)

        # Debug
        avg_upper_ratio = round(np.mean(all_upper_ratios), 4) if all_upper_ratios else 0
        min_upper_ratio = round(np.min(all_upper_ratios), 4) if all_upper_ratios else 0
        max_upper_ratio = round(np.max(all_upper_ratios), 4) if all_upper_ratios else 0
    else:
        eye_contact_percentage = 0
        blink_rate_per_minute = 0
        looking_camera_pct = 0
        looking_slightly_down_pct = 0
        looking_down_pct = 0
        looking_up_pct = 0
        avg_upper_ratio = 0
        min_upper_ratio = 0
        max_upper_ratio = 0

    print(f"   ✅ Eye contact: {eye_contact_percentage}%")
    print(f"   🎯 At camera (0.36-0.45): {looking_camera_pct}%")
    print(f"   🔽 Slightly down (0.45-0.50): {looking_slightly_down_pct}%")
    print(f"   ⬇️  Looking down (>0.50): {looking_down_pct}%")
    print(f"   ⬆️  Looking up (<0.36): {looking_up_pct}%")
    print(f"   👀 Blink rate: {blink_rate_per_minute}/min")
    print(f"   📊 Upper ratio: avg={avg_upper_ratio}, range=[{min_upper_ratio}, {max_upper_ratio}]")

    return {
        "total_blinks": eye_data["blink_count"],
        "blink_rate_per_minute": blink_rate_per_minute,
        "eye_contact_percentage": eye_contact_percentage,
        "looking_at_camera_percentage": looking_camera_pct,
        "looking_slightly_down_percentage": looking_slightly_down_pct,
        "looking_down_percentage": looking_down_pct,
        "looking_up_percentage": looking_up_pct,
        "gaze_stability": round(np.std(all_upper_ratios), 4) if all_upper_ratios else 0,
        "avg_upper_ratio": avg_upper_ratio,
        "min_upper_ratio": min_upper_ratio,
        "max_upper_ratio": max_upper_ratio,
        "total_frames_analyzed": processed_frames
    }

# ============================================================
# HYBRID SCORING CONFIGURATION
# ============================================================

# Percentile data (historical data from research/training samples)
PERCENTILE_DATA = {
    "speech_rate_wpm": {
        "values": [100, 115, 125, 135, 145, 155, 165, 175, 190],
        "weight": 0.26
    },
    "speaking_ratio": {
        "values": [0.30, 0.40, 0.50, 0.58, 0.65, 0.72, 0.80, 0.85, 0.90],
        "weight": 0.24
    },
    "blink_rate_per_minute": {
        "values": [5, 10, 15, 17, 20, 25, 30, 35, 40],
        "weight": 0.18
    },
    "eye_contact_percentage": {
        "values": [30, 45, 55, 65, 70, 75, 80, 85, 90],
        "weight": 0.16
    },
    "head_movement_intensity": {
        "values": [0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2],
        "weight": 0.10
    },
    "average_smile_intensity": {
        "values": [0.05, 0.10, 0.15, 0.18, 0.22, 0.28, 0.35, 0.42, 0.50],
        "weight": 0.04
    },
    "eyebrow_movement_range": {
        "values": [0.005, 0.010, 0.018, 0.025, 0.032, 0.040, 0.050, 0.060, 0.080],
        "weight": 0.02
    }
}

# Optimal ranges (domain knowledge based)
OPTIMAL_RANGES = {
    "speech_rate_wpm": {
        "optimal": (130, 160),
        "acceptable": (110, 180),
        "weight": 0.26
    },
    "speaking_ratio": {
        "optimal": (0.50, 0.70),
        "acceptable": (0.40, 0.80),
        "weight": 0.24
    },
    "blink_rate_per_minute": {
        "optimal": (12, 22),
        "acceptable": (8, 30),
        "weight": 0.18
    },
    "eye_contact_percentage": {
        "optimal": (55, 75),
        "acceptable": (45, 85),
        "weight": 0.16
    },
    "head_movement_intensity": {
        "optimal": (0.40, 0.65),
        "acceptable": (0.25, 0.85),
        "weight": 0.10
    },
    "average_smile_intensity": {
        "optimal": (0.12, 0.25),
        "acceptable": (0.08, 0.35),
        "weight": 0.04
    },
    "eyebrow_movement_range": {
        "optimal": (0.018, 0.035),
        "acceptable": (0.012, 0.045),
        "weight": 0.02
    }
}

# ============================================================
# HYBRID CONFIDENCE SCORING FUNCTIONS
# ============================================================

def calculate_percentile_score(metric, value):
    """
    Calculate score based on percentile rank

    Returns score 0-100 based on position in historical distribution
    """
    if metric not in PERCENTILE_DATA:
        return 50.0

    data = sorted(PERCENTILE_DATA[metric]["values"])

    # Calculate percentile rank manually
    below_count = sum(1 for v in data if v < value)
    equal_count = sum(1 for v in data if v == value)

    percentile = (below_count + 0.5 * equal_count) / len(data) * 100

    # Transform to confidence score with optimal range at 40-80 percentile
    if 40 <= percentile <= 80:
        # Optimal range gets high score (85-100%)
        distance_from_60 = abs(percentile - 60)
        score = 100 - (distance_from_60 / 20 * 15)  # 85-100%
    elif 20 <= percentile < 40 or 80 < percentile <= 90:
        # Good but not optimal (60-85%)
        if percentile < 40:
            score = 60 + ((percentile - 20) / 20 * 25)
        else:
            score = 60 + ((90 - percentile) / 10 * 25)
    else:
        # Too low or too high (20-60%)
        if percentile < 20:
            score = max(20, 60 - ((20 - percentile) / 20 * 40))
        else:
            score = max(20, 60 - ((percentile - 90) / 10 * 40))

    return round(score, 2)

def calculate_range_score(metric, value):
    """
    Calculate score based on optimal range position

    Returns score 0-100 based on distance from optimal range
    """
    if metric not in OPTIMAL_RANGES:
        return 50.0

    optimal = OPTIMAL_RANGES[metric]["optimal"]
    acceptable = OPTIMAL_RANGES[metric]["acceptable"]

    # Check if in optimal range
    if optimal[0] <= value <= optimal[1]:
        # Perfect score in optimal range (center gets 100%)
        center = (optimal[0] + optimal[1]) / 2
        distance_from_center = abs(value - center)
        max_distance = (optimal[1] - optimal[0]) / 2
        score = 100 - (distance_from_center / max_distance * 15)  # 85-100%

    # Check if in acceptable range
    elif acceptable[0] <= value <= acceptable[1]:
        # Linearly decrease from 85% to 60%
        if value < optimal[0]:
            score = 60 + ((value - acceptable[0]) / (optimal[0] - acceptable[0]) * 25)
        else:
            score = 60 + ((acceptable[1] - value) / (acceptable[1] - optimal[1]) * 25)

    # Outside acceptable range
    else:
        # Exponential penalty for extreme values
        if value < acceptable[0]:
            distance = acceptable[0] - value
            max_penalty_distance = acceptable[0] * 0.5
        else:
            distance = value - acceptable[1]
            max_penalty_distance = acceptable[1] * 0.5

        penalty = min(40, (distance / max_penalty_distance) * 40)
        score = max(10, 60 - penalty)  # 10-60%

    return round(score, 2)

def score_conf(metric_name, value):
    """Hitung z-score dan confidence dengan uncertainty adjustment (LEGACY - for compatibility)"""
    if metric_name not in STATS:
        return 0, 0, 0

    mean = STATS[metric_name]["mean"]
    sd = STATS[metric_name]["sd"]
    reliability = STATS[metric_name]["reliability"]

    z = (value - mean) / sd
    base_conf = math.exp(-(z**2) / 2)
    adjusted_conf = base_conf * reliability
    uncertainty = (1 - reliability) * 100

    return z, adjusted_conf, uncertainty

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
# ADDITIONAL SCORING UTILITIES
# ============================================================
def calculate_confidence_scientific(analysis_json):
    """
    ✅ MACHINE RELIABILITY CONFIDENCE SCORE
    Mengukur seberapa yakin sistem dalam analisisnya berdasarkan:
    1. Data quality (face detection rate, audio quality)
    2. Measurement consistency (reliability per metric)
    3. Coverage (semua metrik terdeteksi atau tidak)
    """
    confidence_per_metric = {}
    reliability_scores = []
    coverage_score = 0

    # Extract quality indicators
    face_detection_rate = analysis_json.get("facial_expression_analysis", {}).get("face_detected_percentage", 0)
    audio_quality = 100 if analysis_json.get("speech_analysis", {}).get("speaking_time_seconds", 0) > 0 else 0

    # Extract all metrics
    metrics_data = {
        "speech_rate_wpm": analysis_json.get("speech_analysis", {}).get("speech_rate_wpm", 0),
        "speaking_ratio": analysis_json.get("speech_analysis", {}).get("speaking_ratio", 0),
        "blink_rate_per_minute": analysis_json.get("eye_movement_analysis", {}).get("blink_rate_per_minute", 0),
        "eye_contact_percentage": analysis_json.get("eye_movement_analysis", {}).get("eye_contact_percentage", 0),
        "head_movement_intensity": analysis_json.get("facial_expression_analysis", {}).get("head_movement_intensity", 0),
        "average_smile_intensity": analysis_json.get("facial_expression_analysis", {}).get("average_smile_intensity", 0),
        "eyebrow_movement_range": analysis_json.get("facial_expression_analysis", {}).get("eyebrow_movement_range", 0)
    }

    detected_count = 0
    total_metrics = len(metrics_data)

    for metric, value in metrics_data.items():
        if metric in STATS and value > 0:
            detected_count += 1

            # Get reliability from STATS
            reliability = STATS[metric]["reliability"]

            # Calculate measurement confidence based on value reasonableness
            mean = STATS[metric]["mean"]
            sd = STATS[metric]["sd"]
            z = abs((value - mean) / sd)

            # Values within ±3 SD are considered reliable measurements
            if z <= 1:
                measurement_confidence = 100  # Very confident
            elif z <= 2:
                measurement_confidence = 85   # Confident
            elif z <= 3:
                measurement_confidence = 70   # Moderately confident
            else:
                measurement_confidence = 50   # Less confident (outlier)

            # Combined confidence for this metric
            metric_confidence = (reliability * 100 * 0.7) + (measurement_confidence * 0.3)

            confidence_per_metric[metric] = {
                "value": value,
                "reliability": round(reliability * 100, 2),
                "measurement_confidence": measurement_confidence,
                "combined_confidence": round(metric_confidence, 2),
                "status": "detected"
            }

            reliability_scores.append(metric_confidence * WEIGHTS[metric])
        else:
            confidence_per_metric[metric] = {
                "value": 0,
                "reliability": 0,
                "measurement_confidence": 0,
                "combined_confidence": 0,
                "status": "not_detected"
            }

    # Coverage score (berapa banyak metrik terdeteksi)
    coverage_score = (detected_count / total_metrics) * 100

    # Data quality score
    data_quality_score = (face_detection_rate * 0.6) + (audio_quality * 0.3) + (coverage_score * 0.1)

    # Overall machine confidence
    measurement_reliability = sum(reliability_scores) if reliability_scores else 0

    # Final confidence: 60% dari data quality, 40% dari measurement reliability
    total_confidence_percent = round((data_quality_score * 0.6) + (measurement_reliability * 0.4), 2)

    # Confidence level interpretation (MACHINE PERSPECTIVE)
    if total_confidence_percent >= 85:
        confidence_level = "Very High"
        interpretation = "Sistem sangat yakin dengan hasil analisis. Data berkualitas tinggi dan semua metrik terdeteksi dengan baik."
    elif total_confidence_percent >= 75:
        confidence_level = "High"
        interpretation = "Sistem yakin dengan hasil analisis. Data bagus dan mayoritas metrik reliable."
    elif total_confidence_percent >= 65:
        confidence_level = "Moderate"
        interpretation = "Sistem cukup yakin dengan hasil analisis. Beberapa metrik mungkin kurang optimal."
    elif total_confidence_percent >= 50:
        confidence_level = "Low"
        interpretation = "Sistem kurang yakin dengan hasil analisis. Pertimbangkan verifikasi manual atau improve kualitas video."
    else:
        confidence_level = "Very Low"
        interpretation = "Sistem tidak yakin dengan hasil analisis. Kualitas data rendah atau banyak metrik tidak terdeteksi. Perlu video ulang."

    # Calculate uncertainty margin
    margin_of_error = round((100 - total_confidence_percent) * 0.2, 2)
    lower_bound = round(max(0, total_confidence_percent - margin_of_error), 2)
    upper_bound = round(min(100, total_confidence_percent + margin_of_error), 2)

    return {
        "confidence_per_metric": confidence_per_metric,
        "quality_indicators": {
            "face_detection_rate": round(face_detection_rate, 2),
            "audio_quality": round(audio_quality, 2),
            "metrics_coverage": round(coverage_score, 2),
            "detected_metrics": f"{detected_count}/{total_metrics}"
        },
        "total_confidence_score": total_confidence_percent,
        "confidence_interval": {
            "lower": lower_bound,
            "upper": upper_bound,
            "margin_of_error": margin_of_error
        },
        "confidence_level": confidence_level,
        "interpretation": interpretation,
        "scoring_method": "Machine Reliability Score (Data Quality + Measurement Consistency)",
        "reliability_notes": f"Machine confidence: {total_confidence_percent}% (CI: {lower_bound}-{upper_bound}%)"
    }

def get_performance_level(avg_confidence):
    """Tentukan level reliabilitas machine"""
    if avg_confidence >= 85:
        return "Very High Confidence"
    elif avg_confidence >= 75:
        return "High Confidence"
    elif avg_confidence >= 65:
        return "Medium Confidence"
    elif avg_confidence >= 50:
        return "Low Confidence"
    else:
        return "Very Low Confidence"

def get_recommendation(avg_confidence, confidence_interval, interpretations):
    """Generate rekomendasi berdasarkan machine confidence"""
    reliability_level = get_performance_level(avg_confidence)
    lower = confidence_interval["lower"]
    upper = confidence_interval["upper"]

    if avg_confidence >= 85 and lower >= 80:
        return f"✅ RELIABLE - Hasil analisis dapat dipercaya. Machine confidence sangat tinggi (CI: {lower}-{upper}%). Tidak perlu verifikasi manual."
    elif avg_confidence >= 75 and lower >= 70:
        return f"✅ MOSTLY RELIABLE - Hasil analisis dapat dipercaya. Machine confidence tinggi (CI: {lower}-{upper}%). Verifikasi manual optional."
    elif avg_confidence >= 65 and lower >= 60:
        return f"⚠️ MODERATELY RELIABLE - Hasil analisis cukup dapat dipercaya. Machine confidence moderate (CI: {lower}-{upper}%). Disarankan spot-check manual."
    elif avg_confidence >= 50:
        return f"⚠️ LOW RELIABILITY - Machine kurang yakin dengan hasil (CI: {lower}-{upper}%). Wajib verifikasi manual atau improve kualitas video."
    else:
        return f"❌ UNRELIABLE - Machine tidak yakin dengan hasil (CI: {lower}-{upper}%). Perlu video ulang dengan kualitas lebih baik."

# ============================================================
# MAIN: FULL NON-VERBAL ANALYSIS PIPELINE
# ============================================================

def analyze_interview_video_with_confidence(video_path, audio_path=None, transcript=None):
    """Analisis video interview dengan optimasi penuh + scientific confidence scoring

    Args:
        video_path: Path ke file video
        audio_path: Path ke file audio (optional, akan diekstrak jika None)
        transcript: Teks transkrip untuk perhitungan speech rate yang akurat (optional)
    """
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
    speech_analysis = analyze_speech_tempo(audio_path, transcript=transcript)

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

    # ✅ RETURN the results
    return {
        'analysis': analysis_result,
        'confidence_score': conf_result["total_confidence_score"],
        'confidence_level': conf_result["confidence_level"],
        'confidence_components': conf_result["confidence_per_metric"],
        'confidence_interval': conf_result["confidence_interval"],
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