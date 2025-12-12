import os
import uuid
import shutil
import traceback
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

import threading as th

from app.utils.json_builder import process_transcriptions_sync
from app.utils.gd_json_download import download_and_process_videos
from .state import processing_status, processing_lock, UPLOAD_DIR, RESULTS_DIR

router = APIRouter()


# ======================================================
# POST /upload
# ======================================================
@router.post('/upload')
async def receive_videos_and_process(
    request: Request,
    candidate_name: str = Form(...),
    language: str = Form("en"),
    videos: List[UploadFile] = File(...),
    questions: List[str] = Form(...)
):
    session_id = uuid.uuid4().hex
    base_url = str(request.base_url).rstrip('/')

    # Validate
    if len(questions) != len(videos):
        return JSONResponse(
            {"success": False,
             "error": f'Questions count ({len(questions)}) must match videos count ({len(videos)})'},
            status_code=400
        )

    # Init status
    with processing_lock:
        processing_status[session_id] = {
            "status": "uploading",
            "progress": "0/0",
            "message": "Uploading videos..."
        }

    # Upload videos
    uploaded_videos = []
    try:
        for idx, (video, question) in enumerate(zip(videos, questions), 1):
            ext = os.path.splitext(video.filename)[1] or ".webm"
            filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}{ext}"
            
            # ✅ Ensure UPLOAD_DIR is string
            dest = os.path.join(str(UPLOAD_DIR), filename)

            # Save file
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)

            uploaded_videos.append({
                "positionId": idx,
                "question": question,
                "isVideoExist": True,
                "recordedVideoUrl": f"{base_url}/uploads/{filename}",
                "filename": filename
            })

            with processing_lock:
                processing_status[session_id]["progress"] = f"{idx}/{len(videos)}"
                processing_status[session_id]["message"] = f"Uploading {idx}/{len(videos)}..."

    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)}, status_code=500
        )

    # Start background processing
    with processing_lock:
        processing_status[session_id] = {
            'status': 'processing',
            'progress': f'0/{len(uploaded_videos)}',
            'message': 'Starting transcription...',
            'uploaded_videos': len(uploaded_videos)
        }

    thread = th.Thread(
        target=process_transcriptions_sync,
        args=(session_id, candidate_name, uploaded_videos, base_url, language),
        daemon=True
    )
    thread.start()

    return {
        "success": True,
        "session_id": session_id,
        "uploaded_videos": len(uploaded_videos)
    }

# ======================================================
# POST /upload_json
# ======================================================

@router.post('/upload_json')
async def receive_json_and_download_videos(request: Request):
    """Receive JSON with Google Drive URLs, download videos, then process"""
    session_id = uuid.uuid4().hex
    
    try:
        # Parse JSON
        json_data = await request.json()
        
        print(f'\n🔵 NEW JSON UPLOAD REQUEST - Session: {session_id}')
        
        # Validate structure
        if not json_data.get('success') or not json_data.get('data'):
            return JSONResponse(
                {'success': False, 'error': 'Invalid JSON: missing success or data'},
                status_code=400,
                headers={'Access-Control-Allow-Origin': '*'}
            )
        
        data = json_data['data']
        
        # Extract candidate
        if not data.get('candidate') or not data['candidate'].get('name'):
            return JSONResponse(
                {'success': False, 'error': 'Missing candidate name'},
                status_code=400,
                headers={'Access-Control-Allow-Origin': '*'}
            )
        
        candidate_name = data['candidate']['name']
        candidate_email = data['candidate'].get('email', 'N/A')
        
        # Extract interviews
        if not data.get('reviewChecklists') or not data['reviewChecklists'].get('interviews'):
            return JSONResponse(
                {'success': False, 'error': 'Missing interviews data'},
                status_code=400,
                headers={'Access-Control-Allow-Origin': '*'}
            )
        
        interviews = data['reviewChecklists']['interviews']
        
        if not isinstance(interviews, list) or len(interviews) == 0:
            return JSONResponse(
                {'success': False, 'error': 'Interviews array is empty'},
                status_code=400,
                headers={'Access-Control-Allow-Origin': '*'}
            )
        
        # Get language
        language = json_data.get('language', 'en')
        
        print(f'   Candidate: {candidate_name} ({candidate_email})')
        print(f'   Videos: {len(interviews)} video(s)')
        print(f'   Language: {language}')
        
        # Validate language
        if language not in ["en", "id"]:
            return JSONResponse(
                {'success': False, 'error': f'Invalid language: {language}'},
                status_code=400,
                headers={'Access-Control-Allow-Origin': '*'}
            )
        
        # Log certification info
        if data.get('certification'):
            cert = data['certification']
            print(f'   Certification: {cert.get("abbreviatedType", "N/A")} - {cert.get("status", "N/A")}')
        
        # Initialize status
        with processing_lock:
            processing_status[session_id] = {
                'status': 'downloading',
                'progress': '0/' + str(len(interviews)),
                'message': 'Downloading videos from Google Drive...'
            }
        
        # Start background thread
        thread = th.Thread(
            target=download_and_process_videos,
            args=(session_id, candidate_name, interviews, language, str(request.base_url).rstrip('/')),
            daemon=True
        )
        thread.start()
        
        print(f'✅ JSON received. Background download thread started.')
        print(f'📤 Returning immediate response with session_id: {session_id}')
        
        return JSONResponse(
            content={
                'success': True,
                'session_id': session_id,
                'message': 'JSON received. Downloading videos from Google Drive...',
                'video_count': len(interviews)
            },
            status_code=200,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': '*',
            }
        )
    
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f'❌ Error processing JSON:\n{error_detail}')
        
        return JSONResponse(
            content={'success': False, 'error': str(e)},
            status_code=500,
            headers={'Access-Control-Allow-Origin': '*'}
        )
        
# ======================================================
# GET /status/{session_id}
# ======================================================
@router.get('/status/{session_id}')
async def get_processing_status(session_id: str):

    with processing_lock:
        if session_id not in processing_status:
            return JSONResponse(
                {"status": "not_found", "message": "Session not found"},
                status_code=404
            )

        status_copy = processing_status[session_id].copy()

    if status_copy.get("status") == "completed":
        status_copy["redirect"] = f"halaman_dasboard.html?session={session_id}"

    return status_copy


# ======================================================
# GET /results/{session_id}
# ======================================================
@router.get('/results/{session_id}')
async def get_results(session_id: str):
    # ✅ Ensure RESULTS_DIR is string
    results_path = os.path.join(str(RESULTS_DIR), f"{session_id}.json")

    if not os.path.exists(results_path):
        return JSONResponse(
            {"success": False, "message": "Results not found"},
            status_code=404
        )

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = f.read()
        return JSONResponse(content=data)

    except Exception as e:
        return JSONResponse(
            {"success": False, "message": str(e)},
            status_code=500
        )


# ======================================================
# GET /
# ======================================================
@router.get("/")
async def index():
    return {
        "message": "AI Interview Assessment System",
        "model": "faster-whisper large-v3",
        "endpoints": {
            "upload": "POST /upload",
            "status": "GET /status/{session_id}",
            "results": "GET /results/{session_id}",
        }
    }