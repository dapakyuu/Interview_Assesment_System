# ===== HELPER: Download video from Google Drive =====
import gdown
import requests
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
import os
import traceback
import uuid

from app.utils.json_builder import process_transcriptions_sync

from ..state import (
    get_local_file_path,
    processing_status,
    processing_lock,
    UPLOAD_DIR,
    TRANSCRIPTION_DIR
)


def download_video_from_google_drive(video_url, dest_folder):
    """Download video from Google Drive URL"""
    try:
        # Extract file ID from Google Drive URL
        if 'drive.google.com' in video_url:
            # Format 1: https://drive.google.com/file/d/FILE_ID/view?usp=...
            if '/file/d/' in video_url:
                file_id = video_url.split('/file/d/')[1].split('/')[0]
            # Format 2: https://drive.google.com/open?id=FILE_ID
            elif 'id=' in video_url:
                parsed = urlparse(video_url)
                file_id = parse_qs(parsed.query)['id'][0]
            else:
                raise ValueError(f"Unsupported Google Drive URL format")
            
            # Generate download URL
            download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
            
            # Generate safe filename
            safe_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}.mp4"
            dest_path = os.path.join(dest_folder, safe_name)
            
            print(f"      📥 Downloading from Google Drive (ID: {file_id[:20]}...)")
            
            # Download with gdown
            gdown.download(download_url, dest_path, quiet=False)
            
            # Verify file exists and has content
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
                raise ValueError("Downloaded file is empty or doesn't exist")
            
            file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"      ✅ Downloaded: {safe_name} ({file_size_mb:.2f} MB)")
            
            return safe_name, dest_path
        
        else:
            # Direct URL download (fallback)
            safe_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}.mp4"
            dest_path = os.path.join(dest_folder, safe_name)
            
            print(f"      📥 Downloading from direct URL")
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"      ✅ Downloaded: {safe_name} ({file_size_mb:.2f} MB)")
            
            return safe_name, dest_path
    
    except Exception as e:
        print(f"      ❌ Download failed: {str(e)}")
        raise
    
# ===== BACKGROUND THREAD: Download and Process Videos =====
def download_and_process_videos(session_id, candidate_name, interviews, language, base_url):
    """
    Background thread: Download videos from Google Drive URLs, then process identically to /upload endpoint.
    """
    print(f"\n🔽 [Thread-{session_id[:8]}] Starting video downloads...")
    
    try:
        uploaded_videos = []
        
        # PHASE 1: Download all videos
        print(f'\n📥 Downloading {len(interviews)} video(s) from Google Drive...')
        for idx, interview in enumerate(interviews, 1):
            try:
                # Update download progress
                with processing_lock:
                    processing_status[session_id]['message'] = f'Downloading video {idx}/{len(interviews)}...'
                    processing_status[session_id]['progress'] = f'{idx}/{len(interviews)}'
                
                # Extract fields from interview
                position_id = interview.get('positionId', idx)
                question = interview.get('question', '')
                is_video_exist = interview.get('isVideoExist', False)
                video_url = interview.get('recordedVideoUrl', '')
                
                print(f'\n   📹 Video {idx}/{len(interviews)}:')
                print(f'      Position ID: {position_id}')
                print(f'      Question: {question[:60]}{"..." if len(question) > 60 else ""}')
                print(f'      Video exists: {is_video_exist}')
                print(f'      URL: {video_url[:80]}{"..." if len(video_url) > 80 else ""}')
                
                # Validate
                if not question:
                    print(f'      ⚠️ Missing question, skipping')
                    uploaded_videos.append({
                        'positionId': position_id,
                        'question': '',
                        'isVideoExist': False,
                        'recordedVideoUrl': None,
                        'error': 'Missing question field'
                    })
                    continue
                
                if not is_video_exist or not video_url:
                    print(f'      ⚠️ No video URL, skipping')
                    uploaded_videos.append({
                        'positionId': position_id,
                        'question': question,
                        'isVideoExist': False,
                        'recordedVideoUrl': None,
                        'error': 'No video URL provided'
                    })
                    continue
                
                # Download video from Google Drive
                safe_name, dest_path = download_video_from_google_drive(video_url, UPLOAD_DIR)
                
                # Create local file URL (same as /upload endpoint)
                file_url = f"{base_url}/uploads/{safe_name}"
                
                uploaded_videos.append({
                    'positionId': position_id,
                    'question': question,
                    'isVideoExist': True,
                    'recordedVideoUrl': file_url,
                    'filename': safe_name
                })
                
            except Exception as e:
                print(f'      ❌ Failed to download video {idx}: {str(e)}')
                uploaded_videos.append({
                    'positionId': interview.get('positionId', idx),
                    'question': interview.get('question', ''),
                    'isVideoExist': False,
                    'recordedVideoUrl': None,
                    'error': str(e)
                })
        
        successful_downloads = len([v for v in uploaded_videos if v['isVideoExist']])
        print(f"\n✅ Download complete: {successful_downloads}/{len(interviews)} successful")
        
        # PHASE 2: Update status to processing (same as /upload endpoint)
        with processing_lock:
            processing_status[session_id] = {
                'status': 'processing',
                'progress': '0/' + str(len(uploaded_videos)),
                'message': 'Starting transcription...',
                'uploaded_videos': len(uploaded_videos)
            }
        
        # PHASE 3: Process transcriptions (IDENTICAL to /upload endpoint)
        print(f'\n🔄 Starting transcription process (identical to /upload endpoint)...')
        process_transcriptions_sync(session_id, candidate_name, uploaded_videos, base_url, language)
    
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"❌ Download thread error:\n{error_detail}")
        
        with processing_lock:
            processing_status[session_id] = {
                'status': 'error',
                'error': str(e),
                'error_detail': error_detail
            }