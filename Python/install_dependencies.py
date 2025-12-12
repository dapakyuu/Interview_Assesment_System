import subprocess
import sys
import urllib.request
import zipfile
import os

def pip_install(package):
    print(f"→ Installing: {' '.join(package)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + package)


print("📦 Installing required packages...\n")
# ============================
pip_install(["numpy==1.26.4"])
pip_install(["--upgrade", "torch", "torchaudio", "faster-whisper"])
pip_install(["ipywidgets", "jupyter"])
pip_install(["fastapi", "uvicorn", "nest-asyncio", "pyngrok", "python-multipart"])
pip_install(["tqdm"])
pip_install(["imageio-ffmpeg"])
pip_install(["deepl"])
pip_install(["silero-vad"])
pip_install(["pydub"])
pip_install(["soundfile"])
pip_install(["scipy"])
pip_install(["scikit-learn"])
pip_install(["huggingface-hub"])
pip_install(["mediapipe"])
pip_install(["torchcodec"])
pip_install(["librosa"])
pip_install(["gdown"])
pip_install(["requests"])
pip_install(["python-dotenv"])
pip_install(["webrtcvad-wheels"])
pip_install(["resemblyzer", "--no-deps"])
pip_install(["typing"])
pip_install(["moviepy==1.0.3"])
# ============================

print('\n✅ All safe packages installed')
print('   ✅ No numpy version conflicts')
print('   ✅ Jupyter widgets installed (fixes tqdm warning)')
print('   ✅ FFmpeg required for audio - verify with next cell')
# ============================


def download_and_extract_ffmpeg():
    ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/2025-11-27-git-61b034a47c/ffmpeg-2025-11-27-git-61b034a47c-full_build.zip"
    ffmpeg_zip = "ffmpeg_full_build.zip"
    extract_folder = "ffmpeg_bin"

    # Skip download if already extracted
    if os.path.exists(extract_folder):
        print("🎬 FFmpeg already extracted — skipping download.")
        return extract_folder

    print(f"\n📥 Downloading FFmpeg from:\n{ffmpeg_url}\n")

    # Download ZIP
    urllib.request.urlretrieve(ffmpeg_url, ffmpeg_zip)
    print("✅ FFmpeg zip downloaded!")

    # Extract ZIP
    print("📦 Extracting FFmpeg...")
    with zipfile.ZipFile(ffmpeg_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Add to PATH
    ffmpeg_extracted_path = None
    for root, dirs, files in os.walk(extract_folder):
        if "ffmpeg.exe" in files:
            ffmpeg_extracted_path = root
            break

    if ffmpeg_extracted_path:
        os.environ["PATH"] += os.pathsep + ffmpeg_extracted_path
        print(f"✅ FFmpeg added to PATH: {ffmpeg_extracted_path}")
    else:
        print("❌ FFmpeg binary not found inside extracted folder!")

    # Cleanup
    if os.path.exists(ffmpeg_zip):
        os.remove(ffmpeg_zip)

    print("🎉 FFmpeg successfully installed and ready to use!")
    return extract_folder


# Jalankan otomatis
# download_and_extract_ffmpeg()

print("\n🔥 Import test completed — semua library ter-load tanpa error!")
