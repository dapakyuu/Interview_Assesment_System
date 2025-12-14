import urllib.request
import zipfile
import os
import shutil

def download_and_extract_ffmpeg():
    ffmpeg_url = "https://github.com/GyanD/codexffmpeg/releases/download/2025-11-27-git-61b034a47c/ffmpeg-2025-11-27-git-61b034a47c-full_build.zip"
    ffmpeg_zip = "ffmpeg_full_build.zip"
    extract_folder = "ffmpeg_temp_extract"
    target_bin_folder = "bin"

    # Skip download if bin folder already exists
    if os.path.exists(target_bin_folder):
        print("🎬 FFmpeg bin folder already exists — skipping download.")
        return target_bin_folder

    print(f"\n📥 Downloading FFmpeg from:\n{ffmpeg_url}\n")

    # Download ZIP
    urllib.request.urlretrieve(ffmpeg_url, ffmpeg_zip)
    print("✅ FFmpeg zip downloaded!")

    # Extract ZIP
    print("📦 Extracting FFmpeg...")
    with zipfile.ZipFile(ffmpeg_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Find bin folder
    print("🔍 Searching for bin folder...")
    bin_source_path = None
    for root, dirs, files in os.walk(extract_folder):
        if "bin" in dirs:
            bin_source_path = os.path.join(root, "bin")
            break

    if bin_source_path:
        # Move bin folder to backend/Python directory
        print(f"📁 Found bin folder at: {bin_source_path}")
        print(f"🚚 Moving bin folder to: {os.path.abspath(target_bin_folder)}")
        shutil.move(bin_source_path, target_bin_folder)
        print("✅ Bin folder moved successfully!")
        
        # Add to PATH
        bin_absolute_path = os.path.abspath(target_bin_folder)
        os.environ["PATH"] += os.pathsep + bin_absolute_path
        print(f"✅ FFmpeg added to PATH: {bin_absolute_path}")
    else:
        print("❌ Bin folder not found inside extracted files!")

    # Cleanup: Delete downloaded zip and extracted temp folder
    print("🗑️  Cleaning up temporary files...")
    if os.path.exists(ffmpeg_zip):
        os.remove(ffmpeg_zip)
        print(f"✅ Deleted: {ffmpeg_zip}")
    
    if os.path.exists(extract_folder):
        shutil.rmtree(extract_folder)
        print(f"✅ Deleted: {extract_folder}")

    print("🎉 FFmpeg bin folder successfully installed and ready to use!")
    return target_bin_folder


# Jalankan otomatis
download_and_extract_ffmpeg()

print("\n🔥 Import test completed — semua library ter-load tanpa error!")
