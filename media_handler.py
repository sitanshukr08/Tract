import os
import glob
import uuid
import tempfile
import yt_dlp

def validate_media_url(url: str) -> bool:
    """Basic validation to ensure the string is a URL."""
    return url.startswith("http://") or url.startswith("https://")

def download_audio_temp(url: str, max_duration_secs: int = 7200) -> str:
    """Downloads audio to a system temp folder. Rejects oversized videos early."""
    if not validate_media_url(url):
        raise ValueError("Invalid URL format. Must start with http:// or https://")

    # Check duration before downloading
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            if duration > max_duration_secs:
                raise ValueError(f"Video is too long ({duration//60} mins). Max allowed is {max_duration_secs//60} mins to respect AI limits.")
    except yt_dlp.utils.DownloadError:
        raise ValueError("Unable to fetch video. It may be private, age-restricted, or invalid.")

    # Safely download to OS Temp Directory
    temp_dir = tempfile.gettempdir()
    path_base = os.path.join(temp_dir, f"tract_audio_{uuid.uuid4().hex}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': path_base + '.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'postprocessor_args': [
            '-ac', '1',
            '-ar', '16000',
            '-b:a', '16k'
        ],
        'quiet': True,
        'no_warnings': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        downloaded_files = glob.glob(path_base + ".*")
        if not downloaded_files:
            raise FileNotFoundError("Download succeeded but no audio file was generated. Is FFmpeg installed?")
        
        final_path = downloaded_files[0]
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        
        # Final safety net for Groq's strict 25MB limit
        if size_mb > 24.5:
            os.remove(final_path)
            raise ValueError(f"Audio file is {size_mb:.1f} MB. Groq strictly requires < 25 MB.")
            
        return final_path
    except Exception as e:
        for f in glob.glob(path_base + ".*"):
            try: os.remove(f)
            except OSError: pass
        raise e