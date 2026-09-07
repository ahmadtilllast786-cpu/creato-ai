"""ffmpeg_env.py
Ensures FFmpeg, FFprobe, and FFplay binaries are in os.environ["PATH"] across all environments.
Importing this module automatically configures PATH.
"""
import os
import sys
import shutil

def setup_ffmpeg_path():
    venv_dir = os.path.dirname(sys.executable)
    candidates = [
        venv_dir,
        r"C:\ffmpeg\bin",
        r"C:\Program Files\FuseClip\resources\ffmpeg",
        r"C:\Users\PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin",
    ]
    
    current_path = os.environ.get("PATH", "")
    paths = current_path.split(os.pathsep)
    normalized = {os.path.normcase(os.path.normpath(p)) for p in paths if p}
    
    for candidate in candidates:
        if os.path.isdir(candidate):
            norm_c = os.path.normcase(os.path.normpath(candidate))
            if norm_c not in normalized:
                os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
                normalized.add(norm_c)

setup_ffmpeg_path()
