"""ffmpeg_env.py
Ensures FFmpeg, FFprobe, and FFplay binaries are in os.environ["PATH"] across all environments.
Importing this module automatically configures PATH.
"""
import os
import sys
import shutil

# Force UTF-8 across all streams and Python subprocesses on Windows to prevent log mojibake / glitches
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

def setup_ffmpeg_path():
    candidates = [
        r"C:\ffmpeg\bin",
        r"C:\Users\PC\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin",
        os.path.dirname(sys.executable),
    ]
    
    current_path = os.environ.get("PATH", "")
    paths = current_path.split(os.pathsep)
    # Filter out any old FuseClip ffmpeg paths that crash on filtergraphs
    filtered_paths = [p for p in paths if "fuseclip" not in p.lower()]
    
    valid_candidates = [c for c in candidates if os.path.isdir(c)]
    all_paths = valid_candidates + [p for p in filtered_paths if p not in valid_candidates]
    os.environ["PATH"] = os.pathsep.join(all_paths)

setup_ffmpeg_path()
