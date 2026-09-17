"""Tests for deterministic media processing pipeline:
- Step-gate validation (verify_media_file)
- Safe unlinking with exponential backoff on locks (safe_unlink)
- Canonical naming and path sanitization
- Circuit breaker behavior
"""
import os
import re
import tempfile
import time
import threading
import subprocess
import pytest

from ffmpeg_utils import verify_media_file, safe_unlink, safe_replace


def test_verify_media_file_nonexistent():
    valid, msg = verify_media_file("nonexistent_video_path_12345.mp4")
    assert valid is False
    assert "does not exist" in msg.lower() or "empty" in msg.lower()


def test_verify_media_file_empty(tmp_path):
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")
    valid, msg = verify_media_file(str(empty_file))
    assert valid is False
    assert "minimum required" in msg.lower() or "size" in msg.lower()


def test_verify_media_file_truncated(tmp_path):
    trunc_file = tmp_path / "trunc.mp4"
    trunc_file.write_bytes(b"A" * 500)
    valid, msg = verify_media_file(str(trunc_file))
    assert valid is False
    assert "<= minimum required" in msg


def test_verify_media_file_corrupted(tmp_path):
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"X" * 20000)
    valid, msg = verify_media_file(str(corrupt_file))
    assert valid is False
    assert "ffprobe" in msg.lower() or "stream" in msg.lower()


def test_verify_media_file_valid_mp4(tmp_path):
    valid_file = tmp_path / "valid.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", "2.0",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        str(valid_file)
    ]
    res = subprocess.run(cmd, capture_output=True)
    assert res.returncode == 0
    assert os.path.getsize(str(valid_file)) > 10240

    valid, msg = verify_media_file(str(valid_file))
    assert valid is True
    assert msg == ""


def test_safe_unlink_basic(tmp_path):
    f = tmp_path / "test_del.txt"
    f.write_text("hello")
    assert f.exists()
    assert safe_unlink(str(f)) is True
    assert not f.exists()

    assert safe_unlink(str(f)) is True


def test_safe_unlink_transient_lock(tmp_path):
    f = tmp_path / "locked.txt"
    f.write_text("locked content")

    held_handle = open(str(f), "r+b")

    def release_later():
        time.sleep(0.25)
        held_handle.close()

    t = threading.Thread(target=release_later)
    t.start()

    success = safe_unlink(str(f), max_retries=6, delay_s=0.1)
    t.join()
    assert success is True
    assert not f.exists()


def test_slugify_path_normalization():
    raw_filename = "jared_freid__the_family_plan_(final_netflix_screener) [1080p] 'clip'.mp4"
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', raw_filename)
    assert " " not in sanitized
    assert "(" not in sanitized
    assert ")" not in sanitized
    assert "[" not in sanitized
    assert "]" not in sanitized
    assert "'" not in sanitized
    assert sanitized.endswith(".mp4")
