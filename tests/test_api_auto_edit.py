"""Integration tests for Stage 2 on-demand Auto Edit API endpoints:
- POST /api/clip/auto-edit
- POST /api/clip/revert-base
- POST /api/edit fallback
"""

import asyncio
import json
import os
import httpx
import pytest

app_module = pytest.importorskip("app")
import auto_editor

JOB_ID = "auto-edit-test-job"


def _request(method, path, json_body=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=json_body)
    return asyncio.run(_do())


@pytest.fixture()
def job(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    job_dir = out_root / JOB_ID
    job_dir.mkdir(parents=True)
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))

    clip = {
        "start": 10.0,
        "end": 40.0,
        "video_title_for_youtube_short": "test clip",
        "video_url": f"/videos/{JOB_ID}/testvideo_clip_1.mp4",
    }
    meta = {
        "shorts": [clip],
        "transcript": {"words": [{"w": "Hello", "s": 10.0, "e": 10.5}]},
        "source_video": "src.mp4",
        "output_format": "auto"
    }
    (job_dir / "testvideo_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (job_dir / "testvideo_clip_1.mp4").write_bytes(b"dummy base video content")

    app_module.jobs[JOB_ID] = {
        "status": "completed",
        "result": {"clips": [clip]},
        "output_dir": str(job_dir),
    }

    yield job_dir


class TestAutoEditEndpoints:
    def test_auto_edit_non_destructive_and_revert(self, job, monkeypatch):
        # Mock auto_editor.auto_edit_clip to avoid requiring FFmpeg video rendering in CI
        def mock_auto_edit_clip(input_clip_path, output_clip_path, **kwargs):
            # Verify input is base cut and output path is distinct
            assert os.path.basename(input_clip_path) == "testvideo_clip_1.mp4"
            assert "auto_edited_" in os.path.basename(output_clip_path)
            with open(output_clip_path, "wb") as f:
                f.write(b"dummy edited video content")
            return {
                "success": True,
                "output_path": output_clip_path,
                "silence_cuts": 1,
                "zooms_applied": True,
                "duration": 28.5
            }

        monkeypatch.setattr(auto_editor, "auto_edit_clip", mock_auto_edit_clip)

        # 1. Trigger Auto Edit
        res = _request("POST", "/api/clip/auto-edit", {
            "job_id": JOB_ID,
            "clip_index": 0
        })
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["success"] is True
        assert "auto_edited_" in data["new_video_url"]
        assert data["base_video_url"] == f"/videos/{JOB_ID}/testvideo_clip_1.mp4"
        assert data["is_auto_edited"] is True

        # Ensure base cut still exists intact on disk (non-destructive)
        base_file = job / "testvideo_clip_1.mp4"
        assert base_file.exists()
        assert base_file.read_bytes() == b"dummy base video content"

        # 2. Trigger Revert to Base
        rev_res = _request("POST", "/api/clip/revert-base", {
            "job_id": JOB_ID,
            "clip_index": 0
        })
        assert rev_res.status_code == 200, rev_res.text
        rev_data = rev_res.json()
        assert rev_data["success"] is True
        assert rev_data["new_video_url"] == f"/videos/{JOB_ID}/testvideo_clip_1.mp4"
        assert rev_data["is_auto_edited"] is False

    def test_legacy_edit_endpoint_falls_back_to_auto_edit_when_no_api_key(self, job, monkeypatch):
        def mock_auto_edit_clip(input_clip_path, output_clip_path, **kwargs):
            with open(output_clip_path, "wb") as f:
                f.write(b"dummy edited video content")
            return {"success": True, "output_path": output_clip_path, "silence_cuts": 0, "zooms_applied": True}

        monkeypatch.setattr(auto_editor, "auto_edit_clip", mock_auto_edit_clip)

        res = _request("POST", "/api/edit", {
            "job_id": JOB_ID,
            "clip_index": 0
        })
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["success"] is True
        assert "auto_edited_" in data["new_video_url"]
