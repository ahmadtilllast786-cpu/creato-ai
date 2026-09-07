"""Unit tests for metadata_extractor module."""

import os
import json
import pytest
from metadata_extractor import (
    extract_speech_and_words,
    load_clip_metadata,
    extract_clip_metadata,
)


class TestSpeechExtraction:
    def test_slices_existing_transcript_into_clip_relative_ms(self):
        parent_transcript = {
            "language": "en",
            "segments": [
                {
                    "start": 5.0,
                    "end": 9.0,
                    "text": "Wait listen to this incredible story!",
                    "words": [
                        {"word": "Wait", "start": 5.1, "end": 5.4},
                        {"word": "listen", "start": 5.5, "end": 5.9},
                        {"word": "to", "start": 6.0, "end": 6.1},
                        {"word": "this", "start": 6.2, "end": 6.5},
                        {"word": "incredible", "start": 6.6, "end": 7.3},
                        {"word": "story!", "start": 7.4, "end": 8.0},
                    ]
                },
                {
                    "start": 10.0,
                    "end": 14.0,
                    "text": "Nobody expected this to happen.",
                    "words": [
                        {"word": "Nobody", "start": 10.1, "end": 10.6},
                        {"word": "expected", "start": 10.7, "end": 11.2},
                        {"word": "this", "start": 11.3, "end": 11.5},
                        {"word": "to", "start": 11.6, "end": 11.8},
                        {"word": "happen.", "start": 11.9, "end": 12.5},
                    ]
                }
            ]
        }

        # Clip spans 5.0 to 15.0
        data = extract_speech_and_words(
            video_path="dummy.mp4",
            existing_transcript=parent_transcript,
            clip_start=5.0,
            clip_end=15.0
        )

        assert data["language"] == "en"
        words = data["words"]
        assert len(words) == 11

        # First word "Wait" at 5.1s -> relative to clip is 0.1s -> 100ms
        assert abs(words[0]["s"] - 0.1) < 0.01
        assert words[0]["startMs"] == 100
        assert words[0]["w"] == "Wait"

        # Opening hook
        hook = data["opening_hook"]
        assert hook is not None
        assert "Wait listen" in hook["text"]
        assert hook["startMs"] >= 0

        # Key sentences
        key_sents = data["key_sentences"]
        assert len(key_sents) >= 1
        assert key_sents[0]["is_emphatic"] is True


class TestLoadClipMetadata:
    def test_loads_from_dedicated_file(self, tmp_path):
        out_dir = str(tmp_path)
        meta = {
            "clip_index": 0,
            "filename": "myclip_1.mp4",
            "duration": 30.0,
            "words": [{"w": "hello", "startMs": 0, "endMs": 500}],
            "silence_intervals": [{"start": 10.0, "end": 11.0}],
            "is_extracted": True
        }
        meta_path = os.path.join(out_dir, "myclip_1_real_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        loaded = load_clip_metadata(out_dir, "myclip_1.mp4")
        assert loaded is not None
        assert loaded["filename"] == "myclip_1.mp4"
        assert loaded["words"][0]["w"] == "hello"

    def test_returns_none_when_no_metadata_found(self, tmp_path):
        loaded = load_clip_metadata(str(tmp_path), "non_existent.mp4")
        assert loaded is None
