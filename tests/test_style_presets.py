"""Tests for the Style Preset system and Semantic Matching Engine."""
import json
import os
import pytest

from styles.style_matcher import (
    StylePresetRegistry,
    ContentProfile,
    PresetMatch,
    build_content_profile,
    match_presets,
    preset_to_config,
    get_preset_and_config,
    auto_match_and_config,
)

PRESETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "styles", "presets.json")


class TestPresetDatabase:
    """Validate presets.json schema and data integrity."""

    def test_presets_json_loads(self):
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "presets" in data
        assert "categories" in data
        assert "version" in data

    def test_at_least_30_presets(self):
        registry = StylePresetRegistry()
        assert len(registry.all()) >= 30, f"Only {len(registry.all())} presets, need ≥30"

    def test_five_categories(self):
        registry = StylePresetRegistry()
        assert len(registry.categories) == 5

    def test_all_categories_have_presets(self):
        registry = StylePresetRegistry()
        for cat in registry.categories:
            presets = registry.list_by_category(cat)
            assert len(presets) >= 5, f"Category '{cat}' only has {len(presets)} presets"

    def test_all_presets_have_required_fields(self):
        registry = StylePresetRegistry()
        required_top = {"id", "name", "category", "description",
                        "targetPacing", "cameraMotion", "captions", "assets", "colorGrade"}
        for p in registry.all():
            missing = required_top - set(p.keys())
            assert not missing, f"Preset '{p.get('id')}' missing: {missing}"

    def test_all_presets_have_valid_pacing(self):
        registry = StylePresetRegistry()
        for p in registry.all():
            pacing = p["targetPacing"]
            assert 0.05 <= pacing["silenceCutThresholdSec"] <= 1.0, \
                f"{p['id']}: silenceCutThreshold out of range"
            assert pacing["jumpCutFrequency"] in ("none", "low", "medium", "high")
            assert isinstance(pacing["bRollFrequencySec"], list) and len(pacing["bRollFrequencySec"]) == 2

    def test_all_presets_have_valid_camera(self):
        registry = StylePresetRegistry()
        for p in registry.all():
            cam = p["cameraMotion"]
            assert cam["zoomFrequency"] in ("none", "occasional", "frequent")
            assert 1.0 <= cam["zoomScale"] <= 1.5
            assert 0.03 <= cam["trackingSmoothing"] <= 0.20

    def test_all_presets_have_valid_captions(self):
        registry = StylePresetRegistry()
        for p in registry.all():
            cap = p["captions"]
            assert isinstance(cap["fontFamily"], str) and len(cap["fontFamily"]) > 0
            assert 20 <= cap["fontSize"] <= 80
            assert cap["casing"] in ("uppercase", "capitalize", "lowercase")
            assert 1 <= cap["wordsPerChunk"] <= 8
            assert cap["animation"] in ("bounce", "fade", "pop", "slide", "none")

    def test_unique_ids(self):
        registry = StylePresetRegistry()
        ids = [p["id"] for p in registry.all()]
        assert len(ids) == len(set(ids)), "Duplicate preset IDs found"

    def test_get_by_id(self):
        registry = StylePresetRegistry()
        p = registry.get("mrbeast-energy")
        assert p is not None
        assert p["name"] == "MrBeast Energy"


class TestContentProfiler:
    """Validate semantic profiling of video content."""

    def test_basic_profile(self):
        profile = build_content_profile(
            transcript_text="This is a gaming stream highlight from Twitch",
            duration_seconds=30.0,
        )
        assert profile.word_count == 8
        assert profile.speaking_rate_wpm > 0
        assert profile.detected_category == "gaming_meme"

    def test_business_detection(self):
        profile = build_content_profile(
            transcript_text="Our startup revenue grew 200% this quarter with a strong product market fit and enterprise customers",
            duration_seconds=15.0,
        )
        assert profile.detected_category == "business_corporate"

    def test_empty_transcript(self):
        profile = build_content_profile(transcript_text="", duration_seconds=10.0)
        assert profile.word_count == 0
        assert profile.detected_category is None

    def test_fast_speaker_high_energy(self):
        # 200 words in 60s = 200 wpm (very fast)
        words = " ".join(["word"] * 200)
        profile = build_content_profile(transcript_text=words, duration_seconds=60.0)
        assert profile.energy_variance >= 0.7


class TestStyleMatcher:
    """Validate preset matching and scoring."""

    def test_gaming_content_matches_gaming_presets(self):
        profile = build_content_profile(
            transcript_text="Let's go this Twitch stream is insane clutch moment in Valorant esports tournament",
            duration_seconds=30.0,
            motion_level="high",
        )
        matches = match_presets(profile, top_k=5)
        assert len(matches) > 0
        # Top match should be from gaming category
        categories = [m.category for m in matches[:3]]
        assert "gaming_meme" in categories

    def test_corporate_content_matches_business_presets(self):
        profile = build_content_profile(
            transcript_text="In this webinar we discuss enterprise strategy and quarterly revenue growth for our SaaS product",
            duration_seconds=45.0,
            motion_level="low",
        )
        matches = match_presets(profile, top_k=5)
        categories = [m.category for m in matches[:3]]
        assert "business_corporate" in categories

    def test_category_filter(self):
        profile = build_content_profile(
            transcript_text="Some general content here",
            duration_seconds=30.0,
        )
        matches = match_presets(profile, top_k=10, category_filter="aesthetic_minimal")
        for m in matches:
            assert m.category == "aesthetic_minimal"

    def test_scores_between_0_and_1(self):
        profile = build_content_profile(
            transcript_text="Testing score bounds",
            duration_seconds=30.0,
        )
        matches = match_presets(profile, top_k=40)
        for m in matches:
            assert 0.0 <= m.score <= 1.0

    def test_results_sorted_descending(self):
        profile = build_content_profile(
            transcript_text="Some content about various topics",
            duration_seconds=30.0,
        )
        matches = match_presets(profile, top_k=10)
        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)


class TestPresetToConfig:
    """Validate translation from StylePreset to AutoEditConfig."""

    def test_basic_translation(self):
        registry = StylePresetRegistry()
        preset = registry.get("mrbeast-energy")
        config = preset_to_config(preset)

        assert config["min_silence_s"] == 0.20
        assert config["punch_in_zooms"] is True
        assert config["max_zoom"] == 1.25
        assert config["burn_subtitles"] is True
        assert "subtitle_settings" in config
        assert config["subtitle_settings"]["font_name"] == "Anton"

    def test_quiet_preset_disables_sfx(self):
        registry = StylePresetRegistry()
        preset = registry.get("nature-documentary")
        config = preset_to_config(preset)

        assert config["transition_sfx"] is False
        assert config["sfx_volume"] == 0.0
        assert config["punch_in_zooms"] is False

    def test_all_presets_translate_without_error(self):
        registry = StylePresetRegistry()
        for preset in registry.all():
            config = preset_to_config(preset)
            assert isinstance(config, dict)
            assert "min_silence_s" in config
            assert "subtitle_settings" in config

    def test_get_preset_and_config_returns_pair(self):
        preset, config = get_preset_and_config("hormozi-authority")
        assert preset is not None
        assert preset["name"] == "Hormozi Authority"
        assert config is not None
        assert isinstance(config["subtitle_settings"], dict)

    def test_missing_preset_returns_none(self):
        preset, config = get_preset_and_config("nonexistent-id")
        assert preset is None
        assert config is None


class TestAutoMatchAndConfig:
    """Validate the one-call convenience function."""

    def test_returns_match_and_config(self):
        match, config = auto_match_and_config(
            transcript_text="Subscribe and smash that like button for more insane content",
            duration_seconds=30.0,
        )
        assert isinstance(match, PresetMatch)
        assert isinstance(config, dict)
        assert match.score > 0

    def test_with_category_filter(self):
        match, config = auto_match_and_config(
            transcript_text="Some content",
            duration_seconds=30.0,
            category_filter="documentary_cinematic",
        )
        assert match.category == "documentary_cinematic"
