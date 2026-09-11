"""Semantic Profiling & Style Matching Engine.

Automatically selects the best style preset for a video clip by analyzing:
  1. Transcript content (keywords, tone, topic detection)
  2. Audio energy profile (pacing, silence ratio, speaking rate)
  3. Visual characteristics (face count, motion level)

The engine scores each preset against the input profile and returns a ranked
list of matches.  It also provides ``preset_to_config()`` to translate any
StylePreset into an ``AutoEditConfig`` + subtitle settings dict that the
auto_editor pipeline consumes directly.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json")

# ─── Keyword → category affinity tables ───────────────────────────────────────
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "creator_hype": [
        "subscribe", "smash", "like", "comment", "follow", "viral", "trending",
        "insane", "crazy", "unbelievable", "story time", "reaction", "challenge",
        "prank", "giveaway", "motivation", "grind", "hustle", "mindset", "money",
        "millions", "views", "podcast", "clip", "episode", "content", "creator",
    ],
    "documentary_cinematic": [
        "history", "documentary", "war", "ancient", "civilization", "science",
        "evolution", "universe", "nature", "ocean", "wildlife", "research",
        "study", "discovery", "exploration", "travel", "journey", "culture",
        "analysis", "essay", "film", "cinema", "narrative", "investigate",
    ],
    "business_corporate": [
        "revenue", "startup", "investment", "roi", "saas", "enterprise",
        "leadership", "strategy", "meeting", "webinar", "quarterly", "growth",
        "product", "market", "customer", "b2b", "linkedin", "professional",
        "ceo", "founder", "company", "business", "corporate", "finance",
        "crypto", "bitcoin", "real estate", "property", "pitch", "demo",
    ],
    "gaming_meme": [
        "game", "gaming", "stream", "twitch", "esports", "fps", "mmo",
        "league", "valorant", "fortnite", "minecraft", "speedrun", "clutch",
        "gg", "rage", "noob", "pro", "meme", "reddit", "4chan", "bruh",
        "sus", "based", "cringe", "compilation", "montage", "highlight",
    ],
    "aesthetic_minimal": [
        "aesthetic", "minimal", "clean", "simple", "pastel", "vibe", "mood",
        "lofi", "chill", "relaxing", "cozy", "asmr", "wellness", "meditation",
        "mindfulness", "yoga", "calm", "peaceful", "nature", "organic",
        "vintage", "retro", "dreamy", "soft", "gentle", "cottagecore",
    ],
}

# Specific preset affinity keywords (id → extra keyword list)
_PRESET_KEYWORDS: Dict[str, List[str]] = {
    "mrbeast-energy": ["mrbeast", "challenge", "giveaway", "million", "insane"],
    "hormozi-authority": ["hormozi", "offer", "leads", "acquisition", "100m"],
    "tiktok-viral": ["tiktok", "fyp", "viral", "trend", "duet", "stitch"],
    "podcast-clip-fire": ["podcast", "episode", "interview", "guest", "host"],
    "reaction-king": ["reaction", "react", "watching", "reacting", "first time"],
    "true-crime": ["crime", "murder", "investigation", "suspect", "evidence", "case"],
    "finance-crypto": ["crypto", "bitcoin", "ethereum", "nft", "trading", "stocks"],
    "reddit-story": ["reddit", "aita", "tifu", "askreddit", "story"],
    "meme-edit": ["meme", "shitpost", "dank", "deep fried", "ironic"],
    "twitch-clip-chaos": ["twitch", "stream", "clip", "chat", "donation"],
    "esports-highlight": ["esports", "tournament", "pro", "competitive", "finals"],
    "asmr-whisper": ["asmr", "whisper", "tingles", "trigger", "relaxing"],
    "lofi-study": ["lofi", "lo-fi", "study", "focus", "concentration", "beats"],
    "wellness-mindfulness": ["wellness", "meditation", "mindfulness", "breathe"],
    "real-estate-showcase": ["property", "listing", "bedroom", "square feet", "tour"],
    "linkedin-thought-leader": ["linkedin", "leadership", "thought leader", "professional"],
}


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ContentProfile:
    """Semantic + audio + visual profile of a video clip."""
    # Transcript
    transcript_text: str = ""
    word_count: int = 0
    speaking_rate_wpm: float = 0.0      # words per minute
    duration_seconds: float = 0.0

    # Audio energy
    silence_ratio: float = 0.0          # fraction of clip that is silent
    avg_silence_gap_sec: float = 0.0    # average silence duration
    energy_variance: float = 0.0        # high = dynamic, low = monotone

    # Visual
    face_count: int = 1
    motion_level: str = "medium"        # "low", "medium", "high"
    is_screencast: bool = False

    # Detected topic/keywords
    detected_category: Optional[str] = None
    keyword_hits: Dict[str, int] = field(default_factory=dict)


@dataclass
class PresetMatch:
    """A scored match between a content profile and a style preset."""
    preset_id: str
    preset_name: str
    category: str
    score: float                        # 0.0 – 1.0
    reasons: List[str] = field(default_factory=list)


# ─── Preset Registry ─────────────────────────────────────────────────────────

class StylePresetRegistry:
    """Loads and indexes the preset database."""

    def __init__(self, presets_path: Optional[str] = None):
        path = presets_path or _PRESETS_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.version: str = data.get("version", "1.0.0")
        self.categories: List[str] = data.get("categories", [])
        self.presets: List[Dict[str, Any]] = data.get("presets", [])
        self._by_id: Dict[str, Dict[str, Any]] = {p["id"]: p for p in self.presets}
        self._by_category: Dict[str, List[Dict[str, Any]]] = {}
        for p in self.presets:
            self._by_category.setdefault(p["category"], []).append(p)

    def get(self, preset_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(preset_id)

    def list_ids(self) -> List[str]:
        return list(self._by_id.keys())

    def list_by_category(self, category: str) -> List[Dict[str, Any]]:
        return self._by_category.get(category, [])

    def all(self) -> List[Dict[str, Any]]:
        return list(self.presets)


# ─── Semantic Profiler ────────────────────────────────────────────────────────

def build_content_profile(
    transcript_text: str = "",
    duration_seconds: float = 60.0,
    silence_ratio: float = 0.0,
    avg_silence_gap_sec: float = 0.0,
    face_count: int = 1,
    motion_level: str = "medium",
    is_screencast: bool = False,
) -> ContentProfile:
    """Build a ContentProfile from available signals."""
    words = transcript_text.lower().split()
    word_count = len(words)
    wpm = (word_count / max(1.0, duration_seconds)) * 60.0

    # Keyword detection
    text_lower = transcript_text.lower()
    keyword_hits: Dict[str, int] = {}
    category_scores: Dict[str, float] = {}

    for category, keywords in _CATEGORY_KEYWORDS.items():
        hits = 0
        for kw in keywords:
            count = text_lower.count(kw)
            if count > 0:
                hits += count
                keyword_hits[kw] = keyword_hits.get(kw, 0) + count
        category_scores[category] = hits

    detected_category = None
    if category_scores:
        best = max(category_scores, key=category_scores.get)
        if category_scores[best] > 0:
            detected_category = best

    # Energy variance heuristic: fast speakers with short silences = high energy
    energy_variance = 0.0
    if wpm > 160:
        energy_variance = 0.8
    elif wpm > 130:
        energy_variance = 0.5
    elif wpm > 100:
        energy_variance = 0.3
    else:
        energy_variance = 0.1

    return ContentProfile(
        transcript_text=transcript_text,
        word_count=word_count,
        speaking_rate_wpm=wpm,
        duration_seconds=duration_seconds,
        silence_ratio=silence_ratio,
        avg_silence_gap_sec=avg_silence_gap_sec,
        energy_variance=energy_variance,
        face_count=face_count,
        motion_level=motion_level,
        is_screencast=is_screencast,
        detected_category=detected_category,
        keyword_hits=keyword_hits,
    )


# ─── Style Matching Engine ───────────────────────────────────────────────────

def _pacing_score(profile: ContentProfile, preset: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Score pacing alignment (silence threshold vs. speaking rate)."""
    pacing = preset.get("targetPacing", {})
    threshold = pacing.get("silenceCutThresholdSec", 0.35)
    jump_freq = pacing.get("jumpCutFrequency", "medium")
    reasons = []

    score = 0.0

    # Fast speakers match aggressive silence cuts
    wpm = profile.speaking_rate_wpm
    if wpm > 150 and threshold <= 0.25:
        score += 0.35
        reasons.append("Fast speaking rate matches aggressive pacing")
    elif wpm > 120 and 0.25 < threshold <= 0.45:
        score += 0.35
        reasons.append("Moderate speaking rate matches balanced pacing")
    elif wpm <= 120 and threshold > 0.45:
        score += 0.35
        reasons.append("Deliberate speaking rate matches relaxed pacing")
    else:
        score += 0.10

    # Jump cut frequency vs motion
    motion = profile.motion_level
    if motion == "high" and jump_freq in ("high", "medium"):
        score += 0.15
        reasons.append("High motion matches frequent cuts")
    elif motion == "low" and jump_freq in ("none", "low"):
        score += 0.15
        reasons.append("Low motion matches minimal cuts")
    else:
        score += 0.05

    return min(1.0, score), reasons


def _category_score(profile: ContentProfile, preset: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Score topic/category alignment via keyword matching."""
    reasons = []
    preset_category = preset.get("category", "")

    score = 0.0

    # Direct category match
    if profile.detected_category == preset_category:
        score += 0.30
        reasons.append(f"Topic matches '{preset_category}' category")

    # Preset-specific keyword hits
    preset_id = preset.get("id", "")
    specific_keywords = _PRESET_KEYWORDS.get(preset_id, [])
    specific_hits = 0
    text_lower = profile.transcript_text.lower()
    for kw in specific_keywords:
        if kw in text_lower:
            specific_hits += 1
    if specific_hits > 0:
        bonus = min(0.30, specific_hits * 0.08)
        score += bonus
        reasons.append(f"Matched {specific_hits} preset-specific keywords for '{preset_id}'")

    return min(1.0, score), reasons


def _visual_score(profile: ContentProfile, preset: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Score visual characteristics alignment."""
    camera = preset.get("cameraMotion", {})
    zoom_freq = camera.get("zoomFrequency", "occasional")
    reasons = []
    score = 0.0

    # Multi-face = podcast/group → speaker tracking presets
    if profile.face_count > 1:
        if preset.get("id") in ("podcast-clip-fire", "webinar-highlight"):
            score += 0.20
            reasons.append("Multi-person matches podcast/group preset")

    # Screencast = no zooms needed
    if profile.is_screencast and zoom_freq == "none":
        score += 0.15
        reasons.append("Screencast matches static camera preset")
    elif not profile.is_screencast and zoom_freq != "none":
        score += 0.10

    return min(1.0, score), reasons


def match_presets(
    profile: ContentProfile,
    registry: Optional[StylePresetRegistry] = None,
    top_k: int = 5,
    category_filter: Optional[str] = None,
) -> List[PresetMatch]:
    """Score all presets against a content profile and return the top-k matches.

    Args:
        profile: The content profile to match against.
        registry: Preset registry (loads default if None).
        top_k: Number of top results to return.
        category_filter: Optional category to restrict matching to.

    Returns:
        Sorted list of PresetMatch (best first).
    """
    if registry is None:
        registry = StylePresetRegistry()

    presets = registry.all()
    if category_filter:
        presets = [p for p in presets if p.get("category") == category_filter]

    matches: List[PresetMatch] = []

    for preset in presets:
        reasons: List[str] = []

        # Weighted scoring: pacing (35%), category (40%), visual (25%)
        p_score, p_reasons = _pacing_score(profile, preset)
        c_score, c_reasons = _category_score(profile, preset)
        v_score, v_reasons = _visual_score(profile, preset)

        reasons.extend(p_reasons)
        reasons.extend(c_reasons)
        reasons.extend(v_reasons)

        total = p_score * 0.35 + c_score * 0.40 + v_score * 0.25
        total = round(min(1.0, total), 4)

        matches.append(PresetMatch(
            preset_id=preset["id"],
            preset_name=preset["name"],
            category=preset["category"],
            score=total,
            reasons=reasons,
        ))

    matches.sort(key=lambda m: m.score, reverse=True)
    return matches[:top_k]


# ─── Preset → AutoEditConfig Translation ─────────────────────────────────────

def preset_to_config(preset: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a StylePreset dict into an AutoEditConfig-compatible dict.

    Returns a dict that can be passed directly to
    ``auto_editor.get_auto_edit_config(config_data=result)``.
    """
    pacing = preset.get("targetPacing", {})
    camera = preset.get("cameraMotion", {})
    captions = preset.get("captions", {})
    color = preset.get("colorGrade", {})
    assets = preset.get("assets", {})

    # Map jump cut frequency to boolean
    jump_freq = pacing.get("jumpCutFrequency", "medium")
    jump_cut_disguises = jump_freq in ("medium", "high")

    # Map zoom frequency to punch_in_zooms + cadence
    zoom_freq = camera.get("zoomFrequency", "occasional")
    punch_in_zooms = zoom_freq != "none"
    if zoom_freq == "frequent":
        max_zooms_per_minute = 8
        min_zoom_cooldown = 5.0
    elif zoom_freq == "occasional":
        max_zooms_per_minute = 4
        min_zoom_cooldown = 10.0
    else:
        max_zooms_per_minute = 0
        min_zoom_cooldown = 60.0

    # Map SFX density to volume/toggle
    sfx_density = assets.get("sfxDensity", "medium")
    transition_sfx = sfx_density != "none"
    sfx_volume = {"none": 0.0, "low": 0.15, "medium": 0.22, "high": 0.30}.get(sfx_density, 0.22)

    # Map caption animation to subtitle effect
    anim = captions.get("animation", "pop")
    effect_map = {"bounce": "pop", "fade": "none", "pop": "pop", "slide": "none", "none": "none"}
    subtitle_effect = effect_map.get(anim, "pop")

    # Map casing
    casing = captions.get("casing", "uppercase")
    uppercase = casing == "uppercase"

    config = {
        # Pacing
        "min_silence_s": pacing.get("silenceCutThresholdSec", 0.35),
        "smart_silence_trimming": True,

        # Camera
        "punch_in_zooms": punch_in_zooms,
        "max_zoom": camera.get("zoomScale", 1.15),
        "ema_alpha": camera.get("trackingSmoothing", 0.08),
        "max_zooms_per_minute": max_zooms_per_minute,
        "min_zoom_cooldown_s": min_zoom_cooldown,
        "zoom_cadence_mode": "controlled",

        # Cuts & transitions
        "jump_cut_disguises": jump_cut_disguises,
        "motion_transitions": jump_cut_disguises,

        # Color
        "saturation": color.get("saturation", 1.06),
        "vignette": color.get("vignette", 0.0) > 0.15,
        "vignette_angle": color.get("vignette", 0.25),
        "contrast": 1.05 if color.get("saturation", 1.0) > 1.0 else 1.00,

        # Audio
        "transition_sfx": transition_sfx,
        "sfx_volume": sfx_volume,

        # Captions
        "burn_subtitles": True,
        "caption_font_size": captions.get("fontSize", 50),
        "caption_max_chars": max(10, captions.get("wordsPerChunk", 3) * 6),
        "caption_margin_v": int((1.0 - captions.get("screenAnchor", {}).get("y", 0.82)) * 1920),
        "caption_margin_l": 65,
        "caption_margin_r": 86,
        "subtitle_settings": {
            "font_name": captions.get("fontFamily", "Anton"),
            "font_size": captions.get("fontSize", 50),
            "font_color": captions.get("baseColor", "#FFFFFF"),
            "highlight_color": captions.get("activeWordColor", "#FFE500"),
            "border_color": "#000000",
            "border_width": captions.get("strokeWidth", 4),
            "style": "karaoke",
            "uppercase": uppercase,
            "position": "bottom",
        },
    }

    return config


def get_preset_and_config(
    preset_id: str,
    registry: Optional[StylePresetRegistry] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Look up a preset by ID and return (preset_dict, auto_edit_config_dict).

    Returns (None, None) if the preset ID is not found.
    """
    if registry is None:
        registry = StylePresetRegistry()
    preset = registry.get(preset_id)
    if preset is None:
        return None, None
    return preset, preset_to_config(preset)


def auto_match_and_config(
    transcript_text: str = "",
    duration_seconds: float = 60.0,
    silence_ratio: float = 0.0,
    face_count: int = 1,
    motion_level: str = "medium",
    category_filter: Optional[str] = None,
) -> Tuple[PresetMatch, Dict[str, Any]]:
    """One-call convenience: profile → match → config.

    Returns (best_match, auto_edit_config_dict).
    """
    registry = StylePresetRegistry()
    profile = build_content_profile(
        transcript_text=transcript_text,
        duration_seconds=duration_seconds,
        silence_ratio=silence_ratio,
        face_count=face_count,
        motion_level=motion_level,
    )
    matches = match_presets(profile, registry=registry,
                           top_k=1, category_filter=category_filter)
    best = matches[0] if matches else PresetMatch(
        preset_id="youtube-shorts-punchy",
        preset_name="YouTube Shorts Punchy",
        category="creator_hype",
        score=0.0,
        reasons=["Fallback default"],
    )
    preset = registry.get(best.preset_id) or registry.all()[0]
    config = preset_to_config(preset)
    return best, config
