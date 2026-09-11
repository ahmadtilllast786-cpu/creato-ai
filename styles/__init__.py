"""Style Preset System — multi-style editing presets with semantic auto-matching."""
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

__all__ = [
    "StylePresetRegistry",
    "ContentProfile",
    "PresetMatch",
    "build_content_profile",
    "match_presets",
    "preset_to_config",
    "get_preset_and_config",
    "auto_match_and_config",
]
