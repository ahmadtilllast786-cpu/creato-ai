import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
    Link2, Upload, FileVideo, X, Info, Loader2, ChevronDown, Music, Volume2,
    Sparkles, RefreshCw, Wand2, Move, GripHorizontal, Eye, Sliders, Check,
    RotateCcw, ShieldCheck, ShieldAlert
} from 'lucide-react';
import { getApiUrl } from '../config';

const SUPPORTED_PLATFORMS = [
    'YouTube', 'Vimeo', 'TikTok', 'X / Twitter', 'Twitch',
    'Facebook', 'Instagram', 'Dailymotion', 'Reddit', 'Streamable',
];

const CLIP_TARGET_PRESETS = [
    { value: '', label: 'Auto' },
    { value: '3', label: '3' },
    { value: '5', label: '5' },
    { value: '7', label: '7' },
    { value: '10', label: '10' },
];

const SUBTITLE_PRESET_CONFIGS = {
    shorts: {
        id: 'shorts',
        label: 'Shorts Pop',
        desc: 'Red Accent / Bold Pop',
        fontName: 'Anton',
        fontSize: 44,
        fontColor: '#FFFFFF',
        highlightColor: '#FF2222',
        borderColor: '#000000',
        borderWidth: 4,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'pop',
        uppercase: true,
        baseOpacity: 0.85,
    },
    tiktok: {
        id: 'tiktok',
        label: 'TikTok Viral',
        desc: 'Cyan & Red Accent',
        fontName: 'Verdana',
        fontSize: 40,
        fontColor: '#FFFFFF',
        highlightColor: '#FE2C55',
        borderColor: '#000000',
        borderWidth: 3,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'none',
        uppercase: false,
        baseOpacity: 0.8,
    },
    reels: {
        id: 'reels',
        label: 'Reels Modern',
        desc: 'Pink Gradient Accent',
        fontName: 'Verdana',
        fontSize: 40,
        fontColor: '#FFFFFF',
        highlightColor: '#E1306C',
        borderColor: '#000000',
        borderWidth: 3,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'none',
        uppercase: false,
        baseOpacity: 0.8,
    },
    beast: {
        id: 'beast',
        label: 'MrBeast',
        desc: 'Bold Impact Yellow Pop',
        fontName: 'Impact',
        fontSize: 48,
        fontColor: '#FFFFFF',
        highlightColor: '#FFE500',
        borderColor: '#000000',
        borderWidth: 5,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'pop',
        uppercase: true,
        baseOpacity: 1.0,
    },
    gold: {
        id: 'gold',
        label: 'Gold Luxury',
        desc: 'Warm Gold Glow',
        fontName: 'Verdana',
        fontSize: 40,
        fontColor: '#FFFFFF',
        highlightColor: '#FFD700',
        borderColor: '#000000',
        borderWidth: 3,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'glow',
        uppercase: false,
        baseOpacity: 0.75,
    },
    neon: {
        id: 'neon',
        label: 'Cyber Neon',
        desc: 'Vibrant Green Glow',
        fontName: 'Verdana',
        fontSize: 40,
        fontColor: '#FFFFFF',
        highlightColor: '#00FF88',
        borderColor: '#000000',
        borderWidth: 3,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'glow',
        uppercase: false,
        baseOpacity: 0.7,
    },
    cyber: {
        id: 'cyber',
        label: 'Cyber Cyan',
        desc: 'Cyan Neon Glow',
        fontName: 'Verdana',
        fontSize: 40,
        fontColor: '#FFFFFF',
        highlightColor: '#00FFFF',
        borderColor: '#000000',
        borderWidth: 3,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'glow',
        uppercase: false,
        baseOpacity: 0.7,
    },
    karaoke: {
        id: 'karaoke',
        label: 'Karaoke',
        desc: 'Coral Red Accent',
        fontName: 'Verdana',
        fontSize: 40,
        fontColor: '#FFFFFF',
        highlightColor: '#FF6B6B',
        borderColor: '#000000',
        borderWidth: 3,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'none',
        uppercase: false,
        baseOpacity: 0.75,
    },
    minimal: {
        id: 'minimal',
        label: 'Minimalist',
        desc: 'Clean White Outline',
        fontName: 'Verdana',
        fontSize: 36,
        fontColor: '#FFFFFF',
        highlightColor: '#FFFFFF',
        borderColor: '#000000',
        borderWidth: 2,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'none',
        uppercase: false,
        baseOpacity: 0.75,
    },
    boxed: {
        id: 'boxed',
        label: 'Boxed Pill',
        desc: 'Purple Frosted Pill',
        fontName: 'Verdana',
        fontSize: 40,
        fontColor: '#FFFFFF',
        highlightColor: '#FFFFFF',
        borderColor: '#000000',
        borderWidth: 2,
        bgColor: '#7C3AED',
        bgOpacity: 0.85,
        effect: 'box',
        uppercase: false,
        baseOpacity: 0.85,
    },
    classic: {
        id: 'classic',
        label: 'Classic',
        desc: 'Standard Yellow / White',
        fontName: 'Verdana',
        fontSize: 38,
        fontColor: '#FFFFFF',
        highlightColor: '#FFE500',
        borderColor: '#000000',
        borderWidth: 2,
        bgColor: '#000000',
        bgOpacity: 0.0,
        effect: 'none',
        uppercase: false,
        baseOpacity: 1.0,
    },
    none: {
        id: 'none',
        label: 'No Captions',
        desc: 'Deliver without subtitles',
    },
};

const SUBTITLE_STYLE_OPTIONS = Object.entries(SUBTITLE_PRESET_CONFIGS).map(([k, v]) => ({
    value: k,
    label: `${v.label} (${v.desc})`,
}));

const FONT_OPTIONS = ['Anton', 'Impact', 'Verdana', 'Arial', 'Helvetica', 'Georgia', 'Courier New'];
const EFFECT_OPTIONS = [
    { value: 'pop', label: 'Pop Scale (Dynamic Zoom)' },
    { value: 'glow', label: 'Neon Glow (Vibrant Radiance)' },
    { value: 'box', label: 'Box Border (Highlighted Outline)' },
    { value: 'none', label: 'Standard Highlight' },
];

const HOOK_STYLE_OPTIONS = [
    { value: 'yellow', label: 'Viral Yellow', desc: 'TikTok punchy bright yellow box with bold black text', bg: '#FFD600', text: '#000000', border: '#E6C200' },
    { value: 'classic', label: 'Black & White', desc: 'Sleek classic dark card with white text', bg: '#121214', text: '#FFFFFF', border: '#27272A' },
    { value: 'neon', label: 'Cyber Neon', desc: 'Electric cyan glow on deep navy box', bg: '#0A192F', text: '#00F0FF', border: '#00F0FF55' },
    { value: 'emerald', label: 'Tech Emerald', desc: 'Deep green card with vivid mint text', bg: '#064E3B', text: '#34D399', border: '#34D39955' },
    { value: 'red', label: 'Breaking Red', desc: 'Urgent breaking news red box with white text', bg: '#DC2626', text: '#FFFFFF', border: '#EF4444' },
    { value: 'purple', label: 'Violet Glow', desc: 'Royal violet card with bright white text', bg: '#6366F1', text: '#FFFFFF', border: '#818CF8' },
    { value: 'orange', label: 'Sunset Orange', desc: 'Warm sunset orange card with bold white text', bg: '#EA580C', text: '#FFFFFF', border: '#F97316' },
    { value: 'white_card', label: 'White Card', desc: 'High-contrast clean white card with black text', bg: '#FFFFFF', text: '#000000', border: '#E4E4E7' },
    { value: 'pill', label: 'Minimal Pill', desc: 'Translucent floating slate pill card', bg: '#1E293B', text: '#F1F5F9', border: '#334155' },
    { value: 'breaking_news', label: 'News Banner', desc: 'Crimson banner with yellow highlight headline', bg: '#B91C1C', text: '#FEF08A', border: '#DC2626' },
    { value: 'outline', label: 'White Outline', desc: 'No box, bold white text with black stroke (MrBeast)', bg: 'transparent', text: '#FFFFFF', border: '#52525B', outline: true },
    { value: 'outline_yellow', label: 'Yellow Outline', desc: 'No box, bold yellow text with black stroke', bg: 'transparent', text: '#FFD600', border: '#52525B', outline: true },
    { value: 'dark', label: 'Dark Sleek', desc: 'Subtle dark card with soft white text', bg: '#18181B', text: '#F4F4F5', border: '#3F3F46' },
];

const HOOK_DURATION_OPTIONS = [
    { value: 'forever', label: 'Whole Video (Until End - Recommended)' },
    { value: '5', label: 'First 5 Seconds' },
    { value: '8', label: 'First 8 Seconds' },
    { value: '10', label: 'First 10 Seconds' },
];

const HOOK_POSITION_OPTIONS = [
    { value: 'top', label: 'Top Header Safe Zone (Recommended)' },
    { value: 'safe_top', label: 'Top Header Safe Zone (Slight Margin)' },
    { value: 'center', label: 'Center' },
    { value: 'bottom', label: 'Bottom' },
];

export default function MediaInput({ onProcess, isProcessing }) {
    const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
    // File upload is the primary path; the link is secondary.
    const [mode, setMode] = useState('file'); // 'file' | 'url'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [acknowledged, setAcknowledged] = useState(false);
    const [outputFormat, setOutputFormat] = useState('vertical'); // vertical | horizontal | square
    const [showInfo, setShowInfo] = useState(false);
    // Advanced generation controls — empty string means "let the AI decide",
    // which keeps the default pipeline behavior untouched.
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [targetClips, setTargetClips] = useState('');
    const [clipMinSeconds, setClipMinSeconds] = useState('');
    const [clipMaxSeconds, setClipMaxSeconds] = useState('');
    const [scanZoneCount, setScanZoneCount] = useState(() => {
        try { return localStorage.getItem('os_scan_zone_count') || '3'; } catch { return '3'; }
    });
    // Auto-hook: burn the AI hook text into every clip. On by default; the
    // choice persists so turning it off sticks across sessions.
    const [autoHook, setAutoHook] = useState(() => {
        try { return localStorage.getItem('os_auto_hook') !== '0'; } catch { return true; }
    });
    const [autoHookStyle, setAutoHookStyle] = useState(() => {
        try { return localStorage.getItem('os_auto_hook_style') || 'yellow'; } catch { return 'yellow'; }
    });
    const [autoHookDuration, setAutoHookDuration] = useState(() => {
        try {
            const saved = localStorage.getItem('os_auto_hook_duration');
            if (!saved || saved === '5' || saved === '0') return 'forever';
            return saved;
        } catch { return 'forever'; }
    });
    const [autoHookPosition, setAutoHookPosition] = useState(() => {
        try { return localStorage.getItem('os_auto_hook_position') || 'top'; } catch { return 'top'; }
    });
    const [subtitlesAppliedMsg, setSubtitlesAppliedMsg] = useState(false);
    const [hookAppliedMsg, setHookAppliedMsg] = useState(false);
    // Layout: 'auto' lets the AI pick per video (server default); the others
    // force one on so a podcast host who knows what they uploaded doesn't
    // depend on the detector, and 'none' keeps the plain single crop.
    const [layout, setLayout] = useState(() => {
        try { return localStorage.getItem('os_layout') || 'auto'; } catch { return 'auto'; }
    });
    // Subtitle Style & Draggable Position Studio state
    const [subtitleStyle, setSubtitleStyle] = useState(() => {
        try { return localStorage.getItem('os_subtitle_style') || 'shorts'; } catch { return 'shorts'; }
    });
    const [subtitleYOffset, setSubtitleYOffset] = useState(() => {
        try {
            const v = localStorage.getItem('os_subtitle_y_offset');
            return v != null ? parseFloat(v) : 78;
        } catch { return 78; }
    });
    const [subtitleFont, setSubtitleFont] = useState(() => {
        try { return localStorage.getItem('os_subtitle_font') || 'Anton'; } catch { return 'Anton'; }
    });
    const [subtitleFontSize, setSubtitleFontSize] = useState(() => {
        try { return parseInt(localStorage.getItem('os_subtitle_font_size') || '44'); } catch { return 44; }
    });
    const [subtitleFontColor, setSubtitleFontColor] = useState(() => {
        try { return localStorage.getItem('os_subtitle_font_color') || '#FFFFFF'; } catch { return '#FFFFFF'; }
    });
    const [subtitleHighlightColor, setSubtitleHighlightColor] = useState(() => {
        try { return localStorage.getItem('os_subtitle_highlight_color') || '#FF2222'; } catch { return '#FF2222'; }
    });
    const [subtitleBorderColor, setSubtitleBorderColor] = useState(() => {
        try { return localStorage.getItem('os_subtitle_border_color') || '#000000'; } catch { return '#000000'; }
    });
    const [subtitleBorderWidth, setSubtitleBorderWidth] = useState(() => {
        try { return parseInt(localStorage.getItem('os_subtitle_border_width') || '4'); } catch { return 4; }
    });
    const [subtitleBgColor, setSubtitleBgColor] = useState(() => {
        try { return localStorage.getItem('os_subtitle_bg_color') || '#000000'; } catch { return '#000000'; }
    });
    const [subtitleBgOpacity, setSubtitleBgOpacity] = useState(() => {
        try { return parseFloat(localStorage.getItem('os_subtitle_bg_opacity') || '0.0'); } catch { return 0.0; }
    });
    const [subtitleUppercase, setSubtitleUppercase] = useState(() => {
        try { return localStorage.getItem('os_subtitle_uppercase') !== '0'; } catch { return true; }
    });
    const [subtitleEffect, setSubtitleEffect] = useState(() => {
        try { return localStorage.getItem('os_subtitle_effect') || 'pop'; } catch { return 'pop'; }
    });
    const [subtitleBaseOpacity, setSubtitleBaseOpacity] = useState(() => {
        try { return parseFloat(localStorage.getItem('os_subtitle_base_opacity') || '0.85'); } catch { return 0.85; }
    });

    const [isDraggingSubtitle, setIsDraggingSubtitle] = useState(false);
    const [showSafeZones, setShowSafeZones] = useState(true);
    const [showCustomSubtitleOptions, setShowCustomSubtitleOptions] = useState(false);
    const [showSubtitleStudio, setShowSubtitleStudio] = useState(true);
    const [wordBeat, setWordBeat] = useState(0);
    const previewScreenRef = useRef(null);

    // Live word highlight ticker
    useEffect(() => {
        if (subtitleStyle === 'none') return;
        const interval = setInterval(() => {
            setWordBeat((prev) => (prev + 1) % 4);
        }, 550);
        return () => clearInterval(interval);
    }, [subtitleStyle]);

    // Local object URL for uploaded video file preview
    const previewVideoUrl = useMemo(() => {
        if (file && file.type && file.type.startsWith('video/')) {
            try { return URL.createObjectURL(file); } catch { return null; }
        }
        return null;
    }, [file]);

    const handleSelectPreset = (presetKey) => {
        setSubtitleStyle(presetKey);
        const preset = SUBTITLE_PRESET_CONFIGS[presetKey];
        if (preset && presetKey !== 'none') {
            setSubtitleFont(preset.fontName);
            setSubtitleFontSize(preset.fontSize);
            setSubtitleFontColor(preset.fontColor);
            setSubtitleHighlightColor(preset.highlightColor);
            setSubtitleBorderColor(preset.borderColor);
            setSubtitleBorderWidth(preset.borderWidth);
            setSubtitleBgColor(preset.bgColor);
            setSubtitleBgOpacity(preset.bgOpacity);
            setSubtitleEffect(preset.effect);
            setSubtitleUppercase(preset.uppercase);
            setSubtitleBaseOpacity(preset.baseOpacity);
        }
    };

    const updateDragPosition = (clientY) => {
        if (!previewScreenRef.current || clientY == null) return;
        const rect = previewScreenRef.current.getBoundingClientRect();
        const relY = clientY - rect.top;
        let pct = (relY / rect.height) * 100;
        pct = Math.max(8, Math.min(92, Math.round(pct * 10) / 10));
        // Magnetic snap points
        if (Math.abs(pct - 15) <= 2.5) pct = 15;
        else if (Math.abs(pct - 50) <= 2.5) pct = 50;
        else if (Math.abs(pct - 78) <= 2.5) pct = 78;
        setSubtitleYOffset(pct);
    };

    const handleSubtitlePointerDown = (e) => {
        e.preventDefault();
        setIsDraggingSubtitle(true);
        const clientY = e.clientY != null ? e.clientY : (e.touches && e.touches[0] ? e.touches[0].clientY : null);
        updateDragPosition(clientY);
    };

    useEffect(() => {
        if (!isDraggingSubtitle) return;
        const onPointerMove = (e) => {
            const clientY = e.clientY != null ? e.clientY : (e.touches && e.touches[0] ? e.touches[0].clientY : null);
            updateDragPosition(clientY);
        };
        const onPointerUp = () => {
            setIsDraggingSubtitle(false);
        };
        window.addEventListener('mousemove', onPointerMove);
        window.addEventListener('mouseup', onPointerUp);
        window.addEventListener('touchmove', onPointerMove, { passive: false });
        window.addEventListener('touchend', onPointerUp);
        return () => {
            window.removeEventListener('mousemove', onPointerMove);
            window.removeEventListener('mouseup', onPointerUp);
            window.removeEventListener('touchmove', onPointerMove);
            window.removeEventListener('touchend', onPointerUp);
        };
    }, [isDraggingSubtitle]);

    const getZoneLabel = (y) => {
        if (y <= 22) return 'Top Safe (Above Subject)';
        if (y <= 40) return 'Upper Third';
        if (y <= 62) return 'Center Safe Area';
        if (y <= 74) return 'Lower Center';
        if (y <= 86) return 'Bottom Safe (Recommended)';
        return 'Lower Edge';
    };

    // Background Audio options
    const [bgAudioFile, setBgAudioFile] = useState(null);
    const [bgAudioVolume, setBgAudioVolume] = useState('0.18');
    // Fresh clip extraction (bypass cache to get new clips)
    const [freshClips, setFreshClips] = useState(true);
    const infoRef = useRef(null);

    // Close the compatibility popover on any outside click.
    useEffect(() => {
        if (!showInfo) return;
        const onClick = (e) => {
            if (infoRef.current && !infoRef.current.contains(e.target)) setShowInfo(false);
        };
        document.addEventListener('mousedown', onClick);
        return () => document.removeEventListener('mousedown', onClick);
    }, [showInfo]);

    useEffect(() => {
        fetch(getApiUrl('/api/config'))
            .then((r) => r.ok ? r.json() : null)
            .then((cfg) => {
                if (cfg && cfg.youtubeUrlEnabled === false) {
                    setYoutubeUrlEnabled(false);
                    setMode('file');
                }
            })
            .catch(() => {});
    }, []);

    // A link pasted in the landing hero: preload it here so the user picks up
    // where they left off. Not auto-submitted — the rights attestation below
    // has to be ticked by the user.
    useEffect(() => {
        let pending = null;
        try {
            pending = localStorage.getItem('os_pending_url');
            if (pending) localStorage.removeItem('os_pending_url');
        } catch { /* ignore */ }
        if (pending) {
            setMode('url');
            setUrl(pending);
        }
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!acknowledged) return;
        const subtitlePosition = subtitleYOffset <= 25 ? 'top' : subtitleYOffset <= 65 ? 'middle' : 'bottom';
        const subtitleConfig = {
            style: subtitleStyle,
            font_name: subtitleFont,
            font_size: subtitleFontSize,
            font_color: subtitleFontColor,
            highlight_color: subtitleHighlightColor,
            border_color: subtitleBorderColor,
            border_width: subtitleBorderWidth,
            bg_color: subtitleBgColor,
            bg_opacity: subtitleBgOpacity,
            effect: subtitleEffect,
            uppercase: subtitleUppercase,
            base_opacity: subtitleBaseOpacity,
            manual_y_offset: subtitleYOffset,
            position: subtitlePosition,
        };
        const advanced = {
            targetClips: targetClips || null,
            clipMinSeconds: clipMinSeconds || null,
            clipMaxSeconds: clipMaxSeconds || null,
            trackScanZones: scanZoneCount || null,
            autoHook,
            autoHookStyle,
            autoHookDuration,
            autoHookPosition,
            layout,
            subtitleStyle,
            subtitleYOffset,
            subtitlePosition,
            subtitleConfig,
            bgAudio: bgAudioFile || null,
            bgAudioVolume,
            freshClips,
        };
        try {
            localStorage.setItem('os_auto_hook', autoHook ? '1' : '0');
            localStorage.setItem('os_auto_hook_style', autoHookStyle);
            localStorage.setItem('os_auto_hook_duration', autoHookDuration);
            localStorage.setItem('os_auto_hook_position', autoHookPosition);
            localStorage.setItem('os_layout', layout);
            localStorage.setItem('os_scan_zone_count', scanZoneCount || '3');
            localStorage.setItem('os_subtitle_style', subtitleStyle);
            localStorage.setItem('os_subtitle_y_offset', String(subtitleYOffset));
            localStorage.setItem('os_subtitle_font', subtitleFont);
            localStorage.setItem('os_subtitle_font_size', String(subtitleFontSize));
            localStorage.setItem('os_subtitle_font_color', subtitleFontColor);
            localStorage.setItem('os_subtitle_highlight_color', subtitleHighlightColor);
            localStorage.setItem('os_subtitle_border_color', subtitleBorderColor);
            localStorage.setItem('os_subtitle_border_width', String(subtitleBorderWidth));
            localStorage.setItem('os_subtitle_bg_color', subtitleBgColor);
            localStorage.setItem('os_subtitle_bg_opacity', String(subtitleBgOpacity));
            localStorage.setItem('os_subtitle_uppercase', subtitleUppercase ? '1' : '0');
            localStorage.setItem('os_subtitle_effect', subtitleEffect);
            localStorage.setItem('os_subtitle_base_opacity', String(subtitleBaseOpacity));
        } catch { /* ignore */ }
        if (mode === 'url' && url) {
            onProcess({ type: 'url', payload: url, acknowledged: true, outputFormat, ...advanced });
        } else if (mode === 'file' && file) {
            onProcess({ type: 'file', payload: file, acknowledged: true, outputFormat, ...advanced });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    return (
        <div className="card p-4 sm:p-6 animate-fade">
            <div className="flex gap-4 sm:gap-6 mb-6 border-b border-rule" data-tutorial="source-tabs">
                <button
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'file'
                        ? 'text-ink border-brass'
                        : 'text-muted border-transparent hover:text-ink2'
                        }`}
                >
                    <Upload size={16} className={`hidden sm:block ${mode === 'file' ? 'text-brass' : ''}`} />
                    Upload File
                </button>
                {youtubeUrlEnabled && (
                    <button
                        onClick={() => setMode('url')}
                        className={`flex items-center gap-2 pb-3 px-1 -mb-px border-b-2 text-sm lowercase whitespace-nowrap transition-colors ${mode === 'url'
                            ? 'text-ink border-brass'
                            : 'text-muted border-transparent hover:text-ink2'
                            }`}
                    >
                        <Link2 size={16} className={`hidden sm:block ${mode === 'url' ? 'text-brass' : ''}`} />
                        Video URL
                    </button>
                )}
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' ? (
                    <div className="space-y-4" data-tutorial="drop-zone">
                        <div className="relative">
                            <input
                                type="url"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="https://... paste a video link"
                                className="input-field pr-11"
                                required
                            />
                            <div className="absolute inset-y-0 right-2 flex items-center" ref={infoRef}>
                                <button
                                    type="button"
                                    onClick={() => setShowInfo((v) => !v)}
                                    aria-label="Supported platforms"
                                    className="p-1.5 text-muted hover:text-brass transition-colors"
                                >
                                    <Info size={16} />
                                </button>
                                {showInfo && (
                                    <div className="absolute right-0 top-full mt-2 w-64 z-20 card p-4 text-left animate-fade">
                                        <p className="eyebrow mb-2">Paste a link from</p>
                                        <div className="flex flex-wrap gap-1.5">
                                            {SUPPORTED_PLATFORMS.map((p) => (
                                                <span key={p} className="text-xs px-2 py-0.5 rounded-full bg-paper3 text-ink2">
                                                    {p}
                                                </span>
                                            ))}
                                        </div>
                                        <p className="text-xs text-muted mt-2.5 leading-relaxed">
                                            …and 1,000+ more sites. If a link has a public video, we can usually fetch it.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div
                        data-tutorial="drop-zone"
                        className={`border-2 border-dashed rounded-card p-6 sm:p-8 text-center transition-colors ${file ? 'border-brass' : 'border-rule2 hover:border-brass'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-ok min-w-0">
                                <FileVideo size={18} className="shrink-0" />
                                <span className="font-medium truncate">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 text-muted hover:text-ink hover:bg-paper3 rounded-full transition-colors"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-muted" size={18} />
                                <p className="text-ink2 lowercase">Click to upload or drag and drop</p>
                                <p className="readout mt-2">MP4, MOV up to 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                {/* Output format selector */}
                <div className="mt-5" data-tutorial="output-format">
                    <p className="eyebrow mb-2">Output format</p>
                    <div className="grid grid-cols-3 gap-2">
                        {[
                            { value: 'vertical', label: '9:16', hint: 'Shorts · Reels · TikTok', w: 18, h: 32 },
                            { value: 'square', label: '1:1', hint: 'Feed posts', w: 28, h: 28 },
                            { value: 'horizontal', label: '16:9', hint: 'Keep landscape · YouTube', w: 36, h: 20 },
                        ].map((f) => {
                            const active = outputFormat === f.value;
                            return (
                                <button
                                    key={f.value}
                                    type="button"
                                    onClick={() => setOutputFormat(f.value)}
                                    className={`py-3 px-2 rounded-input border flex flex-col items-center gap-2 transition-colors
                                        ${active ? 'border-[color:var(--color-accent)] text-ink' : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                                >
                                    {/* Aspect-ratio glyph */}
                                    <span
                                        className="rounded-[3px] border-2 transition-colors"
                                        style={{
                                            width: `${f.w}px`,
                                            height: `${f.h}px`,
                                            borderColor: active ? 'var(--color-accent)' : 'var(--color-rule-2)',
                                            backgroundColor: active ? 'color-mix(in srgb, var(--color-accent) 22%, transparent)' : 'transparent',
                                        }}
                                    />
                                    <span className="block font-mono text-sm leading-none">{f.label}</span>
                                    <span className="block text-[11px] sm:text-[10px] leading-tight text-center text-muted">{f.hint}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Number of videos to generate (Presets: Auto, 3, 5, 7, 10) */}
                <div className="mt-5">
                    <div className="flex items-center justify-between mb-2">
                        <p className="eyebrow">Videos to generate</p>
                        <span className="text-[11px] text-muted">{targetClips ? `${targetClips} clips target` : 'AI Auto-Detect'}</span>
                    </div>
                    <div className="grid grid-cols-5 gap-1.5" role="group" aria-label="clips to generate">
                        {CLIP_TARGET_PRESETS.map((option) => {
                            const active = targetClips === option.value;
                            return (
                                <button
                                    key={option.label}
                                    type="button"
                                    aria-pressed={active}
                                    onClick={() => setTargetClips(option.value)}
                                    className={`py-2 rounded-input border text-xs font-mono transition-colors ${active
                                        ? 'border-[color:var(--color-accent)] text-ink bg-[color-mix(in_srgb,var(--color-accent)_12%,transparent)] font-semibold'
                                        : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                                >
                                    {option.label}
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Subtitle & Caption Live Studio with Draggable 9:16 Preview */}
                <div className="mt-5 p-3.5 sm:p-4 rounded-card bg-paper2/50 border border-rule space-y-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Sparkles size={16} className="text-brass" />
                            <div>
                                <h3 className="text-xs font-semibold text-ink uppercase tracking-wider">
                                    Captions & Subtitle Studio
                                </h3>
                                <p className="text-[11px] text-muted">
                                    Live draggable preview • Applied across all generated clips
                                </p>
                            </div>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-brass/10 border border-brass/25 text-brass">
                                {SUBTITLE_PRESET_CONFIGS[subtitleStyle]?.label || 'Shorts Pop'} • Y: {subtitleYOffset}%
                            </span>
                            <button
                                type="button"
                                onClick={() => setShowSubtitleStudio(!showSubtitleStudio)}
                                className="text-muted hover:text-ink transition-colors p-1"
                                title="Toggle Subtitle Studio"
                            >
                                <ChevronDown size={15} className={`transition-transform duration-200 ${showSubtitleStudio ? 'rotate-180' : ''}`} />
                            </button>
                        </div>
                    </div>

                    {showSubtitleStudio && (
                        <div className="space-y-4 pt-1 animate-fade">
                            {/* Live Interactive Screen & Presets Section */}
                            <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-start">
                                {/* Left: 9:16 Live Interactive Draggable Screen */}
                                <div className="md:col-span-5 flex flex-col items-center">
                                    <div className="w-full flex items-center justify-between mb-1.5 px-1 text-[11px] text-muted">
                                        <span className="flex items-center gap-1 font-mono text-[10px] text-ink">
                                            <Eye size={12} className="text-brass" />
                                            9:16 Live Canvas
                                        </span>
                                        <button
                                            type="button"
                                            onClick={() => setShowSafeZones(!showSafeZones)}
                                            className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition-colors ${
                                                showSafeZones
                                                    ? 'border-emerald-500/40 text-emerald-400 bg-emerald-500/10'
                                                    : 'border-rule text-muted hover:text-ink'
                                            }`}
                                        >
                                            {showSafeZones ? 'Safe Zones ON' : 'Safe Zones OFF'}
                                        </button>
                                    </div>

                                    {/* 9:16 Phone Mockup Viewport */}
                                    <div
                                        ref={previewScreenRef}
                                        className="relative w-full max-w-[240px] sm:max-w-[260px] aspect-[9/16] rounded-2xl overflow-hidden bg-neutral-950 border-2 border-rule2 shadow-2xl select-none"
                                        style={{ touchAction: 'none' }}
                                    >
                                        {/* Background: Video if uploaded, else modern mesh gradient */}
                                        {previewVideoUrl ? (
                                            <video
                                                src={previewVideoUrl}
                                                className="absolute inset-0 w-full h-full object-cover opacity-60 pointer-events-none"
                                                muted
                                                autoPlay
                                                loop
                                                playsInline
                                            />
                                        ) : (
                                            <div className="absolute inset-0 pointer-events-none overflow-hidden">
                                                {/* Simulated Studio Background */}
                                                <div className="absolute inset-0 bg-gradient-to-b from-slate-900 via-neutral-950 to-black" />
                                                <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-48 h-48 rounded-full bg-blue-600/15 blur-2xl" />
                                                <div className="absolute bottom-1/3 left-1/2 -translate-x-1/2 w-44 h-44 rounded-full bg-purple-600/15 blur-2xl" />
                                                {/* Mesh Grid */}
                                                <div className="absolute inset-0 opacity-[0.07] bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:12px_12px]" />
                                                {/* Speaker Silhouette Guide */}
                                                <div className="absolute top-[28%] left-1/2 -translate-x-1/2 flex flex-col items-center opacity-25">
                                                    <div className="w-16 h-16 rounded-full border border-dashed border-white/60 mb-2" />
                                                    <div className="w-28 h-20 rounded-t-3xl border-t border-x border-dashed border-white/60" />
                                                </div>
                                            </div>
                                        )}

                                        {/* Platform Safe Zone Overlays (TikTok / Reels / Shorts) */}
                                        {showSafeZones && (
                                            <div className="absolute inset-0 pointer-events-none z-10">
                                                {/* Top Safe Zone (Y: 0-12%) */}
                                                <div className="absolute top-0 inset-x-0 h-[12%] bg-red-500/10 border-b border-red-500/30 flex items-center justify-center">
                                                    <span className="text-[8px] font-mono text-red-300 tracking-wider uppercase">
                                                        Top Header Safe Zone
                                                    </span>
                                                </div>

                                                {/* Right Action Rail (X: 84-100%, Y: 40-78%) */}
                                                <div className="absolute top-[40%] right-0 w-[16%] bottom-[22%] flex flex-col items-center justify-around py-2 opacity-50">
                                                    <div className="w-4 h-4 rounded-full bg-white/20 border border-white/30" />
                                                    <div className="w-4 h-4 rounded-full bg-white/20 border border-white/30" />
                                                    <div className="w-4 h-4 rounded-full bg-white/20 border border-white/30" />
                                                    <div className="w-3 h-3 rounded-full bg-white/20 border border-white/30" />
                                                </div>

                                                {/* Bottom Platform UI / Music Zone (Y: 78-100%) */}
                                                <div className="absolute bottom-0 inset-x-0 h-[22%] bg-red-500/10 border-t border-red-500/30 flex flex-col justify-end p-2">
                                                    <span className="text-[8px] font-mono text-red-300 text-center tracking-wider uppercase">
                                                        Bottom UI / Comments Safe Zone
                                                    </span>
                                                </div>

                                                {/* Magnetic Reference Guidelines */}
                                                <div className="absolute top-[15%] inset-x-4 border-b border-dashed border-white/20" />
                                                <div className="absolute top-[50%] inset-x-4 border-b border-dashed border-white/20" />
                                                <div className="absolute top-[78%] inset-x-4 border-b border-dashed border-white/20" />
                                            </div>
                                        )}

                                        {/* Viral Hook Headline Live Preview (Anchored in Top Header Safe Zone) */}
                                        {autoHook && (() => {
                                            const selectedHook = HOOK_STYLE_OPTIONS.find((h) => h.value === autoHookStyle) || HOOK_STYLE_OPTIONS[0];
                                            const hookTop = autoHookPosition === 'center' ? '50%' : autoHookPosition === 'bottom' ? '78%' : '4.5%';
                                            return (
                                                <div
                                                    style={{
                                                        top: hookTop,
                                                        left: '50%',
                                                        transform: autoHookPosition === 'center' ? 'translate(-50%, -50%)' : 'translateX(-50%)',
                                                        maxWidth: '92%',
                                                        width: 'max-content',
                                                    }}
                                                    className="absolute z-20 flex flex-col items-center pointer-events-none select-none transition-all duration-150"
                                                >
                                                    <div
                                                        style={{
                                                            backgroundColor: selectedHook.bg || '#FFD600',
                                                            color: selectedHook.text || '#000000',
                                                            borderRadius: '7px',
                                                            padding: selectedHook.bg === 'transparent' ? '2px 6px' : '4px 10px',
                                                            border: selectedHook.border ? `1px solid ${selectedHook.border}` : 'none',
                                                            boxShadow: selectedHook.bg !== 'transparent' ? '0 4px 12px rgba(0,0,0,0.5)' : 'none',
                                                            textShadow: selectedHook.outline ? '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000' : 'none',
                                                            fontFamily: 'Noto Serif, Georgia, serif',
                                                        }}
                                                        className="text-center font-bold text-[10.5px] leading-tight shadow"
                                                    >
                                                        VIRAL HOOK HEADLINE 🎯
                                                    </div>
                                                    <div className="flex items-center gap-1 mt-0.5 px-1.5 py-0.5 rounded-full bg-black/85 border border-emerald-500/30 shadow">
                                                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                                                        <span className="text-[7.5px] font-mono text-emerald-300">
                                                            Top Safe Zone • Whole Video
                                                        </span>
                                                    </div>
                                                </div>
                                            );
                                        })()}

                                        {/* Interactive Draggable Caption Box */}
                                        {subtitleStyle !== 'none' && (
                                            <div
                                                onMouseDown={handleSubtitlePointerDown}
                                                onTouchStart={handleSubtitlePointerDown}
                                                style={{
                                                    top: `${subtitleYOffset}%`,
                                                    left: '50%',
                                                    transform: 'translate(-50%, -50%)',
                                                    fontFamily: subtitleFont,
                                                    fontSize: `${Math.round(subtitleFontSize * 0.38)}px`,
                                                    lineHeight: 1.25,
                                                    textAlign: 'center',
                                                    cursor: isDraggingSubtitle ? 'grabbing' : 'grab',
                                                }}
                                                className={`absolute z-20 w-[88%] px-2.5 py-1.5 rounded-lg transition-shadow duration-150 flex flex-col items-center justify-center select-none ${
                                                    isDraggingSubtitle
                                                        ? 'ring-2 ring-brass ring-offset-2 ring-offset-black/50 shadow-2xl scale-105'
                                                        : 'hover:ring-1 hover:ring-brass/60'
                                                }`}
                                            >
                                                {/* Drag handle badge */}
                                                <div className="flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-black/75 border border-white/20 text-[8px] font-mono text-amber-300 mb-1 opacity-90 shadow">
                                                    <GripHorizontal size={10} />
                                                    <span>drag {subtitleYOffset}%</span>
                                                </div>

                                                {/* Caption Text with live word-beat karaoke effect */}
                                                <div
                                                    style={{
                                                        backgroundColor: subtitleBgOpacity > 0
                                                            ? `${subtitleBgColor}${Math.round(subtitleBgOpacity * 255).toString(16).padStart(2, '0')}`
                                                            : 'transparent',
                                                        borderRadius: '6px',
                                                        padding: subtitleBgOpacity > 0 ? '4px 10px' : '2px 4px',
                                                        textTransform: subtitleUppercase ? 'uppercase' : 'none',
                                                    }}
                                                    className="w-full flex items-center justify-center flex-wrap gap-1.5 font-bold"
                                                >
                                                    {['AI', 'CREATES', 'VIRAL', 'CLIPS'].map((word, idx) => {
                                                        const isActive = idx === wordBeat;
                                                        const subBw = Math.max(subtitleBorderWidth, 0);
                                                        const subBc = subtitleBorderColor;
                                                        const outline = subBw > 0
                                                            ? `-${subBw}px -${subBw}px 0 ${subBc}, ${subBw}px -${subBw}px 0 ${subBc}, -${subBw}px ${subBw}px 0 ${subBc}, ${subBw}px ${subBw}px 0 ${subBc}, 0 2px 4px rgba(0,0,0,0.8)`
                                                            : 'none';
                                                        return (
                                                            <span
                                                                key={word}
                                                                style={{
                                                                    color: isActive ? subtitleHighlightColor : subtitleFontColor,
                                                                    opacity: isActive ? 1.0 : subtitleBaseOpacity,
                                                                    textShadow: isActive && subtitleEffect === 'glow'
                                                                        ? `0 0 12px ${subtitleHighlightColor}, ${outline}`
                                                                        : outline,
                                                                    transform: isActive && subtitleEffect === 'pop' ? 'scale(1.15)' : 'scale(1)',
                                                                    transition: 'all 0.12s ease-out',
                                                                }}
                                                            >
                                                                {word}
                                                            </span>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                        )}

                                        {/* Status message when captions disabled */}
                                        {subtitleStyle === 'none' && (
                                            <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-muted text-xs">
                                                <span>Captions turned off for this generation</span>
                                            </div>
                                        )}
                                    </div>

                                    {/* Position Quick Snap Buttons */}
                                    <div className="w-full max-w-[260px] flex items-center justify-between gap-1 mt-2.5">
                                        <button
                                            type="button"
                                            onClick={() => setSubtitleYOffset(15)}
                                            className={`flex-1 py-1 text-[10px] font-mono rounded border transition-colors ${
                                                subtitleYOffset === 15
                                                    ? 'border-brass bg-brass/15 text-brass font-bold'
                                                    : 'border-rule text-muted hover:border-rule2 hover:text-ink'
                                            }`}
                                        >
                                            Top (15%)
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setSubtitleYOffset(50)}
                                            className={`flex-1 py-1 text-[10px] font-mono rounded border transition-colors ${
                                                subtitleYOffset === 50
                                                    ? 'border-brass bg-brass/15 text-brass font-bold'
                                                    : 'border-rule text-muted hover:border-rule2 hover:text-ink'
                                            }`}
                                        >
                                            Center (50%)
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setSubtitleYOffset(78)}
                                            className={`flex-1 py-1 text-[10px] font-mono rounded border transition-colors ${
                                                subtitleYOffset === 78
                                                    ? 'border-brass bg-brass/15 text-brass font-bold'
                                                    : 'border-rule text-muted hover:border-rule2 hover:text-ink'
                                            }`}
                                        >
                                            Bottom (78%)
                                        </button>
                                    </div>
                                    <p className="text-[10px] text-muted text-center mt-1">
                                        Drag the caption box anywhere on screen to adjust height
                                    </p>
                                </div>

                                {/* Right: Presets Grid & Fine-Tuning */}
                                <div className="md:col-span-7 space-y-3">
                                    <div className="flex items-center justify-between">
                                        <span className="eyebrow">Select Visual Style</span>
                                        <span className="text-[11px] font-mono text-muted">11 Presets</span>
                                    </div>

                                    {/* Style Presets Grid */}
                                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
                                        {Object.entries(SUBTITLE_PRESET_CONFIGS).map(([key, preset]) => {
                                            const active = subtitleStyle === key;
                                            return (
                                                <button
                                                    key={key}
                                                    type="button"
                                                    onClick={() => handleSelectPreset(key)}
                                                    className={`p-2 rounded-input border text-left transition-all relative overflow-hidden flex flex-col justify-between min-h-[58px] ${
                                                        active
                                                            ? 'border-brass bg-brass/10 ring-1 ring-brass text-ink shadow-sm'
                                                            : 'border-rule2 bg-paper hover:border-rule hover:bg-paper2/50 text-muted'
                                                    }`}
                                                >
                                                    <div className="flex items-center justify-between w-full mb-1">
                                                        <span className={`text-xs font-semibold truncate ${active ? 'text-brass' : 'text-ink'}`}>
                                                            {preset.label}
                                                        </span>
                                                        {active && <Check size={12} className="text-brass shrink-0" />}
                                                    </div>
                                                    <span className="text-[10px] text-muted leading-tight truncate">
                                                        {preset.desc}
                                                    </span>
                                                    {preset.highlightColor && (
                                                        <div
                                                            className="w-full h-1 rounded-full mt-1.5 opacity-80"
                                                            style={{ backgroundColor: preset.highlightColor }}
                                                        />
                                                    )}
                                                </button>
                                            );
                                        })}
                                    </div>

                                    {/* Custom Styling Expander */}
                                    {subtitleStyle !== 'none' && (
                                        <div className="pt-2 border-t border-rule">
                                            <button
                                                type="button"
                                                onClick={() => setShowCustomSubtitleOptions(!showCustomSubtitleOptions)}
                                                className="w-full flex items-center justify-between py-1.5 text-xs text-muted hover:text-ink transition-colors"
                                            >
                                                <span className="flex items-center gap-1.5 font-medium">
                                                    <Sliders size={13} className="text-brass" />
                                                    Customize Font, Colors & Effects
                                                </span>
                                                <span className={`transition-transform duration-200 ${showCustomSubtitleOptions ? 'rotate-180' : ''}`}>
                                                    ▾
                                                </span>
                                            </button>

                                            {showCustomSubtitleOptions && (
                                                <div className="space-y-3 pt-2 animate-fade">
                                                    {/* Font & Effect */}
                                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                                                        <div>
                                                            <p className="eyebrow mb-1">Font Family</p>
                                                            <select
                                                                value={subtitleFont}
                                                                onChange={(e) => setSubtitleFont(e.target.value)}
                                                                className="input-field w-full text-xs py-1.5"
                                                            >
                                                                {FONT_OPTIONS.map((f) => (
                                                                    <option key={f} value={f}>{f}</option>
                                                                ))}
                                                            </select>
                                                        </div>
                                                        <div>
                                                            <p className="eyebrow mb-1">Highlight Effect</p>
                                                            <select
                                                                value={subtitleEffect}
                                                                onChange={(e) => setSubtitleEffect(e.target.value)}
                                                                className="input-field w-full text-xs py-1.5"
                                                            >
                                                                {EFFECT_OPTIONS.map((eff) => (
                                                                    <option key={eff.value} value={eff.value}>{eff.label}</option>
                                                                ))}
                                                            </select>
                                                        </div>
                                                    </div>

                                                    {/* Colors: Highlight Color & Text Color */}
                                                    <div className="grid grid-cols-2 gap-2.5">
                                                        <div>
                                                            <p className="eyebrow mb-1">Active Word Color</p>
                                                            <div className="flex items-center gap-2">
                                                                <input
                                                                    type="color"
                                                                    value={subtitleHighlightColor}
                                                                    onChange={(e) => setSubtitleHighlightColor(e.target.value)}
                                                                    className="w-7 h-7 rounded border border-rule cursor-pointer p-0 bg-transparent"
                                                                />
                                                                <span className="text-[11px] font-mono text-muted uppercase">
                                                                    {subtitleHighlightColor}
                                                                </span>
                                                            </div>
                                                        </div>
                                                        <div>
                                                            <p className="eyebrow mb-1">Base Text Color</p>
                                                            <div className="flex items-center gap-2">
                                                                <input
                                                                    type="color"
                                                                    value={subtitleFontColor}
                                                                    onChange={(e) => setSubtitleFontColor(e.target.value)}
                                                                    className="w-7 h-7 rounded border border-rule cursor-pointer p-0 bg-transparent"
                                                                />
                                                                <span className="text-[11px] font-mono text-muted uppercase">
                                                                    {subtitleFontColor}
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    {/* Sliders: Border Width & Drag Height */}
                                                    <div className="space-y-2">
                                                        <div className="space-y-1">
                                                            <div className="flex justify-between text-[11px]">
                                                                <span className="eyebrow">Border Outline</span>
                                                                <span className="font-mono text-muted">{subtitleBorderWidth}px</span>
                                                            </div>
                                                            <input
                                                                type="range"
                                                                min="0"
                                                                max="6"
                                                                value={subtitleBorderWidth}
                                                                onChange={(e) => setSubtitleBorderWidth(parseInt(e.target.value))}
                                                                className="w-full accent-[var(--color-accent)]"
                                                            />
                                                        </div>

                                                        <div className="space-y-1">
                                                            <div className="flex justify-between text-[11px]">
                                                                <span className="eyebrow">Vertical Height (Y%)</span>
                                                                <span className="font-mono text-brass">{subtitleYOffset}%</span>
                                                            </div>
                                                            <input
                                                                type="range"
                                                                min="10"
                                                                max="90"
                                                                value={subtitleYOffset}
                                                                onChange={(e) => setSubtitleYOffset(parseFloat(e.target.value))}
                                                                className="w-full accent-[var(--color-accent)]"
                                                            />
                                                        </div>
                                                    </div>

                                                    {/* Toggles: Uppercase & Dim Inactive Words */}
                                                    <div className="flex items-center justify-between pt-1">
                                                        <label className="flex items-center gap-2 text-xs text-ink cursor-pointer">
                                                            <input
                                                                type="checkbox"
                                                                checked={subtitleUppercase}
                                                                onChange={(e) => setSubtitleUppercase(e.target.checked)}
                                                                className="w-3.5 h-3.5 accent-[var(--color-accent)] cursor-pointer"
                                                            />
                                                            <span>UPPERCASE Text</span>
                                                        </label>
                                                        <button
                                                            type="button"
                                                            onClick={() => handleSelectPreset(subtitleStyle)}
                                                            className="text-[11px] text-muted hover:text-brass flex items-center gap-1 transition-colors"
                                                            title="Reset customization to preset default"
                                                        >
                                                            <RotateCcw size={11} />
                                                            <span>Reset to preset</span>
                                                        </button>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Subtitles Apply & Reset Action Bar */}
                            <div className="pt-3 border-t border-rule flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2.5">
                                <button
                                    type="button"
                                    onClick={() => {
                                        handleSelectPreset('shorts');
                                        setSubtitleYOffset(78);
                                        setSubtitleUppercase(true);
                                        setSubtitlesAppliedMsg(false);
                                    }}
                                    className="px-3 py-1.5 rounded text-xs text-muted hover:text-ink border border-rule hover:border-rule2 transition-colors flex items-center justify-center gap-1.5"
                                >
                                    <RotateCcw size={12} />
                                    <span>Cancel / Reset Subtitles</span>
                                </button>
                                <div className="flex items-center justify-end gap-2">
                                    {subtitlesAppliedMsg && (
                                        <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1 animate-fade">
                                            <Check size={13} /> Subtitle Settings Applied
                                        </span>
                                    )}
                                    <button
                                        type="button"
                                        onClick={() => {
                                            try {
                                                localStorage.setItem('os_subtitle_style', subtitleStyle);
                                                localStorage.setItem('os_subtitle_y_offset', String(subtitleYOffset));
                                            } catch { /* ignore */ }
                                            setSubtitlesAppliedMsg(true);
                                            setTimeout(() => setSubtitlesAppliedMsg(false), 3000);
                                        }}
                                        className="px-4 py-1.5 rounded bg-brass text-black font-semibold text-xs hover:bg-brass/90 transition-colors flex items-center justify-center gap-1.5 shadow-sm"
                                    >
                                        <Check size={13} />
                                        <span>Apply Subtitle Settings</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Viral Hook Headline & Visual Style Options */}
                <div className="mt-5 p-3.5 sm:p-4 rounded-card bg-paper2/50 border border-rule space-y-4">
                    <div className="flex items-center justify-between">
                        <label className="flex items-center gap-2 text-xs font-semibold text-ink uppercase tracking-wider cursor-pointer select-none">
                            <input
                                type="checkbox"
                                checked={autoHook}
                                onChange={(e) => setAutoHook(e.target.checked)}
                                className="w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                            />
                            <Wand2 size={16} className="text-brass" />
                            <span>Burn Hook Headlines on Clip</span>
                        </label>
                        <div className="flex items-center gap-2">
                            {autoHook && (
                                <span className="text-[10px] text-brass uppercase font-mono px-2 py-0.5 rounded bg-brass/10 border border-brass/25">
                                    {HOOK_STYLE_OPTIONS.find((h) => h.value === autoHookStyle)?.label || 'Viral Yellow'} • {autoHookDuration === 'forever' ? 'Whole Video' : `${autoHookDuration}s`}
                                </span>
                            )}
                        </div>
                    </div>

                    {autoHook && (
                        <div className="space-y-4 pt-1 animate-fade">
                            <div>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="eyebrow">Select Hook Visual Style</span>
                                    <span className="text-[11px] font-mono text-muted">13 Styles (Like Subtitles)</span>
                                </div>

                                {/* Hook Visual Styles Grid (13 Selectable Cards) */}
                                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                                    {HOOK_STYLE_OPTIONS.map((opt) => {
                                        const active = autoHookStyle === opt.value;
                                        return (
                                            <button
                                                key={opt.value}
                                                type="button"
                                                onClick={() => setAutoHookStyle(opt.value)}
                                                className={`p-2.5 rounded-input border text-left transition-all relative overflow-hidden flex flex-col justify-between min-h-[82px] ${
                                                    active
                                                        ? 'border-brass bg-brass/10 ring-1 ring-brass text-ink shadow-sm'
                                                        : 'border-rule2 bg-paper hover:border-rule hover:bg-paper2/50 text-muted'
                                                }`}
                                            >
                                                {/* Visual Badge Preview */}
                                                <div
                                                    style={{
                                                        backgroundColor: opt.bg,
                                                        color: opt.text,
                                                        border: opt.border ? `1px solid ${opt.border}` : 'none',
                                                        textShadow: opt.outline ? '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000' : 'none',
                                                    }}
                                                    className="w-full py-1 px-1.5 rounded text-center text-[10px] font-bold tracking-tight mb-1.5 shadow-sm truncate"
                                                >
                                                    HOOK PREVIEW
                                                </div>

                                                <div className="flex items-center justify-between w-full mb-0.5">
                                                    <span className={`text-xs font-semibold truncate ${active ? 'text-brass' : 'text-ink'}`}>
                                                        {opt.label}
                                                    </span>
                                                    {active && <Check size={12} className="text-brass shrink-0" />}
                                                </div>
                                                <span className="text-[10px] text-muted leading-tight line-clamp-2">
                                                    {opt.desc}
                                                </span>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* Duration & Position Controls */}
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-rule">
                                <div>
                                    <div className="flex items-center justify-between mb-1.5">
                                        <p className="eyebrow">Hook Duration</p>
                                        <span className="text-[10px] font-mono text-emerald-400">Whole Video Default</span>
                                    </div>
                                    <select
                                        value={autoHookDuration}
                                        onChange={(e) => setAutoHookDuration(e.target.value)}
                                        className="input-field w-full text-xs py-2"
                                        aria-label="Hook Duration"
                                    >
                                        {HOOK_DURATION_OPTIONS.map((opt) => (
                                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                                        ))}
                                    </select>
                                    <p className="text-[10px] text-muted mt-1">
                                        Permanently visible until the video ends to maximize short-form hook retention.
                                    </p>
                                </div>

                                <div>
                                    <div className="flex items-center justify-between mb-1.5">
                                        <p className="eyebrow">Hook Position</p>
                                        <span className="text-[10px] font-mono text-brass">Top Safe Zone</span>
                                    </div>
                                    <select
                                        value={autoHookPosition}
                                        onChange={(e) => setAutoHookPosition(e.target.value)}
                                        className="input-field w-full text-xs py-2"
                                        aria-label="Hook Position"
                                    >
                                        {HOOK_POSITION_OPTIONS.map((opt) => (
                                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                                        ))}
                                    </select>
                                    <p className="text-[10px] text-muted mt-1">
                                        Anchored inside the Top Header Safe Zone so it never overlaps speaker faces or captions.
                                    </p>
                                </div>
                            </div>

                            {/* Hook Apply & Reset Action Bar */}
                            <div className="pt-3 border-t border-rule flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-2.5">
                                <button
                                    type="button"
                                    onClick={() => {
                                        setAutoHookStyle('yellow');
                                        setAutoHookDuration('forever');
                                        setAutoHookPosition('top');
                                        setHookAppliedMsg(false);
                                    }}
                                    className="px-3 py-1.5 rounded text-xs text-muted hover:text-ink border border-rule hover:border-rule2 transition-colors flex items-center justify-center gap-1.5"
                                >
                                    <RotateCcw size={12} />
                                    <span>Cancel / Reset Hook</span>
                                </button>
                                <div className="flex items-center justify-end gap-2">
                                    {hookAppliedMsg && (
                                        <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1 animate-fade">
                                            <Check size={13} /> Hook Settings Applied
                                        </span>
                                    )}
                                    <button
                                        type="button"
                                        onClick={() => {
                                            try {
                                                localStorage.setItem('os_auto_hook_style', autoHookStyle);
                                                localStorage.setItem('os_auto_hook_duration', autoHookDuration);
                                                localStorage.setItem('os_auto_hook_position', autoHookPosition);
                                            } catch { /* ignore */ }
                                            setHookAppliedMsg(true);
                                            setTimeout(() => setHookAppliedMsg(false), 3000);
                                        }}
                                        className="px-4 py-1.5 rounded bg-brass text-black font-semibold text-xs hover:bg-brass/90 transition-colors flex items-center justify-center gap-1.5 shadow-sm"
                                    >
                                        <Check size={13} />
                                        <span>Apply Hook Settings</span>
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Background Audio (BGM) */}
                <div className="mt-5 p-3.5 rounded-input bg-paper2/50 border border-rule2">
                    <div className="flex items-center justify-between mb-2">
                        <label className="flex items-center gap-2 text-xs font-medium text-ink cursor-pointer">
                            <Music size={14} className="text-brass" />
                            Background Audio (Low Voice)
                        </label>
                        {bgAudioFile && (
                            <button
                                type="button"
                                onClick={() => setBgAudioFile(null)}
                                className="text-[11px] text-muted hover:text-ink flex items-center gap-1 transition-colors"
                            >
                                <X size={12} /> Remove
                            </button>
                        )}
                    </div>
                    {bgAudioFile ? (
                        <div className="space-y-3">
                            <div className="flex items-center gap-2 text-xs bg-paper3 p-2 rounded border border-rule truncate">
                                <Music size={14} className="text-brass shrink-0" />
                                <span className="truncate flex-1 font-mono text-[11px]">{bgAudioFile.name}</span>
                                <span className="text-muted text-[10px] shrink-0">{(bgAudioFile.size / 1024 / 1024).toFixed(1)} MB</span>
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="flex items-center gap-1.5 text-muted shrink-0 text-xs">
                                    <Volume2 size={13} className="text-brass" />
                                    <span>Low Voice:</span>
                                </div>
                                <input
                                    type="range"
                                    min="0.05"
                                    max="0.40"
                                    step="0.01"
                                    value={bgAudioVolume}
                                    onChange={(e) => setBgAudioVolume(e.target.value)}
                                    className="flex-1 accent-[var(--color-accent)] h-1.5 bg-paper3 rounded-lg cursor-pointer"
                                />
                                <span className="font-mono text-[11px] text-ink2 w-9 text-right">{Math.round(parseFloat(bgAudioVolume) * 100)}%</span>
                            </div>
                            <p className="text-[11px] text-muted leading-tight">
                                Intelligently sliced: each generated video gets a different part of this audio softly mixed in background with smooth fade-in/out.
                            </p>
                        </div>
                    ) : (
                        <label className="flex items-center justify-center gap-2 py-2.5 px-3 border border-dashed border-rule2 hover:border-brass rounded cursor-pointer transition-colors text-xs text-muted hover:text-ink2">
                            <input
                                type="file"
                                accept="audio/*"
                                onChange={(e) => {
                                    if (e.target.files && e.target.files[0]) {
                                        setBgAudioFile(e.target.files[0]);
                                    }
                                }}
                                className="hidden"
                            />
                            <Music size={13} className="text-brass" />
                            <span>Add background audio file (MP3, WAV, M4A)</span>
                        </label>
                    )}
                </div>

                {/* Fresh Clips Toggle */}
                <div className="mt-4 flex items-center justify-between">
                    <label className="flex items-center gap-2 text-xs text-ink2 cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={freshClips}
                            onChange={(e) => setFreshClips(e.target.checked)}
                            className="w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                        />
                        <RefreshCw size={13} className="text-brass" />
                        <span>Generate new & fresh clips every time (different moments)</span>
                    </label>
                </div>

                {/* Advanced generation controls — collapsed by default; blank = AI decides */}
                <div className="mt-4">
                    <button
                        type="button"
                        onClick={() => setShowAdvanced((v) => !v)}
                        className="flex items-center gap-1.5 text-xs text-muted hover:text-ink2 lowercase transition-colors"
                    >
                        <ChevronDown size={14} className={`transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                        advanced options
                        {(clipMinSeconds || clipMaxSeconds || scanZoneCount !== '3' || !autoHook) && (
                            <span className="text-brass">·</span>
                        )}
                    </button>
                    {showAdvanced && (
                        /* Stacked on a phone: three number fields side by side leaves
                           ~100px each, which crushes both label and value. */
                        <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-2 animate-fade">
                            <div>
                                <p className="eyebrow mb-1.5">custom clip target</p>
                                <input
                                    type="number" min="1" max="15" step="1"
                                    value={targetClips}
                                    onChange={(e) => setTargetClips(e.target.value)}
                                    placeholder="custom (1–15)"
                                    aria-label="custom clip count"
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <p className="eyebrow mb-1.5">min length (s)</p>
                                <input
                                    type="number" min="5" max="175" step="1"
                                    value={clipMinSeconds}
                                    onChange={(e) => setClipMinSeconds(e.target.value)}
                                    placeholder="15"
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <p className="eyebrow mb-1.5">max length (s)</p>
                                <input
                                    type="number" min="10" max="180" step="1"
                                    value={clipMaxSeconds}
                                    onChange={(e) => setClipMaxSeconds(e.target.value)}
                                    placeholder="60"
                                    className="input-field"
                                />
                            </div>
                            <p className="col-span-1 sm:col-span-3 text-[11px] leading-relaxed text-muted">
                                Targets, not guarantees: the AI returns fewer clips when the
                                material doesn't hold them. Leave blank to let it decide.
                            </p>
                            <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                <span className="text-xs text-ink2">vertical layout</span>
                                <select
                                    value={layout}
                                    onChange={(e) => setLayout(e.target.value)}
                                    className="input-field !w-auto text-xs py-1.5"
                                    aria-label="vertical layout"
                                >
                                    <option value="auto">Auto (AI picks per video)</option>
                                    <option value="split">Two speakers stacked</option>
                                    <option value="screencast">Screen over presenter</option>
                                    <option value="none">Single crop only</option>
                                </select>
                            </div>
                            <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                <div>
                                    <span className="text-xs text-ink2">tracking scan bands</span>
                                    <p className="text-[11px] text-muted mt-0.5">More bands help catch side objects.</p>
                                </div>
                                <select
                                    value={scanZoneCount}
                                    onChange={(e) => setScanZoneCount(e.target.value)}
                                    className="input-field !w-auto text-xs py-1.5"
                                    aria-label="tracking scan bands"
                                >
                                    {[3, 4, 5, 6, 7].map((count) => (
                                        <option key={count} value={String(count)}>{count} bands</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    )}
                </div>

                <label className="flex items-start gap-2.5 mt-5 text-left text-[13px] sm:text-xs leading-relaxed text-muted cursor-pointer select-none">
                    <input
                        type="checkbox"
                        checked={acknowledged}
                        onChange={(e) => setAcknowledged(e.target.checked)}
                        className="mt-0.5 w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                    />
                    <span>
                        I confirm I own this content or have the rights to process it. I am responsible for any content I submit. See our <a href="/terms" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Terms</a> and <a href="/privacy" target="_blank" rel="noopener noreferrer" className="text-ink2 underline underline-offset-2 hover:text-brass transition-colors" onClick={(e) => e.stopPropagation()}>Privacy Policy</a>.
                    </span>
                </label>

                <button
                    type="submit"
                    data-tutorial="generate"
                    disabled={isProcessing || !acknowledged || (mode === 'url' && !url) || (mode === 'file' && !file)}
                    className="w-full btn-primary mt-4"
                >
                    {isProcessing ? (
                        <>
                            <Loader2 size={16} className="animate-spin" />
                            Processing Video...
                        </>
                    ) : (
                        <>
                            Generate Clips
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
