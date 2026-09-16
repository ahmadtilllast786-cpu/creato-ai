import React, { useState, useEffect, useRef } from 'react';
import { Link2, Upload, FileVideo, X, Info, Loader2, ChevronDown, Music, Volume2, Sparkles, RefreshCw } from 'lucide-react';
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

const SUBTITLE_STYLE_OPTIONS = [
    { value: 'shorts', label: 'Shorts Pop (Red Accent / Bold)' },
    { value: 'tiktok', label: 'TikTok (Cyan & Red Accent)' },
    { value: 'reels', label: 'Reels (Pink Gradient Accent)' },
    { value: 'beast', label: 'Beast (Bold Impact Yellow)' },
    { value: 'gold', label: 'Gold Glow (Warm Glow)' },
    { value: 'neon', label: 'Neon (Vibrant Green Glow)' },
    { value: 'cyber', label: 'Cyber (Cyan Glow)' },
    { value: 'karaoke', label: 'Karaoke (Red Accent)' },
    { value: 'minimal', label: 'Minimal (Clean White Outline)' },
    { value: 'boxed', label: 'Boxed (Purple Pill Box)' },
    { value: 'classic', label: 'Classic (Standard Outline)' },
    { value: 'none', label: 'None (No Subtitles)' },
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
        try { return localStorage.getItem('os_auto_hook_style') || 'classic'; } catch { return 'classic'; }
    });
    // Layout: 'auto' lets the AI pick per video (server default); the others
    // force one on so a podcast host who knows what they uploaded doesn't
    // depend on the detector, and 'none' keeps the plain single crop.
    const [layout, setLayout] = useState(() => {
        try { return localStorage.getItem('os_layout') || 'auto'; } catch { return 'auto'; }
    });
    // Subtitle Style selection before processing
    const [subtitleStyle, setSubtitleStyle] = useState(() => {
        try { return localStorage.getItem('os_subtitle_style') || 'shorts'; } catch { return 'shorts'; }
    });
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
        const advanced = {
            targetClips: targetClips || null,
            clipMinSeconds: clipMinSeconds || null,
            clipMaxSeconds: clipMaxSeconds || null,
            trackScanZones: scanZoneCount || null,
            autoHook,
            autoHookStyle,
            layout,
            subtitleStyle,
            bgAudio: bgAudioFile || null,
            bgAudioVolume,
            freshClips,
        };
        try {
            localStorage.setItem('os_auto_hook', autoHook ? '1' : '0');
            localStorage.setItem('os_auto_hook_style', autoHookStyle);
            localStorage.setItem('os_layout', layout);
            localStorage.setItem('os_scan_zone_count', scanZoneCount || '3');
            localStorage.setItem('os_subtitle_style', subtitleStyle);
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

                {/* Subtitle Style Preset */}
                <div className="mt-5">
                    <div className="flex items-center justify-between mb-2">
                        <p className="eyebrow flex items-center gap-1.5">
                            <Sparkles size={13} className="text-brass" />
                            Subtitle Style
                        </p>
                    </div>
                    <select
                        value={subtitleStyle}
                        onChange={(e) => setSubtitleStyle(e.target.value)}
                        className="input-field w-full text-xs sm:text-sm py-2"
                        aria-label="Subtitle Style"
                    >
                        {SUBTITLE_STYLE_OPTIONS.map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                    </select>
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
                            <div className="col-span-1 sm:col-span-3 flex flex-wrap items-center justify-between gap-3 pt-3 sm:pt-1 border-t border-rule">
                                <label className="flex items-center gap-2 text-xs text-ink2 cursor-pointer select-none">
                                    <input
                                        type="checkbox"
                                        checked={autoHook}
                                        onChange={(e) => setAutoHook(e.target.checked)}
                                        className="w-4 h-4 shrink-0 accent-[var(--color-accent)] cursor-pointer"
                                    />
                                    auto hook titles on clips
                                </label>
                                {autoHook && (
                                    <select
                                        value={autoHookStyle}
                                        onChange={(e) => setAutoHookStyle(e.target.value)}
                                        className="input-field !w-auto text-xs py-1.5"
                                    >
                                        <option value="classic">Black & White (Default)</option>
                                        <option value="dark">Dark</option>
                                        <option value="white_card">White Card</option>
                                        <option value="yellow">Yellow</option>
                                        <option value="red">Red</option>
                                        <option value="outline">Outline</option>
                                        <option value="outline_yellow">Outline+</option>
                                    </select>
                                )}
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
