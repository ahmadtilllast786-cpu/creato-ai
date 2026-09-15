import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Loader2, Sparkles, Clock, Type } from 'lucide-react';
import RemotionPreview from './RemotionPreview';
import Modal from './ui/Modal';
import SegmentedControl from './ui/SegmentedControl';
import { ActivePlaybackController } from '../lib/activePlayback';
import InspectorActionBar from './ui/InspectorActionBar';

const ENTRANCE_OPTIONS = [
    { value: 'spring', label: 'Bounce' },
    { value: 'fade', label: 'Fade' },
    { value: 'slide-up', label: 'Slide Up' },
    { value: 'none', label: 'None' },
];

// 12 distinct, high-retention short-form styles matching hooks.py and HookOverlay.tsx
const HOOK_STYLES = [
    { value: 'classic', label: 'Black & White', box: 'rgba(18,18,20,0.94)', text: '#FFFFFF' },
    { value: 'dark', label: 'Dark Sleek', box: 'rgba(18,18,20,0.92)', text: '#FFFFFF' },
    { value: 'white_card', label: 'White Card', box: 'rgba(255,255,255,0.96)', text: '#000000' },
    { value: 'yellow', label: 'Viral Yellow', box: 'rgba(255,214,0,0.96)', text: '#000000' },
    { value: 'red', label: 'Breaking Red', box: 'rgba(220,38,38,0.96)', text: '#FFFFFF' },
    { value: 'neon', label: 'Cyber Neon', box: 'rgba(10,25,47,0.95)', text: '#00F0FF' },
    { value: 'emerald', label: 'Tech Emerald', box: 'rgba(6,78,59,0.95)', text: '#34D399' },
    { value: 'purple', label: 'Violet Glow', box: 'rgba(99,102,241,0.95)', text: '#FFFFFF' },
    { value: 'orange', label: 'Sunset Orange', box: 'rgba(234,88,12,0.95)', text: '#FFFFFF' },
    { value: 'pill', label: 'Minimal Pill', box: 'rgba(15,23,42,0.82)', text: '#F1F5F9' },
    { value: 'breaking_news', label: 'News Banner', box: 'rgba(185,28,28,0.98)', text: '#FEF08A' },
    { value: 'outline', label: 'White Outline', box: 'transparent', text: '#FFFFFF', outline: true },
    { value: 'outline_yellow', label: 'Yellow Outline', box: 'transparent', text: '#FFD600', outline: true },
];

const POSITION_OPTIONS = [
    { value: 'top', label: 'top (above video)' },
    { value: 'center', label: 'center' },
    { value: 'bottom', label: 'bottom' },
];

const SIZE_OPTIONS = [
    { value: 'S', label: 'Small' },
    { value: 'M', label: 'Medium' },
    { value: 'L', label: 'Large' },
];

const FONT_OPTIONS = [
    { value: 'Noto Serif, Georgia, serif', label: 'Noto Serif (Viral)' },
    { value: 'Impact, sans-serif', label: 'Impact (Punchy)' },
    { value: 'Arial, Helvetica, sans-serif', label: 'Arial (Clean)' },
    { value: 'Verdana, sans-serif', label: 'Verdana' },
    { value: 'Georgia, serif', label: 'Georgia' },
    { value: 'Courier New, monospace', label: 'Courier' },
];

const COLOR_PRESETS = [
    { color: '#FFFFFF', label: 'White' },
    { color: '#FFD600', label: 'Yellow' },
    { color: '#00F0FF', label: 'Neon Blue' },
    { color: '#34D399', label: 'Mint' },
    { color: '#FF4D4D', label: 'Red' },
    { color: '#FFA500', label: 'Orange' },
    { color: '#000000', label: 'Black' },
];

const DURATION_MODE_OPTIONS = [
    { value: 'forever', label: 'whole video (forever)' },
    { value: 'custom', label: 'custom seconds' },
];

// Last-used hook settings, restored on the next open
function loadHookPrefs() {
    try { return JSON.parse(localStorage.getItem('os_hook_prefs')) || {}; } catch { return {}; }
}

export default function HookModal({
    isOpen,
    onClose,
    onGenerate,
    onRemove,
    isProcessing,
    videoUrl,
    initialText,
    durationInSeconds,
    existingSubtitles,
    hasCaptions,
    serverRender,
    burnedHook
}) {
    const prefs = loadHookPrefs();
    const [text, setText] = useState(initialText || 'POV: You are using the viral hook feature');
    const [position, setPosition] = useState(prefs.position || 'top'); // default top above video
    const [size, setSize] = useState(prefs.size || 'M');
    const [style, setStyle] = useState(prefs.style || 'classic');
    const [entranceAnimation, setEntranceAnimation] = useState(prefs.entranceAnimation || 'spring');
    const [durationMode, setDurationMode] = useState(prefs.durationMode || 'forever'); // default forever for whole video
    const [displayDuration, setDisplayDuration] = useState(prefs.displayDuration || 5);
    const [fontName, setFontName] = useState(prefs.fontName || 'Noto Serif, Georgia, serif');
    const [fontColor, setFontColor] = useState(prefs.fontColor || '#FFFFFF');
    const [bgColor, setBgColor] = useState(prefs.bgColor || '');
    const [uppercase, setUppercase] = useState(prefs.uppercase || false);

    // Snapshot of applied settings for clean cancel/restore
    const appliedSnapshotRef = useRef(null);

    useEffect(() => {
        if (isOpen) {
            ActivePlaybackController.stopAll();
            appliedSnapshotRef.current = {
                text,
                position,
                size,
                style,
                entranceAnimation,
                durationMode,
                displayDuration,
                fontName,
                fontColor,
                bgColor,
                uppercase,
            };
        }
    }, [isOpen]);

    const handleCancel = () => {
        if (appliedSnapshotRef.current) {
            const s = appliedSnapshotRef.current;
            setText(s.text);
            setPosition(s.position);
            setSize(s.size);
            setStyle(s.style);
            setEntranceAnimation(s.entranceAnimation);
            setDurationMode(s.durationMode);
            setDisplayDuration(s.displayDuration);
            setFontName(s.fontName);
            setFontColor(s.fontColor);
            setBgColor(s.bgColor);
            setUppercase(s.uppercase);
        }
        onClose();
    };

    const isDirty = useMemo(() => {
        if (!appliedSnapshotRef.current) return false;
        const s = appliedSnapshotRef.current;
        return (
            text !== s.text ||
            position !== s.position ||
            size !== s.size ||
            style !== s.style ||
            entranceAnimation !== s.entranceAnimation ||
            durationMode !== s.durationMode ||
            displayDuration !== s.displayDuration ||
            fontName !== s.fontName ||
            fontColor !== s.fontColor ||
            bgColor !== s.bgColor ||
            uppercase !== s.uppercase
        );
    }, [text, position, size, style, entranceAnimation, durationMode, displayDuration, fontName, fontColor, bgColor, uppercase]);

    if (!isOpen) return null;

    const displayForever = durationMode === 'forever';

    // Hook config for Remotion preview
    const activeStyleObj = HOOK_STYLES.find(s => s.value === style) || HOOK_STYLES[0];
    const resolvedBoxColor = bgColor !== '' ? bgColor : (activeStyleObj.box || 'rgba(18, 18, 20, 0.94)');
    const resolvedTextColor = fontColor !== '#FFFFFF' ? fontColor : (activeStyleObj.text || '#FFFFFF');

    const hookConfig = {
        text: text || 'Enter your text...',
        position,
        size,
        style,
        entranceAnimation,
        displayForever,
        displayDurationSec: displayForever ? null : displayDuration,
        fontName,
        fontColor: resolvedTextColor,
        bgColor: resolvedBoxColor,
        uppercase,
    };

    const useRemotionPreview = !!videoUrl;

    const getPositionClass = () => {
        switch (position) {
            case 'center': return 'items-center justify-center';
            case 'bottom': return 'items-center justify-end pb-[20%]';
            case 'top': default: return 'items-center justify-start pt-[3%]';
        }
    };

    const getSizeStyle = () => {
        switch (size) {
            case 'S': return { fontSize: '14px', maxWidth: '80%' };
            case 'L': return { fontSize: '24px', maxWidth: '95%' };
            case 'M': default: return { fontSize: '18px', maxWidth: '90%' };
        }
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="xl" eyebrow="EDITOR · HOOK" title="viral hook headline">
            <div className="flex flex-col md:flex-row gap-6">
                {/* Left: Live Preview */}
                <div className="flex-1 flex flex-col items-center justify-center bg-black rounded-card border border-rule overflow-hidden relative aspect-[9/16] max-h-[600px]">
                    {useRemotionPreview ? (
                        <RemotionPreview
                            videoUrl={videoUrl}
                            durationInSeconds={durationInSeconds || 30}
                            hook={hookConfig}
                            subtitles={existingSubtitles || null}
                        />
                    ) : (
                        <>
                            <video src={videoUrl} className="w-full h-full object-contain opacity-50" muted playsInline />
                            <div className={`absolute w-full px-8 text-center transition-all duration-300 pointer-events-none flex flex-col h-full ${getPositionClass()}`}>
                                <div
                                    className="font-bold px-3 py-2 rounded-xl shadow-2xl text-center whitespace-pre-wrap transition-all duration-200"
                                    style={{
                                        ...getSizeStyle(),
                                        backgroundColor: resolvedBoxColor,
                                        color: resolvedTextColor,
                                        fontFamily: fontName,
                                        textTransform: uppercase ? 'uppercase' : 'none',
                                        boxShadow: activeStyleObj.outline ? 'none' : '0 4px 15px rgba(0,0,0,0.5)',
                                        textShadow: activeStyleObj.outline ? '-2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000' : 'none',
                                        padding: resolvedBoxColor !== 'transparent' ? '10px 14px' : '4px 8px',
                                    }}
                                >
                                    {text || "Enter your text..."}
                                </div>
                            </div>
                        </>
                    )}
                </div>

                {/* Right: Controls & Inspector */}
                <div className="w-full md:w-96 flex flex-col">
                    <div className="space-y-4 flex-1 overflow-y-auto custom-scrollbar pr-1 max-h-[580px]">
                        {/* Text Input */}
                        <div>
                            <div className="flex items-center justify-between mb-1.5">
                                <p className="eyebrow">Hook Headline Text</p>
                                <button
                                    type="button"
                                    onClick={() => setUppercase(!uppercase)}
                                    className={`px-2 py-0.5 rounded text-[11px] font-semibold border transition-colors ${uppercase ? 'bg-[color:var(--color-accent)] text-black border-[color:var(--color-accent)]' : 'border-rule2 text-muted hover:text-ink'}`}
                                    title="Toggle uppercase"
                                >
                                    {uppercase ? 'ALL CAPS' : 'Aa normal'}
                                </button>
                            </div>
                            <textarea
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                rows={3}
                                className="input-field resize-none"
                                style={{ fontFamily: fontName }}
                                placeholder="Enter headline text that hooks viewers immediately..."
                            />
                        </div>

                        {/* Duration Mode: Forever (Whole Video) vs Custom */}
                        <div>
                            <div className="flex items-center justify-between mb-1.5">
                                <p className="eyebrow flex items-center gap-1.5">
                                    <Clock size={12} /> Duration
                                </p>
                                {displayForever && (
                                    <span className="text-[11px] font-medium text-[color:var(--color-accent)]">
                                        forever (whole video)
                                    </span>
                                )}
                            </div>
                            <SegmentedControl
                                options={DURATION_MODE_OPTIONS}
                                value={durationMode}
                                onChange={setDurationMode}
                                size="sm"
                            />
                            {!displayForever && (
                                <div className="mt-2.5 p-2.5 bg-paper2/50 border border-rule rounded-input space-y-1.5 animate-fade">
                                    <div className="flex items-center justify-between">
                                        <span className="text-xs text-muted">Disappear after</span>
                                        <span className="readout font-bold">{displayDuration}s</span>
                                    </div>
                                    <input
                                        type="range"
                                        min="2"
                                        max="30"
                                        value={displayDuration}
                                        onChange={(e) => setDisplayDuration(parseInt(e.target.value))}
                                        className="w-full accent-[var(--color-accent)]"
                                    />
                                    <div className="flex justify-between text-[10px] text-muted">
                                        <span>2s</span>
                                        <span>30s</span>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Style Presets Palette */}
                        <div>
                            <p className="eyebrow mb-1.5">Headline Style ({HOOK_STYLES.length} Looks)</p>
                            <div className="grid grid-cols-3 gap-1.5">
                                {HOOK_STYLES.map((s) => (
                                    <button
                                        key={s.value}
                                        type="button"
                                        onClick={() => {
                                            setStyle(s.value);
                                            setFontColor(s.text);
                                            setBgColor(s.box);
                                        }}
                                        className={`px-1.5 py-2 rounded-input border text-[11px] transition-all text-left flex flex-col justify-between
                                            ${style === s.value ? 'border-[color:var(--color-accent)] ring-1 ring-[color:var(--color-accent)] bg-paper2' : 'border-rule2 hover:border-rule hover:bg-paper2/50'}`}
                                        title={s.label}
                                    >
                                        <div
                                            className="rounded px-1.5 py-0.5 font-bold text-center text-xs truncate w-full shadow-sm"
                                            style={{
                                                backgroundColor: s.box,
                                                color: s.text,
                                                textShadow: s.outline ? '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000' : 'none',
                                            }}
                                        >
                                            Hook
                                        </div>
                                        <span className="block mt-1 text-[10px] text-muted truncate">{s.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Font Family Selection */}
                        <div>
                            <p className="eyebrow mb-1.5 flex items-center gap-1.5">
                                <Type size={12} /> Font
                            </p>
                            <select
                                value={fontName}
                                onChange={(e) => setFontName(e.target.value)}
                                className="input-field text-xs py-1.5"
                            >
                                {FONT_OPTIONS.map((f) => (
                                    <option key={f.value} value={f.value} style={{ fontFamily: f.value }}>
                                        {f.label}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Custom Colors */}
                        <div className="flex gap-4">
                            <div className="flex-1">
                                <p className="eyebrow mb-1.5">Text Color</p>
                                <div className="flex items-center gap-1.5 flex-wrap">
                                    {COLOR_PRESETS.slice(0, 5).map((c) => (
                                        <button
                                            key={c.color}
                                            type="button"
                                            onClick={() => setFontColor(c.color)}
                                            className={`w-5 h-5 rounded-full border transition-transform ${fontColor === c.color ? 'scale-125 border-[color:var(--color-accent)]' : 'border-rule2'}`}
                                            style={{ backgroundColor: c.color }}
                                            title={c.label}
                                        />
                                    ))}
                                    <label className="w-5 h-5 rounded-full border border-dashed border-rule2 cursor-pointer flex items-center justify-center hover:border-[color:var(--color-accent)] relative" title="Custom color">
                                        <span className="text-[10px] text-muted">+</span>
                                        <input type="color" value={fontColor} onChange={(e) => setFontColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                    </label>
                                </div>
                            </div>
                            <div className="flex-1">
                                <p className="eyebrow mb-1.5">Card Background</p>
                                <div className="flex items-center gap-2">
                                    <label className="relative w-7 h-7 rounded-input border border-rule2 cursor-pointer overflow-hidden shrink-0" title="Custom background color">
                                        <div className="w-full h-full" style={{ backgroundColor: resolvedBoxColor }} />
                                        <input type="color" value={resolvedBoxColor.startsWith('#') ? resolvedBoxColor : '#121214'} onChange={(e) => setBgColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                    </label>
                                    <button
                                        type="button"
                                        onClick={() => setBgColor('transparent')}
                                        className={`px-2 py-1 text-[10px] rounded border transition-colors ${bgColor === 'transparent' ? 'border-[color:var(--color-accent)] text-[color:var(--color-accent)]' : 'border-rule2 text-muted'}`}
                                    >
                                        no box
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Position Control (Default: Top Safe Zone) */}
                        <div>
                            <p className="eyebrow mb-1.5">Position</p>
                            <SegmentedControl
                                options={POSITION_OPTIONS}
                                value={position}
                                onChange={setPosition}
                                size="sm"
                            />
                            {position === 'top' && (
                                <p className="text-[11px] text-muted mt-1 leading-tight">
                                    Anchored permanently above the video (safe header zone, never covers faces).
                                </p>
                            )}
                            {position === 'bottom' && hasCaptions && (
                                <p className="text-[11px] text-warn mt-1 leading-relaxed">
                                    This clip has captions near the bottom — top is the recommended safe zone.
                                </p>
                            )}
                        </div>

                        {/* Size Control */}
                        <div>
                            <p className="eyebrow mb-1.5">Size</p>
                            <SegmentedControl
                                options={SIZE_OPTIONS}
                                value={size}
                                onChange={setSize}
                                size="sm"
                            />
                        </div>

                        {/* Entrance Animation */}
                        <div className={serverRender ? 'opacity-50' : ''}>
                            <p className="eyebrow mb-1.5">Entrance Animation</p>
                            <SegmentedControl
                                options={ENTRANCE_OPTIONS}
                                value={entranceAnimation}
                                onChange={setEntranceAnimation}
                                columns={2}
                                size="sm"
                            />
                        </div>

                        {burnedHook && (
                            <div className="p-2.5 border border-rule rounded-input text-xs text-muted space-y-1.5">
                                <p>
                                    Clip has active hook: "{burnedHook}".
                                    Applying updates or replaces it cleanly.
                                </p>
                                {onRemove && (
                                    <button
                                        type="button"
                                        onClick={onRemove}
                                        disabled={isProcessing}
                                        className="text-warn underline underline-offset-2 hover:opacity-80 transition-opacity"
                                    >
                                        remove hook completely
                                    </button>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Transactional Apply & Cancel Action Bar */}
                    <div className="mt-4 pt-3 border-t border-rule">
                        <InspectorActionBar
                            isDirty={isDirty}
                            isApplying={isProcessing}
                            applyLabel={isProcessing ? 'applying hook…' : 'apply hook'}
                            cancelLabel="cancel"
                            onApply={() => {
                                try {
                                    localStorage.setItem('os_hook_prefs', JSON.stringify({
                                        style,
                                        position,
                                        size,
                                        entranceAnimation,
                                        durationMode,
                                        displayDuration,
                                        fontName,
                                        fontColor,
                                        bgColor,
                                        uppercase,
                                    }));
                                } catch { /* ignore */ }
                                onGenerate({
                                    text: text.trim(),
                                    position,
                                    size,
                                    style,
                                    fontName,
                                    fontColor: resolvedTextColor,
                                    bgColor: resolvedBoxColor,
                                    uppercase,
                                    displayForever,
                                    duration_seconds: displayForever ? null : displayDuration,
                                    // Remotion preview/burn config
                                    remotion: hookConfig,
                                });
                            }}
                            onCancel={handleCancel}
                            description={isDirty ? 'staged hook edits' : (displayForever ? 'hook active (whole video)' : 'hook active')}
                        />
                    </div>
                </div>
            </div>
        </Modal>
    );
}
