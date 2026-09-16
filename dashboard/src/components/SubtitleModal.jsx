import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Loader2, ShieldAlert, ShieldCheck, Check, RotateCcw } from 'lucide-react';
import { apiFetch } from '../lib/api';
import { detectBurnedInCaptions } from '../lib/captionDetector';
import RemotionPreview from './RemotionPreview';
import Modal from './ui/Modal';
import SegmentedControl from './ui/SegmentedControl';
import { ActivePlaybackController } from '../lib/activePlayback';
import PlatformSafeZoneOverlay, { PlatformSafeZoneControls } from './PlatformSafeZoneOverlay';

const COLLISION_OPTIONS = [
    { value: 'smart_reposition', label: 'smart safe' },
    { value: 'occlusion_mask', label: 'mask old' },
    { value: 'manual_offset', label: 'manual y' },
];

const FONT_OPTIONS = [
    { value: 'Verdana', label: 'Verdana' },
    { value: 'Arial', label: 'Arial' },
    { value: 'Impact', label: 'Impact' },
    { value: 'Helvetica', label: 'Helvetica' },
    { value: 'Georgia', label: 'Georgia' },
    { value: 'Courier New', label: 'Courier New' },
];

const COLOR_PRESETS = [
    { color: '#FFFFFF', label: 'White' },
    { color: '#FFFF00', label: 'Yellow' },
    { color: '#00FFFF', label: 'Cyan' },
    { color: '#00FF00', label: 'Green' },
    { color: '#FF0000', label: 'Red' },
    { color: '#FF69B4', label: 'Pink' },
];

const HIGHLIGHT_PRESETS = [
    { color: '#FFDD00', label: 'Gold' },
    { color: '#FF4444', label: 'Red' },
    { color: '#00FF88', label: 'Green' },
    { color: '#00BBFF', label: 'Blue' },
    { color: '#FF69B4', label: 'Pink' },
];

const ANIMATION_OPTIONS = [
    { value: 'pop', label: 'Pop' },
    { value: 'word-highlight', label: 'Glow' },
    { value: 'karaoke', label: 'Karaoke' },
    { value: 'none', label: 'None' },
];

const POSITION_OPTIONS = [
    { value: 'top', label: 'top' },
    { value: 'middle', label: 'middle' },
    { value: 'bottom', label: 'bottom' },
];

// Ready-made caption looks burned server-side as karaoke ASS (word highlight):
// dimmed base text + strong active word, optional glow/pop/box effect.
const CAPTION_PRESETS = [
    { id: 'tiktok',  label: 'TikTok',     style: 'karaoke', effect: 'none', highlightColor: '#FE2C55', baseOpacity: 0.75, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'reels',   label: 'Reels',      style: 'karaoke', effect: 'none', highlightColor: '#E1306C', baseOpacity: 0.7,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'shorts',  label: 'Shorts Pop', style: 'karaoke', effect: 'pop',  highlightColor: '#FF0000', baseOpacity: 0.7,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'gold',    label: 'Gold Glow',  style: 'karaoke', effect: 'glow', highlightColor: '#FFD700', baseOpacity: 0.6,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'neon',    label: 'Neon',       style: 'karaoke', effect: 'glow', highlightColor: '#00FF88', baseOpacity: 0.55, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'cyber',   label: 'Cyber',      style: 'karaoke', effect: 'glow', highlightColor: '#00FFFF', baseOpacity: 0.5,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'karaoke', label: 'Karaoke',    style: 'karaoke', effect: 'none', highlightColor: '#FF6B6B', baseOpacity: 0.6,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'minimal', label: 'Minimal',    style: 'karaoke', effect: 'none', highlightColor: '#FFFFFF', baseOpacity: 0.65, uppercase: false, fontName: 'Verdana', borderWidth: 1 },
    { id: 'beast',   label: 'Beast',      style: 'karaoke', effect: 'pop',  highlightColor: '#FFD700', baseOpacity: 1.0,  uppercase: true,  fontName: 'Impact',  borderWidth: 3 },
    { id: 'boxed',   label: 'Boxed',      style: 'karaoke', effect: 'box',  highlightColor: '#7C3AED', baseOpacity: 0.85, uppercase: false, fontName: 'Verdana', borderWidth: 2 },
    { id: 'classic', label: 'Classic',    style: 'classic', effect: 'none', highlightColor: '#FFD700', baseOpacity: 1.0,  uppercase: false, fontName: 'Verdana', borderWidth: 2 },
];

const swatchClass = (selected) =>
    `w-6 h-6 rounded-full transition-all ${selected
        ? 'ring-2 ring-[color:var(--color-accent)] ring-offset-2 ring-offset-[color:var(--color-paper-2)]'
        : 'ring-1 ring-[color:var(--color-rule-2)] hover:ring-[color:var(--color-accent)]'}`;

export default function SubtitleModal({ isOpen, onClose, onGenerate, onApplyAll, onRemove, isProcessing, videoUrl, jobId, clipIndex, existingHook, bulkCount = 0, bulkProgress }) {
    const [position, setPosition] = useState('bottom');
    const [fontSize] = useState(24);
    const [fontName, setFontName] = useState('Verdana');
    const [fontColor, setFontColor] = useState('#FFFFFF');
    const [highlightColor, setHighlightColor] = useState('#FFDD00');
    const [borderColor, setBorderColor] = useState('#000000');
    const [borderWidth, setBorderWidth] = useState(2);
    const [bgColor, setBgColor] = useState('#000000');
    const [bgOpacity, setBgOpacity] = useState(0.0);
    const [animation, setAnimation] = useState('pop');
    const [showTextEditor, setShowTextEditor] = useState(false);

    // Collision avoidance and pre-existing subtitle detection
    const [hasBurnedInCaptions, setHasBurnedInCaptions] = useState(false);
    const [isScanning, setIsScanning] = useState(false);
    const [detectionConfidence, setDetectionConfidence] = useState(0);
    const [collisionMode, setCollisionMode] = useState('smart_reposition'); // smart_reposition | occlusion_mask | manual_offset
    const [manualYOffset, setManualYOffset] = useState(48); // % from top

    // Karaoke (server-side ASS burn) state
    const [style, setStyle] = useState('classic'); // classic | karaoke
    const [effect, setEffect] = useState('none'); // none | glow | pop | box
    const [baseOpacity, setBaseOpacity] = useState(1.0);
    const [uppercase, setUppercase] = useState(false);
    const [activePreset, setActivePreset] = useState(null);

    const applyPreset = (p) => {
        setActivePreset(p.id);
        setStyle(p.style);
        setEffect(p.effect);
        setHighlightColor(p.highlightColor);
        setBaseOpacity(p.baseOpacity);
        setUppercase(p.uppercase);
        setFontName(p.fontName);
        setBorderWidth(p.borderWidth);
        setFontColor('#FFFFFF');
        setBgOpacity(0);
        // Keep the Remotion preview roughly in sync with the burned look
        setAnimation(p.style === 'karaoke' ? (p.effect === 'pop' ? 'pop' : p.effect === 'glow' ? 'word-highlight' : 'karaoke') : 'none');
    };

    // Remotion preview & caption state
    const [captions, setCaptions] = useState([]);
    const [originalCaptions, setOriginalCaptions] = useState([]);
    const [editableText, setEditableText] = useState('');
    const [durationSec, setDurationSec] = useState(30);
    const [captionsLoading, setCaptionsLoading] = useState(false);
    const [useRemotionPreview, setUseRemotionPreview] = useState(false);
    const [platformSafeZone, setPlatformSafeZone] = useState('off');
    const [showGuides, setShowGuides] = useState(false);

    // Snapshot of applied settings for clean cancel/restore
    const appliedSnapshotRef = useRef(null);

    useEffect(() => {
        if (isOpen) {
            ActivePlaybackController.stopAll();
            appliedSnapshotRef.current = {
                position, fontSize, fontName, fontColor, highlightColor,
                borderColor, borderWidth, bgColor, bgOpacity, animation,
                style, effect, baseOpacity, uppercase, activePreset,
                collisionMode, manualYOffset,
                editableText,
                captions: [...captions],
            };
        }
    }, [isOpen]);

    const handleCancel = () => {
        if (appliedSnapshotRef.current) {
            const s = appliedSnapshotRef.current;
            setPosition(s.position);
            setFontName(s.fontName);
            setFontColor(s.fontColor);
            setHighlightColor(s.highlightColor);
            setBorderColor(s.borderColor);
            setBorderWidth(s.borderWidth);
            setBgColor(s.bgColor);
            setBgOpacity(s.bgOpacity);
            setAnimation(s.animation);
            setStyle(s.style);
            setEffect(s.effect);
            setBaseOpacity(s.baseOpacity);
            setUppercase(s.uppercase);
            setActivePreset(s.activePreset);
            setCollisionMode(s.collisionMode);
            setManualYOffset(s.manualYOffset);
            if (s.editableText !== undefined) setEditableText(s.editableText);
            if (s.captions !== undefined) setCaptions(s.captions);
        }
        onClose();
    };

    const isDirty = useMemo(() => {
        if (!appliedSnapshotRef.current) return false;
        const s = appliedSnapshotRef.current;
        return (
            position !== s.position ||
            fontName !== s.fontName ||
            fontColor !== s.fontColor ||
            highlightColor !== s.highlightColor ||
            borderWidth !== s.borderWidth ||
            bgOpacity !== s.bgOpacity ||
            style !== s.style ||
            effect !== s.effect ||
            uppercase !== s.uppercase ||
            activePreset !== s.activePreset ||
            collisionMode !== s.collisionMode ||
            manualYOffset !== s.manualYOffset ||
            (s.editableText !== undefined && editableText !== s.editableText)
        );
    }, [position, fontName, fontColor, highlightColor, borderWidth, bgOpacity, style, effect, uppercase, activePreset, collisionMode, manualYOffset, editableText]);

    // Fetch word-level captions when modal opens
    useEffect(() => {
        if (!isOpen || !jobId || clipIndex === undefined) return;

        setCaptionsLoading(true);
        apiFetch(`/api/clip/${jobId}/${clipIndex}/transcript`)
            .then((res) => res.ok ? res.json() : null)
            .then((data) => {
                if (data && data.captions && data.captions.length > 0) {
                    setCaptions(data.captions);
                    setOriginalCaptions(data.captions);
                    const fullText = data.captions.map(c => c.text).join(' ');
                    setEditableText(fullText);
                    setDurationSec(data.durationSec || 30);
                    setUseRemotionPreview(true);
                    if (appliedSnapshotRef.current && !appliedSnapshotRef.current.editableText) {
                        appliedSnapshotRef.current.editableText = fullText;
                        appliedSnapshotRef.current.captions = [...data.captions];
                    }
                } else {
                    setUseRemotionPreview(false);
                }
            })
            .catch(() => setUseRemotionPreview(false))
            .finally(() => setCaptionsLoading(false));
    }, [isOpen, jobId, clipIndex]);

    // Canvas frame pre-scan to detect pre-existing burned-in video subtitles
    useEffect(() => {
        if (!isOpen || !videoUrl) return;

        let active = true;
        setIsScanning(true);

        detectBurnedInCaptions(videoUrl)
            .then((res) => {
                if (!active) return;
                if (res && res.hasBurnedInCaptions) {
                    setHasBurnedInCaptions(true);
                    setDetectionConfidence(res.confidence || 0.8);
                    setCollisionMode('smart_reposition');
                } else {
                    setHasBurnedInCaptions(false);
                    setDetectionConfidence(0);
                }
            })
            .catch(() => {
                if (active) {
                    setHasBurnedInCaptions(false);
                    setDetectionConfidence(0);
                }
            })
            .finally(() => {
                if (active) setIsScanning(false);
            });

        return () => {
            active = false;
        };
    }, [isOpen, videoUrl]);

    // When user edits or enters text, distribute words across timestamps
    const handleTextEdit = (newText) => {
        setEditableText(newText);
        const newWords = newText.split(/\s+/).filter(w => w.length > 0);
        if (newWords.length === 0) {
            setCaptions([]);
            return;
        }

        if (originalCaptions.length === 0) {
            // New caption addition on clips that did not have captions previously
            const totalDurationMs = (durationSec || 30) * 1000;
            const wordDurationMs = Math.max(200, Math.floor(totalDurationMs / newWords.length));
            const newCaptions = newWords.map((word, i) => ({
                text: word,
                startMs: Math.round(i * wordDurationMs),
                endMs: Math.round(Math.min((i + 1) * wordDurationMs, totalDurationMs)),
            }));
            setCaptions(newCaptions);
            if (!useRemotionPreview && videoUrl) setUseRemotionPreview(true);
            return;
        }

        // Distribute new words across the time span of original captions
        const totalDurationMs = originalCaptions[originalCaptions.length - 1].endMs - originalCaptions[0].startMs;
        const startMs = originalCaptions[0].startMs;
        const wordDurationMs = totalDurationMs / newWords.length;

        const newCaptions = newWords.map((word, i) => ({
            text: word,
            startMs: Math.round(startMs + i * wordDurationMs),
            endMs: Math.round(startMs + (i + 1) * wordDurationMs),
        }));
        setCaptions(newCaptions);
    };

    if (!isOpen) return null;

    // Build subtitle config for Remotion
    const subtitleConfig = {
        captions,
        position,
        hasBurnedInCaptions,
        collisionMode,
        manualYOffset,
        style: {
            fontFamily: fontName,
            fontSize: fontSize * 2.2, // Scale up for 1080p (modal fontSize is for small preview)
            fontColor,
            highlightColor,
            borderColor,
            borderWidth: borderWidth * 1.5,
            bgColor,
            bgOpacity,
            animation,
            // Karaoke look reflected live in the playable preview.
            baseOpacity: style === 'karaoke' ? baseOpacity : 1,
            uppercase: style === 'karaoke' ? uppercase : false,
        },
    };

    // Fallback: static CSS preview (same as original)
    const bw = Math.max(borderWidth, 0);
    const bc = borderColor;
    const outlineShadow = bw > 0 ? [
        `-${bw}px -${bw}px 0 ${bc}`, `${bw}px -${bw}px 0 ${bc}`,
        `-${bw}px ${bw}px 0 ${bc}`, `${bw}px ${bw}px 0 ${bc}`,
        `0 -${bw}px 0 ${bc}`, `0 ${bw}px 0 ${bc}`,
        `-${bw}px 0 0 ${bc}`, `${bw}px 0 0 ${bc}`,
    ].join(', ') : 'none';

    const isMaskMode = hasBurnedInCaptions && collisionMode === 'occlusion_mask';

    const fallbackPreviewStyle = {
        fontFamily: fontName,
        color: fontColor,
        fontSize: '20px',
        fontWeight: 'bold',
        maxWidth: '85%',
        padding: isMaskMode ? '12px 24px' : '6px 12px',
        borderRadius: isMaskMode ? '12px' : '4px',
        textAlign: 'center',
        lineHeight: '1.3',
        ...(isMaskMode
            ? {
                backgroundColor: 'rgba(10, 11, 16, 0.94)',
                backdropFilter: 'blur(16px)',
                WebkitBackdropFilter: 'blur(16px)',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.1)',
                textShadow: outlineShadow,
            }
            : bgOpacity > 0
                ? {
                    backgroundColor: `${bgColor}${Math.round(bgOpacity * 255).toString(16).padStart(2, '0')}`,
                    textShadow: 'none',
                }
                : { textShadow: outlineShadow }
        ),
    };

    let fallbackPositionClasses = '';
    let fallbackPositionInline = {};

    if (hasBurnedInCaptions) {
        if (collisionMode === 'smart_reposition') {
            fallbackPositionClasses = 'top-0 bottom-0';
        } else if (collisionMode === 'occlusion_mask') {
            fallbackPositionClasses = 'bottom-[18%]';
        } else if (collisionMode === 'manual_offset') {
            fallbackPositionInline = { top: `${manualYOffset}%`, transform: 'translateY(-50%)' };
        }
    } else {
        if (position === 'top') fallbackPositionClasses = 'top-20';
        else if (position === 'middle') fallbackPositionClasses = 'top-0 bottom-0';
        else fallbackPositionClasses = 'bottom-[18%]';
    }

    // Text edits must survive the server render path too (issue #69):
    // send edited words whenever text differs from transcript output.
    const textEdited = (originalCaptions.length > 0
        && editableText.trim() !== originalCaptions.map((c) => c.text).join(' ').trim())
        || (originalCaptions.length === 0 && editableText.trim().length > 0);

    const styleOptions = {
        position, fontSize, fontName, fontColor, borderColor, borderWidth, bgColor, bgOpacity,
        // Karaoke burn (server-side ASS render)
        style, effect, baseOpacity, uppercase, highlightColor,
        // Collision avoidance & safe zones
        hasBurnedInCaptions,
        collisionMode,
        manualYOffset,
        clearPreviousSubtitles: true,
        // Remotion data
        remotion: useRemotionPreview ? subtitleConfig : null,
        captions: textEdited ? captions : (captions.length > 0 ? captions : null),
    };

    const bulkRunning = bulkProgress?.running;

    const modalFooter = (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 w-full">
            <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                    <span
                        className={`w-2.5 h-2.5 rounded-full transition-colors ${
                            isDirty || textEdited ? 'bg-warn animate-pulse' : 'bg-ok/70'
                        }`}
                    />
                    <span className="text-xs font-mono lowercase text-muted">
                        {isDirty || textEdited ? 'staged caption changes pending' : 'captions applied'}
                    </span>
                </div>
                {onRemove && (
                    <button
                        type="button"
                        onClick={onRemove}
                        disabled={isProcessing}
                        className="text-xs text-muted hover:text-warn transition-colors underline underline-offset-2 lowercase disabled:opacity-50 ml-2 cursor-pointer"
                        title="Remove burned captions from this clip"
                    >
                        remove captions
                    </button>
                )}
            </div>

            <div className="flex items-center gap-2.5 shrink-0">
                {onApplyAll && bulkCount > 1 && (
                    <button
                        type="button"
                        onClick={() => onApplyAll({ ...styleOptions, captions: null })}
                        disabled={isProcessing}
                        className="btn-ghost py-2 px-3 text-xs flex items-center gap-1.5 cursor-pointer"
                    >
                        {bulkRunning ? (
                            <><Loader2 size={13} className="animate-spin" /> applying to all… {bulkProgress.current}/{bulkProgress.total}</>
                        ) : (
                            `apply this style to all ${bulkCount} clips`
                        )}
                    </button>
                )}
                <button
                    type="button"
                    onClick={handleCancel}
                    disabled={isProcessing}
                    className="btn-ghost py-2 px-4 text-xs font-medium flex items-center gap-1.5 hover:text-warn transition-colors cursor-pointer"
                    title="Discard staged changes and restore previous settings"
                >
                    <RotateCcw size={13} />
                    <span>Cancel</span>
                </button>
                <button
                    type="button"
                    onClick={() => onGenerate(styleOptions)}
                    disabled={isProcessing}
                    className={`btn-primary py-2 px-5 text-xs font-semibold flex items-center justify-center gap-2 transition-all cursor-pointer ${
                        isDirty || textEdited
                            ? 'ring-2 ring-brass shadow-md opacity-100'
                            : 'opacity-90 hover:opacity-100'
                    }`}
                    title="Apply new subtitles to this clip (replaces previous subtitles)"
                >
                    {isProcessing && !bulkRunning ? (
                        <>
                            <Loader2 size={14} className="animate-spin text-brassink" />
                            <span>Applying Captions…</span>
                        </>
                    ) : (
                        <>
                            <Check size={14} />
                            <span>Apply Captions</span>
                        </>
                    )}
                </button>
            </div>
        </div>
    );

    return (
        <Modal isOpen={isOpen} onClose={handleCancel} size="xl" eyebrow="EDITOR · SUBTITLES" title="subtitles" footer={modalFooter}>
            <div className="flex flex-col md:flex-row gap-6">
                {/* Left: Preview */}
                <div className="flex-1 flex flex-col items-center justify-center bg-black rounded-card border border-rule overflow-hidden relative aspect-[9/16] max-h-[580px]">
                    {/* Platform Safe Zone & Guides Toolbar */}
                    <div className="absolute top-2 z-30">
                        <PlatformSafeZoneControls
                            platform={platformSafeZone}
                            onPlatformChange={setPlatformSafeZone}
                            showGuides={showGuides}
                            onToggleGuides={setShowGuides}
                        />
                    </div>

                    {/* Platform Safe Zone Collision Mask & Alignment Guides */}
                    <PlatformSafeZoneOverlay platform={platformSafeZone} showGuides={showGuides} />

                    {captionsLoading ? (
                        <div className="flex items-center gap-2 text-muted">
                            <Loader2 size={16} className="animate-spin" />
                            <span className="text-sm lowercase">Loading preview...</span>
                        </div>
                    ) : useRemotionPreview ? (
                        <RemotionPreview
                            videoUrl={videoUrl}
                            durationInSeconds={durationSec}
                            subtitles={subtitleConfig}
                            hook={existingHook || null}
                        />
                    ) : (
                        <>
                            <video src={videoUrl} className="w-full h-full object-contain opacity-50" muted playsInline />
                            <div
                                style={fallbackPositionInline}
                                className={`absolute w-full px-8 text-center transition-all duration-300 pointer-events-none flex flex-col items-center justify-center ${fallbackPositionClasses}`}
                            >
                                <span style={fallbackPreviewStyle}>
                                    This is how your subtitles<br/>will appear on the video
                                </span>
                            </div>
                        </>
                    )}
                </div>

                {/* Right: Controls */}
                <div className="w-full md:w-80 flex flex-col max-h-[580px]">
                    {/* Top Action Header directly inside Subtitle Options */}
                    <div className="p-2.5 mb-2 rounded-card bg-paper2 border border-rule flex items-center justify-between gap-2 shrink-0">
                        <div className="flex items-center gap-1.5 min-w-0">
                            <span className={`w-2 h-2 rounded-full shrink-0 transition-colors ${
                                isDirty || textEdited ? 'bg-warn animate-pulse' : 'bg-ok/70'
                            }`} />
                            <span className="text-[11px] font-mono lowercase text-muted truncate">
                                {isDirty || textEdited ? 'staged style' : 'captions applied'}
                            </span>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                            <button
                                type="button"
                                onClick={handleCancel}
                                disabled={isProcessing}
                                className="btn-ghost py-1 px-2 text-xs flex items-center gap-1 hover:text-warn transition-colors cursor-pointer"
                                title="Discard changes and close"
                            >
                                <RotateCcw size={12} />
                                <span>Cancel</span>
                            </button>
                            <button
                                type="button"
                                onClick={() => onGenerate(styleOptions)}
                                disabled={isProcessing}
                                className={`btn-primary py-1 px-3 text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                                    isDirty || textEdited ? 'ring-2 ring-brass shadow-sm' : 'opacity-90 hover:opacity-100'
                                }`}
                                title="Apply this subtitle style to the clip"
                            >
                                {isProcessing && !bulkRunning ? (
                                    <>
                                        <Loader2 size={12} className="animate-spin text-brassink" />
                                        <span>Applying…</span>
                                    </>
                                ) : (
                                    <>
                                        <Check size={12} />
                                        <span>Apply Style</span>
                                    </>
                                )}
                            </button>
                        </div>
                    </div>

                    <div className="space-y-5 flex-1 overflow-y-auto custom-scrollbar pr-1 pb-2">
                        {/* Caption presets (server-side karaoke burn) */}
                        <div>
                            <p className="eyebrow mb-2">Preset</p>
                            <div className="grid grid-cols-3 gap-1.5">
                                {CAPTION_PRESETS.map((p) => (
                                    <button
                                        key={p.id}
                                        onClick={() => applyPreset(p)}
                                        className={`px-2 py-1.5 rounded-input border text-xs transition-colors flex items-center gap-1.5 justify-center
                                            ${activePreset === p.id
                                                ? 'border-[color:var(--color-accent)] text-ink ring-1 ring-[color:var(--color-accent)]'
                                                : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                                        title={p.label}
                                    >
                                        <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: p.highlightColor }} />
                                        {p.label}
                                    </button>
                                ))}
                            </div>

                            {/* Direct Apply button for selected preset */}
                            <div className="flex items-center justify-between gap-2 mt-2 pt-2 border-t border-rule2/60">
                                <span className="text-[11px] text-muted truncate">
                                    {activePreset ? `Preset: ${CAPTION_PRESETS.find(p => p.id === activePreset)?.label || activePreset}` : 'Custom style'}
                                </span>
                                <button
                                    type="button"
                                    onClick={() => onGenerate(styleOptions)}
                                    disabled={isProcessing}
                                    className="btn-primary py-1 px-2.5 text-xs font-semibold flex items-center gap-1 transition-all cursor-pointer shrink-0 shadow-sm"
                                    title="Apply selected style preset now"
                                >
                                    {isProcessing && !bulkRunning ? (
                                        <>
                                            <Loader2 size={12} className="animate-spin text-brassink" />
                                            <span>Applying…</span>
                                        </>
                                    ) : (
                                        <>
                                            <Check size={12} />
                                            <span>Apply This Style</span>
                                        </>
                                    )}
                                </button>
                            </div>
                            {style === 'karaoke' && (
                                <div className="mt-3 space-y-3 animate-fade">
                                    <div className="flex items-center justify-between">
                                        <span className="readout">UPPERCASE</span>
                                        <label className="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" checked={uppercase} onChange={(e) => setUppercase(e.target.checked)} className="sr-only peer" />
                                            <div className="w-8 h-4 rounded-full bg-paper3 peer-checked:bg-brass transition-colors after:content-[''] after:absolute after:top-0 after:left-0 after:h-4 after:w-4 after:rounded-full after:bg-ink after:transition-all peer-checked:after:translate-x-full"></div>
                                        </label>
                                    </div>
                                    <div>
                                        <div className="flex justify-between mb-1">
                                            <span className="readout">Dim inactive words</span>
                                            <span className="readout">{Math.round(baseOpacity * 100)}%</span>
                                        </div>
                                        <input
                                            type="range"
                                            min="30"
                                            max="100"
                                            value={Math.round(baseOpacity * 100)}
                                            onChange={(e) => setBaseOpacity(parseInt(e.target.value) / 100)}
                                            className="w-full accent-[var(--color-accent)]"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Position Selector */}
                        <div>
                            <p className="eyebrow mb-2">Position</p>
                            <SegmentedControl
                                options={POSITION_OPTIONS}
                                value={position}
                                onChange={setPosition}
                                size="sm"
                            />
                        </div>

                        {/* Collision Avoidance & Pre-existing Text Guard */}
                        <div className="p-3 rounded-input bg-paper2 border border-rule space-y-2.5">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-1.5 min-w-0">
                                    <span className={`w-2 h-2 rounded-full shrink-0 ${isScanning ? 'bg-muted animate-pulse' : hasBurnedInCaptions ? 'bg-amber-400' : 'bg-emerald-400'}`} />
                                    <span className="text-xs font-medium text-ink truncate">
                                        {isScanning ? 'scanning canvas…' : hasBurnedInCaptions ? 'hardcoded text detected' : 'no burned-in text'}
                                    </span>
                                </div>
                                <label className="relative inline-flex items-center cursor-pointer shrink-0 ml-2" title="Toggle collision avoidance">
                                    <input
                                        type="checkbox"
                                        checked={hasBurnedInCaptions}
                                        onChange={(e) => setHasBurnedInCaptions(e.target.checked)}
                                        className="sr-only peer"
                                    />
                                    <div className="w-8 h-4 rounded-full bg-paper3 peer-checked:bg-brass transition-colors after:content-[''] after:absolute after:top-0 after:left-0 after:h-4 after:w-4 after:rounded-full after:bg-ink after:transition-all peer-checked:after:translate-x-full"></div>
                                </label>
                            </div>

                            {hasBurnedInCaptions && (
                                <div className="space-y-2 pt-1 border-t border-rule2 animate-fade">
                                    <div className="flex justify-between items-center">
                                        <span className="eyebrow">Collision Mode</span>
                                        {detectionConfidence > 0 && (
                                            <span className="readout text-muted">conf {Math.round(detectionConfidence * 100)}%</span>
                                        )}
                                    </div>
                                    <SegmentedControl
                                        options={COLLISION_OPTIONS}
                                        value={collisionMode}
                                        onChange={setCollisionMode}
                                        size="sm"
                                    />
                                    {collisionMode === 'smart_reposition' && (
                                        <p className="text-[11px] text-muted leading-tight">
                                            Auto-elevates captions to center safe zone to prevent text overlapping.
                                        </p>
                                    )}
                                    {collisionMode === 'occlusion_mask' && (
                                        <p className="text-[11px] text-muted leading-tight">
                                            Renders an opaque frosted backdrop behind captions to cleanly mask old subtitles.
                                        </p>
                                    )}
                                    {collisionMode === 'manual_offset' && (
                                        <div className="space-y-1 pt-1">
                                            <div className="flex justify-between">
                                                <span className="readout">Height Offset</span>
                                                <span className="readout">{manualYOffset}%</span>
                                            </div>
                                            <input
                                                type="range"
                                                min="10"
                                                max="90"
                                                value={manualYOffset}
                                                onChange={(e) => setManualYOffset(parseInt(e.target.value))}
                                                className="w-full accent-[var(--color-accent)]"
                                            />
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>

                        {/* Animation Style (new) */}
                        <div>
                            <p className="eyebrow mb-2">Animation</p>
                            <SegmentedControl
                                options={ANIMATION_OPTIONS}
                                value={animation}
                                onChange={setAnimation}
                                columns={2}
                                size="sm"
                            />
                        </div>

                        {/* Caption text & words addition/editing */}
                        <div>
                            <button
                                type="button"
                                onClick={() => setShowTextEditor(!showTextEditor)}
                                className="w-full flex items-center justify-between mb-2 hover:opacity-80 transition-opacity"
                            >
                                <span className="eyebrow">
                                    Caption Text {captions.length > 0 ? `(${captions.length} words)` : '(add new)'}
                                </span>
                                <span className={`text-muted transition-transform ${showTextEditor ? 'rotate-180' : ''}`}>▾</span>
                            </button>
                            {showTextEditor && (
                                <div className="space-y-1.5 animate-fade">
                                    <textarea
                                        value={editableText}
                                        onChange={(e) => handleTextEdit(e.target.value)}
                                        rows={4}
                                        className="input-field resize-none leading-relaxed"
                                        placeholder="Type or paste caption text here to add or replace subtitles..."
                                    />
                                    <p className="text-[10px] text-muted leading-tight">
                                        Words are timed across the video. Applying will replace previous subtitles.
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Font Family */}
                        <div>
                            <p className="eyebrow mb-2">Font</p>
                            <select
                                value={fontName}
                                onChange={(e) => setFontName(e.target.value)}
                                className="input-field"
                            >
                                {FONT_OPTIONS.map((f) => (
                                    <option key={f.value} value={f.value} style={{ fontFamily: f.value }}>{f.label}</option>
                                ))}
                            </select>
                        </div>

                        {/* Text Color */}
                        <div>
                            <p className="eyebrow mb-2">Text color</p>
                            <div className="flex flex-wrap items-center gap-2.5">
                                {COLOR_PRESETS.map((c) => (
                                    <button
                                        key={c.color}
                                        onClick={() => setFontColor(c.color)}
                                        className={swatchClass(fontColor === c.color)}
                                        style={{ backgroundColor: c.color }}
                                        title={c.label}
                                    />
                                ))}
                                <label className="w-6 h-6 rounded-full border border-dashed border-rule2 cursor-pointer flex items-center justify-center hover:border-brass transition-colors overflow-hidden relative" title="Custom color">
                                    <span className="text-xs text-muted leading-none">+</span>
                                    <input type="color" value={fontColor} onChange={(e) => setFontColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                </label>
                            </div>
                        </div>

                        {/* Highlight Color (new) */}
                        <div>
                            <p className="eyebrow mb-2">Highlight</p>
                            <div className="flex flex-wrap items-center gap-2.5">
                                {HIGHLIGHT_PRESETS.map((c) => (
                                    <button
                                        key={c.color}
                                        onClick={() => setHighlightColor(c.color)}
                                        className={swatchClass(highlightColor === c.color)}
                                        style={{ backgroundColor: c.color }}
                                        title={c.label}
                                    />
                                ))}
                            </div>
                        </div>

                        {/* Border / Outline */}
                        <div>
                            <p className="eyebrow mb-2">Border</p>
                            <div className="flex items-center gap-3">
                                <label className="relative w-8 h-8 rounded-input border border-rule2 cursor-pointer overflow-hidden shrink-0" title="Border color">
                                    <div className="w-full h-full" style={{ backgroundColor: borderColor }} />
                                    <input type="color" value={borderColor} onChange={(e) => setBorderColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                </label>
                                <div className="flex-1">
                                    <input
                                        type="range"
                                        min="0"
                                        max="5"
                                        value={borderWidth}
                                        onChange={(e) => setBorderWidth(parseInt(e.target.value))}
                                        className="w-full accent-[var(--color-accent)]"
                                    />
                                    <div className="flex justify-between">
                                        <span className="readout">None</span>
                                        <span className="readout">Thick</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Background Box */}
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <p className="eyebrow">Background</p>
                                <label className="relative inline-flex items-center cursor-pointer">
                                    <input type="checkbox" checked={bgOpacity > 0} onChange={(e) => setBgOpacity(e.target.checked ? 0.5 : 0)} className="sr-only peer" />
                                    <div className="w-8 h-4 rounded-full bg-paper3 peer-checked:bg-brass transition-colors after:content-[''] after:absolute after:top-0 after:left-0 after:h-4 after:w-4 after:rounded-full after:bg-ink after:transition-all peer-checked:after:translate-x-full"></div>
                                </label>
                            </div>
                            {bgOpacity > 0 && (
                                <div className="space-y-3 animate-fade">
                                    <div className="flex items-center gap-3">
                                        <label className="relative w-8 h-8 rounded-input border border-rule2 cursor-pointer overflow-hidden shrink-0" title="Background color">
                                            <div className="w-full h-full" style={{ backgroundColor: bgColor }} />
                                            <input type="color" value={bgColor} onChange={(e) => setBgColor(e.target.value)} className="absolute inset-0 opacity-0 cursor-pointer" />
                                        </label>
                                        <div className="flex-1">
                                            <input
                                                type="range"
                                                min="10"
                                                max="100"
                                                value={Math.round(bgOpacity * 100)}
                                                onChange={(e) => setBgOpacity(parseInt(e.target.value) / 100)}
                                                className="w-full accent-[var(--color-accent)]"
                                            />
                                            <div className="flex justify-between">
                                                <span className="readout">Transparent</span>
                                                <span className="readout">{Math.round(bgOpacity * 100)}%</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Sticky Bottom Action Bar inside Subtitle Options */}
                    <div className="sticky bottom-0 bg-paper/95 backdrop-blur-sm border-t border-rule pt-2.5 pb-1 mt-2 space-y-2 z-30 shrink-0">
                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                onClick={handleCancel}
                                disabled={isProcessing}
                                className="btn-ghost py-1.5 px-3 text-xs flex items-center justify-center gap-1.5 flex-1 cursor-pointer hover:text-warn transition-colors"
                            >
                                <RotateCcw size={13} />
                                <span>Cancel</span>
                            </button>
                            <button
                                type="button"
                                onClick={() => onGenerate(styleOptions)}
                                disabled={isProcessing}
                                className={`btn-primary py-1.5 px-4 text-xs font-semibold flex items-center justify-center gap-2 flex-[2] transition-all cursor-pointer ${
                                    isDirty || textEdited ? 'ring-2 ring-brass shadow-md' : 'opacity-90 hover:opacity-100'
                                }`}
                            >
                                {isProcessing && !bulkRunning ? (
                                    <>
                                        <Loader2 size={13} className="animate-spin text-brassink" />
                                        <span>Applying Style…</span>
                                    </>
                                ) : (
                                    <>
                                        <Check size={13} />
                                        <span>Apply Style</span>
                                    </>
                                )}
                            </button>
                        </div>
                        {onApplyAll && bulkCount > 1 && (
                            <button
                                type="button"
                                onClick={() => onApplyAll({ ...styleOptions, captions: null })}
                                disabled={isProcessing}
                                className="btn-ghost w-full py-1 px-2 text-[11px] flex items-center justify-center gap-1.5 cursor-pointer"
                            >
                                {bulkRunning ? (
                                    <><Loader2 size={12} className="animate-spin" /> applying to all…</>
                                ) : (
                                    `apply this style to all ${bulkCount} clips`
                                )}
                            </button>
                        )}
                        {onRemove && (
                            <div className="text-center pt-0.5">
                                <button
                                    type="button"
                                    onClick={onRemove}
                                    disabled={isProcessing}
                                    className="text-[11px] text-muted hover:text-warn underline underline-offset-2 lowercase cursor-pointer"
                                >
                                    remove captions from clip
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </Modal>
    );
}
