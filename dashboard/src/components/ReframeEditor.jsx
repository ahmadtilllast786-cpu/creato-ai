import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
    X, Loader2, Crosshair, RotateCcw, AlertCircle,
    Play, Pause, Columns2, Eye, EyeOff, Users, Layers
} from 'lucide-react';
import { getApiUrl } from '../config';
import { apiJson } from '../lib/api';
import { ASPECT_RATIOS, FRAMING_MODES } from '../lib/subjectTracker';
import TrackingOverlayCanvas from './TrackingOverlayCanvas';

// Manual reframing & Multi-Person Auto-Reframe Engine:
// Controls 3-layer visual overlays, dynamic aspect ratio fitting (9:16, 1:1, 4:5, 16:9),
// and flexible framing modes (Full-Screen Focus vs. Split/Half-Screen).

const fmt = (s) => {
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, '0')}`;
};

export default function ReframeEditor({ jobId, clipIndex, clipTitle, onClose, onReframed }) {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [data, setData] = useState(null);
    const [overrides, setOverrides] = useState({});   // idx -> number | {top,bottom}
    const [playing, setPlaying] = useState(null);     // scene index being played
    const [saving, setSaving] = useState(false);

    // Multi-Person & Reframe Control State
    const [aspectRatio, setAspectRatio] = useState('9:16');
    const [framingMode, setFramingMode] = useState(FRAMING_MODES.FULL_SCREEN);
    const [showTrackingOverlays, setShowTrackingOverlays] = useState(true);

    useEffect(() => {
        let alive = true;
        (async () => {
            try {
                const res = await apiJson(`/api/clip/${jobId}/${clipIndex}/scenes`);
                if (!alive) return;
                setData(res);
                if (res.output_format === 'square') {
                    setAspectRatio('1:1');
                }
                const salvos = res.saved_overrides || {};
                setOverrides(Object.fromEntries(
                    Object.entries(salvos).map(([k, v]) => [Number(k), v])
                ));
            } catch (e) {
                if (alive) setError(e?.message || 'Could not read the scenes of this clip.');
            } finally {
                if (alive) setLoading(false);
            }
        })();
        return () => { alive = false; };
    }, [jobId, clipIndex]);

    const sourceW = data?.source_width || 1920;
    const sourceH = data?.source_height || 1080;
    const sourceAspect = sourceW / sourceH;
    const targetAspectVal = ASPECT_RATIOS[aspectRatio] || (9 / 16);
    const calculatedCropWidthFraction = Math.min(1.0, targetAspectVal / sourceAspect);
    const activeWidthFraction = calculatedCropWidthFraction || (data?.crop_width_fraction ?? 0.3164);
    const half = activeWidthFraction / 2;
    const clamp = useCallback((v) => Math.min(1 - half, Math.max(half, v)), [half]);

    const valueOf = useCallback((scene) => (
        overrides[scene.index] ?? clamp(scene.suggested_center)
    ), [overrides, clamp]);

    const setSingle = useCallback((idx, fraction) => {
        setOverrides((o) => ({ ...o, [idx]: clamp(fraction) }));
    }, [clamp]);

    const setSplitHalf = useCallback((idx, which, fraction) => {
        setOverrides((o) => {
            const cur = o[idx];
            const base = (cur && typeof cur === 'object')
                ? cur
                : { top: { x: clamp(0.3), y: 0.5 }, bottom: { x: clamp(0.7), y: 0.5 } };
            const anterior = base[which] || { y: 0.5 };
            return { ...o, [idx]: {
                ...base,
                [which]: { x: clamp(fraction), y: anterior.y ?? 0.5 },
            } };
        });
    }, [clamp]);

    const toggleSplit = useCallback((idx, scene) => {
        setOverrides((o) => {
            const cur = o[idx];
            if (cur && typeof cur === 'object') {
                return { ...o, [idx]: cur.top?.x ?? 0.5 };
            }
            const centre = typeof cur === 'number' ? cur : clamp(scene.suggested_center);
            const y = scene.suggested_center_y ?? 0.5;
            return { ...o, [idx]: {
                top: { x: clamp(centre - 0.2), y },
                bottom: { x: clamp(centre + 0.2), y },
            } };
        });
    }, [clamp]);

    const autoSplitScene = useCallback((idx, scene) => {
        const persons = scene.detected_persons || [];
        if (persons.length >= 2) {
            const sorted = [...persons].sort((a, b) => a.x - b.x);
            const p1 = sorted[0];
            const p2 = sorted[1];
            setOverrides((o) => ({
                ...o,
                [idx]: {
                    top: { x: clamp(p1.x + p1.width / 2), y: Math.min(1.0, Math.max(0.0, p1.y + p1.height / 2)) },
                    bottom: { x: clamp(p2.x + p2.width / 2), y: Math.min(1.0, Math.max(0.0, p2.y + p2.height / 2)) },
                },
            }));
        } else {
            toggleSplit(idx, scene);
        }
    }, [clamp, toggleSplit]);

    const applyFramingModeToAll = useCallback((mode) => {
        setFramingMode(mode);
        if (!data?.scenes) return;

        setOverrides((prev) => {
            const next = { ...prev };
            data.scenes.forEach((scene) => {
                const persons = scene.detected_persons || [];
                if (mode === FRAMING_MODES.SPLIT_LEFT) {
                    const targetX = persons.length > 0 ? persons[0].x + persons[0].width / 2 : 0.30;
                    next[scene.index] = clamp(targetX);
                } else if (mode === FRAMING_MODES.SPLIT_RIGHT) {
                    const targetX = persons.length > 0
                        ? persons[persons.length - 1].x + persons[persons.length - 1].width / 2
                        : 0.70;
                    next[scene.index] = clamp(targetX);
                } else if (mode === FRAMING_MODES.AUTO_SPLIT) {
                    if (persons.length >= 2) {
                        const sorted = [...persons].sort((a, b) => a.x - b.x);
                        next[scene.index] = {
                            top: { x: clamp(sorted[0].x + sorted[0].width / 2), y: sorted[0].y + sorted[0].height / 2 },
                            bottom: { x: clamp(sorted[1].x + sorted[1].width / 2), y: sorted[1].y + sorted[1].height / 2 },
                        };
                    } else {
                        next[scene.index] = clamp(scene.suggested_center);
                    }
                } else {
                    if (scene.group_hull) {
                        next[scene.index] = clamp(scene.group_hull.centerX);
                    } else {
                        next[scene.index] = clamp(scene.suggested_center);
                    }
                }
            });
            return next;
        });
    }, [data, clamp]);

    const resetScene = useCallback((idx) => {
        setOverrides((o) => {
            const next = { ...o };
            delete next[idx];
            return next;
        });
    }, []);

    const adjusted = Object.keys(overrides).length;

    const handleSave = async () => {
        if (!adjusted || saving) return;
        setSaving(true);
        setError(null);
        try {
            const payload = Object.fromEntries(Object.entries(overrides).map(([k, v]) => [
                String(k),
                typeof v === 'object'
                    ? { top: { x: Number(v.top.x.toFixed(4)), y: Number(v.top.y.toFixed(4)) },
                        bottom: { x: Number(v.bottom.x.toFixed(4)), y: Number(v.bottom.y.toFixed(4)) } }
                    : Number(v.toFixed(4)),
            ]));
            const res = await apiJson('/api/clip/reframe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: clipIndex,
                    crop_overrides: payload,
                    aspect_ratio: aspectRatio,
                }),
            });
            if (onReframed) onReframed(clipIndex, res);
            onClose();
        } catch (e) {
            setError(e?.message || 'The re-render failed.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-end sm:items-center justify-center p-0 sm:p-4">
            <div className="card w-full max-w-3xl max-h-[92vh] sm:max-h-[90vh] flex flex-col rounded-b-none sm:rounded-card animate-sheet-up sm:animate-none">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-rule">
                    <div className="flex items-center gap-2.5 min-w-0">
                        <Crosshair size={18} className="text-brass shrink-0" />
                        <div className="min-w-0">
                            <h2 className="text-base font-medium text-ink lowercase truncate">reframing & smart tracking</h2>
                            {clipTitle && <p className="text-xs text-muted truncate">{clipTitle}</p>}
                        </div>
                    </div>
                    <button onClick={onClose} className="p-1.5 hover:bg-paper3 rounded-input transition-colors">
                        <X size={18} className="text-muted" />
                    </button>
                </div>

                {/* Sub-header Toolbar: Diagnostic Layers, Aspect Ratio, Framing Modes */}
                <div className="px-4 py-2.5 border-b border-rule bg-paper2/50 flex flex-wrap items-center justify-between gap-2.5 text-xs">
                    {/* Tracking Overlays Toggle */}
                    <div className="flex items-center gap-2">
                        <button
                            type="button"
                            onClick={() => setShowTrackingOverlays(!showTrackingOverlays)}
                            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-input border transition-colors ${
                                showTrackingOverlays
                                    ? 'bg-brass/15 border-brass/40 text-brass font-medium'
                                    : 'bg-paper3/60 border-rule text-muted hover:text-ink'
                            }`}
                            title="Toggle Visual Tracking Overlays (Scanning Zone 20%-80%, Subject Boxes, Viewport, Headroom Guide)"
                        >
                            {showTrackingOverlays ? <Eye size={13} /> : <EyeOff size={13} />}
                            <span>tracking overlays</span>
                        </button>
                        <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono hidden md:inline-flex items-center gap-1">
                            focus: 20%–80%
                        </span>
                        <span className="text-[10px] px-2 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20 font-mono hidden sm:inline-flex items-center gap-1">
                            deadzone: ±15%
                        </span>
                    </div>

                    {/* Target Aspect Ratio Selector */}
                    <div className="flex items-center gap-1">
                        <span className="text-muted text-[11px] mr-1">ratio:</span>
                        {['9:16', '1:1', '4:5', '16:9'].map((ratio) => (
                            <button
                                key={ratio}
                                type="button"
                                onClick={() => setAspectRatio(ratio)}
                                className={`px-2 py-0.5 rounded text-[11px] font-mono transition-colors ${
                                    aspectRatio === ratio
                                        ? 'bg-ink text-paper font-semibold'
                                        : 'bg-paper3 text-muted hover:text-ink'
                                }`}
                            >
                                {ratio}
                            </button>
                        ))}
                    </div>

                    {/* Framing Modes Selector */}
                    <div className="flex items-center gap-1">
                        <span className="text-muted text-[11px] mr-1">framing:</span>
                        {[
                            { id: FRAMING_MODES.FULL_SCREEN, label: 'full' },
                            { id: FRAMING_MODES.SPLIT_LEFT, label: 'left' },
                            { id: FRAMING_MODES.SPLIT_RIGHT, label: 'right' },
                            { id: FRAMING_MODES.AUTO_SPLIT, label: 'auto-split' },
                        ].map((mode) => (
                            <button
                                key={mode.id}
                                type="button"
                                onClick={() => applyFramingModeToAll(mode.id)}
                                className={`px-2 py-0.5 rounded text-[11px] transition-colors ${
                                    framingMode === mode.id
                                        ? 'bg-brass/20 text-brass border border-brass/40 font-medium'
                                        : 'bg-paper3 text-muted hover:text-ink'
                                }`}
                            >
                                {mode.label}
                            </button>
                        ))}
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto overscroll-contain custom-scrollbar p-4 space-y-5">
                    <p className="text-xs text-muted leading-relaxed">
                        Play a scene to hear who is talking, then drag the rectangle over
                        the person you want. Multi-subject bounding boxes and group hulls are
                        tracked in real-time. Scenes you leave alone keep the automatic camera.
                    </p>

                    {loading && (
                        <div className="flex items-center gap-2 text-sm text-muted py-8 justify-center">
                            <Loader2 size={18} className="animate-spin text-brass" />
                            reading the scenes & detecting subjects…
                        </div>
                    )}

                    {error && (
                        <div className="flex items-start gap-2 text-xs text-warn border border-warn/40 rounded-input p-3">
                            <AlertCircle size={14} className="shrink-0 mt-0.5" />
                            <span>{error}</span>
                        </div>
                    )}

                    {data?.scenes?.map((scene) => (
                        <SceneRow
                            key={scene.index}
                            scene={scene}
                            value={valueOf(scene)}
                            widthFraction={activeWidthFraction}
                            aspectRatio={aspectRatio}
                            showOverlays={showTrackingOverlays}
                            previewUrl={data.preview_url}
                            touched={scene.index in overrides}
                            playing={playing === scene.index}
                            onPlayToggle={() => setPlaying((p) => (p === scene.index ? null : scene.index))}
                            onMoveSingle={(f) => setSingle(scene.index, f)}
                            onMoveHalf={(which, f) => setSplitHalf(scene.index, which, f)}
                            onToggleSplit={() => toggleSplit(scene.index, scene)}
                            onAutoSplit={() => autoSplitScene(scene.index, scene)}
                            onReset={() => resetScene(scene.index)}
                        />
                    ))}
                </div>

                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 border-t border-rule">
                    <span className="text-xs text-muted">
                        {adjusted === 0
                            ? 'nothing adjusted yet'
                            : `${adjusted} scene${adjusted > 1 ? 's' : ''} reframed by hand`}
                    </span>
                    <div className="flex items-center gap-2 [&>button]:flex-1 sm:[&>button]:flex-none">
                        <button onClick={onClose} className="btn-quiet py-2 px-4 text-sm">cancel</button>
                        <button
                            onClick={handleSave}
                            disabled={!adjusted || saving}
                            className="btn-primary py-2 px-4 text-sm disabled:opacity-40"
                        >
                            {saving ? <Loader2 size={16} className="animate-spin" /> : null}
                            {saving ? 're-rendering…' : 'apply reframing'}
                        </button>
                    </div>
                </div>
                <div className="sm:hidden safe-bottom" />
            </div>
        </div>
    );
}

function SceneRow({ scene, value, widthFraction, aspectRatio, showOverlays, previewUrl, touched, playing,
                    onPlayToggle, onMoveSingle, onMoveHalf, onToggleSplit, onAutoSplit, onReset }) {
    const boxRef = useRef(null);
    const videoRef = useRef(null);
    const [dragging, setDragging] = useState(null);   // null | 'single' | 'top' | 'bottom'

    const isSplit = value && typeof value === 'object';
    const numPersons = scene.detected_persons ? scene.detected_persons.length : 0;

    // Play only this scene's slice of the shared preview.
    useEffect(() => {
        const v = videoRef.current;
        if (!v) return;
        if (!playing) { v.pause(); return; }
        v.currentTime = scene.start;
        v.play().catch(() => {});
        const stopAtEnd = () => { if (v.currentTime >= scene.end) { v.pause(); } };
        v.addEventListener('timeupdate', stopAtEnd);
        return () => v.removeEventListener('timeupdate', stopAtEnd);
    }, [playing, scene.start, scene.end]);

    const fractionFromEvent = useCallback((clientX) => {
        const el = boxRef.current;
        if (!el) return 0.5;
        const rect = el.getBoundingClientRect();
        return (clientX - rect.left) / rect.width;
    }, []);

    useEffect(() => {
        if (!dragging) return;
        const move = (e) => {
            const x = e.touches ? e.touches[0].clientX : e.clientX;
            const f = fractionFromEvent(x);
            if (dragging === 'single') onMoveSingle(f);
            else onMoveHalf(dragging, f);
        };
        const up = () => setDragging(null);
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
        window.addEventListener('touchmove', move);
        window.addEventListener('touchend', up);
        return () => {
            window.removeEventListener('mousemove', move);
            window.removeEventListener('mouseup', up);
            window.removeEventListener('touchmove', move);
            window.removeEventListener('touchend', up);
        };
    }, [dragging, fractionFromEvent, onMoveSingle, onMoveHalf]);

    const startDrag = (which) => (e) => {
        e.stopPropagation();
        setDragging(which);
        const x = e.touches ? e.touches[0].clientX : e.clientX;
        if (which === 'single') onMoveSingle(fractionFromEvent(x));
        else onMoveHalf(which, fractionFromEvent(x));
    };

    const win = (centre, label, which) => {
        const leftPct = (centre - widthFraction / 2) * 100;
        return (
            <div
                key={which}
                onMouseDown={startDrag(which)}
                onTouchStart={startDrag(which)}
                className="absolute inset-y-0 border-2 border-brass cursor-ew-resize z-20"
                style={{ left: `${leftPct}%`, width: `${widthFraction * 100}%` }}
            >
                {label && (
                    <span className="absolute top-1 left-1 text-[10px] px-1 rounded bg-brass text-paper lowercase font-mono">
                        {label}
                    </span>
                )}
            </div>
        );
    };

    return (
        <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs gap-2">
                <div className="flex items-center gap-2 min-w-0">
                    <button
                        onClick={onPlayToggle}
                        className="flex items-center gap-1 text-ink2 hover:text-brass transition-colors"
                        title="play this scene with sound"
                    >
                        {playing ? <Pause size={13} /> : <Play size={13} />}
                    </button>
                    <span className="readout text-muted truncate">
                        scene {scene.index + 1} · {fmt(scene.start)}–{fmt(scene.end)}
                    </span>
                    {numPersons > 0 && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-paper3 border border-rule text-muted flex items-center gap-1 font-mono">
                            <Users size={10} className="text-brass" />
                            {numPersons} {numPersons === 1 ? 'subject' : 'subjects'}
                        </span>
                    )}
                </div>
                <div className="flex items-center gap-2.5 shrink-0">
                    {numPersons >= 2 && !isSplit && (
                        <button
                            onClick={onAutoSplit}
                            className="flex items-center gap-1 text-[11px] text-brass hover:underline bg-brass/10 px-1.5 py-0.5 rounded border border-brass/20"
                            title="Auto-split dual cameras across detected subjects"
                        >
                            <Columns2 size={11} /> auto-split
                        </button>
                    )}
                    <button
                        onClick={onToggleSplit}
                        className={`flex items-center gap-1 transition-colors ${
                            isSplit ? 'text-brass' : 'text-muted hover:text-ink2'}`}
                        title="stack two regions instead of one window"
                    >
                        <Columns2 size={12} /> {isSplit ? 'single' : 'split'}
                    </button>
                    {touched ? (
                        <button onClick={onReset} className="flex items-center gap-1 text-brass hover:underline">
                            <RotateCcw size={12} /> automatic
                        </button>
                    ) : (
                        <span className="text-muted">automatic</span>
                    )}
                </div>
            </div>

            <div
                ref={boxRef}
                className={`relative overflow-hidden rounded-input select-none border ${
                    touched ? 'border-brass' : 'border-rule'}`}
            >
                {/* 3 Visual UI Diagnostic / Editing Layers (HTML5 Canvas) */}
                <TrackingOverlayCanvas
                    persons={scene.detected_persons || []}
                    groupHull={scene.group_hull || null}
                    cameraViewport={
                        isSplit
                            ? {
                                  x: Math.max(0, value.top.x - widthFraction / 2),
                                  y: 0,
                                  width: widthFraction,
                                  height: 1.0,
                                  centerX: value.top.x,
                                  centerY: 0.5,
                              }
                            : {
                                  x: Math.max(0, value - widthFraction / 2),
                                  y: 0,
                                  width: widthFraction,
                                  height: 1.0,
                                  centerX: value,
                                  centerY: 0.5,
                              }
                    }
                    aspectRatio={aspectRatio}
                    showOverlays={showOverlays}
                    showScanningZone={true}
                    showHeadroom={true}
                    deadzoneRadius={0.15}
                />

                {playing && previewUrl ? (
                    <video
                        ref={videoRef}
                        src={getApiUrl(previewUrl)}
                        playsInline
                        className="w-full block"
                    />
                ) : scene.thumbnail_url ? (
                    <img
                        src={getApiUrl(scene.thumbnail_url)}
                        alt=""
                        draggable={false}
                        className="w-full block pointer-events-none"
                    />
                ) : (
                    <div className="w-full aspect-video bg-paper3" />
                )}

                {/* Everything outside the kept region is dimmed, so what survives
                    the crop is what stays bright. */}
                {!isSplit && (
                    <>
                        <div className="absolute inset-y-0 left-0 bg-black/65 pointer-events-none z-15"
                             style={{ width: `${Math.max(0, (value - widthFraction / 2) * 100)}%` }} />
                        <div className="absolute inset-y-0 right-0 bg-black/65 pointer-events-none z-15"
                             style={{ width: `${Math.max(0, 100 - (value + widthFraction / 2) * 100)}%` }} />
                    </>
                )}

                {isSplit
                    ? [win(value.top.x, 'top', 'top'), win(value.bottom.x, 'bottom', 'bottom')]
                    : win(value, null, 'single')}
            </div>

            {isSplit && (
                <p className="text-[11px] text-muted leading-snug">
                    Two regions stacked in the vertical frame: <strong>top</strong> above,
                    <strong> bottom</strong> below. Drag each one onto the person it should hold.
                </p>
            )}
        </div>
    );
}
