import React, { useEffect } from 'react';
import { 
    Instagram, 
    Youtube, 
    Video, 
    Grid, 
    Eye, 
    Shield, 
    Search, 
    Heart, 
    MessageCircle, 
    Share2, 
    Bookmark, 
    Disc, 
    ThumbsUp, 
    ThumbsDown, 
    Repeat, 
    Camera, 
    MoreVertical, 
    ChevronLeft,
    Plus
} from 'lucide-react';

/**
 * PlatformSafeZoneOverlay
 * 
 * Provides:
 * 1. Platform UI Collision Masking (TikTok, Instagram Reels, YouTube Shorts)
 *    Matching real platform obstruction overlays (headers, bottom metadata, right action rail)
 * 2. Visual Alignment Guides
 *    - Rule of Thirds grid (33.3%, 66.6%)
 *    - Upper Rule of Thirds Eye-Trace Target Zone (30% - 35%, nominal 32.5%)
 *    - Center Crosshair Reticle with +/- 2.5% tolerance tick marks
 *    - 7% Inset Safe Margin wireframe
 */
export default function PlatformSafeZoneOverlay({
    platform = 'off',
    showGuides = false,
    showEyeTrace = true,
    opacity = 0.82,
    className = ''
}) {
    if (platform === 'off' && !showGuides) {
        return null;
    }

    return (
        <div 
            className={`absolute inset-0 pointer-events-none overflow-hidden select-none z-20 ${className}`}
            style={{ width: '100%', height: '100%' }}
        >
            {/* ======================================================== */}
            {/* PLATFORM UI COLLISION MASKS (Red Overlays)              */}
            {/* ======================================================== */}

            {/* --- TIKTOK MASK --- */}
            {platform === 'tiktok' && (
                <div className="absolute inset-0 pointer-events-none text-white font-sans text-[10px]">
                    {/* Top Header Buffer (Y: 0% - 14%) */}
                    <div 
                        className="absolute top-0 inset-x-0 bg-red-600/80 backdrop-blur-[1px] flex flex-col justify-between px-3 pt-2 pb-1.5 shadow-sm"
                        style={{ height: '14%' }}
                    >
                        <div className="flex justify-between items-center text-[9px] opacity-80 font-mono">
                            <span>9:16</span>
                            <span className="tracking-wider uppercase font-semibold text-[8px] bg-black/40 px-1.5 py-0.5 rounded text-white">TikTok Safe Zone</span>
                            <span>100%</span>
                        </div>
                        <div className="flex items-center gap-2 mb-1">
                            <ChevronLeft size={16} className="shrink-0" />
                            <div className="flex-1 bg-white/20 rounded-full py-1 px-2.5 flex items-center gap-1.5 text-white/90 text-[10px]">
                                <Search size={11} className="shrink-0 opacity-70" />
                                <span className="truncate">Find related content</span>
                            </div>
                            <span className="text-[11px] font-medium shrink-0">Search</span>
                        </div>
                    </div>

                    {/* Right Action Rail Buffer (X: 82% - 100%, Y: 42% - 84%) */}
                    <div 
                        className="absolute right-0 bg-red-600/80 backdrop-blur-[1px] flex flex-col items-center justify-around py-2 shadow-sm rounded-l-xl"
                        style={{ top: '42%', bottom: '16%', width: '18%' }}
                    >
                        {/* Profile avatar + plus */}
                        <div className="relative flex flex-col items-center">
                            <div className="w-7 h-7 rounded-full border border-white/60 bg-black/30 flex items-center justify-center font-bold text-[9px]">
                                TT
                            </div>
                            <div className="absolute -bottom-1 w-3.5 h-3.5 bg-red-500 rounded-full flex items-center justify-center text-white border border-white">
                                <Plus size={9} />
                            </div>
                        </div>

                        {/* Heart */}
                        <div className="flex flex-col items-center gap-0.5">
                            <Heart size={16} fill="currentColor" className="drop-shadow" />
                            <span className="text-[8px] font-semibold">142K</span>
                        </div>

                        {/* Comment */}
                        <div className="flex flex-col items-center gap-0.5">
                            <MessageCircle size={16} fill="currentColor" className="drop-shadow" />
                            <span className="text-[8px] font-semibold">1,280</span>
                        </div>

                        {/* Bookmark */}
                        <div className="flex flex-col items-center gap-0.5">
                            <Bookmark size={16} fill="currentColor" className="drop-shadow" />
                            <span className="text-[8px] font-semibold">9.4K</span>
                        </div>

                        {/* Share */}
                        <div className="flex flex-col items-center gap-0.5">
                            <Share2 size={16} className="drop-shadow" />
                            <span className="text-[8px] font-semibold">Share</span>
                        </div>

                        {/* Audio Disc */}
                        <div className="w-6 h-6 rounded-full bg-black/50 border border-white/60 flex items-center justify-center">
                            <Disc size={13} className="animate-spin" style={{ animationDuration: '4s' }} />
                        </div>
                    </div>

                    {/* Bottom Metadata & Nav Buffer (Y: 76% - 100%) */}
                    <div 
                        className="absolute bottom-0 inset-x-0 bg-red-600/80 backdrop-blur-[1px] flex flex-col justify-end px-3 pb-2 pt-2 shadow-sm"
                        style={{ height: '24%' }}
                    >
                        <div className="max-w-[80%] flex flex-col gap-1 mb-2">
                            <div className="flex items-center gap-1.5 font-bold text-[11px]">
                                <span>@creator_account</span>
                                <span className="text-[8px] bg-white/20 px-1 rounded font-normal">Follow</span>
                            </div>
                            <p className="text-[9.5px] leading-tight line-clamp-2 text-white/95">
                                Keep viral captions and faces out of this red zone for maximum engagement... #fyp #viral
                            </p>
                            <div className="flex items-center gap-1 text-[8.5px] text-white/80 mt-0.5">
                                <Disc size={9} className="shrink-0" />
                                <span className="truncate">Original Sound - Trending Audio Track</span>
                            </div>
                        </div>

                        {/* Bottom Nav Bar */}
                        <div className="border-t border-white/20 pt-1 flex justify-between items-center text-[8px] opacity-80">
                            <span>Home</span>
                            <span>Friends</span>
                            <span className="w-6 h-4 bg-white/30 rounded flex items-center justify-center font-bold">+</span>
                            <span>Inbox</span>
                            <span>Profile</span>
                        </div>
                    </div>

                    {/* Safe Zone Indicator Stamp */}
                    <div className="absolute top-[16%] left-3 border border-dashed border-emerald-400/60 bg-emerald-950/40 px-2 py-0.5 rounded text-[8px] text-emerald-300 flex items-center gap-1">
                        <Shield size={9} />
                        <span>TikTok Safe Envelope (10%-82%, 14%-76%)</span>
                    </div>
                </div>
            )}

            {/* --- INSTAGRAM REELS MASK --- */}
            {platform === 'instagram' && (
                <div className="absolute inset-0 pointer-events-none text-white font-sans text-[10px]">
                    {/* Top Header Buffer (Y: 0% - 14.5%) */}
                    <div 
                        className="absolute top-0 inset-x-0 bg-red-600/80 backdrop-blur-[1px] flex items-center justify-between px-3 pt-3 pb-2 shadow-sm"
                        style={{ height: '14.5%' }}
                    >
                        <div className="flex items-center gap-2">
                            <ChevronLeft size={18} />
                            <span className="font-bold text-sm tracking-wide">Reels</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="tracking-wider uppercase font-semibold text-[8px] bg-black/40 px-1.5 py-0.5 rounded text-white">Instagram Safe Zone</span>
                            <Camera size={18} />
                        </div>
                    </div>

                    {/* Right Action Rail Buffer (Y: 52% - 87%, X: 82% - 100%) */}
                    <div 
                        className="absolute right-0 bg-red-600/80 backdrop-blur-[1px] flex flex-col items-center justify-around py-3 shadow-sm rounded-l-xl"
                        style={{ top: '50%', bottom: '13%', width: '18%' }}
                    >
                        {/* Like */}
                        <div className="flex flex-col items-center gap-0.5">
                            <Heart size={18} />
                            <span className="text-[8.5px] font-semibold">14.7K</span>
                        </div>

                        {/* Comment */}
                        <div className="flex flex-col items-center gap-0.5">
                            <MessageCircle size={18} />
                            <span className="text-[8.5px] font-semibold">269</span>
                        </div>

                        {/* Share */}
                        <div className="flex flex-col items-center gap-0.5">
                            <Share2 size={18} />
                            <span className="text-[8.5px] font-semibold">4,447</span>
                        </div>

                        {/* 3 Dots */}
                        <div className="flex flex-col items-center">
                            <MoreVertical size={16} />
                        </div>

                        {/* Audio Thumbnail */}
                        <div className="w-6 h-6 rounded-md border border-white/60 bg-black/40 flex items-center justify-center overflow-hidden">
                            <Disc size={12} className="animate-spin" style={{ animationDuration: '3s' }} />
                        </div>
                    </div>

                    {/* Bottom Metadata Buffer (Y: 76% - 100%) */}
                    <div 
                        className="absolute bottom-0 inset-x-0 bg-red-600/80 backdrop-blur-[1px] flex flex-col justify-end px-3 pb-3 pt-2 shadow-sm"
                        style={{ height: '24%' }}
                    >
                        <div className="max-w-[80%] flex flex-col gap-1">
                            <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded-full bg-white/30 border border-white/60 flex items-center justify-center font-bold text-[8px]">
                                    IG
                                </div>
                                <span className="font-semibold text-[11px]">reels_creator</span>
                                <span className="text-[9px] border border-white/40 px-1.5 py-0.2 rounded font-medium">Follow</span>
                            </div>
                            <p className="text-[9.5px] leading-tight line-clamp-2 text-white/95 mt-0.5">
                                Keep subtitles above this bar to avoid Instagram comment and profile UI collision...
                            </p>
                            <div className="flex items-center gap-1 text-[8.5px] text-white/80">
                                <Disc size={9} className="shrink-0" />
                                <span className="truncate">Original Audio · reels_creator</span>
                            </div>
                        </div>
                    </div>

                    {/* Safe Zone Indicator Stamp */}
                    <div className="absolute top-[16%] left-3 border border-dashed border-emerald-400/60 bg-emerald-950/40 px-2 py-0.5 rounded text-[8px] text-emerald-300 flex items-center gap-1">
                        <Shield size={9} />
                        <span>Reels Safe Envelope (10%-82%, 14.5%-76%)</span>
                    </div>
                </div>
            )}

            {/* --- YOUTUBE SHORTS MASK --- */}
            {platform === 'youtube' && (
                <div className="absolute inset-0 pointer-events-none text-white font-sans text-[10px]">
                    {/* Top Header Buffer (Y: 0% - 13.5%) */}
                    <div 
                        className="absolute top-0 inset-x-0 bg-red-600/80 backdrop-blur-[1px] flex items-center justify-between px-3 pt-3 pb-2 shadow-sm"
                        style={{ height: '13.5%' }}
                    >
                        <ChevronLeft size={18} />
                        <div className="flex items-center gap-3">
                            <span className="tracking-wider uppercase font-semibold text-[8px] bg-black/40 px-1.5 py-0.5 rounded text-white">Shorts Safe Zone</span>
                            <Search size={16} />
                            <MoreVertical size={16} />
                        </div>
                    </div>

                    {/* Right Action Rail Buffer (Y: 38% - 86%, X: 82% - 100%) */}
                    <div 
                        className="absolute right-0 bg-red-600/80 backdrop-blur-[1px] flex flex-col items-center justify-around py-3 shadow-sm rounded-l-xl"
                        style={{ top: '38%', bottom: '14%', width: '18%' }}
                    >
                        {/* Like */}
                        <div className="flex flex-col items-center gap-0.5">
                            <ThumbsUp size={16} />
                            <span className="text-[8.5px] font-semibold">396</span>
                        </div>

                        {/* Dislike */}
                        <div className="flex flex-col items-center gap-0.5">
                            <ThumbsDown size={16} />
                            <span className="text-[8.5px] font-semibold">Dislike</span>
                        </div>

                        {/* Comments */}
                        <div className="flex flex-col items-center gap-0.5">
                            <MessageCircle size={16} />
                            <span className="text-[8.5px] font-semibold">18</span>
                        </div>

                        {/* Share */}
                        <div className="flex flex-col items-center gap-0.5">
                            <Share2 size={16} />
                            <span className="text-[8.5px] font-semibold">Share</span>
                        </div>

                        {/* Remix */}
                        <div className="flex flex-col items-center gap-0.5">
                            <Repeat size={16} />
                            <span className="text-[8.5px] font-semibold">Remix</span>
                        </div>

                        {/* Sound thumbnail */}
                        <div className="w-6 h-6 rounded-md border border-white/60 bg-black/40 flex items-center justify-center overflow-hidden">
                            <div className="w-2.5 h-2.5 bg-red-500 rounded-sm" />
                        </div>
                    </div>

                    {/* Bottom Channel & Title Buffer (Y: 76% - 100%) */}
                    <div 
                        className="absolute bottom-0 inset-x-0 bg-red-600/80 backdrop-blur-[1px] flex flex-col justify-end px-3 pb-3 pt-2 shadow-sm"
                        style={{ height: '24%' }}
                    >
                        <div className="max-w-[80%] flex flex-col gap-1.5">
                            <div className="flex items-center gap-2">
                                <div className="w-6 h-6 rounded-full bg-red-600 border border-white/60 flex items-center justify-center font-bold text-[8px]">
                                    YT
                                </div>
                                <span className="font-bold text-[11px]">@ShortsChannel</span>
                                <span className="bg-white text-black font-semibold text-[9px] px-2 py-0.5 rounded-full">
                                    Subscribe
                                </span>
                            </div>
                            <p className="text-[9.5px] leading-tight line-clamp-2 text-white/95">
                                Keep title & subtitles clear of YouTube Shorts navigation & subscribe banner
                            </p>
                        </div>
                    </div>

                    {/* Safe Zone Indicator Stamp */}
                    <div className="absolute top-[15%] left-3 border border-dashed border-emerald-400/60 bg-emerald-950/40 px-2 py-0.5 rounded text-[8px] text-emerald-300 flex items-center gap-1">
                        <Shield size={9} />
                        <span>Shorts Safe Envelope (10%-82%, 13.5%-76%)</span>
                    </div>
                </div>
            )}

            {/* ======================================================== */}
            {/* VISUAL DIAGNOSTIC GUIDES (Rule of 3rds, Reticle, Gaze)   */}
            {/* ======================================================== */}
            {showGuides && (
                <div className="absolute inset-0 pointer-events-none">
                    {/* 7% Inset Safe Margin Box */}
                    <div 
                        className="absolute border border-dashed border-white/35 rounded-sm pointer-events-none"
                        style={{ top: '7%', bottom: '7%', left: '7%', right: '7%' }}
                    >
                        <span className="absolute -top-3 left-2 font-mono text-[8px] text-white/50 bg-black/60 px-1 rounded">
                            7% Action Safe
                        </span>
                    </div>

                    {/* Rule of Thirds - Horizontal Lines */}
                    <div 
                        className="absolute inset-x-0 border-t border-cyan-400/40 pointer-events-none flex justify-between px-2"
                        style={{ top: '33.333%' }}
                    >
                        <span className="font-mono text-[7.5px] text-cyan-300/70 -translate-y-3 bg-black/60 px-0.5 rounded">1/3 (33%)</span>
                    </div>
                    <div 
                        className="absolute inset-x-0 border-t border-cyan-400/40 pointer-events-none flex justify-between px-2"
                        style={{ top: '66.666%' }}
                    >
                        <span className="font-mono text-[7.5px] text-cyan-300/70 -translate-y-3 bg-black/60 px-0.5 rounded">2/3 (67%)</span>
                    </div>

                    {/* Rule of Thirds - Vertical Lines */}
                    <div 
                        className="absolute inset-y-0 border-l border-cyan-400/40 pointer-events-none"
                        style={{ left: '33.333%' }}
                    />
                    <div 
                        className="absolute inset-y-0 border-l border-cyan-400/40 pointer-events-none"
                        style={{ left: '66.666%' }}
                    />

                    {/* Upper Rule-of-Thirds Eye-Trace Target Zone (Y: 30% - 35%, Target: 32.5%) */}
                    {showEyeTrace && (
                        <div 
                            className="absolute inset-x-0 bg-amber-400/10 border-y border-dashed border-amber-400/60 pointer-events-none flex items-center justify-between px-2"
                            style={{ top: '30%', height: '5%' }}
                        >
                            <span className="font-mono text-[8px] text-amber-300 bg-black/70 px-1 rounded shadow flex items-center gap-1">
                                <Eye size={9} className="text-amber-400" />
                                <span>Eye-Line Anchor (32.5%)</span>
                            </span>
                            {/* Horizontal Midpoint Reticle at (50%, 32.5%) */}
                            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center">
                                <div className="w-3 h-3 border border-amber-400 rounded-full flex items-center justify-center">
                                    <div className="w-1 h-1 bg-amber-400 rounded-full animate-ping" />
                                </div>
                            </div>
                            <span className="font-mono text-[7.5px] text-amber-300/80 bg-black/70 px-0.5 rounded">±2.5% Tol</span>
                        </div>
                    )}

                    {/* Center Crosshair Reticle (50%, 50%) with Tolerance Ticks */}
                    <div 
                        className="absolute pointer-events-none"
                        style={{ left: '50%', top: '50%', transform: 'translate(-50%, -50%)' }}
                    >
                        {/* Center Circle */}
                        <div className="w-6 h-6 -ml-3 -mt-3 border border-cyan-400/70 rounded-full absolute flex items-center justify-center">
                            <div className="w-1 h-1 bg-cyan-400 rounded-full" />
                        </div>
                        {/* Horizontal Reticle Line */}
                        <div className="absolute w-12 h-px bg-cyan-400/70 -left-6 top-0" />
                        {/* Vertical Reticle Line */}
                        <div className="absolute h-12 w-px bg-cyan-400/70 left-0 -top-6" />

                        {/* Precision Ticks at +/- 2.5% */}
                        <div className="absolute w-2 h-px bg-amber-400 -left-1 -top-3" title="+2.5% gaze tolerance" />
                        <div className="absolute w-2 h-px bg-amber-400 -left-1 top-3" title="-2.5% gaze tolerance" />
                        <div className="absolute h-2 w-px bg-amber-400 -top-1 -left-3" title="-2.5% horizontal tolerance" />
                        <div className="absolute h-2 w-px bg-amber-400 -top-1 left-3" title="+2.5% horizontal tolerance" />
                    </div>
                </div>
            )}
        </div>
    );
}

/**
 * PlatformSafeZoneControls
 * Compact toolbar controls to toggle safe zones and diagnostic guides
 */
export function PlatformSafeZoneControls({
    platform = 'off',
    onPlatformChange,
    showGuides = false,
    onToggleGuides,
    className = ''
}) {
    // Keyboard shortcut G to toggle guides
    useEffect(() => {
        const handleKeyDown = (e) => {
            // Only trigger if not inside an input / textarea
            if (['INPUT', 'TEXTAREA'].includes(e.target?.tagName)) return;
            if (e.key === 'g' || e.key === 'G') {
                e.preventDefault();
                onToggleGuides && onToggleGuides(!showGuides);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [showGuides, onToggleGuides]);

    return (
        <div className={`flex items-center gap-1.5 bg-paper2/90 backdrop-blur-sm border border-rule px-2 py-1 rounded-input text-xs ${className}`}>
            <span className="text-[10px] text-muted uppercase font-mono tracking-wider flex items-center gap-1">
                <Shield size={11} className="text-ink2" />
                <span>Mask:</span>
            </span>

            {/* Platform Selector Buttons */}
            <div className="flex items-center bg-paper3 rounded p-0.5 gap-0.5 border border-rule/50">
                {[
                    { id: 'off', label: 'Off' },
                    { id: 'tiktok', label: 'TikTok', icon: Video },
                    { id: 'instagram', label: 'Reels', icon: Instagram },
                    { id: 'youtube', label: 'Shorts', icon: Youtube },
                ].map((p) => {
                    const active = platform === p.id;
                    const Icon = p.icon;
                    return (
                        <button
                            key={p.id}
                            type="button"
                            onClick={() => onPlatformChange && onPlatformChange(p.id)}
                            className={`px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors flex items-center gap-1 ${
                                active 
                                    ? 'bg-ink text-paper shadow-sm' 
                                    : 'text-muted hover:text-ink hover:bg-paper'
                            }`}
                            title={`Toggle ${p.label} safe zone UI obstruction mask`}
                        >
                            {Icon && <Icon size={10} />}
                            <span>{p.label}</span>
                        </button>
                    );
                })}
            </div>

            <div className="h-3.5 w-px bg-rule mx-0.5" />

            {/* Alignment Guides Toggle */}
            <button
                type="button"
                onClick={() => onToggleGuides && onToggleGuides(!showGuides)}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors flex items-center gap-1 border ${
                    showGuides 
                        ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 shadow-sm' 
                        : 'border-transparent text-muted hover:text-ink hover:bg-paper'
                }`}
                title="Toggle Rule-of-Thirds, Eye-Trace Target & Reticle Guides (Shortcut: G)"
            >
                <Grid size={11} />
                <span>Guides</span>
                <kbd className="font-mono text-[8px] bg-black/40 px-1 py-0.2 rounded text-muted ml-0.5 border border-white/10">G</kbd>
            </button>
        </div>
    );
}
