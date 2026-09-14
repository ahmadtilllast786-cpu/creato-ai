import React, { useMemo, useRef, useEffect } from 'react';
import { Player } from '@remotion/player';
import { ShortVideo } from '../remotion/compositions/ShortVideo';
import { ActivePlaybackController } from '../lib/activePlayback';

/**
 * Wraps Remotion's Player component for real-time preview in modals.
 * Accepts the same ShortVideoProps interface as the Remotion composition.
 *
 * @param {object} props
 * @param {string} props.videoUrl - URL to the base clip video
 * @param {number} props.durationInSeconds - Video duration in seconds
 * @param {object|null} props.subtitles - SubtitleConfig or null
 * @param {object|null} props.hook - HookConfig or null
 * @param {object|null} props.effects - EffectsConfig or null
 * @param {string} [props.className] - Additional CSS classes
 * @param {string} [props.playerId] - Unique player ID
 */
export default function RemotionPreview({
    videoUrl,
    durationInSeconds = 30,
    subtitles = null,
    hook = null,
    effects = null,
    className = '',
    playerId: customPlayerId = null,
}) {
    const fps = 30;
    const durationInFrames = Math.max(1, Math.round(durationInSeconds * fps));
    const playerRef = useRef(null);
    const containerRef = useRef(null);
    const playerId = useMemo(
        () => customPlayerId || `remotion-player-${Math.random().toString(36).slice(2, 9)}`,
        [customPlayerId]
    );

    const inputProps = useMemo(
        () => ({
            videoUrl,
            durationInFrames,
            fps,
            width: 1080,
            height: 1920,
            subtitles,
            hook,
            effects,
        }),
        [videoUrl, durationInFrames, subtitles, hook, effects]
    );

    useEffect(() => {
        // Stop any currently playing background media before mounting Remotion player
        ActivePlaybackController.claimPlayback(playerId, {
            pause: () => {
                try {
                    playerRef.current?.pause();
                } catch (e) {}
            },
        });

        const unregister = ActivePlaybackController.register(playerId, {
            pause: () => {
                try {
                    playerRef.current?.pause();
                } catch (e) {}
            },
        });

        const handleStopAll = (e) => {
            if (e.detail?.activeId !== playerId) {
                try {
                    playerRef.current?.pause();
                } catch (e) {}
            }
        };

        window.addEventListener('STOP_ALL_MEDIA', handleStopAll);
        return () => {
            unregister();
            window.removeEventListener('STOP_ALL_MEDIA', handleStopAll);
        };
    }, [playerId]);

    return (
        <div ref={containerRef} data-player-id={playerId} className={`w-full h-full ${className}`}>
            <Player
                ref={playerRef}
                component={ShortVideo}
                inputProps={inputProps}
                durationInFrames={durationInFrames}
                fps={fps}
                compositionWidth={1080}
                compositionHeight={1920}
                style={{
                    width: '100%',
                    height: '100%',
                }}
                controls
                autoPlay
                loop
            />
        </div>
    );
}
