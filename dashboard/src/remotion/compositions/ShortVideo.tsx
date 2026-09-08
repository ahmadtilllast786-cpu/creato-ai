import React from "react";
import { AbsoluteFill } from "remotion";
import { Video } from "@remotion/media";
import type { ShortVideoProps } from "../lib/types";
import { Subtitles } from "./Subtitles";
import { HookOverlay } from "./HookOverlay";
import { VideoEffects } from "./VideoEffects";

/**
 * Main composition that layers all post-processing on top of the base video.
 * Uses @remotion/media Video for browser-side rendering compatibility.
 */
export const ShortVideo: React.FC<Record<string, unknown>> = (rawProps) => {
  const { videoUrl, subtitles, hook, effects } =
    rawProps as unknown as ShortVideoProps;
  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {/* Layer 1: Base video container with motion/zoom effects isolated to footage */}
      <AbsoluteFill
        id="video-motion-viewport"
        style={{
          overflow: "hidden",
          pointerEvents: "auto",
        }}
      >
        <VideoEffects config={effects}>
          <Video
            src={videoUrl}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        </VideoEffects>
      </AbsoluteFill>

      {/* Layer 2: Independent stationary overlay viewport (Decoupled from video motion) */}
      <AbsoluteFill
        id="captions-overlay-viewport"
        style={{
          pointerEvents: "none",
          zIndex: 10,
        }}
      >
        {/* Animated subtitles (Screen-fixed coordinates, no motion-container inheritance) */}
        {subtitles && <Subtitles config={subtitles} />}

        {/* Hook text overlay */}
        {hook && <HookOverlay config={hook} />}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
