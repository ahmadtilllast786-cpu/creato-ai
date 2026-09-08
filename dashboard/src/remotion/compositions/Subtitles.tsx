import React from "react";
import {
  AbsoluteFill,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
} from "remotion";
import type { SubtitleConfig } from "../lib/types";
import { groupCaptionsIntoBlocks, getActiveWordIndex } from "../lib/captions";
import { getFontStack } from "../lib/fonts";

interface SubtitlesProps {
  config: SubtitleConfig;
}

const POSITION_MAP: Record<string, React.CSSProperties> = {
  top: { top: "12%", bottom: "auto", transform: "translate3d(0, 0, 0)" },
  middle: { top: "48%", bottom: "auto", transform: "translate3d(0, -50%, 0)" },
  bottom: { bottom: "17%", top: "auto", transform: "translate3d(0, 0, 0)" },
};

export const Subtitles: React.FC<SubtitlesProps> = ({ config }) => {
  const { fps } = useVideoConfig();
  const blocks = groupCaptionsIntoBlocks(config.captions);

  return (
    <AbsoluteFill style={{ pointerEvents: "none" }}>
      {blocks.map((block, i) => {
        const startFrame = Math.round((block.startMs / 1000) * fps);
        const durationFrames = Math.max(
          1,
          Math.round(((block.endMs - block.startMs) / 1000) * fps)
        );

        return (
          <Sequence
            key={i}
            from={startFrame}
            durationInFrames={durationFrames}
            layout="none"
          >
            <SubtitleBlock
              block={block}
              config={config}
              blockStartMs={block.startMs}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

interface SubtitleBlockProps {
  block: ReturnType<typeof groupCaptionsIntoBlocks>[number];
  config: SubtitleConfig;
  blockStartMs: number;
}

const SubtitleBlock: React.FC<SubtitleBlockProps> = ({
  block,
  config,
  blockStartMs,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const {
    style,
    position,
    hasBurnedInCaptions,
    collisionMode = "smart_reposition",
    manualYOffset,
  } = config;

  // Frame-quantized time relative to composition start (damps microsecond jitter)
  const currentTimeMs = blockStartMs + Math.round((frame / fps) * 1000);
  const activeIndex = getActiveWordIndex(block.words, currentTimeMs);

  // Dynamic Collision Avoidance & Safe Zones
  let positionStyle = POSITION_MAP[position] ?? POSITION_MAP.bottom;
  let isOcclusionMask = false;

  if (hasBurnedInCaptions) {
    if (collisionMode === "smart_reposition") {
      // Auto-elevate above lower-third (Y: 65%-95%) into center-safe zone (Y: 48% or bottom: 38%)
      positionStyle = {
        top: "auto",
        bottom: "38%",
        transform: "translate3d(0, 0, 0)",
      };
    } else if (collisionMode === "occlusion_mask") {
      // Keep lower position but activate high-opacity occlusion backdrop to hide old text
      positionStyle = {
        top: "auto",
        bottom: "17%",
        transform: "translate3d(0, 0, 0)",
      };
      isOcclusionMask = true;
    } else if (collisionMode === "manual_offset" && manualYOffset != null) {
      const clampedOffset = Math.max(5, Math.min(95, Math.round(manualYOffset)));
      positionStyle = {
        top: `${clampedOffset}%`,
        bottom: "auto",
        transform: "translate3d(0, -50%, 0)",
      };
    }
  }

  const fontStack = getFontStack(style.fontFamily);

  // Background style: standard background or occlusion mask pill
  let bgStyle: React.CSSProperties = {};
  if (isOcclusionMask) {
    bgStyle = {
      backgroundColor: "rgba(10, 11, 16, 0.95)",
      backdropFilter: "blur(16px)",
      WebkitBackdropFilter: "blur(16px)",
      borderRadius: 14,
      padding: "14px 28px",
      boxShadow: "0 8px 32px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.1)",
      minWidth: "65%",
      maxWidth: "92%",
    };
  } else if (style.bgOpacity > 0) {
    bgStyle = {
      backgroundColor: `${style.bgColor}${Math.round(style.bgOpacity * 255)
        .toString(16)
        .padStart(2, "0")}`,
      borderRadius: 8,
      padding: "8px 16px",
    };
  }

  return (
    <div
      style={{
        position: "absolute",
        left: "10%",
        right: "10%",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        pointerEvents: "none",
        ...positionStyle,
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          alignItems: "center",
          gap: "6px 8px",
          maxWidth: "100%",
          lineHeight: 1.35,
          ...bgStyle,
        }}
      >
        {block.words.map((word, i) => (
          <WordSpan
            key={i}
            word={word.text}
            isActive={i === activeIndex}
            style={style}
            fontStack={fontStack}
            animation={style.animation}
            frame={frame}
            fps={fps}
            wordStartMs={word.startMs}
            blockStartMs={blockStartMs}
          />
        ))}
      </div>
    </div>
  );
};

interface WordSpanProps {
  word: string;
  isActive: boolean;
  style: SubtitleConfig["style"];
  fontStack: string;
  animation: SubtitleConfig["style"]["animation"];
  frame: number;
  fps: number;
  wordStartMs: number;
  blockStartMs: number;
}

const WordSpan: React.FC<WordSpanProps> = ({
  word,
  isActive,
  style,
  fontStack,
  animation,
  frame,
  fps,
  wordStartMs,
  blockStartMs,
}) => {
  const wordStartFrame = Math.round(
    ((wordStartMs - blockStartMs) / 1000) * fps
  );

  let transform = "";
  let color = style.fontColor;
  let extraStyle: React.CSSProperties = {};

  // Dim inactive words toward the backend's opaque scaled color (matches the
  // burned ASS look; not CSS opacity).
  if (!isActive && style.baseOpacity != null && style.baseOpacity < 1) {
    const m = /^#?([0-9a-fA-F]{6})$/.exec(style.fontColor || "#FFFFFF");
    if (m) {
      const scale = 0.35 + 0.65 * style.baseOpacity;
      const [r, g, b] = [0, 2, 4].map((i) =>
        Math.round(parseInt(m[1].slice(i, i + 2), 16) * scale)
      );
      color = `rgb(${r}, ${g}, ${b})`;
    }
  }

  if (isActive) {
    color = style.highlightColor;

    switch (animation) {
      case "pop": {
        const scale = spring({
          frame: frame - wordStartFrame,
          fps,
          config: { mass: 0.5, stiffness: 280, damping: 14 },
          durationInFrames: 10,
        });
        // Quantize scale to 3 decimals to avoid sub-pixel micro-jitter
        const scaleValue = Math.round(interpolate(scale, [0, 1], [1, 1.18]) * 1000) / 1000;
        transform = `scale(${scaleValue})`;
        break;
      }
      case "karaoke": {
        extraStyle = {
          backgroundColor: style.highlightColor,
          color: style.bgColor || "#000000",
          borderRadius: 4,
          padding: "2px 6px",
        };
        break;
      }
      case "word-highlight": {
        extraStyle = {
          textShadow: `0 0 12px ${style.highlightColor}, 0 0 24px ${style.highlightColor}40`,
        };
        break;
      }
      default:
        break;
    }
  }

  // Text stroke via integer-quantized textShadow (CSS paint-order not reliable in Remotion)
  const bw = Math.max(0, Math.round(style.borderWidth));
  const strokeShadow =
    bw > 0
      ? [
          `${bw}px 0 0 ${style.borderColor}`,
          `-${bw}px 0 0 ${style.borderColor}`,
          `0 ${bw}px 0 ${style.borderColor}`,
          `0 -${bw}px 0 ${style.borderColor}`,
          `${bw}px ${bw}px 0 ${style.borderColor}`,
          `-${bw}px -${bw}px 0 ${style.borderColor}`,
          `${bw}px -${bw}px 0 ${style.borderColor}`,
          `-${bw}px ${bw}px 0 ${style.borderColor}`,
        ].join(", ")
      : "none";

  return (
    <span
      style={{
        fontFamily: fontStack,
        fontSize: Math.round(style.fontSize),
        fontWeight: 700,
        color: animation === "karaoke" && isActive ? undefined : color,
        textShadow:
          animation !== "karaoke"
            ? [strokeShadow, extraStyle.textShadow].filter(Boolean).join(", ")
            : strokeShadow,
        transform,
        transformOrigin: "center bottom",
        willChange: "transform",
        display: "inline-block",
        verticalAlign: "baseline",
        padding: "2px 4px",
        transition: "transform 90ms cubic-bezier(0.2, 0.8, 0.2, 1), color 80ms ease",
        textTransform: style.uppercase ? "uppercase" : "none",
        ...extraStyle,
      }}
    >
      {word}
    </span>
  );
};
