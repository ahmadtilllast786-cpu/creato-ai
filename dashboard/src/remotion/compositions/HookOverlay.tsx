import React from "react";
import {
  AbsoluteFill,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
} from "remotion";
import type { HookConfig } from "../lib/types";
import { notoSerifFontFace, NOTO_SERIF_FONT_FAMILY } from "../lib/fonts";

interface HookOverlayProps {
  config: HookConfig;
}

const SIZE_SCALE: Record<string, number> = {
  S: 0.8,
  M: 1.0,
  L: 1.3,
};

// Permanent Viral Hook Positioning: Anchor the initial hook headline permanently
// in the dedicated safe margin strictly within the top safe zone (Y: 5%–12%), so it never overlays
// speaker faces, eyes, or lower subtitles.
const POSITION_STYLE: Record<string, React.CSSProperties> = {
  top: { top: "7%", bottom: "auto" },
  center: { top: "50%", bottom: "auto", transform: "translateY(-50%)" },
  bottom: { top: "78%", bottom: "auto" },
};

// Must mirror hooks.py HOOK_STYLES (the server-side FFmpeg fallback).
interface HookLook {
  box: string | null;
  text: string;
  outlinePx: number;
  shadow: boolean;
}

const HOOK_LOOKS: Record<string, HookLook> = {
  classic: { box: "rgba(18, 18, 20, 0.94)", text: "#FFFFFF", outlinePx: 0, shadow: true },
  black_white: { box: "rgba(18, 18, 20, 0.94)", text: "#FFFFFF", outlinePx: 0, shadow: true },
  dark: { box: "rgba(18, 18, 20, 0.92)", text: "#FFFFFF", outlinePx: 0, shadow: true },
  white_card: { box: "rgba(255, 255, 255, 0.96)", text: "#000000", outlinePx: 0, shadow: true },
  yellow: { box: "rgba(255, 214, 0, 0.96)", text: "#000000", outlinePx: 0, shadow: true },
  red: { box: "rgba(220, 38, 38, 0.96)", text: "#FFFFFF", outlinePx: 0, shadow: true },
  neon: { box: "rgba(10, 25, 47, 0.95)", text: "#00F0FF", outlinePx: 0, shadow: true },
  emerald: { box: "rgba(6, 78, 59, 0.95)", text: "#34D399", outlinePx: 0, shadow: true },
  purple: { box: "rgba(99, 102, 241, 0.95)", text: "#FFFFFF", outlinePx: 0, shadow: true },
  orange: { box: "rgba(234, 88, 12, 0.95)", text: "#FFFFFF", outlinePx: 0, shadow: true },
  pill: { box: "rgba(15, 23, 42, 0.82)", text: "#F1F5F9", outlinePx: 0, shadow: true },
  breaking_news: { box: "rgba(185, 28, 28, 0.98)", text: "#FEF08A", outlinePx: 0, shadow: true },
  outline: { box: null, text: "#FFFFFF", outlinePx: 8, shadow: false },
  outline_yellow: { box: null, text: "#FFD600", outlinePx: 8, shadow: false },
};

export const HookOverlay: React.FC<HookOverlayProps> = ({ config }) => {
  const { fps, durationInFrames: totalVideoFrames } = useVideoConfig();
  const isForever = Boolean(
    config.displayForever !== false ||
    !config.displayDurationSec ||
    config.displayDurationSec <= 0 ||
    (config.displayDurationSec * fps >= totalVideoFrames)
  );
  const displayFrames = isForever
    ? totalVideoFrames
    : Math.min(Math.round((config.displayDurationSec || 5) * fps), totalVideoFrames);

  return (
    <AbsoluteFill>
      <style>{notoSerifFontFace}</style>
      <Sequence from={0} durationInFrames={displayFrames} layout="none">
        <HookBox config={config} displayFrames={displayFrames} isForever={isForever} />
      </Sequence>
    </AbsoluteFill>
  );
};

interface HookBoxProps {
  config: HookConfig;
  displayFrames: number;
  isForever: boolean;
}

const HookBox: React.FC<HookBoxProps> = ({ config, displayFrames, isForever }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const scale = SIZE_SCALE[config.size] ?? 1.0;

  // Entrance animation
  let animOpacity = 1;
  let animScale = 1;
  let animTranslateY = 0;

  switch (config.entranceAnimation) {
    case "spring": {
      const prog = spring({
        frame,
        fps,
        config: { mass: 0.8, stiffness: 200, damping: 15 },
        durationInFrames: 20,
      });
      animScale = interpolate(prog, [0, 1], [0.7, 1]);
      animOpacity = interpolate(prog, [0, 1], [0, 1]);
      break;
    }
    case "fade": {
      animOpacity = interpolate(frame, [0, 15], [0, 1], {
        extrapolateRight: "clamp",
      });
      break;
    }
    case "slide-up": {
      const prog = spring({
        frame,
        fps,
        config: { mass: 1, stiffness: 150, damping: 18 },
        durationInFrames: 20,
      });
      animTranslateY = interpolate(prog, [0, 1], [60, 0]);
      animOpacity = interpolate(prog, [0, 1], [0, 1]);
      break;
    }
    default:
      break;
  }

  // Exit fade (last 15 frames) only when disappearing before video ends
  if (!isForever) {
    const fadeOutStart = displayFrames - 15;
    if (frame > fadeOutStart) {
      animOpacity *= interpolate(frame, [fadeOutStart, displayFrames], [1, 0], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
    }
  }

  const positionStyle = POSITION_STYLE[config.position] ?? POSITION_STYLE.top;
  const look = HOOK_LOOKS[config.style ?? "classic"] ?? HOOK_LOOKS.classic;

  // Base font size: 5% of 1080 width (matches hooks.py logic)
  const baseFontSize = 1080 * 0.05;
  const fontSize = Math.round(baseFontSize * scale);
  const outlinePx = Math.round(look.outlinePx * scale);

  const customFont = config.fontName || `'${NOTO_SERIF_FONT_FAMILY}', 'Noto Serif', Georgia, serif`;
  const customTextColor = config.fontColor || look.text;
  const customBgBox = config.bgColor !== undefined && config.bgColor !== '' ? config.bgColor : (look.box ?? "transparent");

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        display: "flex",
        justifyContent: "center",
        ...positionStyle,
      }}
    >
      <div
        style={{
          opacity: animOpacity,
          transform: `scale(${animScale}) translateY(${animTranslateY}px)`,
          maxWidth: "90%",
          backgroundColor: customBgBox,
          borderRadius: 20,
          padding: customBgBox !== "transparent" ? `${25 * scale}px ${30 * scale}px` : 0,
          boxShadow: look.shadow && customBgBox !== "transparent" ? "5px 5px 15px rgba(0, 0, 0, 0.25)" : "none",
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontFamily: customFont,
            fontSize,
            fontWeight: 700,
            color: customTextColor,
            lineHeight: 1.4,
            wordBreak: "break-word",
            textTransform: config.uppercase ? "uppercase" : "none",
            ...(outlinePx > 0
              ? {
                  WebkitTextStroke: `${outlinePx}px #000000`,
                  paintOrder: "stroke fill",
                }
              : {}),
          }}
        >
          {config.text}
        </span>
      </div>
    </div>
  );
};
