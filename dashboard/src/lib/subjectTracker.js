/**
 * Multi-Person Subject-Detection & Smart Camera Auto-Reframe Engine
 *
 * Implements:
 * 1. Widened Center Scanning & Focus Band (X: 20%–80%) with overextended coverage.
 * 2. Extended ±15% horizontal dead-zone hysteresis for zero-pan stability.
 * 3. Dynamic 9:16 target viewport transformation with 20%–30% vertical headroom.
 * 4. Framing modes: Full-Screen, Left Half, Right Half, and Auto-Split.
 * 5. Velocity clamping & Exponential Moving Average (EMA) motion damping.
 * 6. Visual 3-layer diagnostic rendering for preview canvas overlays.
 */

export const ASPECT_RATIOS = {
  '9:16': 9 / 16,
  '1:1': 1.0,
  '4:5': 4 / 5,
  '16:9': 16 / 9,
};

export const FRAMING_MODES = {
  FULL_SCREEN: 'full_screen',
  SPLIT_LEFT: 'split_left',
  SPLIT_RIGHT: 'split_right',
  AUTO_SPLIT: 'auto_split',
};

// Widened central scanning & focus envelope (X: 20% to 80%)
export const SCANNING_ZONE = {
  minX: 0.20,
  maxX: 0.80,
  centerX: 0.50,
  width: 0.60,
};

/**
 * Calculates a unified bounding hull encompassing all detected participants
 * with headroom and shoulder expansion padding.
 *
 * @param {Array<{id: number, x: number, y: number, width: number, height: number, confidence: number}>} persons
 * @param {{x: number, y: number}} [padding]
 * @returns {Object|null}
 */
export function computeGroupHull(
  persons,
  padding = { x: 0.05, y: 0.08 }
) {
  if (!persons || persons.length === 0) return null;

  let minX = 1.0;
  let minY = 1.0;
  let maxX = 0.0;
  let maxY = 0.0;

  for (const p of persons) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x + p.width);
    maxY = Math.max(maxY, p.y + p.height);
  }

  // Apply headroom and shoulder expansion margin
  const paddedMinX = Math.max(0.0, minX - padding.x);
  const paddedMinY = Math.max(0.0, minY - padding.y);
  const paddedMaxX = Math.min(1.0, maxX + padding.x);
  const paddedMaxY = Math.min(1.0, maxY + padding.y);

  const width = Math.max(0.01, paddedMaxX - paddedMinX);
  const height = Math.max(0.01, paddedMaxY - paddedMinY);

  return {
    minX: paddedMinX,
    minY: paddedMinY,
    maxX: paddedMaxX,
    maxY: paddedMaxY,
    width,
    height,
    centerX: paddedMinX + width / 2,
    centerY: paddedMinY + height / 2,
  };
}

/**
 * Smart Cameraman Tracker with:
 * - Widened Central Focus Envelope (X: 20% to 80%)
 * - Dynamic 9:16 Aspect Ratio Transformation with 20%–30% Headroom
 * - Extended ±15% Horizontal Dead-Zone Hysteresis (Zero Pan inside deadzone)
 * - Maximum Velocity Clamping (Prevents camera whipping)
 * - Exponential Moving Average (EMA) coordinate smoothing
 */
export class SmartCameramanTracker {
  constructor(options = {}) {
    this.aspectRatioKey = options.aspectRatio || '9:16';
    this.aspectRatio = ASPECT_RATIOS[this.aspectRatioKey] || 9 / 16;
    this.framingMode = options.framingMode || FRAMING_MODES.FULL_SCREEN;

    this.smoothingFactor = options.smoothingFactor !== undefined ? options.smoothingFactor : 0.10;
    // Extended horizontal dead-zone: ±15% of screen width (Zero pan)
    this.deadzoneRatio = options.deadzoneRatio !== undefined ? options.deadzoneRatio : 0.15;
    this.maxVelocity = options.maxVelocity !== undefined ? options.maxVelocity : 0.018;

    this.scanZone = options.scanZone || SCANNING_ZONE;

    this.currentCenter = { x: 0.5, y: 0.5 };
    this.targetCenter = { x: 0.5, y: 0.5 };
    this.anchorCenter = { x: 0.5, y: 0.5 };

    this.currentScale = 1.0;
    this.targetScale = 1.0;
  }

  setAspectRatio(ratioKey) {
    if (ASPECT_RATIOS[ratioKey]) {
      this.aspectRatioKey = ratioKey;
      this.aspectRatio = ASPECT_RATIOS[ratioKey];
    }
  }

  setFramingMode(mode) {
    this.framingMode = mode;
  }

  reset(centerX = 0.5, centerY = 0.5) {
    this.currentCenter = { x: centerX, y: centerY };
    this.targetCenter = { x: centerX, y: centerY };
    this.anchorCenter = { x: centerX, y: centerY };
    this.currentScale = 1.0;
    this.targetScale = 1.0;
  }

  /**
   * Evaluates detected persons and updates the camera viewport.
   * Runs:
   * 1. Widened Scanning Zone Tracking (20% - 80%)
   * 2. Vertical Headroom & Shoulder Margin Alignment (20% - 30%)
   * 3. Extended ±15% Dead-Zone Hysteresis Filter (Zero Pan)
   * 4. Velocity Clamping & EMA Smoothing
   * 5. Final 9:16 Canvas Crop Bounds Output
   */
  update(
    persons,
    videoWidth = 1920,
    videoHeight = 1080,
    forceSnap = false
  ) {
    const sourceAspect = (videoWidth || 1920) / (videoHeight || 1080);
    const hull = computeGroupHull(persons);

    let rawTargetX = 0.5;
    let rawTargetY = 0.5;
    let rawTargetScale = 1.0;

    // Standard 9:16 crop dimensions in normalized coordinates
    let cropWidthFrac = Math.min(1.0, this.aspectRatio / sourceAspect);
    let cropHeightFrac = Math.min(1.0, cropWidthFrac / this.aspectRatio * sourceAspect);

    if (cropHeightFrac > 1.0) {
      cropHeightFrac = 1.0;
      cropWidthFrac = Math.min(1.0, (cropHeightFrac * this.aspectRatio) / sourceAspect);
    }

    if (hull) {
      if (this.framingMode === FRAMING_MODES.SPLIT_LEFT) {
        // Offset framing anchored to left half
        rawTargetX = Math.min(hull.centerX, 0.35);
        rawTargetY = hull.centerY;
      } else if (this.framingMode === FRAMING_MODES.SPLIT_RIGHT) {
        // Offset framing anchored to right half
        rawTargetX = Math.max(hull.centerX, 0.65);
        rawTargetY = hull.centerY;
      } else if (this.framingMode === FRAMING_MODES.AUTO_SPLIT && persons && persons.length >= 2) {
        // Auto-split: center of participants
        rawTargetX = hull.centerX;
        rawTargetY = hull.centerY;
      } else {
        // Full-screen focus:
        // Widened horizontal tracking allows subjects to move across X: 20% to 80%
        rawTargetX = hull.centerX;

        // Dynamic Multi-Person Expansion:
        if (hull.width > cropWidthFrac * 0.90) {
          rawTargetScale = Math.min(1.6, hull.width / (cropWidthFrac * 0.85));
        }

        // Vertical Headroom & Shoulder Margin Alignment (Y: 20%–30% from top):
        // Position detected person's head at ~25% from top of crop box
        const topHeadY = Math.min(...persons.map((p) => p.y));
        const targetHeadroomRatio = 0.25;
        // Ideal viewport top = topHeadY - targetHeadroomRatio * cropHeightFrac
        // Center = top + cropHeightFrac / 2
        rawTargetY = topHeadY - targetHeadroomRatio * cropHeightFrac + cropHeightFrac / 2;
      }
    }

    // Extended Dead-Zone (Hysteresis) Filter:
    // If movement from anchorCenter is within ±15% of screen width, keep camera 100% stationary
    const diffXFromAnchor = Math.abs(rawTargetX - this.anchorCenter.x);
    const diffYFromAnchor = Math.abs(rawTargetY - this.anchorCenter.y);

    if (diffXFromAnchor <= this.deadzoneRatio && diffYFromAnchor <= 0.08) {
      // Keep target strictly at anchor center (Stationary tripod camera)
      rawTargetX = this.anchorCenter.x;
      rawTargetY = this.anchorCenter.y;
    } else {
      // Subject crossed outside widened dead-zone: smoothly advance anchor
      this.anchorCenter.x = rawTargetX;
      this.anchorCenter.y = rawTargetY;
    }

    this.targetCenter.x = rawTargetX;
    this.targetCenter.y = rawTargetY;
    this.targetScale = rawTargetScale;

    if (forceSnap) {
      this.currentCenter.x = this.targetCenter.x;
      this.currentCenter.y = this.targetCenter.y;
      this.currentScale = this.targetScale;
    } else {
      // 1. Velocity Clamping on Raw Deltas (Prevents whipping)
      let deltaX = (this.targetCenter.x - this.currentCenter.x) * this.smoothingFactor;
      let deltaY = (this.targetCenter.y - this.currentCenter.y) * this.smoothingFactor;

      const velocity = Math.hypot(deltaX, deltaY);
      if (velocity > this.maxVelocity && velocity > 0) {
        const factor = this.maxVelocity / velocity;
        deltaX *= factor;
        deltaY *= factor;
      }

      // 2. Exponential Moving Average (EMA) Step
      this.currentCenter.x += deltaX;
      this.currentCenter.y += deltaY;

      // Smooth scale expansion
      this.currentScale += (this.targetScale - this.currentScale) * this.smoothingFactor;
    }

    // Recalculate crop bounds with smoothed scale
    cropWidthFrac = Math.min(1.0, (this.aspectRatio / sourceAspect) * this.currentScale);
    cropHeightFrac = Math.min(1.0, (cropWidthFrac / this.aspectRatio) * sourceAspect);

    if (cropHeightFrac > 1.0) {
      cropHeightFrac = 1.0;
      cropWidthFrac = Math.min(1.0, (cropHeightFrac * this.aspectRatio) / sourceAspect);
    }

    // Clamp coordinates so crop window never leaves frame bounds
    const halfW = cropWidthFrac / 2;
    const halfH = cropHeightFrac / 2;

    const clampedCenterX = Math.max(halfW, Math.min(1.0 - halfW, this.currentCenter.x));
    const clampedCenterY = Math.max(halfH, Math.min(1.0 - halfH, this.currentCenter.y));

    return {
      x: clampedCenterX - halfW,
      y: clampedCenterY - halfH,
      width: cropWidthFrac,
      height: cropHeightFrac,
      centerX: clampedCenterX,
      centerY: clampedCenterY,
    };
  }
}

/**
 * 3 Visual UI Diagnostic / Editing Layers (Rendered on Preview Canvas)
 *
 * Layer 0: Widened Center Scanning & Focus Band (X: 20%–80%) & Dynamic Vertical Framing Guides.
 * Layer 1: Person Bounding Lines (Corner Ticks & Subject IDs).
 * Layer 2: Group Hull / Expansion Boundary (Dashed Border & Translucent Fill).
 * Layer 3: Active 9:16 Camera Viewport Frame (Headroom 20%–30%, Rule of Thirds, ±15% Deadzone).
 */
export function renderTrackingLayers(
  ctx,
  width,
  height,
  state = {},
  options = {}
) {
  const {
    showScanningZone = true,
    showPersons = true,
    showGroupHull = true,
    showViewport = true,
    showRuleOfThirds = true,
    showHeadroom = true,
  } = options;

  ctx.save();

  // -------------------------------------------------------------
  // LAYER 0: Widened Center Scanning & Focus Band (X: 20% - 80%)
  // -------------------------------------------------------------
  if (showScanningZone) {
    const scanMinX = (state.scanZone?.minX ?? 0.20) * width;
    const scanMaxX = (state.scanZone?.maxX ?? 0.80) * width;
    const scanW = scanMaxX - scanMinX;

    ctx.save();
    // Translucent soft blue coverage envelope
    ctx.fillStyle = 'rgba(59, 130, 246, 0.04)';
    ctx.fillRect(scanMinX, 0, scanW, height);

    // Left scanning boundary line (X: 20%)
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.65)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(scanMinX, 0);
    ctx.lineTo(scanMinX, height);
    // Right scanning boundary line (X: 80%)
    ctx.moveTo(scanMaxX, 0);
    ctx.lineTo(scanMaxX, height);
    ctx.stroke();

    // Center focus axis (X: 50%)
    const midX = width * 0.50;
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.35)';
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(midX, 0);
    ctx.lineTo(midX, height);
    ctx.stroke();
    ctx.setLineDash([]);

    // Scanning zone top pill
    const scanTag = 'CENTER FOCUS BAND (20% - 80%)';
    ctx.font = 'bold 9px monospace';
    const tagW = ctx.measureText(scanTag).width + 12;
    ctx.fillStyle = 'rgba(59, 130, 246, 0.85)';
    ctx.fillRect(scanMinX, 4, tagW, 16);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(scanTag, scanMinX + 6, 15);

    ctx.restore();
  }

  // -------------------------------------------------------------
  // LAYER 2: Group Hull / Expansion Boundary (Rendered underneath)
  // -------------------------------------------------------------
  if (showGroupHull && state.groupHull && state.persons && state.persons.length > 1) {
    const gh = state.groupHull;
    const gx = Math.round(gh.minX * width);
    const gy = Math.round(gh.minY * height);
    const gw = Math.round(gh.width * width);
    const gh_h = Math.round(gh.height * height);

    ctx.save();
    // Translucent expansion boundary tint
    ctx.fillStyle = 'rgba(168, 85, 247, 0.08)';
    ctx.fillRect(gx, gy, gw, gh_h);

    // Dashed purple boundary line
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(gx, gy, gw, gh_h);
    ctx.setLineDash([]);

    // Group Hull badge tag
    ctx.fillStyle = 'rgba(168, 85, 247, 0.95)';
    const tagText = `GROUP HULL (${state.persons.length} SUBJECTS)`;
    ctx.font = 'bold 10px sans-serif';
    const tagW = ctx.measureText(tagText).width + 12;
    ctx.fillRect(gx, Math.max(0, gy - 18), tagW, 18);

    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(tagText, gx + 6, Math.max(12, gy - 5));
    ctx.restore();
  }

  // -------------------------------------------------------------
  // LAYER 1: Person Bounding Lines (Corner Ticks & Subject IDs)
  // -------------------------------------------------------------
  if (showPersons && state.persons && state.persons.length > 0) {
    ctx.save();

    state.persons.forEach((person, idx) => {
      const px = Math.round(person.x * width);
      const py = Math.round(person.y * height);
      const pw = Math.round(person.width * width);
      const ph = Math.round(person.height * height);

      const color = idx === 0 ? '#00E5FF' : '#FFE500'; // Cyan for primary, Yellow for secondary
      const tickLen = Math.min(16, Math.max(6, Math.round(Math.min(pw, ph) * 0.25)));

      // Subtle translucent bounding box outline
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px, py, pw, ph);

      // Distinct high-visibility corner ticks
      ctx.lineWidth = 3;
      ctx.beginPath();
      // Top-Left corner
      ctx.moveTo(px, py + tickLen);
      ctx.lineTo(px, py);
      ctx.lineTo(px + tickLen, py);
      // Top-Right corner
      ctx.moveTo(px + pw - tickLen, py);
      ctx.lineTo(px + pw, py);
      ctx.lineTo(px + pw, py + tickLen);
      // Bottom-Left corner
      ctx.moveTo(px, py + ph - tickLen);
      ctx.lineTo(px, py + ph);
      ctx.lineTo(px + tickLen, py + ph);
      // Bottom-Right corner
      ctx.moveTo(px + pw - tickLen, py + ph);
      ctx.lineTo(px + pw, py + ph);
      ctx.lineTo(px + pw, py + ph - tickLen);
      ctx.stroke();

      // Subject ID Badge Pill
      const confPct = Math.round((person.confidence || 0.9) * 100);
      const badgeText = `Subject #${person.id || idx + 1} (${confPct}%)`;
      ctx.font = 'bold 11px sans-serif';
      const badgeW = ctx.measureText(badgeText).width + 12;
      const badgeH = 18;

      ctx.fillStyle = color;
      if (typeof ctx.roundRect === 'function') {
        ctx.beginPath();
        ctx.roundRect(px, Math.max(0, py - badgeH - 2), badgeW, badgeH, 4);
        ctx.fill();
      } else {
        ctx.fillRect(px, Math.max(0, py - badgeH - 2), badgeW, badgeH);
      }

      ctx.fillStyle = '#000000';
      ctx.fillText(badgeText, px + 6, Math.max(badgeH - 4, py - 6));
    });

    ctx.restore();
  }

  // -------------------------------------------------------------
  // LAYER 3: Active 9:16 Camera Viewport Frame
  // -------------------------------------------------------------
  if (showViewport && state.cameraViewport) {
    const vp = state.cameraViewport;
    const vx = Math.round(vp.x * width);
    const vy = Math.round(vp.y * height);
    const vw = Math.round(vp.width * width);
    const vh = Math.round(vp.height * height);

    ctx.save();

    // Darken areas outside the active camera viewport (cinematic mask)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    // Top
    ctx.fillRect(0, 0, width, vy);
    // Bottom
    ctx.fillRect(0, vy + vh, width, height - (vy + vh));
    // Left
    ctx.fillRect(0, vy, vx, vh);
    // Right
    ctx.fillRect(vx + vw, vy, width - (vx + vw), vh);

    // Active Viewport Framing Border (Gold / Brass Studio Border)
    ctx.strokeStyle = '#FACC15';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(vx, vy, vw, vh);

    // Rule of Thirds Grid Lines inside Viewport
    if (showRuleOfThirds) {
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.25)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);

      // Vertical 1/3 and 2/3 lines
      ctx.beginPath();
      ctx.moveTo(vx + vw * (1 / 3), vy);
      ctx.lineTo(vx + vw * (1 / 3), vy + vh);
      ctx.moveTo(vx + vw * (2 / 3), vy);
      ctx.lineTo(vx + vw * (2 / 3), vy + vh);

      // Horizontal 1/3 and 2/3 lines
      ctx.moveTo(vx, vy + vh * (1 / 3));
      ctx.lineTo(vx + vw, vy + vh * (1 / 3));
      ctx.moveTo(vx, vy + vh * (2 / 3));
      ctx.lineTo(vx + vw, vy + vh * (2 / 3));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Headroom Safe Zone Guide Lines (20% & 30% from top of viewport)
    if (showHeadroom) {
      const hrTop = vy + vh * 0.20;
      const hrBottom = vy + vh * 0.30;
      ctx.strokeStyle = 'rgba(236, 72, 153, 0.65)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(vx, hrTop);
      ctx.lineTo(vx + vw, hrTop);
      ctx.moveTo(vx, hrBottom);
      ctx.lineTo(vx + vw, hrBottom);
      ctx.stroke();
      ctx.setLineDash([]);

      const hrTag = 'HEADROOM (20%-30%)';
      ctx.font = 'bold 8px monospace';
      ctx.fillStyle = 'rgba(236, 72, 153, 0.90)';
      ctx.fillRect(vx + vw - 116, hrTop - 13, 114, 13);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(hrTag, vx + vw - 112, hrTop - 3);
    }

    // Extended Horizontal Deadzone Indicator (±15% of screen width)
    const cx = Math.round(vp.centerX * width);
    const cy = Math.round(vp.centerY * height);
    const deadzoneRatio = state.deadzoneRadius !== undefined ? state.deadzoneRadius : 0.15;
    const deadzoneW = Math.round(width * deadzoneRatio);

    ctx.strokeStyle = 'rgba(34, 197, 94, 0.45)'; // Subtle green deadzone box
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 2]);
    ctx.strokeRect(cx - deadzoneW / 2, vy, deadzoneW, vh);
    ctx.setLineDash([]);

    // Deadzone label
    const dzTag = '±15% DEADZONE (ZERO PAN)';
    ctx.font = 'bold 8px monospace';
    ctx.fillStyle = 'rgba(34, 197, 94, 0.85)';
    ctx.fillRect(cx - deadzoneW / 2, vy + vh - 15, 136, 14);
    ctx.fillStyle = '#000000';
    ctx.fillText(dzTag, cx - deadzoneW / 2 + 4, vy + vh - 4);

    // Center crosshair
    ctx.strokeStyle = '#FACC15';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy);
    ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8);
    ctx.lineTo(cx, cy + 8);
    ctx.stroke();

    // Viewport Aspect Ratio Label Badge
    const aspectTag = `VIEWPORT ${state.aspectRatioLabel || '9:16'} (1080x1920)`;
    ctx.font = 'bold 10px monospace';
    const tagW = ctx.measureText(aspectTag).width + 12;

    ctx.fillStyle = '#FACC15';
    ctx.fillRect(vx, vy, tagW, 18);
    ctx.fillStyle = '#000000';
    ctx.fillText(aspectTag, vx + 6, vy + 13);

    ctx.restore();
  }

  ctx.restore();
}
