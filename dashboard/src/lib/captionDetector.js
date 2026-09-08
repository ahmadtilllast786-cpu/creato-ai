/**
 * Lightweight Client-Side Caption & Hardcoded Subtitle Detector
 *
 * Samples keyframes across the lower third of a video using HTML5 Canvas,
 * analyzing Sobel edge gradients, contrast transitions, and horizontal text-band
 * clustering to determine if burned-in subtitles already exist.
 */

export const DEFAULT_SCAN_ZONE = {
  top: 0.65,    // Y: 65% (start of lower third)
  bottom: 0.95, // Y: 95% (end of lower third before extreme edge)
  left: 0.08,   // X: 8% safe margin
  right: 0.92,  // X: 92% safe margin
};

/**
 * Analyzes an ImageData buffer within the specified zone for text-like features.
 * Text in subtitles is characterized by:
 * 1. High-frequency bidirectional edge transitions (Sobel |Gx| + |Gy| > threshold).
 * 2. High local contrast (bright stroke next to dark background/outline).
 * 3. Horizontal line clustering (concentrated text lines 15-45px tall).
 */
export function analyzeFrameForBurnedInText(
  ctx,
  canvasWidth,
  canvasHeight,
  zone = DEFAULT_SCAN_ZONE
) {
  const startX = Math.round(canvasWidth * zone.left);
  const endX = Math.round(canvasWidth * zone.right);
  const startY = Math.round(canvasHeight * zone.top);
  const endY = Math.round(canvasHeight * zone.bottom);

  const scanWidth = endX - startX;
  const scanHeight = endY - startY;

  if (scanWidth <= 0 || scanHeight <= 0) {
    return { hasText: false, score: 0, peakRowPercent: zone.bottom };
  }

  const imgData = ctx.getImageData(startX, startY, scanWidth, scanHeight);
  const data = imgData.data;

  // Convert to grayscale luminance
  const lum = new Uint8Array(scanWidth * scanHeight);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    // Standard Rec. 601 luminance
    lum[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  }

  // Row edge accumulator for horizontal text-band clustering
  const rowEdgeCounts = new Uint32Array(scanHeight);
  let totalHighEdgePixels = 0;
  const EDGE_THRESHOLD = 70; // Sharp letter stroke boundary

  for (let y = 1; y < scanHeight - 1; y++) {
    const rowOffset = y * scanWidth;
    let rowEdges = 0;

    for (let x = 1; x < scanWidth - 1; x++) {
      const idx = rowOffset + x;

      // Fast Sobel approximation
      const gx = Math.abs(lum[idx + 1] - lum[idx - 1]);
      const gy = Math.abs(lum[idx + scanWidth] - lum[idx - scanWidth]);
      const mag = gx + gy;

      if (mag > EDGE_THRESHOLD) {
        // Confirm contrast: check if neighboring pixels have high dynamic range
        const maxLocal = Math.max(
          lum[idx], lum[idx + 1], lum[idx - 1],
          lum[idx + scanWidth], lum[idx - scanWidth]
        );
        const minLocal = Math.min(
          lum[idx], lum[idx + 1], lum[idx - 1],
          lum[idx + scanWidth], lum[idx - scanWidth]
        );

        if (maxLocal - minLocal > 60) {
          rowEdges++;
          totalHighEdgePixels++;
        }
      }
    }

    rowEdgeCounts[y] = rowEdges;
  }

  // Text line identification: check for consecutive rows with concentrated edge density
  const minRowEdgeThreshold = Math.max(12, Math.round(scanWidth * 0.05));
  let peakRowIdx = 0;
  let maxEdgesInRow = 0;
  let textBandHeight = 0;
  let inBand = false;
  let bandCount = 0;

  for (let y = 0; y < scanHeight; y++) {
    const count = rowEdgeCounts[y];
    if (count > maxEdgesInRow) {
      maxEdgesInRow = count;
      peakRowIdx = y;
    }

    if (count >= minRowEdgeThreshold) {
      if (!inBand) {
        inBand = true;
        bandCount++;
      }
      textBandHeight++;
    } else {
      inBand = false;
    }
  }

  const totalPixels = scanWidth * scanHeight;
  const edgeDensity = totalHighEdgePixels / (totalPixels || 1);

  // Subtitles usually form 1 or 2 distinct horizontal lines with band height between 8 and 80px
  const hasTextPattern =
    (bandCount >= 1 && bandCount <= 4) &&
    textBandHeight >= 10 &&
    edgeDensity > 0.02 &&
    maxEdgesInRow >= minRowEdgeThreshold;

  const score = Math.min(1, edgeDensity * 18 + (hasTextPattern ? 0.4 : 0));
  const peakRowPercent = zone.top + (peakRowIdx / scanHeight) * (zone.bottom - zone.top);

  return {
    hasText: hasTextPattern && score > 0.35,
    score: Math.round(score * 100) / 100,
    peakRowPercent: Math.round(peakRowPercent * 100) / 100,
  };
}

/**
 * Scans a video URL or HTMLVideoElement across keyframes to detect pre-existing subtitles.
 */
export async function detectBurnedInCaptions(
  videoSource,
  options = {}
) {
  const {
    sampleCount = 5,
    zone = DEFAULT_SCAN_ZONE,
    signal,
  } = options;

  return new Promise((resolve) => {
    let videoEl;
    let cleanup = () => {};

    if (typeof videoSource === "string") {
      videoEl = document.createElement("video");
      videoEl.crossOrigin = "anonymous";
      videoEl.muted = true;
      videoEl.playsInline = true;
      videoEl.preload = "auto";
      videoEl.src = videoSource;

      cleanup = () => {
        try {
          videoEl.pause();
          videoEl.removeAttribute("src");
          videoEl.load();
        } catch (_) {}
      };
    } else {
      videoEl = videoSource;
    }

    const offscreenCanvas = document.createElement("canvas");
    // Use fixed 360x640 scan resolution for high performance and consistent metrics
    offscreenCanvas.width = 360;
    offscreenCanvas.height = 640;
    const ctx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

    if (!ctx) {
      resolve({
        hasBurnedInCaptions: false,
        confidence: 0,
        detectedZone: { topPercent: zone.top, bottomPercent: zone.bottom },
      });
      return;
    }

    const onMetadataLoaded = async () => {
      if (signal?.aborted) {
        cleanup();
        resolve({
          hasBurnedInCaptions: false,
          confidence: 0,
          detectedZone: { topPercent: zone.top, bottomPercent: zone.bottom },
        });
        return;
      }

      const duration = videoEl.duration || 10;
      // Sample keyframe fractions across the video duration
      const sampleFractions = [0.15, 0.35, 0.55, 0.75, 0.88].slice(0, sampleCount);
      const sampleResults = [];
      let positiveCount = 0;
      let totalConfidence = 0;
      let peakRowAcc = 0;

      for (const fraction of sampleFractions) {
        if (signal?.aborted) break;

        const targetTime = Math.max(0.1, Math.min(duration - 0.2, duration * fraction));

        await new Promise((seekResolve) => {
          let timeoutId;
          const onSeeked = () => {
            clearTimeout(timeoutId);
            videoEl.removeEventListener("seeked", onSeeked);
            seekResolve();
          };
          timeoutId = setTimeout(() => {
            videoEl.removeEventListener("seeked", onSeeked);
            seekResolve();
          }, 600); // 600ms timeout per seek

          videoEl.addEventListener("seeked", onSeeked);
          videoEl.currentTime = targetTime;
        });

        if (signal?.aborted) break;

        // Draw frame to canvas
        try {
          ctx.drawImage(videoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
          const analysis = analyzeFrameForBurnedInText(
            ctx,
            offscreenCanvas.width,
            offscreenCanvas.height,
            zone
          );

          sampleResults.push({
            time: Math.round(targetTime * 10) / 10,
            hasText: analysis.hasText ? 1 : 0,
            score: analysis.score,
          });

          if (analysis.hasText) {
            positiveCount++;
            totalConfidence += analysis.score;
            peakRowAcc += analysis.peakRowPercent;
          }
        } catch (_) {
          // Canvas cross-origin taint or draw error fallback
        }
      }

      cleanup();

      // At least 40% of sampled frames must detect text to flag burned-in captions
      const requiredPositives = Math.max(1, Math.ceil(sampleResults.length * 0.4));
      const hasBurnedInCaptions = positiveCount >= requiredPositives;
      const confidence =
        sampleResults.length > 0
          ? Math.round((totalConfidence / (positiveCount || 1)) * 100) / 100
          : 0;

      const avgPeakRow =
        positiveCount > 0 ? peakRowAcc / positiveCount : (zone.top + zone.bottom) / 2;

      resolve({
        hasBurnedInCaptions,
        confidence,
        detectedZone: {
          topPercent: Math.max(zone.top, Math.round((avgPeakRow - 0.08) * 100) / 100),
          bottomPercent: Math.min(zone.bottom, Math.round((avgPeakRow + 0.08) * 100) / 100),
        },
        sampleResults,
      });
    };

    if (videoEl.readyState >= 1) {
      onMetadataLoaded();
    } else {
      videoEl.addEventListener("loadedmetadata", onMetadataLoaded, { once: true });
      videoEl.addEventListener(
        "error",
        () => {
          cleanup();
          resolve({
            hasBurnedInCaptions: false,
            confidence: 0,
            detectedZone: { topPercent: zone.top, bottomPercent: zone.bottom },
          });
        },
        { once: true }
      );
    }
  });
}
