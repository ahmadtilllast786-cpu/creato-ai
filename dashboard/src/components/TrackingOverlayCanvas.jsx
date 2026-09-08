import React, { useRef, useEffect } from 'react';
import { renderTrackingLayers, computeGroupHull, SCANNING_ZONE } from '../lib/subjectTracker';

/**
 * High-performance HTML5 Canvas tracking overlay for Multi-Person Detection & Smart Reframe.
 * Renders:
 * - Layer 0: Widened Center Scanning & Focus Band (X: 20%–80%) & Dynamic Vertical Framing Guides
 * - Layer 1: Corner-ticked bounding boxes with subject IDs & detection confidence
 * - Layer 2: Unified group expansion hull with translucent purple tint
 * - Layer 3: Active 9:16 camera viewport with 20%–30% headroom, rule-of-thirds & ±15% deadzone
 */
export default function TrackingOverlayCanvas({
  persons = [],
  groupHull = null,
  cameraViewport = null,
  aspectRatio = '9:16',
  showOverlays = true,
  showScanningZone = true,
  showPersons = true,
  showGroupHull = true,
  showViewport = true,
  showRuleOfThirds = true,
  showHeadroom = true,
  deadzoneRadius = 0.15,
  scanZone = SCANNING_ZONE,
  className = '',
}) {
  const canvasRef = useRef(null);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high-DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const displayWidth = Math.round(rect.width);
    const displayHeight = Math.round(rect.height);

    if (displayWidth === 0 || displayHeight === 0) return;

    if (canvas.width !== displayWidth * dpr || canvas.height !== displayHeight * dpr) {
      canvas.width = displayWidth * dpr;
      canvas.height = displayHeight * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    if (!showOverlays) {
      ctx.restore();
      return;
    }

    // Fallback: compute group hull if not provided
    const effectiveHull = groupHull || (persons && persons.length > 0 ? computeGroupHull(persons) : null);

    renderTrackingLayers(
      ctx,
      displayWidth,
      displayHeight,
      {
        persons: persons || [],
        groupHull: effectiveHull,
        cameraViewport: cameraViewport || null,
        deadzoneRadius,
        aspectRatioLabel: aspectRatio,
        scanZone: scanZone || SCANNING_ZONE,
      },
      {
        showScanningZone,
        showPersons,
        showGroupHull,
        showViewport,
        showRuleOfThirds,
        showHeadroom,
      }
    );

    ctx.restore();
  };

  useEffect(() => {
    draw();
  }, [
    persons,
    groupHull,
    cameraViewport,
    aspectRatio,
    showOverlays,
    showScanningZone,
    showPersons,
    showGroupHull,
    showViewport,
    showRuleOfThirds,
    showHeadroom,
    deadzoneRadius,
    scanZone,
  ]);

  // Handle dynamic resize of parent container
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !window.ResizeObserver) return;

    const observer = new ResizeObserver(() => {
      draw();
    });

    observer.observe(canvas);
    return () => observer.disconnect();
  }, [
    persons,
    groupHull,
    cameraViewport,
    aspectRatio,
    showOverlays,
    showScanningZone,
    showPersons,
    showGroupHull,
    showViewport,
    showRuleOfThirds,
    showHeadroom,
    deadzoneRadius,
    scanZone,
  ]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 w-full h-full pointer-events-none z-10 ${className}`}
      style={{ display: showOverlays ? 'block' : 'none' }}
    />
  );
}
