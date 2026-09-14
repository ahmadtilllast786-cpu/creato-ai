# Creato incremental implementation plan

The work is intentionally delivered in small, verified slices. Each slice is
implemented, tested, committed, and pushed before the next one starts.

## Slice 1 — tracking foundation (this run)

- Match multiple face detections with IoU, x/y continuity, size, and one-to-one
  assignment instead of horizontal distance alone.
- Let motion/context detection scan 3–7 configurable bands while retaining the
  existing left/center/right editorial decisions.
- Expose that band count in the dashboard and forward it per job, without
  changing the three-zone public API.
- Remove isolated crop spikes and cap camera movement per frame, independently
  per scene so hard scene cuts remain intentional.
- Enable the existing frame-based automatic layout/context picker by default;
  missing Gemini or failed analysis still falls back safely.
- Expose deterministic clip-count presets: Auto, 3, 5, 10, and 15.

## Slice 2 — word-accurate clip boundaries

- Add transcript-word search with start-word and end-word selectors.
- Snap boundaries to complete words and sentence/pause boundaries.
- Preserve absolute timestamps through metadata, rerender, recut, and webhook
  payloads.

## Slice 3 — audio and finishing controls

- Add a licensed stock SFX catalog and word-triggered SFX markers.
- Add optional background music before generation, with ducking and loop/fade
  rules.
- Add watermark presets and safe placement that avoids faces, captions, and
  platform UI zones.

## Slice 4 — styles, transitions, and filters

- Wire the existing 36-preset matcher into the API/editor instead of leaving it
  as a library-only feature.
- Add style-aware transitions and FFmpeg filter presets with deterministic seeds,
  so repeat renders are varied but debuggable.
- Add an explicit “new variation” action; do not silently mutate an existing
  exported video.

## Slice 5 — model evaluation and release hardening

- Benchmark MediaPipe/YOLO against open trackers (for example ByteTrack or
  BoT-SORT) only after checking their licenses and measuring CPU/GPU cost on a
  representative Creato corpus.
- Add golden-video tests for face retention, text/context retention, split/merge
  decisions, frame geometry, and camera shake metrics.
- Pin all JavaScript workspaces and make CI run lint, renderer tests, and builds.
- Finish the Creato/OpenShorts identity and licensing audit before public SaaS
  deployment.

## Guardrails

- No third-party code is copied into production without a compatible license,
  attribution, and a benchmark showing it improves the current pipeline.
- Every slice keeps the existing fallback path and must leave a readable error
  when a model, codec, or API is unavailable.
- “AI-generated” variation is bounded by reproducible presets and seeds; it is
  not an uncontrolled self-modifying training loop.
