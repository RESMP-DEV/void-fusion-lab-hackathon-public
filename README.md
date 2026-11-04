# Void Fusion: visible tire replacement for finished advertising footage

Void Fusion lets an advertising team replace one tire in already-shot footage:
Meta SAM 3.1 tracks the existing wheel frame by frame, VOID Pass 1 removes it and
rebuilds the wheel well, a visible Michelin-style replacement follows the same
motion, and insertion-aware VOID Pass 2 refines only the cleanup seam.

## Watch the v2 demo

- [107.21-second long-form demo](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v2.0.0/void-fusion-advertising-2026-11-03.mp4)
- [Five-stage comparison master](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v2.0.0/void-fusion-five-stage-comparison-2026-11-03.mp4)
- [Four-stage campaign contact sheet](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v2.0.0/void-fusion-campaign-four-stage-proof-2026-11-03.jpg)

The long-form demo deliberately shows five uninterrupted blocks: original
commercial footage, explicit red tire selection, clean plate, visible branded
replacement, and refined result. It is not a subtle invisible edit.

## Result

| Measure | Result |
| --- | ---: |
| Continuous tire masks recovered | 17 runs / 465 frames |
| VOID Pass 1 successes | 17 / 17 |
| VOID Pass 2 successes | 17 / 17 |
| Retained after final visibility gate | 16 shots / 457 frames |
| Underlying selected action | 19.042 seconds at 24 fps |
| Long-form presentation | 2,573 frames / 107.208 seconds / 1920x1080 |

One eight-frame Pass 2 run was excluded because one frame had zero detectable
brand pixels. Every frame in the retained campaign passed the brand-visibility
check. The minimum retained brand-pixel count is 466; the mean is approximately
9,378 pixels per frame.

The retained Pass 2 runs have:

- object-preservation changed ratio: minimum 5.61%, mean 16.25%, maximum 44.21%;
- cleanup-ring changed ratio: minimum 9.51%, mean 23.46%, maximum 40.19%;
- background changed ratio: minimum 0.80%, mean 3.81%, maximum 14.45%.

These are change measurements, not claims of photorealism or artistic quality.

## Why this is not one whole-scene video generator

The real advertising footage remains the baseline. SAM 3.1 identifies where the
selected tire appears in every retained frame. VOID Pass 1 changes the selected
tire region into a clean plate. The replacement renderer fits the tire trajectory
and remains visible. The insertion-aware quadmask marks that replacement as keep
and exposes only a bounded cleanup ring to VOID Pass 2. Comparator reports then
separate object preservation, cleanup-ring change, background change, and
temporal warnings.

This keeps the useful parts of the original shoot: camera motion, road, vehicle,
performance, lighting, and editing. It changes the advertising object rather than
asking a generator to reinvent the entire commercial.

## Model roles

- **Meta SAM 3.1** provides dense frame-aligned tire masks and the visible red
  selection overlay.
- **Netflix VOID Pass 1** removes the tracked tire and reconstructs the road and
  wheel well.
- **The deterministic replacement lane** fits a visible Michelin-style tire to
  the tracked ellipse trajectory. It is not claimed to be learned image-to-3D
  reconstruction or physics simulation.
- **Netflix VOID Pass 2** sees an insertion-aware quadmask: replacement keep,
  untouched background keep, and cleanup ring affected.
- **Comparator and assembly gates** reject missing stages, cadence mismatches,
  near-invisible insertions, and refined frames without visible brand pixels.

## Reproduce the software gates

```bash
uv sync
just check
just quality
uv run pytest scripts/tests/test_advertising_tire_asset.py \
  scripts/tests/test_assemble_advertising_reel.py -q
```

The heavy SAM 3.1 and VOID lanes require their documented CUDA environments and
checkpoints; the repository tests intentionally do not download checkpoints or
call external APIs.

## Evidence and limitations

- Sanitized campaign evidence: [`evidence/advertising-campaign-evidence.json`](evidence/advertising-campaign-evidence.json)
- Judge-facing presentation guide: [`docs/presentation.md`](docs/presentation.md)

Known limitations are explicit: the replacement renderer is deterministic and
2.5D-looking rather than a learned 3D reconstruction; Pass 2 changes the
replacement more than a production compositor would want in some shots; temporal
pop warnings remain; and the source is transformed fair-use advertising footage
for this research demonstration. No source video or model checkpoint is committed
to the repository.
