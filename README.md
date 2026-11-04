# Void Fusion: true 3D tire replacement for finished advertising footage

Void Fusion turns a plain-language advertising request into a measurable video
replacement: Meta SAM 3.1 finds and tracks the real tire, VOID Pass 1 removes it,
a fitted Blender-native 3D wheel is posed into the clean shot, and
insertion-aware VOID Pass 2 refines only the surrounding cleanup ring.

## Watch the v3 demo

- [108.875-second true-3D demo](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v3.0.0/void-fusion-true3d-final.mp4)
- [Five-stage comparison master](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v3.0.0/void-fusion-true3d-five-stage-comparison.mp4)
- [Five-stage proof sheet](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v3.0.0/void-fusion-true3d-five-stage-proof.jpg)

The demo shows five uninterrupted 465-frame blocks: original commercial action,
SAM selection, VOID Pass 1 clean plate, Blender-native 3D insertion, and
preserved VOID Pass 2. Explanatory cards label every stage.

## Result

| Measure | Result |
| --- | ---: |
| Continuous tire runs recovered | 17 runs / 465 frames |
| VOID Pass 1 successes | 17 / 17 |
| Insertion-aware VOID Pass 2 successes | 17 / 17 |
| Preserved frames passing brand gate | 465 / 465 |
| Campaign minimum brand-band pixels | 37 per frame |
| Inserted-object changed ratio after preservation | 0.0 |
| Cleanup-ring changed ratio | mean 27.65% / max 52.48% |
| Final presentation | 2,613 frames / 108.875 seconds / 1920x1080 |

The replacement is a real Blender mesh: 38 objects, 2,700 vertices, seven
materials, with persisted `.blend` and valid `.glb` exports. Every inserted and
cleanup mask decodes as nonempty binary ROI-black `{0,255}` across all 465
frames.

## Why the replacement remains visible

Diffusion does not always obey a keep mask reliably enough for advertising.
Qualification showed seeds and stronger prompts could reduce early brand-band
pixels to almost zero. Void Fusion therefore enforces the quadmask contract
after VOID Pass 2: the exact Blender object is retained inside its KEEP region,
while the model-refined cleanup ring and background remain from Pass 2.

The comparator then verifies the intended separation: object changed ratio is
zero, every cleanup ring changed, and background change is measured separately.
Yellow brand-band pixels are a visibility gate, not OCR or an artistic-quality
claim.

## Model and runtime roles

- **Meta SAM 3.1** provides the frame-aligned tire perception used to recover
  and track one conceptual object.
- **Netflix VOID Pass 1** removes the tracked tire and reconstructs the wheel
  well and road.
- **Blender 5.2.2 Cycles** renders the requested tire as true 3D geometry with
  per-frame position, scale, tilt, and rolling spin.
- **Netflix VOID Pass 2** refines only the cleanup ring through an
  insertion-aware quadmask.
- **Comparator and assembler gates** reject missing stages, cadence mismatch,
  no-op insertion, mask-value violations, and frames below the brand threshold.

## Reproduce the software gates

```bash
uv sync
just check
just quality
uv run pytest scripts/tests/test_advertising_tire_3d.py \
  scripts/tests/test_blender_tire_render.py \
  scripts/tests/test_preserve_pass2_insertion.py \
  scripts/tests/test_assemble_advertising_reel.py -q
```

The SAM 3.1 and VOID lanes require their documented CUDA environments and
checkpoints. Repository tests do not download checkpoints, commit source media,
or call external APIs.

## Evidence and limitations

- Sanitized true-3D campaign evidence: [`evidence/true3d-tire-campaign-2026-11-03.json`](evidence/true3d-tire-campaign-2026-11-03.json)
- Judge-facing presentation guide: [`docs/presentation.md`](docs/presentation.md)

The procedural wheel is not a scan of the filmed tire. Its pose is
ellipse-derived rather than recovered 3D ground truth. Pixel measurements do
not prove photorealism or artistic quality. Source media, model checkpoints,
and generated videos are intentionally absent from this repository.
