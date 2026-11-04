# Architecture and status

Void Fusion is a visible advertising-object replacement runtime. The v2 campaign
uses Meta SAM 3.1 to track one tire through finished advertising footage, VOID
Pass 1 to remove that tire and reconstruct the wheel well, a deterministic fitted
Michelin-style replacement, and insertion-aware VOID Pass 2 to refine only a
bounded cleanup ring.

## Current campaign pipeline

1. Forward-only SAM 3.1 runs recover continuous tire masks from verified tire
   anchors. Empty, unstable, road-sized, mascot, and otherwise non-tire runs are
   rejected before CUDA inpainting.
2. The selected tire becomes the exact VOID Pass 1 quadmask: tire `0` remove,
   bounded cleanup ring `127` affected, and untouched background `255` keep.
3. VOID Pass 1 produces a 12 fps output. The downstream lane trims temporal
   padding and halves PTS to recover the exact selected frame count at the
   original 24 fps cadence.
4. The replacement lane fits the tire trajectory, renders a visible black tire
   with a yellow Michelin-style brand band, and composites it over the clean
   plate. It writes an ROI-black inserted mask and a full 16-pixel ROI-black
   cleanup ring.
5. The insertion-aware quadmask marks the replacement and untouched background
   keep, and only the cleanup ring affected. Frame zero follows the affected-ring
   contract and is fully keep.
6. VOID Pass 2 refines the staged sequences. The comparator aligns exact
   geometry, frame count, FPS, and timestamps, then measures object preservation,
   cleanup-ring change, background change, and temporal warnings.
7. A campaign-level brand audit rejects a refined run if the inserted brand band
   is no longer detectable in any frame. The assembler separately rejects a
   missing stage, a cadence mismatch, or a pre-Pass2 insertion nearly identical
   to its clean plate.

## V2 campaign result

| Stage | Result |
| --- | ---: |
| Continuous tire runs recovered | 17 / 465 frames |
| VOID Pass 1 successes | 17 / 17 |
| VOID Pass 2 successes | 17 / 17 |
| Rejected by final brand gate | 1 run / 8 frames |
| Retained campaign | 16 shots / 457 frames / 19.042 s |
| Retained brand-visible frames | 457 / 457 |
| Long-form presentation | 2,573 frames / 107.208 s / 1920x1080 |

The retained Pass 2 object-preservation changed ratio ranges from 5.61% to
44.21%, the cleanup-ring changed ratio from 9.51% to 40.19%, and the background
changed ratio from 0.80% to 14.45%. These aggregate ranges are recorded in
`evidence/advertising-campaign-evidence.json`.

## Owned code boundaries

| Area | Owner |
| --- | --- |
| Fitted replacement renderer and pre-Pass2 brand check | `scripts/advertising_tire_asset.py` |
| Multi-shot reel assembler and insertion gate | `scripts/assemble_advertising_reel.py` |
| Pass 2 comparison | `scripts/pass2_compare.py` |
| VOID sequence packaging | `scripts/package_quadmask.py` |
| Media and quadmask contracts | `src/samurai/media.py`, `src/samurai/quadmask.py` |

## Important boundaries

- The replacement is deterministic and 2.5D-looking. It is not learned
  image-to-3D reconstruction, physical simulation, or a claim of photorealism.
- Pass 2 materially changes the replacement in some retained shots. The changed
  ratios and temporal warnings are retained rather than hidden.
- Comparator values quantify pixel change, not artistic quality.
- The SAM 3.1 execution used the community MuggledSAM runtime; its masks are not presented as ground truth.
- Source media, checkpoints, private paths, credentials, and private review
  history are not committed.
- The earlier humanoid full-chain lane measured lower-body ownership on only
  frames `0`, `1`, `7`, and `13` of 96. That historical lane does not claim continuous pose ownership; the tire campaign likewise claims no full-body physics.
