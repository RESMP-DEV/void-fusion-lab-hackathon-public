# Judge presentation guide: advertising tire replacement

## One sentence

**Void Fusion swaps a selected tire in already-shot advertising footage: SAM 3.1
tracks the original wheel, VOID removes it, a visible Michelin-style replacement
follows the same motion, and a second VOID pass cleans only the seam.**

A layperson version: **we can replace the tire in a finished car commercial
without reshooting the commercial or regenerating the whole scene.**

## Demo map

The submitted long-form demo is 107.208 seconds, 1920x1080, 24 fps, and contains
2,573 frames. Player rounding can move a boundary by a few frames.

| Time | What the judge sees | Point |
| --- | --- | --- |
| 0:00-0:03 | Title and original-footage title card | This is a finished advertising lane, not a generic editor. |
| 0:03-0:22 | Original multi-shot commercial action | Real footage, real camera motion, visible existing tires. |
| 0:22-0:43 | Red SAM target block | The same selected tire is explicitly identified in every retained frame. |
| 0:43-1:03 | VOID Pass 1 clean plate | The original tire is removed and the wheel well/road is reconstructed. |
| 1:03-1:24 | Visible replacement block | A Michelin-style tire follows the selected motion; the brand band remains visible. |
| 1:24-1:45 | VOID Pass 2 block | Only a cleanup ring is regenerated while the replacement is marked keep. |
| 1:45-1:47 | Result card | 16 retained shots, 457 aligned frames, and a frame-level visibility gate. |

The separate five-stage comparison master shows the same sequence with original,
selection, clean plate, insertion, and refined result side by side for every
retained shot.

## Rubric-facing presentation

### Impact, 30%

Advertising teams often need to update a sponsor's tire, change a product
version, or localize a campaign after the shoot. The expensive options are a
reshoot or broad manual roto/compositing. A whole-scene generator risks changing
camera, vehicle performance, lighting, and action that were already approved.

Void Fusion changes the selected tire while preserving the rest of the finished
spot. The submitted campaign demonstrates the workflow across 16 retained shots
from multiple advertising setups rather than one cherry-picked clip.

### Use of Meta AI tooling, 25%

Meta SAM 3.1 is not decorative. It supplies the frame-aligned target that drives
the VOID quadmask and the replacement trajectory. The red selection block makes
that role visible to a non-technical viewer.

The campaign also treats model output skeptically:

1. midpoint detections were re-run from frame zero because the runtime is
   forward-only;
2. empty, unstable, road-sized, mascot, and non-tire masks were rejected;
3. continuous runs were semantically reviewed before VOID work;
4. refined outputs were rejected frame-by-frame if the replacement's brand band
   disappeared.

That is why 465 recovered frames became 457 retained frames: one eight-frame
Pass 2 result failed the final visibility gate.

### Technical depth and execution, 20%

The executable chain is:

1. recover continuous tire masks with SAM 3.1;
2. encode the tire as VOID remove class `0` and a bounded cleanup ring as `127`;
3. run VOID Pass 1 for all 17 selected runs;
4. retime the 12 fps VOID output back to the exact source frame cadence;
5. fit and render a visible replacement tire over each clean plate;
6. emit an inserted-object mask and 16-pixel cleanup ring;
7. build an insertion-aware quadmask where the replacement is keep and only the
   ring is affected;
8. run VOID Pass 2 for all 17 runs;
9. compare no-Pass2 and Pass2 by object, cleanup ring, background, and temporal
   measurements;
10. reject missing stages, cadence mismatches, and near-invisible pre-Pass2
    insertions in the assembler, then apply a campaign-level brand audit to the
    refined frames before final selection.

All 17 Pass 1 runs and all 17 Pass 2 runs completed. The final 16-shot assembly
has 457 aligned frames.

### Demo quality, 25%

The demo is deliberately explicit. It shows the original, red target, clean
plate, visible replacement, and refined result as separate full-frame blocks.
No segment asks the viewer to infer an invisible edit. The closing card reports
the retained scale and visibility gate rather than claiming perfection.

## Evidence

| Artifact | Value |
| --- | --- |
| Long-form demo | 2,573 frames, 107.208 s, 1920x1080, SHA-256 `821eadc309574bf286c92172c31f493b3f0acb062f8633def778f0f5aca3c0cf` |
| Comparison master | 457 frames, 19.042 s, 3360x440, SHA-256 `2d74acfa057debf29bbbca7552a1f8843706807a46b03594e6fcd4f2e9e63ac5` |
| Four-stage proof sheet | SHA-256 `89c55d13f3e6df056e41af255ef42b1882709051fc6c4a426d642ab084d98202` |
| Recovered continuous runs | 17 / 465 frames |
| Pass 1 | 17 / 17 succeeded |
| Pass 2 | 17 / 17 succeeded |
| Retained | 16 shots / 457 frames |
| Final brand gate | 457 / 457 retained frames brand-visible |

## Plain-language glossary

| Term | Meaning |
| --- | --- |
| SAM 3.1 | Draws and follows the outline of the selected tire. |
| Clean plate | The scene after the original tire is removed. |
| Quadmask | A pixel instruction map: remove, affected, or keep. |
| VOID Pass 1 | Rebuilds the missing road and wheel well. |
| Cleanup ring | A narrow area around the replacement where Pass 2 may repair the seam. |
| VOID Pass 2 | Refines that ring while preserving the inserted tire. |
| Visibility gate | A frame-level check that the branded replacement is still visibly present. |

## Expected questions

**Why not use SAM's text prompt alone?**
SAM 3.1 can find tire-like objects, but it can also return a road band, mascot,
or an empty mask after a cut. The campaign ran continuous-mask and semantic
checks before spending VOID time, then applied a final brand-visibility gate
after Pass 2.

**Why is the tire renderer not learned image-to-3D?**
It is deterministic and fits the tracked ellipse trajectory. This is disclosed
as a limitation and avoids pretending that a full physics or 3D reconstruction
system exists.

**Does Pass 2 perfectly preserve the replacement?**
No. Retained object-preservation changed ratios range from 5.61% to 44.21%. The
numbers are reported instead of hidden. The cleanup ring changes more than the
background on average, which matches the intended local-refinement contract.

**Is the result production-ready?**
No. It is a real multi-shot research campaign with explicit gates, not a claim
of photoreal advertising delivery.

## Presenter fallback

If asked for a one-minute explanation:

> We start with finished car-commercial footage. SAM 3.1 follows one tire through
> each selected shot and draws it in red. VOID Pass 1 removes that tire and
> rebuilds the wheel well. We place a clearly branded Michelin-style tire on the
> same trajectory, then tell VOID Pass 2 to preserve that replacement and repair
> only the ring around it. The final demo keeps sixteen shots and 457 frames;
> every retained frame still shows the replacement brand, and one short shot was
> rejected when that failed.
