# Judge presentation guide: true 3D advertising tire replacement

## One sentence

**Void Fusion replaces a selected tire in finished advertising footage: SAM 3.1
tracks the real wheel, VOID removes it, Blender poses a true 3D replacement, and
a second VOID pass refines only the cleanup ring.**

Layperson version: **we can put a different 3D tire into a finished car
commercial without reshooting the commercial or regenerating the whole scene.**

## Demo map

The submitted demo is 108.875 seconds, 1920x1080, 24 fps, and contains 2,613
frames. Player rounding can move a boundary by a few frames.

| Time | What the judge sees | Point |
| --- | --- | --- |
| 0:00-0:02 | Original-footage card | This is a finished advertising lane, not a generic editor. |
| 0:02-0:21 | Original multi-shot action | Real footage, real camera motion, visible existing tires. |
| 0:21-0:23 | SAM 3.1 card | One conceptual tire is the selected object. |
| 0:23-0:43 | SAM selection block | Frame-aligned perception tracks the tire across cuts and motion. |
| 0:43-0:45 | VOID Pass 1 card | Removal is isolated from the rest of the scene. |
| 0:45-1:04 | Clean plate block | The real tire disappears without regenerating the whole commercial. |
| 1:04-1:06 | Blender card | The requested target becomes actual 3D geometry. |
| 1:06-1:25 | True 3D insertion block | The wheel follows position, scale, tilt, and rolling spin. |
| 1:25-1:28 | Pass 2 card | Only the cleanup ring is exposed. |
| 1:28-1:47 | Preserved Pass 2 block | The refined surroundings survive while the brand object stays measurable. |
| 1:47-1:49 | Result card | 17 shots, 465 frames, and every frame passes the visibility gate. |

## Rubric-facing presentation

### Impact, 30%

Advertising teams need to change a sponsor, product version, localization, or
unsafe visual after a shoot has already been approved. The alternatives are
expensive reshoots or broad manual rotoscoping and compositing. A whole-scene
generator risks changing camera, vehicle performance, lighting, and action.

Void Fusion changes the selected object while preserving the finished spot. The
campaign spans 17 continuous runs from multiple advertising setups, not one
cherry-picked clip.

### Use of Meta AI tooling, 25%

Meta SAM 3.1 grounds selection as a frame-aligned, conceptual tire. Its output
is converted to the exact remove/overlap/affected/keep semantics required by
VOID and the insertion lane. The campaign treats perception skeptically:
frame-zero anchors, continuous-run checks, semantic rejection of non-tires,
quadmask readback, and final per-frame brand visibility are all required.

### Technical depth and execution, 20%

The executable chain is:

1. recover continuous tire masks with SAM 3.1;
2. encode removal and affected regions as VOID quadmasks;
3. run VOID Pass 1 and retime native temporal padding to exact 24 fps;
4. compile a bounded English target into a Blender wheel specification;
5. fit per-frame 3D position, scale, tilt, and rolling spin;
6. render transparent RGBA from a 38-object / 2,700-vertex / seven-material
   mesh and export `.blend` plus `.glb`;
7. composite over the clean plate and emit ROI-black inserted/cleanup masks;
8. run insertion-aware VOID Pass 2 with the replacement marked KEEP;
9. enforce the KEEP contract after diffusion and encode QP0 `yuv444p`;
10. measure object, cleanup-ring, background, and brand visibility before
    assembly.

### Demo quality, 25%

The video deliberately shows every stage as a full-frame block. The replacement
is not an invisible or ambiguous effect. The closing card reports the measured
scale, and the comparison master shows all five stages simultaneously for every
shot.

## Evidence

| Artifact | Value |
| --- | --- |
| Long-form demo | 2,613 frames, 108.875 s, 1920x1080, SHA-256 `be8c783362b39e89105a934dfcc97fae4a2c66beac97aa3a975e8590093f0e26` |
| Comparison master | 465 frames, 19.375 s, 3360x440, SHA-256 `57531ddea479fc5dfe6ecfc4e02f3e150420f1ff4ced8775fa053ce54f6c2bfe` |
| Ad-result reel | 465 frames, 19.375 s, 672x384, SHA-256 `c66cb9fa9d5ab265b401badd9f559282eb95401b2ddbc720cb898b417250deb0` |
| Proof sheet | SHA-256 `3a11ec460ea9c393cbfca7ddd5981c7e83159515b72a38f50ba8feec6336142f` |
| Preserved campaign | 17 shots / 465 frames |
| Brand gate | 465 / 465 frames pass |
| Object preservation | Changed ratio 0.0 in every shot |
| Cleanup-ring refinement | All 17 shots changed; mean 27.65%, max 52.48% |

## Plain-language glossary

| Term | Meaning |
| --- | --- |
| SAM 3.1 | Draws and follows the outline of the selected tire. |
| Clean plate | The scene after the original tire is removed. |
| True 3D asset | A modeled wheel with depth and geometry, not a flat sticker. |
| Quadmask | A pixel instruction map: remove, affected, or keep. |
| VOID Pass 2 | Repairs the narrow ring around the replacement. |
| KEEP contract | The exact 3D object is restored after diffusion so it cannot silently disappear. |
| Visibility gate | A frame-level check that the branded band remains present. |

## Expected questions

**Why not let SAM choose everything?**
SAM 3.1 is excellent dense perception, but a video edit also needs object
identity over time, affected-region semantics, insertion bounds, and failure
gates. Void Fusion keeps those contracts explicit instead of asking one model to
infer the whole pipeline.

**Why enforce the KEEP contract after Pass 2?**
Qualification showed diffusion could suppress the brand band even when the mask
said keep. Restoring the exact 3D object inside its mask preserves the
advertising requirement while retaining Pass 2 for the cleanup ring and
background.

**Is the tire learned image-to-3D?**
No. It is procedural Blender geometry fitted to the tracked trajectory. That
limitation is explicit; the project does not claim 3D reconstruction or physics
simulation.

**Is this production-ready?**
No. It is a real multi-shot research campaign with measurable gates. Photoreal
advertising delivery would require more scenes, lighting estimation, occlusion
testing, and human review.
