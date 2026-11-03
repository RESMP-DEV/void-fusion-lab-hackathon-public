# Judge presentation guide

## One-sentence premise

**Void Fusion changes who or what appears in an existing video by automatically
finding one selected object, erasing it, inserting a replacement that follows
the original motion, and measuring whether the result stays believable.**

A shorter version is: **Void Fusion is a digital stunt-double system for one
selected object in a real video.**

## Who needs it

A film, advertising, animation, or previsualization team often has a good shot
where one object, product, prop, stand-in, or character is wrong. The normal
options are an expensive reshoot, a manual compositing session, or asking a
generic video generator to invent the entire scene again. Void Fusion keeps the
real action and camera motion, changes only the selected object, and records
what changed.

## Demo map

The submitted demo is 2 minutes 35.8 seconds. The ranges below are presentation
blocks; player rounding may move a boundary by a few frames.

| Approximate time | What the judge sees | Plain-language point |
|---|---|---|
| 0:00-0:18 | Original scene and replacement goal | Change one selected object without reshooting the action. |
| 0:18-0:38 | Target selection and mask overlay | Meta SAM 3.1 supplies the frame-aligned anchor; structured judgments help decide target, preserve, and affected regions. |
| 0:38-0:58 | Object removal and clean plate | VOID Pass 1 reconstructs the background where the selected object was. |
| 0:58-1:18 | Fitted 3D replacement | A model-free capsule proxy is posed over the clean plate. |
| 1:18-1:38 | Remasking and second refinement | The replacement is marked keep; only its cleanup ring is regenerated. |
| 1:38-1:58 | Four-candidate measurement | Every complete candidate is tracked, removed, inserted, remasked, refined, and measured. |
| 1:58-2:18 | Limitations | Pose ownership is sparse and temporal warnings remain visible. |
| 2:18-2:36 | Four-panel result and close | Original, target mask, before-refinement insertion, and selected result are directly comparable. |

## Rubric-facing presentation

### Impact, 30%

**Claim:** Void Fusion addresses the cost of reshooting or manually reworking a
real shot when only one object must change.

Presentation sequence:

1. Start with the one-sentence premise.
2. Name the user: a film, advertising, animation, or previsualization team.
3. Name the old cost: reshoot, manual compositing, or regenerating the scene.
4. Show the four-panel result.
5. End with the measured outcome: four complete candidate chains, with
   `smooth-00` selected by a declared formula.

Evidence: the public demo, `evidence/submission-evidence.json`, and the result
table in `README.md`.

### Use of Meta AI tooling, 25%

**Claim:** Meta models have distinct, auditable roles rather than one thin
text-to-video call.

- **Meta SAM 3.1** provides the dense, frame-aligned mask for the selected
  object.
- **Meta Muse Spark 1.3** participates in a bounded target/preserve/affected
  prompt loop with strict JSON validation.
- Semantic decisions do not masquerade as pixels. Coordinates and concepts are
  converted into aligned probability fields and enter the same numerical fusion
  contract as the dense expert.
- Ambiguous selections escalate instead of silently changing target.

Judge-facing distinction: Muse helps decide *what the user means*; SAM 3.1
helps find *where that meaning appears in each frame*; VOID changes only the
approved region.

Evidence: model identities, three successful Muse calls, one disclosed
uncaptured generation, execution parameters, and output digests are recorded in
`evidence/submission-evidence.json`.

### Technical depth and execution, 20%

**Claim:** The project is a reproducible control system around perception,
insertion, and video inpainting.

1. Select and track one conceptual object.
2. Convert aligned expert opinions into probability fields and fuse them.
3. Build the exact VOID quadmask: `0` remove, `63` overlap, `127` affected,
   `255` keep.
4. Run VOID Pass 1 to create the clean plate.
5. Fit and composite a 3D proxy.
6. Remask the replacement as keep and mark only a cleanup ring for change.
7. Run VOID Pass 2.
8. Compare no-Pass2 and Pass2 with exact geometry, timestamps, object change,
   background change, and temporal stability.
9. Rank only complete candidate chains.

The public evidence carries the exact objective formula and pre-Pass2 scores
needed to reproduce the ranking:

```text
0.35 * pre_pass2_objective
+ 0.25 * (1 - object_preservation_changed_ratio)
+ 0.20 * (1 - background_changed_ratio)
+ 0.20 * (1 - temporal_pop_score)
```

Evidence: public source, local checks, candidate digests, comparator-report
digests, and formula/ranking tests.

### Demo quality, 25%

**Claim:** The demo is short, real, structured, and honest.

- It opens with the problem rather than a technical diagram.
- It shows the original, mask, insertion, and refined result.
- It explains every model role in plain language.
- It shows the four-candidate measurement.
- It names the selected candidate.
- It closes with known limitations rather than hiding them.
- The presentation derivative is 1280x720, 155.798549 seconds, and SHA-256
  `b3e0ca9530662da0a59826e185e3cd801b9f2dcb85524d93ef2eeb8080cfccf1`.

## Plain-language glossary

| Term | Layperson meaning |
|---|---|
| SAM 3.1 | Finds the outline of the chosen object frame by frame. |
| Muse | Helps understand what should be removed, kept, or allowed to change. |
| Expert fusion | Turns several model opinions into one numerical decision. |
| Clean plate | The background after the original object is removed. |
| Quadmask | An instruction map marking remove, overlap, affected, and keep pixels. |
| VOID Pass 1 | Rebuilds the background where the selected object was. |
| VOID Pass 2 | Cleans only the seam around the inserted replacement. |
| Best-of-N | Runs several complete versions and chooses the strongest measured result. |
| Temporal warning | A visible flicker or sudden frame-to-frame change. |

## Expected questions and answers

**Why not use one video generator for everything?**
A whole-scene generator can change camera work, identity, lighting, and action
that were already correct. Void Fusion changes the selected object while
preserving the real footage and then measures the result.

**Why is SAM 3.1 not the only model?**
Dense segmentation finds an object, but target intent, preserved subjects, and
affected shadows or contact can be ambiguous. Separating semantic selection
from dense perception makes those roles auditable.

**Is the replacement learned image-to-3D?**
No. This submission uses a model-free fitted capsule proxy. That limitation is
explicit.

**Does the system claim continuous physics?**
No. Sparse pose ownership passes only 4 of 96 source frames, and every current
comparator reports the maximum temporal warning.

**Do the metrics prove artistic quality?**
No. They quantify change. The project keeps the warnings and does not present
the selected score as a human-quality score.

**What is the strongest technical result?**
Four complete chains passed all 50 VOID Pass2 steps. The public formula
reproduces the `smooth-00 > smooth-06 > smooth-02 > smooth-04` ranking.

## Presenter fallback

If asked for a one-minute explanation, use this:

> Void Fusion is a digital stunt-double system. We point at one object in a
> real video. SAM 3.1 follows it frame by frame, structured model reviewers
> help decide what to remove and preserve, and their judgments become a pixel
> mask. VOID erases the original object and rebuilds the background. We insert
> a fitted replacement, mark it as preserved, and ask VOID to clean only the
> seam. The system runs several complete versions, measures object change,
> background change, and temporal stability, and selects the strongest result.
> The current demo is real, but intentionally reports its temporal and pose
> limitations.

## Presentation non-claims

- Do not call the result photorealistic, production-ready, or continuous
  full-body physics.
- Do not call comparator metrics artistic quality.
- Do not claim learned image-to-3D reconstruction.
- Do not describe the system as autonomous whole-scene video generation.
- Do not omit the 4/96 ownership result or temporal warning.
