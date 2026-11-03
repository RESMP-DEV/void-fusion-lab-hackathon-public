# Credits and external boundaries

## Team

Kearm (representative), Eden Chan, and Jason Lu.

## Original code

Unless credited below, source in `src/samurai/`, `scripts/`, documentation,
and configuration is original RESMP-DEV work under Apache-2.0. The historical
`samurai` package name is retained for command compatibility and does not claim
ownership of the upstream SAMURAI project.

## External models and projects

- **Meta SAM 3 / SAM 3.1**: pinned external source and checkpoints governed by
  Meta's SAM License. Used as the dense perception anchor.
- **Meta Muse Spark 1.3**: commercial Meta Model API service used for bounded
  target/affected reasoning and prompt-candidate validation. No model weights
  are redistributed.
- **Netflix VOID**: pinned external revision
  `e3914f8f551dd4b880661991fd6b28cd1699a97a`; its source is Apache-2.0, while
  checkpoints, base models, generated assets, and benchmarks remain external.
- **SAM 2** and **SAMURAI**: historical comparison context only; neither is
  redistributed or imported by canonical code.
- **CogVideoX-Fun**: VOID's external base-model lineage; no checkpoint or base
  model is redistributed here.
- **Blender / bpy**: optional separate execution tool used by the insertion
  proxy renderer. This repository distributes an Apache-2.0 script that calls a
  separately installed Blender runtime; it does not redistribute Blender.
- **PyTorch, Diffusers, NumPy, Pillow, and FFmpeg**: runtime and media
  dependencies under their own licenses.

## Demo rights boundary

The representative reviewed the transformed demonstration under a fair-use
boundary for this hackathon. The public presentation derivative removes the
source-stream watermark region and does not redistribute unmodified source
footage, private datasets, model checkpoints, or private pipeline output media.

## Methods

The label-free calibration implementation uses a bounded Bernoulli
Dawid-Skene estimation formulation credited to Alexander Philip Dawid and Allan
Skene (1979). No external implementation is claimed as incorporated unless a
receipt says so.
