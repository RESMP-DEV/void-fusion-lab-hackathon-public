# VOID Auto-Edit public workspace

This repository contains the public source, checks, evidence, and demo material
for VOID Auto-Edit. Keep the repository release-safe: no credentials, private
machine paths, model checkpoints, datasets, or unlicensed source media.

## Development contract

- Use `uv`, never pip.
- Run `just check` and `just quality` before reporting a change.
- Keep canonical Python in `src/samurai/`; it must not import `sam2`.
- Keep SAM 3.1 and VOID in separate execution lanes.
- Never enable GitHub Actions; local checks are the repository gate.
- Open bounded pull requests from feature branches; `master` is merge-only.
- A non-author reviewer must inspect substantive changes before merge.

## Model and media boundaries

- Meta SAM 3.1 follows the selected object through the processed shot.
- Muse contributes a second perception view while the object moves.
- Reasoning outputs become aligned coordinate or probability fields; prose is
  never treated as pixels.
- Gray masks are lossless H.264 `yuv444p` at `qp=0`.
- VOID quadmask values are exactly `0` remove, `63` overlap, `127` affected,
  and `255` keep.
- Comparator numbers quantify change, not artistic quality. Preserve measured
  warnings and boundaries.
