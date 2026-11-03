# Void Fusion public agent instructions

This repository is the public, single-commit hackathon export. It contains the
release-clean source tree and sanitized evidence, not the team's private Git
history, pull-request discussion, credentials, model checkpoints, datasets, or
source media.

## Baseline and eligibility

The source starts from the tagged pre-hackathon baseline documented in
`docs/contribution-manifest.md`. The anonymized cumulative timeline in
`provenance/daily/` and the tracked receipts distinguish event-built work from
that baseline. Do not claim baseline code as hackathon work.

## Development contract

- Use `uv`, never pip.
- Run `just check` and `just quality` before reporting a change.
- Keep canonical Python in `src/samurai/`; it must not import `sam2`.
- Keep SAM 3.1 and VOID in separate execution lanes.
- Never commit credentials, private paths, model checkpoints, datasets, or
  media without recorded rights.
- Never enable GitHub Actions; local checks are the repository gate.
- Open bounded pull requests from feature branches; `master` is merge-only.
- A non-author reviewer must inspect substantive changes before merge.

## Model and media boundaries

- Meta SAM 3.1 is the dense anchor. Reasoning outputs must become aligned
  coordinate/probability fields and pass the same fusion contract as other
  experts; prose is never treated as pixels.
- Gray masks are lossless H.264 `yuv444p` at `qp=0`.
- VOID quadmask values are exactly `0` remove, `63` overlap, `127` affected,
  and `255` keep.
- The insertion order is selection/tracking, removal, VOID Pass 1 clean plate,
  fitted insertion, remask, insertion-aware quadmask, VOID Pass 2 cleanup, and
  no-Pass2 versus Pass2 comparison.
- Comparator numbers quantify change, not artistic quality. Preserve measured
  warnings and non-claims.
