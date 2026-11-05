# VOID Auto-Edit

Click one object in a video. VOID Auto-Edit removes it and runs automatic checks across the processed shot.

Video teams can review flagged moments instead of scrubbing through an entire AI edit.

## How it works

1. The creator clicks the object to remove.
2. SAM 3.1 follows that object through the shot.
3. Muse adds a second view of the moving object.
4. VOID rebuilds the background.
5. VOID Auto-Edit checks the processed shot automatically.

## Demo

- [Watch the demo](https://github.com/RESMP-DEV/void-fusion-lab-hackathon-public/releases/download/v4.0.0/void-auto-edit-removal-only.mp4)
- [Presentation script](docs/presentation.md)
- [Project details](evidence/auto-edit-submission-2026-11-04.json)

## Run the checks

```bash
uv sync
just check
just quality
```
