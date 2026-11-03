#!/usr/bin/env bash
set -euo pipefail

VOID_REVISION="${1:-e3914f8f551dd4b880661991fd6b28cd1699a97a}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/runtime/void-model"

mkdir -p "$ROOT/runtime"
if [ ! -d "$DEST/.git" ]; then
    git clone https://github.com/Netflix/void-model.git "$DEST"
fi

actual_revision="$(git -C "$DEST" rev-parse HEAD)"
if [ "$actual_revision" != "$VOID_REVISION" ]; then
    printf 'VOID checkout is %s, expected %s\n' "$actual_revision" "$VOID_REVISION" >&2
    exit 1
fi

uv venv --python 3.12 "$DEST/.venv"
uv pip install --python "$DEST/.venv/bin/python" -r "$DEST/requirements.txt"
printf 'VOID lane ready: %s\n' "$DEST"
