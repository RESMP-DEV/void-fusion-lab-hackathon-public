from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    name: str
    command: list[str]
    exit_code: int
    elapsed_seconds: float
    log: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def receipt_path(mode: str) -> Path:
    return repository_root() / "build/reports" / mode / "receipt.json"


def prepare_receipt(mode: str) -> None:
    path = receipt_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)


def source_revision() -> dict[str, object]:
    root = repository_root()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        revision = "unknown"
        status = [f"git unavailable: {error}"]
    return {"revision": revision, "dirty": bool(status)}


def run_check(name: str, command: list[str]) -> CheckResult:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=repository_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    log = (completed.stdout + completed.stderr).strip()
    return CheckResult(name, command, completed.returncode, elapsed, log[-20000:])


def write_receipt(
    mode: str,
    scope: str,
    checks: list[CheckResult],
    *,
    startup_error: str | None = None,
) -> bool:
    passed = (
        startup_error is None and bool(checks) and all(check.exit_code == 0 for check in checks)
    )
    receipt = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "source": source_revision(),
        "scope": scope,
        "checks": [asdict(check) for check in checks],
        "aggregate_passed": passed,
    }
    if startup_error is not None:
        receipt["startup_error"] = startup_error
    path = receipt_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return passed


def environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if "SAMURAI" in key}
