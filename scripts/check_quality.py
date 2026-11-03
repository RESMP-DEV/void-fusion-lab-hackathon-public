from __future__ import annotations

import argparse
import sys

from check_common import prepare_receipt, run_check, write_receipt

CHECK_COMMANDS = [
    ("ruff-check", [sys.executable, "-m", "ruff", "check", "src", "scripts"]),
    (
        "ruff-format",
        [sys.executable, "-m", "ruff", "format", "--check", "src", "scripts"],
    ),
    ("shellcheck", ["shellcheck", "scripts/setup_void_lane.sh"]),
    ("ty", ["ty", "check", "src", "scripts"]),
    ("pytest", [sys.executable, "-m", "pytest", "-q"]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    prepare_receipt("quality")
    checks = [run_check(name, command) for name, command in CHECK_COMMANDS]
    passed = write_receipt(
        "quality",
        "Static quality checks and local media/contract tests; no SAM 3.1 or VOID CUDA run.",
        checks,
    )
    for check in checks:
        status = "PASS" if check.exit_code == 0 else "FAIL"
        print(f"{check.name}: {status} ({check.elapsed_seconds:.3f}s)")
    print(f"aggregate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
