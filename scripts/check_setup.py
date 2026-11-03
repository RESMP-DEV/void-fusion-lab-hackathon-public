from __future__ import annotations

import argparse
import ast
import sys

from check_common import CheckResult, prepare_receipt, repository_root, write_receipt


def instruction_contract() -> tuple[int, str]:
    root = repository_root()
    agents = root / "AGENTS.md"
    claude = root / "CLAUDE.md"
    problems = []
    if not agents.is_file():
        problems.append("AGENTS.md is missing")
    if not claude.is_symlink() or claude.readlink().as_posix() != "AGENTS.md":
        problems.append("CLAUDE.md must be a relative symlink to AGENTS.md")
    for filename in ("README.md", "CONTRIBUTING.md", "docs/architecture.md"):
        if not (root / filename).is_file():
            problems.append(f"{filename} is missing")
    return (0, "instruction contract passed") if not problems else (1, "; ".join(problems))


def workflow_contract() -> tuple[int, str]:
    workflows = [
        path.relative_to(repository_root()).as_posix()
        for path in repository_root().rglob(".github/workflows")
        if path.is_dir()
    ]
    return (
        (0, "no GitHub Actions workflows")
        if not workflows
        else (
            1,
            "forbidden workflow directories: " + ", ".join(workflows),
        )
    )


def canonical_import_contract() -> tuple[int, str]:
    problems = []
    for path in [
        *(repository_root() / "src/samurai").glob("*.py"),
        *repository_root().glob("scripts/*.py"),
    ]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            problems.append(f"{path.name}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "sam2" or alias.name.startswith("sam2.") for alias in node.names
            ):
                problems.append(f"{path.name}: canonical code imports sam2")
            if isinstance(node, ast.ImportFrom) and (
                node.module == "sam2" or (node.module or "").startswith("sam2.")
            ):
                problems.append(f"{path.name}: canonical code imports sam2")
    return (
        (0, "canonical SAM 3.1 boundary has no sam2 imports")
        if not problems
        else (
            1,
            "; ".join(problems),
        )
    )


def internal(name: str, result: tuple[int, str]) -> CheckResult:
    return CheckResult(
        name=name,
        command=[sys.argv[0], f"internal:{name}"],
        exit_code=result[0],
        elapsed_seconds=0.0,
        log=result[1],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    prepare_receipt("setup")
    checks = [
        internal("instruction-contract", instruction_contract()),
        internal("workflow-contract", workflow_contract()),
        internal("canonical-import-contract", canonical_import_contract()),
    ]
    passed = write_receipt(
        "setup",
        "Scaffold and model-boundary contracts; no SAM 3.1, VOID, CUDA, or real-video run.",
        checks,
    )
    for check in checks:
        print(f"{check.name}: {'PASS' if check.exit_code == 0 else 'FAIL'} - {check.log}")
    print(f"aggregate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
