default:
    @just --list

# Install the base CLI and development tools.
setup:
    uv sync

# Scaffold, instruction, workflow, and model-boundary contracts.
check:
    uv run python scripts/check_setup.py

# Lint, format check, type check, and contract/media tests.
quality:
    uv run python scripts/check_quality.py

# Install the optional SAM 3.1 execution lane.
setup-sam31:
    uv sync --extra sam31

# Clone the pinned Netflix VOID reference and build its separate environment.
setup-void revision='e3914f8f551dd4b880661991fd6b28cd1699a97a':
    ./scripts/setup_void_lane.sh {{revision}}

# Local doctor output; use --strict only in a configured execution lane.
doctor *args:
    uv run samurai doctor {{args}}
