from __future__ import annotations

from check_quality import CHECK_COMMANDS
from check_setup import canonical_import_contract, instruction_contract, workflow_contract


def test_setup_contracts_pass_in_repository():
    assert instruction_contract() == (0, "instruction contract passed")
    assert workflow_contract() == (0, "no GitHub Actions workflows")
    assert canonical_import_contract()[0] == 0


def test_quality_owns_expected_domains():
    assert [name for name, _command in CHECK_COMMANDS] == [
        "ruff-check",
        "ruff-format",
        "shellcheck",
        "ty",
        "pytest",
    ]
