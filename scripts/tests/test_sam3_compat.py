from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast

COMPAT_PATH = Path(__file__).resolve().parents[1] / "sam31_pipe" / "sam3_compat.py"


def _load_compat():
    spec = importlib.util.spec_from_file_location("sam3_compat_under_test", COMPAT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_mutate_hf_environment(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("HF_HUB_")}
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib.util, os\n"
        f"spec = importlib.util.spec_from_file_location('compat', {str(COMPAT_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print(os.environ.get('HF_HUB_CACHE'), os.environ.get('HF_HUB_OFFLINE'))\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(probe)], env=env, check=True, capture_output=True, text=True
    )
    assert completed.stdout.strip() == "None None"


def test_build_predictor_tolerates_missing_init_state(monkeypatch, tmp_path):
    compat = _load_compat()

    class Predictor:
        def __init__(self):
            self.model = types.SimpleNamespace()

    predictor = Predictor()
    model_builder = types.SimpleNamespace(build_sam3_predictor=lambda **kwargs: predictor)
    sam3 = cast(Any, types.ModuleType("sam3"))
    sam3.model_builder = model_builder
    huggingface_hub = cast(Any, types.ModuleType("huggingface_hub"))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    huggingface_hub.hf_hub_download = lambda repo, filename: str(checkpoint)
    monkeypatch.setitem(sys.modules, "sam3", sam3)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    assert compat.build_predictor(version="sam3.1") is predictor


def test_resolve_does_not_override_the_caller_hf_environment(monkeypatch, tmp_path):
    compat = _load_compat()
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    checkpoint = tmp_path / "checkpoint.pt"
    huggingface_hub = cast(Any, types.ModuleType("huggingface_hub"))
    huggingface_hub.hf_hub_download = lambda repo, filename: str(checkpoint)
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    assert compat.resolve("sam3.1") == ("sam3.1", "facebook/sam3.1", str(checkpoint))
    assert "HF_HUB_CACHE" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ
