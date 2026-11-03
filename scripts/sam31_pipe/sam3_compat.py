# ruff: noqa: UP031
# Formatting note: the percent-style formatting and the long prompt and data
# strings are kept verbatim; converting them is a follow-up cleanup.
"""SAM 3.x loader shim for CSC eval jobs. Do not edit site-packages.

Model choice: env SAM_MODEL (or seg.py --model), default "sam3.1".
  sam3.1 -> HF facebook/sam3.1, sam3.1_multiplex.pt   (default)
  sam3   -> HF facebook/sam3,   sam3.pt               (legacy 3.0; only if those weights are cached)
Weights are resolved with hf_hub_download under the caller's standard Hugging Face
environment (including HF_HUB_CACHE and HF_HUB_OFFLINE when set). The loader never
overrides either variable. Logs one line: "SAM model: ...".

Bug worked around: sam3 @7f473cf start_session passes offload_state_to_cpu, but
Sam3MultiplexTrackingWithInteractivity.init_state rejects it.
"""

import os

DEFAULT_MODEL = "sam3.1"
MODELS = {
    "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt"),
    "sam3": ("facebook/sam3", "sam3.pt"),
}


def resolve(version=None):
    version = version or os.environ.get("SAM_MODEL", DEFAULT_MODEL)
    if version not in MODELS:
        raise ValueError(
            "unknown SAM model version %r; valid versions: %s" % (version, ", ".join(MODELS))
        )
    repo, fn = MODELS[version]
    from huggingface_hub import hf_hub_download  # ty: ignore[unresolved-import]

    return version, repo, hf_hub_download(repo, fn)


def build_predictor(version=None, **kw):
    from sam3 import model_builder  # ty: ignore[unresolved-import]

    version, repo, ckpt = resolve(version)
    kw.setdefault("compile", False)
    kw.setdefault("warm_up", False)
    kw.setdefault("use_fa3", False)  # flash_attn_interface is not installed
    print(
        "SAM model: version=%s hf_id=%s ckpt=%s" % (version, repo, os.path.realpath(ckpt)),
        flush=True,
    )
    p = model_builder.build_sam3_predictor(version=version, checkpoint_path=ckpt, **kw)
    orig = getattr(p.model, "init_state", None)
    if orig is None:
        return p

    def init_state(*a, **k):
        k.pop("offload_state_to_cpu", None)
        return orig(*a, **k)

    p.model.init_state = init_state
    return p
