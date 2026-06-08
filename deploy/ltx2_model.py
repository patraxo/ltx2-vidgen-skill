from __future__ import annotations  # PEP-563 lazy annotations — needed so local
# Python 3.9 can parse the Modal entry script (uses PEP-604 `T | None` syntax in
# method signatures introduced by W1/W8). The Modal container itself runs Python
# 3.12 (see `add_python="3.12"` below) where this is a no-op.

import logging
import os
import tempfile
from pathlib import Path

from fastapi import Body, HTTPException, Request, Response, status

import modal

# Make the co-located `deploy/utils/` package importable — both at deploy time
# (so add_local_python_source("utils") resolves locally) and inside the container.
import sys as _sys
_DEPLOY_DIR = Path(__file__).resolve().parent          # deploy/ — holds utils/
if str(_DEPLOY_DIR) not in _sys.path:
    _sys.path.insert(0, str(_DEPLOY_DIR))
_REPO_ROOT = _DEPLOY_DIR.parent                        # repo root (kept for any root-relative use)
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_NAME = "Lightricks/LTX-2.3"
MODEL_REVISION = "76730e634e70a28f4e8d51f5e29c08e40e2d8e74"

# 2026-05-23 K5 — upstream Lightricks pipelines (`TI2VidTwoStagesPipeline`,
# `TI2VidTwoStagesHQPipeline`, `KeyframeInterpolationPipeline`, `RetakePipeline`,
# `DistilledPipeline`) all accept `torch_compile: bool = False`. When True,
# `COMPILE_TRANSFORMER` (ltx_core/.../compiling.py) is appended to module_ops →
# each of the 48 transformer_blocks is replaced with `torch.compile(block)` and
# `forward` is wrapped with a dynamo context. Projected wins per K5: 1.3-1.6x
# transformer-forward speedup → ~15-25% end-to-end latency reduction on warm
# requests. Cold-start cost per pipeline variant: ~2-5 min the FIRST time on
# any container; the TorchInductor cache (mounted to /models/.torchinductor_cache
# above) eliminates that cost on subsequent boots.
#
# FBCache compatibility: confirmed safe via inspection — `compile_transformer`
# only wraps individual blocks + sets a dynamo context on forward; our
# `attach_to_ltx_model` swaps `_process_transformer_blocks` (the iterator over
# `transformer_blocks`), which is invoked from the un-compiled outer forward.
# FBCache calls the compiled blocks one at a time and short-circuits as usual.
#
# Set `LTX_ENABLE_TORCH_COMPILE=0` to opt out (e.g. for cold-start-sensitive
# debugging traffic with no warm pool).
ENABLE_TORCH_COMPILE = os.environ.get("LTX_ENABLE_TORCH_COMPILE", "1") == "1"

# 2026-05-23: registry-mode selector. The original `StateDictRegistry`
# (`gpu` mode) OOMs the H100 80GB by stacking 44GB stage_1 + 44GB stage_2
# + 24GB Gemma in GPU memory. `cpu_pinned` (R2 patch, default for the new
# CpuPinnedStateDictRegistry) caches state-dict tensors in CPU pinned
# memory and streams a FRESH GPU copy on every `.get()`, so the registry
# contributes ZERO bytes of permanent GPU pressure. Saves ~13-15s per
# warm request because Gemma (~24GB on disk) streams from pinned CPU at
# PCIe ~25 GB/s (~1s) instead of safetensors disk (~16s).
#
#   LTX_REGISTRY=off          → no registry (legacy default, every call
#                                reads from disk).
#   LTX_REGISTRY=cpu_pinned   → CpuPinnedStateDictRegistry (recommended).
#   LTX_REGISTRY=gpu          → original StateDictRegistry (DO NOT USE
#                                in prod — OOM-prone, kept for forensic
#                                comparison only).
#
# Legacy env var `LTX_USE_REGISTRY=1` still works and now maps to
# `cpu_pinned` (the safe variant) for backward compatibility.
_LTX_REGISTRY_MODE = os.environ.get("LTX_REGISTRY", "").lower().strip()
if not _LTX_REGISTRY_MODE:
    if os.environ.get("LTX_USE_REGISTRY", "0") == "1":
        _LTX_REGISTRY_MODE = "cpu_pinned"
    else:
        _LTX_REGISTRY_MODE = "off"

ENABLE_STATE_DICT_REGISTRY = _LTX_REGISTRY_MODE != "off"

# LTX_PREWARM_COMPILE=1 runs a tiny forward pass in @modal.enter() to
# populate the torch.compile / TorchInductor / Triton caches. Saves
# ~30-40s on the first user request, BUT leaks the entire two-stage
# model into the allocator's reserved pool, causing CUBLAS_STATUS_ALLOC_FAILED
# on the very next request (Gemma can't allocate cuBLAS handle).
# Stays in the codebase for use once we have a cuda.empty_cache+reset path.
ENABLE_PREWARM_COMPILE = os.environ.get("LTX_PREWARM_COMPILE", "0") == "1"
GEMMA_MODEL = "google/gemma-3-12b-it-qat-q4_0-unquantized"  # LTX-2 requires this specific Gemma variant
# 2026-05-23: pin Gemma to a specific HF commit. Was "main", which would
# silently drift if Google republished the repo. Commit fetched via
# `curl https://huggingface.co/api/models/google/gemma-3-12b-it-qat-q4_0-unquantized`.
GEMMA_REVISION = "68f7ee4fbd59087436ada77ed2d62f373fdd4482"

# 2026-05-23: Upstream-canonical default negative prompt mirrored verbatim
# from ltx_pipelines/utils/constants.py:135-147 @ commit 76730e6. This is the
# prompt Lightricks themselves tuned CFG against — 15+ artifact categories
# covering exposure, motion, anatomy, audio, cinematography. We copy it
# here as a module-level constant so the FastAPI Body() defaults can use it
# without depending on the ltx_pipelines import (which only resolves inside
# `initialize()` after we extend sys.path on the volume-mounted clone).
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

# Pin LTX-2 source code to a known-good commit so an upstream regression on
# `main` (e.g. another `multigpu` removal) cannot break our deploys.
# This is the head of `main` from 2026-05-11, against which our test suite
# passed end-to-end (i2v_url, i2v_base64, keyframe_interp, v2v_retake).
LTX2_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
LTX2_REPO_REVISION = "1799988521d5e9725a4fbd4533d0cc12f07c07ed"

# 2026-06-06: 22B IC-LoRA weight repos (verified present via HF model API on
# 2026-06-06). Each repo ships a single .safetensors trained on the LTX-2.3-22b
# distilled base. `ref0.5` in the filename = reference_downscale_factor 2 (the
# control/reference video is supplied at 0.5x the output resolution); the
# upstream ICLoraPipeline reads this factor out of the LoRA metadata at load
# time (ltx_pipelines.iclora_utils.read_lora_reference_downscale_factor).
#   * Motion-Track-Control — point/trajectory motion control.
#   * Union-Control        — unified Canny + Depth + Pose control. This is the
#                            camera/motion-control path that SUPERSEDES the
#                            non-loadable 19B camera LoRAs (see _snap_init note).
IC_LORA_MOTION_TRACK_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control"
IC_LORA_MOTION_TRACK_FILE = "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"
IC_LORA_MOTION_TRACK_REVISION = "572bb9c9a1ba3d8e8724cce69783ffc2422386db"
IC_LORA_UNION_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
IC_LORA_UNION_FILE = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
IC_LORA_UNION_REVISION = "b4d1c4d8c9e544e9bbbd6811bb4363708b6093ff"

# 2026-06-06: fp8 base — env-gated, default bf16 (quality stays the default).
# When LTX_FP8=1 we load the transformer from the OFFICIAL pre-quantized
# `Lightricks/LTX-2.3-fp8` repo (2 files: a dev fp8 + a distilled fp8, verified
# via the HF model API on 2026-06-06) and pass `quantization=QuantizationPolicy
# .fp8_cast()` to the pipeline ctors. fp8 is near-lossless on H100 (Hopper has
# native fp8 tensor cores; `fp8_cast` stores linear weights in float8_e4m3fn and
# upcasts during inference) — but it is STRICTLY OPT-IN; the bf16 dev/distilled
# checkpoints remain the default. We deliberately DO NOT use the `fp8-scaled-mm`
# policy: it requires `tensorrt_llm` (not in this image) and a `torch.ops
# .tensorrt_llm` custom op. `fp8_cast`'s downcast sd-op is `value.to(float8_e4m3fn)`,
# which is a no-op on the already-fp8 official files and a real downcast on bf16,
# so it composes correctly with the pre-quantized weights here.
LTX_FP8 = os.environ.get("LTX_FP8", "0") == "1"

# =============================================================================
# 2026-06-07 ZERO-COMPROMISE OPTIMIZATION STACK — shipped to prod.
# Verified bit-identical + 1.96x warm on RTX PRO 6000 (deploy/ltx/bench/STACK_VERIFIED.md).
# Three opt-in levers, all monkey-patches installed once in `_snap_init`:
#   1. emb-cache       — LRU output-cache on PromptEncoder.__call__ (default ON).
#   2. audio-skip      — per-request `skip_audio` (default OFF; audio KEPT).
#   3. persist-pipeline— keep stage transformer(s) GPU-resident across requests,
#                        RESOLUTION-KEYED + LRU-bounded (default `both`).
# =============================================================================

# --- LEVER 1: emb-cache ---------------------------------------------------
# Cache the Gemma text-embedding (EmbeddingsProcessorOutput) keyed by
# (prompt_text, encoder_version). The CONSTANT DEFAULT_NEGATIVE_PROMPT is the top
# hit — its full Gemma forward (~3.6-3.8s) is skipped on every warm request after
# the first. BIT-IDENTICAL: same text + same deterministic weights → same embedding
# tensor → unchanged diffusion. Default ON (safe). Negative always cacheable; the
# positive (idx 0) is cached only when enhance_prompt is OFF (enhancement rewrites
# prompts[0] via a sampled Gemma generate). See bench/EMB_CACHE_VERIFIED.md.
LTX_CACHE_TEXT_EMB = os.environ.get("LTX_CACHE_TEXT_EMB", "1") == "1"
_EMB_CACHE_MAX = int(os.environ.get("LTX_CACHE_TEXT_EMB_MAX", "8"))
_EMB_CACHE: dict = {}
_EMB_CACHE_ORDER: list = []
_EMB_CACHE_STATE: dict = {
    "enabled": LTX_CACHE_TEXT_EMB,
    "version": None,   # set per-request to the loaded-weights identity
    "hits": 0,
    "misses": 0,
}

# --- LEVER 2: audio-skip --------------------------------------------------
# PER-REQUEST `skip_audio` (default FALSE = audio KEPT, so existing audio reels are
# unaffected). When TRUE: wrap AudioDecoder.__call__ → None so the audio VAE decoder
# + vocoder (~1.3s) are NEVER built; the pipeline's `audio = self.audio_decoder(...)`
# becomes None → encode_video(audio=None) writes a clean video-only mp4. The audio
# DENOISE context (carried through the transformer) is untouched, so VIDEO pixels are
# byte-identical to the audio-on run. NOT a global kill-switch for everyone — the env
# only sets the default; the per-request flag is authoritative.
LTX_SKIP_AUDIO = os.environ.get("LTX_SKIP_AUDIO", "0") == "1"
_AUDIO_SKIP_STATE: dict = {"enabled": LTX_SKIP_AUDIO, "skipped": 0}

# --- LEVER 3: persist-pipeline (RESOLUTION-KEYED + LRU) --------------------
# DiffusionStage.__call__ rebuilds + re-fuses the transformer (stage_2 also fuses the
# distilled LoRA) on EVERY request, then frees it via `gpu_model(...)` → `.to("meta")`.
# ~2.6s (stage_1 build) + ~3.9s (stage_2 build/fuse) per warm request for only ~2.7s of
# real denoise. This lever builds the stage transformer ONCE per resolution and keeps it
# resident, skipping the per-request rebuild + LoRA refuse.
#
#   LTX_PERSIST_PIPELINE=off      → baseline (rebuild + free every call).
#   LTX_PERSIST_PIPELINE=stage2   → persist ONLY stage_2 (the 3.9s build/fuse slice;
#                                   the conservative, OOM-safe choice).
#   LTX_PERSIST_PIPELINE=both     → persist stage_1 AND stage_2 (the full 1.96x prize;
#                                   runtime-downgrades to stage2 → off on OOM).
#   LTX_PERSIST_PIPELINE=1/on     → alias for "both".
#
# ★ RESOLUTION-KEYED FIX (vs the bench's id(stage)-only key): the resident transformer
# is keyed by (id(stage), stage_role, height, width). The bench locked the resident to
# the FIRST request's shape, which would feed a wrong-shape module to a 768x1280 portrait
# reel after a 768x512 request (or vice-versa). Keying by resolution means each shape gets
# its OWN resident transformer and a 2nd request at a different resolution builds a fresh
# (correct-shape) resident instead of reusing the wrong one.
#
# ★ LRU EVICTION (bounds VRAM): each resident transformer is ~35 GB. Keeping every
# resolution resident would OOM even the 96 GB card. We cap the number of resident
# (stage, resolution) entries at LTX_PERSIST_LRU_MAX (default 2 resolution-pairs ⇒ up to
# 4 transformers in `both` mode, but typically 2 active resolutions). On overflow the
# OLDEST resident is `.to("meta")`-freed + cleaned up BEFORE the new build, so a cached +
# a rebuilt module never coexist (the original gpu_resident double-alloc OOM guard).
#
# CRITICAL OOM GUARD: the build-once path syncs + logs memory_allocated() before/after;
# the reuse path never builds a second copy. On a build OOM it frees the partial alloc,
# downgrades the mode (both→stage2→off), and falls back to the original build+free so the
# request still completes (verbatim logs, as the bench did).
_LTX_PERSIST_RAW = os.environ.get("LTX_PERSIST_PIPELINE", "both").lower().strip()
if _LTX_PERSIST_RAW in ("1", "on", "true", "yes", "both"):
    LTX_PERSIST_PIPELINE = "both"
elif _LTX_PERSIST_RAW in ("stage2", "stage_2", "2"):
    LTX_PERSIST_PIPELINE = "stage2"
else:
    LTX_PERSIST_PIPELINE = "off"

# LRU cap = number of distinct (stage, resolution) residents kept on the warm
# container. Each resident transformer is ~35 GB; a single resolution-pair
# (stage_1 + stage_2) = ~70.8 GB, peak ~72.4 GB on the 96 GB card (measured). TWO
# resolution-pairs would be ~140 GB → guaranteed OOM. So the cap is 2 = exactly ONE
# resolution-pair resident at a time. Switching resolution (e.g. 768x512 → 768x1280)
# LRU-evicts the previous resolution's pair BEFORE building the new one, holding peak
# at ~72 GB regardless of which resolution is active. 768x1280 portrait has ~2.5x the
# activation footprint of 512, so even one extra resident would push past the 95 GB
# card — VERIFIED by the 2026-06-07 ship run (cap=4 OOM'd at 1280; cap=2 keeps a
# single pair resident and switches cleanly). Do NOT raise above 2 on this card.
_PERSIST_LRU_MAX = int(os.environ.get("LTX_PERSIST_LRU_MAX", "2"))

# Free VRAM (GB) kept available before BUILDING a new resident transformer or
# loading the non-persist RetakePipeline (v2v). When free VRAM drops below this,
# LRU residents are evicted (→ pinned host weights, rebuilt on next miss) until
# the headroom is met. This is the PROACTIVE companion to the count cap above:
# the cap bounds the NUMBER of residents, this bounds FREE space so a different
# mode/resolution (or RetakePipeline, which can't reuse the persisted stages)
# never stacks onto a near-full 96 GB card and OOMs the forward-pass activations.
# Default 40 GB. This is sized to the COST OF THE THING WE'RE MAKING ROOM FOR,
# not to leftover slack: a single LTX-2.3 stage transformer is ~34 GB and a
# build/forward needs activation room on top. With 16 GB the guard saw the
# ~20 GB of steady-state slack after a warm pair, decided no eviction was
# needed, then tried to BUILD a ~34 GB transformer into 20 GB free → OOM
# (verified 2026-06-08: keyframe→i2v switch, 94.8/95 GB, 1.88 GB activation
# alloc failed). 40 GB forces eviction of a resident from the OTHER pipeline
# family BEFORE a cross-family/cross-resolution build, so there is always room
# to build the new ~34 GB stage + run its forward. Same-mode warm repeats are a
# cache HIT (no build, guard not consulted) so warm speed is preserved; only the
# first build after a switch pays the eviction. At heavy-base modes (i2v/t2v load
# the audio + Gemma stack, ~+17 GB) this naturally self-limits to one resident
# stage at a time, which is the most the 96 GB card can hold alongside them.
_VRAM_HEADROOM_GB = float(os.environ.get("LTX_VRAM_HEADROOM_GB", "40"))

# Resident-transformer registry (lives on the warm container). An LRU OrderedDict keyed
# by (id(stage), role, height, width) → the built transformer kept on GPU. Guarded so a
# cached module + a rebuilt module can never coexist.
import collections as _collections
_PERSIST_TRANSFORMERS: "_collections.OrderedDict" = _collections.OrderedDict()
# Current request's shape, set at the top of `_process_video` so the _transformer_ctx
# wrapper (which only receives **kwargs) can build the cache key. (stage, role) alone is
# resolution-blind — this carries the (height, width) the wrapper needs.
_PERSIST_CURRENT_SHAPE: dict = {"height": None, "width": None}
_PERSIST_STATE: dict = {
    "mode": LTX_PERSIST_PIPELINE,   # may be downgraded at runtime on OOM
    "stage1_hits": 0,
    "stage2_hits": 0,
    "stage1_builds": 0,
    "stage2_builds": 0,
    "evictions": 0,
    "oom_fallback": False,
    "lru_max": _PERSIST_LRU_MAX,   # activation-aware cap, set per-request
}


def _free_vram_gb() -> float:
    """Free GPU VRAM in GB via the CUDA driver. Returns +inf when CUDA is
    unavailable or the query fails, so headroom checks never block in those
    cases. Reads mem_get_info()[0] (free bytes) which reflects what the caching
    allocator can hand out — accurate even with expandable_segments:True."""
    try:
        import torch as _t
        if not _t.cuda.is_available():
            return float("inf")
        return _t.cuda.mem_get_info()[0] / (1024 ** 3)
    except Exception:
        return float("inf")


def _evict_lru_resident() -> bool:
    """Module-level LRU eviction of the oldest persisted transformer. Mirrors
    the closure `_evict_one_lru` (which is not visible outside _gpu_init) so the
    retake path and the build-miss path can both free residents. Moves the module
    to meta, then gc.collect() → empty_cache() → synchronize() so the freed CUDA
    blocks actually return to the allocator (ref cycles in the pipeline need gc
    before empty_cache reclaims anything). Returns False when nothing to evict."""
    try:
        old_key, old_mod = _PERSIST_TRANSFORMERS.popitem(last=False)
    except KeyError:
        return False
    try:
        old_mod.to("meta")
    except Exception:
        pass
    _PERSIST_STATE["evictions"] += 1
    print(f"   [PERSIST] 🗑️  headroom evict resident {old_key}; freeing")
    try:
        import gc as _gc
        import torch as _t
        _gc.collect()
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
            _t.cuda.synchronize()
    except Exception:
        pass
    return True


def _ensure_vram_headroom(required_gb: float) -> None:
    """Proactively evict LRU residents until at least `required_gb` of VRAM is
    free. No-op on the fast path (enough headroom already → no eviction, warm
    same-mode speed preserved). Used before BUILDING a new resident transformer
    and before loading the non-persist RetakePipeline (v2v). Transient: does NOT
    change _PERSIST_STATE['mode']; the next same-mode call rebuilds/reuses
    normally. If residents run out before headroom is met, the existing reactive
    OOM fallback is the last-resort net."""
    before = _free_vram_gb()
    if before >= required_gb:
        return
    print(f"   [PERSIST] headroom {before:.1f}GB < {required_gb:.1f}GB target "
          f"-> evicting residents (have {len(_PERSIST_TRANSFORMERS)})")
    while _free_vram_gb() < required_gb and _PERSIST_TRANSFORMERS:
        if not _evict_lru_resident():
            break
    print(f"   [PERSIST] headroom now {_free_vram_gb():.1f}GB "
          f"(residents={len(_PERSIST_TRANSFORMERS)})")


def _free_all_residents() -> None:
    """Evict every persisted transformer. Used before the single-stage
    RetakePipeline (v2v), which cannot reuse the two-stage residents and needs
    the room for its own ~35 GB weights + forward."""
    n = len(_PERSIST_TRANSFORMERS)
    if n == 0:
        return
    print(f"   [PERSIST] freeing ALL {n} residents (v2v needs the room)")
    while _PERSIST_TRANSFORMERS:
        if not _evict_lru_resident():
            break


# VAE-decode tiling. The decode of the final latent back to pixels is the
# activation spike that scales with height*width*frames (the ~24 GB term in the
# 2*35+24=94 GB OOM). LTX's TilingConfig.default() = spatial 768px/64 overlap,
# temporal 80 frames/24 overlap. SMALLER tiles shrink the decode peak (bf16-exact
# up to overlap blend) — letting more transformers stay resident. Env-tunable so
# the tile size can be tuned/swept without code changes; per-request override
# (tile_px / temporal_frames) lets a sweep run in one warm container.
_VAE_TILE_PX = int(os.environ.get("LTX_VAE_TILE_PX", "768"))
_VAE_TILE_OVERLAP = int(os.environ.get("LTX_VAE_TILE_OVERLAP", "64"))
_VAE_TEMPORAL_FRAMES = int(os.environ.get("LTX_VAE_TEMPORAL_FRAMES", "80"))
_VAE_TEMPORAL_OVERLAP = int(os.environ.get("LTX_VAE_TEMPORAL_OVERLAP", "24"))


def _make_tiling_config(tile_px=None, temporal_frames=None):
    """Build a VAE TilingConfig from env defaults with optional per-request
    overrides. Clamps to LTX's validity rules (spatial: >=64 & /32, overlap /32
    & < tile; temporal: >=16 & /8, overlap /8 & < tile) so a swept value can
    never raise. Falls back to TilingConfig.default() on any import/build error."""
    try:
        from ltx_core.model.video_vae import TilingConfig
        from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig
        tp = int(tile_px) if tile_px else _VAE_TILE_PX
        to = _VAE_TILE_OVERLAP
        tf = int(temporal_frames) if temporal_frames else _VAE_TEMPORAL_FRAMES
        tov = _VAE_TEMPORAL_OVERLAP
        tp = max(64, (tp // 32) * 32)
        to = min(max(0, (to // 32) * 32), tp - 32)
        tf = max(16, (tf // 8) * 8)
        tov = min(max(0, (tov // 8) * 8), tf - 8)
        return TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=tp, tile_overlap_in_pixels=to),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=tf, tile_overlap_in_frames=tov),
        )
    except Exception as _e:
        print(f"   [VAE-TILE] config build failed ({_e}); falling back to default()")
        from ltx_core.model.video_vae import TilingConfig
        return TilingConfig.default()


# Per-stage transformer is ~35 GB resident. The two-stage forward also needs an
# ACTIVATION + audio/VAE/text working set that scales with height*width*frames
# (measured ~24 GB at 1280x768x97, ~9-10 GB at 512x768x97). On a 96 GB card you
# cannot hold BOTH 35 GB stage transformers resident AND run a high-res forward:
# 2*35 + 24 = 94 GB -> OOM (verified 2026-06-08: warm i2v at 1280x768 hit
# 93.96/95 GB with 2 residents). Cold generation only fits because the stages
# build sequentially (stage_2 isn't resident while stage_1 runs). So the resident
# cap MUST be activation-aware: keep as many stage transformers resident as fit
# alongside the projected forward working set. At 1280x768x97 -> 1 (rotates the
# two stages, slower warm but never OOM); at <=512 height -> 2 (both resident,
# fast warm). Never exceed _PERSIST_LRU_MAX. LTX_VRAM_USABLE_GB / LTX_XFMR_GB let
# the math be retuned without code changes.
_VRAM_USABLE_GB = float(os.environ.get("LTX_VRAM_USABLE_GB", "91"))
_XFMR_GB = float(os.environ.get("LTX_XFMR_GB", "35"))


def _max_residents_for(height: int, width: int, num_frames: int) -> int:
    """How many ~35 GB stage transformers can stay resident alongside this
    request's projected activation working set. Clamped to [1, _PERSIST_LRU_MAX]."""
    try:
        mpx_frames = (float(height) * float(width) * float(num_frames)) / 1.0e6
        extras_gb = 0.25 * mpx_frames + 4.0  # activation + audio/VAE/text + margin
        n = int((_VRAM_USABLE_GB - extras_gb) / _XFMR_GB)
    except Exception:
        n = 1
    return max(1, min(_PERSIST_LRU_MAX, n))


def _purge_stale_residents(height: int, width: int, cap: int) -> None:
    """Run at the START of every request, BEFORE any allocation. Drops residents
    whose resolution != this request's (a different-resolution resident can't be
    reused and only burns ~35 GB — e.g. a 512-batch warm container hit by a 1280
    request, which used to OOM the first call), then trims any remaining excess to
    the activation-aware `cap`. Same-resolution residents within the cap are kept
    (warm-speed preserved). Frees with gc + empty_cache + synchronize so the VRAM
    is actually reclaimed before the forward builds anything."""
    removed = 0
    for k in list(_PERSIST_TRANSFORMERS.keys()):
        # key = (id(stage), role, height, width)
        if (k[2], k[3]) != (height, width):
            mod = _PERSIST_TRANSFORMERS.pop(k, None)
            if mod is not None:
                try:
                    mod.to("meta")
                except Exception:
                    pass
                _PERSIST_STATE["evictions"] += 1
                removed += 1
    while len(_PERSIST_TRANSFORMERS) > max(0, cap):
        if not _evict_lru_resident():
            break
        removed += 1
    if removed:
        print(f"   [PERSIST] purged {removed} stale/excess residents before forward "
              f"(res={width}x{height}, cap={cap}, residents={len(_PERSIST_TRANSFORMERS)})")
        try:
            import gc as _gc
            import torch as _t
            _gc.collect()
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
                _t.cuda.synchronize()
        except Exception:
            pass

# 2026-06-06: SageAttention-3 (Blackwell FP4 attention) toggle. Default OFF.
# When "1", _gpu_init monkey-patches LTX's `Attention.forward` so the UNMASKED
# self-attention path (attn1 / audio_attn1 — q==k==v, no context, no mask)
# routes through `sageattn3_blackwell(q, k, v, is_causal=False)` (thu-ml
# SageAttention3, FP4, sm_120 only). Masked cross-attention (attn2, with
# `video.context_mask`) and the audio<->video cross-attn stay on the original
# attention_function (SDPA) because SageAttention rejects arbitrary attn_mask.
#
# HARD REQUIREMENT (verified 2026-06-06 against thu-ml/SageAttention
# sageattention3_blackwell/README.md + jt-zhang/SageAttention3 HF card):
#   * GPU: Blackwell sm_120 ONLY (RTX PRO 6000 qualifies).
#   * torch>=2.8.0 (our build resolves torch 2.12.0+cu130 — OK).
#   * CUDA toolkit >=12.8 to BUILD (setup.py uses torch.utils.cpp_extension,
#     which asserts the container's nvcc matches torch's compiled CUDA = 13.0).
#     The prod `ltx2` image base is `nvidia/cuda:12.4.0-devel` → nvcc 12.4 →
#     Sage3 source build FAILS with the SAME version-mismatch that already
#     breaks flash-attn (see W7 note below). Building Sage3 therefore requires
#     a CUDA-13 devel image base. We DO NOT change the prod image base here;
#     the Sage3 install + CUDA-13 base live ONLY in the bench variant
#     (deploy/ltx/bench/sage/ltx2_sage_model.py, app `ltx2-sage-bench`). Prod
#     stays bf16-SDPA on the 12.4 image until the bench validates Sage3.
# No prebuilt wheel exists and the HF repo is gated; the bench builds from the
# thu-ml `sageattention3_blackwell` source dir at image-build time.
#
# 2026-06-06 SAGE-2.2 ADDITION: SageAttention-2.2.0 (INT8-QK + FP16-PV,
# `sageattn_qk_int8_pv_fp16_cuda`) is wired as a SEPARATE opt-in branch,
# selected by `LTX_SAGE_ATTN=2` (default OFF). Unlike SA3 FP4 (value "1", which
# is build-blocked on the public sm_120 toolchain — see SAGE3_BENCHMARK.md),
# SA2.2 is the PROVEN Blackwell path (~30-35% faster diffusion on RTX 5090,
# measured) and runs on sm_120 today. It is benchmarked in the bench variant
# (deploy/ltx/bench/sage/ltx2_sage22_model.py, app `ltx2-sage22-bench`,
# LTX_SAGE_ATTN=2, CUDA-13 base). This branch is present so prod can be flipped
# to SA2.2 LATER once the bench validates speedup + frame quality — but it is
# NOT the default and requires (a) flipping the prod env `LTX_SAGE_ATTN` to "2"
# AND (b) a CUDA-13 prod image base with sageattention==2.2.0 installed (the
# 12.4 prod image cannot build it). Both are deliberate, deferred changes.
#
# Flag semantics (string-valued):
#   LTX_SAGE_ATTN=0  → bf16-SDPA (default, unchanged prod behavior).
#   LTX_SAGE_ATTN=1  → SA3 FP4 self-attn (build-blocked; bench-only).
#   LTX_SAGE_ATTN=2  → SA2.2 INT8-QK/FP16-PV self-attn (opt-in; mask-safe SDPA
#                      cross-attn + per-call try/except → SDPA fallback).
_LTX_SAGE_ATTN_MODE = os.environ.get("LTX_SAGE_ATTN", "0").strip()
LTX_SAGE_ATTN = _LTX_SAGE_ATTN_MODE == "1"   # SA3 FP4 (legacy bench toggle)
LTX_SAGE22 = _LTX_SAGE_ATTN_MODE == "2"      # SA2.2 INT8-QK/FP16-PV (opt-in)

LTX_FP8_REPO = "Lightricks/LTX-2.3-fp8"
LTX_FP8_REVISION = "1d756cd27fa11c0896c4dfee093cd1bf36c7f7a1"
LTX_FP8_DEV_FILE = "ltx-2.3-22b-dev-fp8.safetensors"
LTX_FP8_DISTILLED_FILE = "ltx-2.3-22b-distilled-fp8.safetensors"

# 2026-06-06: community 22B LoRAs (selectable per-request via the `lora` body
# param on /generate). All trained on the LTX-2.3-22b-dev base, so they load as
# plain `loras=[LoraPathStrengthAndSDOps(path, strength, RENAMING_MAP)]` entries
# on the standard two-stage i2v pipeline — the SAME native pattern already used
# for `distilled_lora` and the IC-LoRA control path. Filenames + revisions
# verified via the HF model API on 2026-06-06 (do not edit without re-verifying).
#   * joyfox Transition  — shot transitions / first↔last-frame morphs. Generalizes
#                          to plain i2v per its model card.
#   * Licon VBVR-I2V     — video-reasoning i2v consistency (motion/temporal). The
#                          390K-R32 checkpoint is the latest/largest release of
#                          the three rank-32 variants in the repo.
# NOTE: `LiconStudio/LTX-2.3-Multiple-Subject-Reference` (MSR) is intentionally
# NOT wired — it is an IC-LoRA that requires the custom `ComfyUI-Licon-MSR`
# plugin to fold multiple reference images into a pseudo-video latent sequence;
# it does NOT fit the simple `loras=` path (see DELIVER notes). Deferred.
COMMUNITY_LORAS = {
    "transition": {
        "repo": "joyfox/LTX-2.3-Transition-LORA",
        "file": "ltx2.3-transition.safetensors",
        "revision": "685118b10fdc2aea63b47bbc6c395fe8c602db53",
        "strength": 0.9,
    },
    "vbvr": {
        "repo": "LiconStudio/Ltx2.3-VBVR-lora-I2V",
        "file": "Ltx2.3-Licon-VBVR-I2V-390K-R32.safetensors",
        "revision": "584c96e3dd5b211670e6b573b37fdd09d75d9aa4",
        "strength": 0.9,
    },
}

MODEL_DIR = Path("/models")
REPO_DIR = Path("/ltx2")

model_volume = modal.Volume.from_name("ltx2-model-cache", create_if_missing=True)
volumes = {MODEL_DIR: model_volume}

MINUTES = 60
HOURS = 60 * MINUTES

# =============================================================================
# Image Build Functions
# =============================================================================

def clone_and_setup_repo():
    """Clone LTX-2 repo (pinned commit) and install packages.

    Why the various workarounds:

    1. We clone a PINNED commit (LTX2_REPO_REVISION) rather than `main` so
       that an upstream regression on `main` cannot break our deploys overnight.
    2. The pinned commit ships with a broken import in
       ``ltx_pipelines/utils/blocks.py`` that references a non-existent
       ``ltx_pipelines.multigpu`` subpackage (Lightricks/LTX-2 issue #216).
       We write a minimal stub so the import resolves; the missing class is
       only used as a type hint and is never instantiated in single-GPU paths.
    3. ``ltx-pipelines`` declares ``openimageio`` as a transitive dep and
       imports it eagerly in ``media_io.py``. We install it explicitly at
       image-build time so `pip install -e` resolution is not the only thing
       keeping it on PATH.

    CACHE_BUSTER_V8_REPO_PIN
    """
    import shutil
    import subprocess

    print(f"🚀 Cloning LTX-2 repository @ {LTX2_REPO_REVISION[:10]}...")
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)

    subprocess.run(
        ["git", "clone", LTX2_REPO_URL, str(REPO_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "checkout", LTX2_REPO_REVISION],
        check=True,
    )
    head_sha = subprocess.run(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"   ✅ Checked out {head_sha[:10]}")

    # ----------------------------------------------------------------------
    # Workaround for Lightricks/LTX-2 issue #216:
    # ltx_pipelines/utils/blocks.py imports DelegatingBuilder from a
    # `ltx_pipelines.multigpu` subpackage that doesn't ship in the OSS repo.
    # The class is only used as a type hint (never instantiated), so a
    # minimal stub satisfies the import without altering runtime behavior.
    # ----------------------------------------------------------------------
    multigpu_dir = (
        REPO_DIR / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "multigpu"
    )
    if not multigpu_dir.exists():
        print("🔧 Writing multigpu stub package (issue #216 workaround)...")
        multigpu_dir.mkdir(parents=True, exist_ok=True)
        (multigpu_dir / "__init__.py").write_text("")
        (multigpu_dir / "delegating_builder.py").write_text(
            '"""Stub for missing upstream ltx_pipelines.multigpu.delegating_builder.\n\n'
            'Only used as a type hint in ltx_pipelines.utils.blocks; never instantiated\n'
            'in single-GPU inference. See https://github.com/Lightricks/LTX-2/issues/216\n'
            '"""\n'
            'from __future__ import annotations\n\n'
            'from typing import Generic, TypeVar\n\n'
            '_T = TypeVar("_T")\n\n\n'
            'class DelegatingBuilder(Generic[_T]):\n'
            '    """Placeholder; the real multi-GPU builder is not part of the OSS release."""\n\n'
            '    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover\n'
            '        raise NotImplementedError(\n'
            '            "DelegatingBuilder is a stub; multi-GPU code paths are not "\n'
            '            "available in the open-source LTX-2 release."\n'
            '        )\n'
        )
        print("   ✅ multigpu stub written")
    else:
        print("   ✓ multigpu package already exists, skipping stub")

    print("📦 Installing ltx-core...")
    subprocess.run(
        ["pip", "install", "-e", str(REPO_DIR / "packages" / "ltx-core")],
        check=True,
    )

    print("📦 Installing matching torchvision (MUST match torch ABI)...")
    torch_version = subprocess.run(
        ["python", "-c", "import torch; print(torch.__version__.split('+')[0])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"   Detected torch version: {torch_version}")
    subprocess.run(["pip", "uninstall", "-y", "torchvision"], check=False)
    subprocess.run(
        ["pip", "install", "--no-cache-dir", "torchvision"],
        check=True,
    )

    print("🔍 Verifying torch/torchvision versions...")
    subprocess.run(
        [
            "python", "-c",
            "import torch, torchvision; "
            "print(f'torch: {torch.__version__}, torchvision: {torchvision.__version__}')",
        ],
        check=True,
    )

    print("📦 Installing ltx-pipelines (incl. openimageio for media_io.py)...")
    # `openimageio` is imported eagerly at the top of
    # ltx_pipelines/utils/media_io.py, so without it ANY pipeline import fails.
    # It's listed as a transitive dep of ltx-pipelines, but we install it up
    # front so the pip resolver doesn't silently leave it out.
    subprocess.run(
        ["pip", "install", "openimageio"],
        check=True,
    )
    subprocess.run(
        ["pip", "install", "-e", str(REPO_DIR / "packages" / "ltx-pipelines")],
        check=True,
    )

    print("📦 Pinning transformers==4.57.6 (required for Gemma3 rope_local_base_freq)...")
    subprocess.run(["pip", "install", "transformers==4.57.6"], check=True)

    print("🔍 Verifying transformers + OpenImageIO + ltx imports...")
    subprocess.run(
        [
            "python", "-c",
            "import transformers, OpenImageIO; "
            "from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline; "
            "from ltx_pipelines.retake import RetakePipeline; "
            "from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline; "
            "print(f'transformers: {transformers.__version__}'); "
            "print(f'OpenImageIO: {OpenImageIO.__version__}'); "
            "print('✅ All pipeline imports resolved')",
        ],
        check=True,
    )

    print("✅ LTX-2 packages installed successfully (pinned commit)")


def download_models():
    """Download LTX-2.3 weights and Gemma text encoder.

    Strategy: commit incrementally after each big artifact so we never try to
    publish ~50GB in a single Modal volume commit (which triggers DATA_LOSS:
    'failed to publish commit to server'). Skips already-downloaded files so
    re-runs are cheap and resume on failure.
    CACHE_BUSTER_V6_INCREMENTAL_COMMITS
    """
    from huggingface_hub import hf_hub_download, snapshot_download  # snapshot_download used for Gemma
    import os

    model_volume.reload()

    hf_cache = str(MODEL_DIR / "huggingface")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache

    ltx_model_path = MODEL_DIR / "ltx2"
    gemma_path = MODEL_DIR / "gemma"
    ltx_model_path.mkdir(parents=True, exist_ok=True)
    gemma_path.mkdir(parents=True, exist_ok=True)

    OLD_WEIGHT_FILES = [
        "ltx-2-19b-dev.safetensors",
        "ltx-2-19b-distilled-lora-384.safetensors",
        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "ltx-2-temporal-upscaler-x2-1.0.safetensors",
    ]
    cleaned_any = False
    for old_file in OLD_WEIGHT_FILES:
        old_path = ltx_model_path / old_file
        if old_path.exists():
            print(f"🗑️ Deleting old weight: {old_file}")
            old_path.unlink()
            cleaned_any = True
    if cleaned_any:
        model_volume.commit()

    # NOTE: Lightricks/LTX-2.3 is a checkpoint-only repo (just .safetensors files
    # at the root). It does NOT contain a diffusers-style layout with
    # model_index.json or transformer/vae/text_encoder/scheduler/tokenizer/etc.
    # subdirs. Inference is driven by the ltx-core / ltx-pipelines packages,
    # which load these .safetensors files directly by path.
    # 2026-05-23 quality upgrade — add the v1.1 distilled LoRA published by
    # Lightricks. README pins it as the recommended stage-2 refinement LoRA
    # for 2.3, with measurable improvements in fast-motion stability and
    # prompt adherence over the v1.0 file. We still download the v1.0 file so
    # rollback is one-line. See modal/deploy/QUALITY_NOTES.md.
    LTX_SINGLE_FILES = [
        "ltx-2.3-22b-dev.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        "ltx-2.3-22b-distilled-lora-384.safetensors",
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        # 2026-06-06: standalone distilled checkpoint (~46 GB) — REQUIRED by
        # ICLoraPipeline (ltx_pipelines.ic_lora @ pinned commit 1799988), which
        # is built on the distilled base, NOT the dev checkpoint. Also unblocks
        # mode='preview' (DistilledPipeline) which previously 503'd because this
        # file was never downloaded. Verified present in the weights repo at
        # MODEL_REVISION via the HF model API.
        "ltx-2.3-22b-distilled-1.1.safetensors",
    ]
    for fname in LTX_SINGLE_FILES:
        target = ltx_model_path / fname
        if target.exists() and target.stat().st_size > 0:
            print(f"✓ Already present: {fname}")
            continue
        print(f"📥 Downloading {fname} from {MODEL_NAME}...")
        hf_hub_download(
            repo_id=MODEL_NAME,
            revision=MODEL_REVISION,
            filename=fname,
            local_dir=str(ltx_model_path),
        )
        size_mb = target.stat().st_size / 1024 / 1024 if target.exists() else 0
        print(f"   ✅ Got {fname} ({size_mb:.1f} MB) — committing...")
        model_volume.commit()

    # 2026-06-06: fp8 base weights — downloaded ONLY when LTX_FP8=1 so the
    # default bf16 build is not bloated by ~22GB of unused fp8 files. Same
    # per-file incremental-commit pattern. The dev fp8 backs mode='default'/'hq'/
    # KF-interp; the distilled fp8 backs mode='preview' + IC-LoRA when fp8 is on.
    if LTX_FP8:
        print("🔻 LTX_FP8=1 — downloading official pre-quantized fp8 weights...")
        for fname in (LTX_FP8_DEV_FILE, LTX_FP8_DISTILLED_FILE):
            target = ltx_model_path / fname
            if target.exists() and target.stat().st_size > 0:
                print(f"✓ Already present: {fname}")
                continue
            print(f"📥 Downloading {fname} from {LTX_FP8_REPO}...")
            hf_hub_download(
                repo_id=LTX_FP8_REPO,
                revision=LTX_FP8_REVISION,
                filename=fname,
                local_dir=str(ltx_model_path),
            )
            size_mb = target.stat().st_size / 1024 / 1024 if target.exists() else 0
            print(f"   ✅ Got {fname} ({size_mb:.1f} MB) — committing...")
            model_volume.commit()
    else:
        print("⏭️  LTX_FP8=0 — skipping fp8 weight download (bf16 default).")

    # 2026-06-06: community LoRAs (transition + VBVR). Small single-file LoRAs,
    # always downloaded (hundreds of MB each) so they are selectable per-request
    # via the /generate `lora` param without a redeploy. Incremental-commit
    # pattern matches the IC-LoRA + single-file blocks above.
    community_lora_path = MODEL_DIR / "community_loras"
    community_lora_path.mkdir(parents=True, exist_ok=True)
    for lora_key, cfg in COMMUNITY_LORAS.items():
        target = community_lora_path / cfg["file"]
        if target.exists() and target.stat().st_size > 0:
            print(f"✓ Already present: {cfg['file']} ({lora_key})")
            continue
        print(f"📥 Downloading community LoRA {cfg['file']} ({lora_key}) from {cfg['repo']}...")
        hf_hub_download(
            repo_id=cfg["repo"],
            revision=cfg["revision"],
            filename=cfg["file"],
            local_dir=str(community_lora_path),
        )
        size_mb = target.stat().st_size / 1024 / 1024 if target.exists() else 0
        print(f"   ✅ Got {cfg['file']} ({size_mb:.1f} MB) — committing...")
        model_volume.commit()

    gemma_marker = gemma_path / "config.json"
    if not gemma_marker.exists():
        print(f"📥 Downloading Gemma text encoder: {GEMMA_MODEL}...")
        snapshot_download(
            repo_id=GEMMA_MODEL,
            revision=GEMMA_REVISION,
            local_dir=str(gemma_path),
        )
        print("   ✅ Gemma downloaded — committing...")
        model_volume.commit()
    else:
        print("✓ Gemma already present, skipping")

    # 2026-06-06: 22B IC-LoRA weights (Motion-Track + Union control). Mirrors
    # the per-file incremental-commit pattern above so a mid-download failure
    # resumes cheaply and never tries to publish a giant single commit. These
    # are small LoRA files (hundreds of MB), one safetensors each, downloaded
    # into MODEL_DIR/ic_loras and loaded by the ICLoraPipeline factory below.
    ic_lora_path = MODEL_DIR / "ic_loras"
    ic_lora_path.mkdir(parents=True, exist_ok=True)
    IC_LORA_DOWNLOADS = [
        (IC_LORA_MOTION_TRACK_REPO, IC_LORA_MOTION_TRACK_FILE, IC_LORA_MOTION_TRACK_REVISION),
        (IC_LORA_UNION_REPO, IC_LORA_UNION_FILE, IC_LORA_UNION_REVISION),
    ]
    for repo_id, fname, revision in IC_LORA_DOWNLOADS:
        target = ic_lora_path / fname
        if target.exists() and target.stat().st_size > 0:
            print(f"✓ Already present: {fname}")
            continue
        print(f"📥 Downloading IC-LoRA {fname} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename=fname,
            local_dir=str(ic_lora_path),
        )
        size_mb = target.stat().st_size / 1024 / 1024 if target.exists() else 0
        print(f"   ✅ Got {fname} ({size_mb:.1f} MB) — committing...")
        model_volume.commit()

    print("\n📂 Final model directory contents:")
    for item in MODEL_DIR.iterdir():
        if item.is_dir():
            print(f"   📁 {item.name}/")
        else:
            size_mb = item.stat().st_size / 1024 / 1024
            print(f"   📄 {item.name} ({size_mb:.1f} MB)")

    model_volume.commit()
    print("✅ All models downloaded and committed to volume")


# =============================================================================
# Modal Image Definition
# =============================================================================

ltx2_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "git", "cmake", "build-essential", "clang",
        "libgl1-mesa-glx", "libglib2.0-0", "libopengl0", "libglx0",
        "ffmpeg", "libsm6", "libxext6",
    )
    .pip_install("uv")
    .pip_install(
        # Base dependencies (ltx-core will install torch~=2.7)
        "einops",
        "numpy<2",
        "safetensors",
        "accelerate>=1.0.0",
        "scipy>=1.14",
        # Pipeline dependencies
        "av",
        "tqdm",
        "pillow",
        # Additional
        "transformers==4.57.6",
        "huggingface_hub>=0.27.0",
        "hf_transfer",
        "fastapi",
        "httpx",
        "requests",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/models/huggingface",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # 2026-06-07: REVERTED gpu_resident → cpu_pinned. gpu_resident benched
        # great on ephemeral `modal run` (no snapshots, 48-54GB) BUT OOM'd under
        # the REAL deploy (memory-snapshot restore + _gpu_init registry rebuild
        # double-allocates → 94GB/95GB → CUDA OOM). cpu_pinned is the validated,
        # working zero-compromise lever (−13-15s vs disk-read, no quant, no OOM).
        # 2026-06-07 UPDATE: the snapshot double-alloc OOM is FIXED + validated under
        # real snapshots (peak 65.9 GB even under STREAM_OVER=40, warm −34%, output
        # bit-identical — see bench/gpures/DOUBLE_ALLOC_FIX.md). gpu_resident_registry.py
        # now has a hard VRAM-headroom guard that refuses to keep resident any entry
        # that could be LoRA-fusion-doubled past usable VRAM, so the ~39 GB transformer
        # always streams (which is where the −30% win actually comes from — small
        # high-frequency components resident + disk-read elimination, NOT a resident
        # transformer). Prod default LEFT on cpu_pinned here; to flip, set
        # LTX_REGISTRY=gpu_resident (the registry code + _gpu_init rebuild are ready).
        # 2026-06-07: REVERTED gpu_resident → cpu_pinned. The clean all-configs sweep
        # showed gpu_resident TIES cpu_pinned (~20.7 vs 20.8s warm) — the earlier
        # "−34-37%" was measured vs a STALE ~30s cpu_pinned baseline that didn't
        # reproduce. gpu_resident just costs +8.5GB VRAM (48 vs 39.5GB) for ZERO speed
        # gain. Best zero-compromise config = cpu_pinned + FBCache-on (17s, 39.5GB,
        # bit-identical registry weights). gpu_resident code stays ready (no harm), unused.
        "LTX_REGISTRY": "cpu_pinned",
        # gpu_resident budget knobs (only read when LTX_REGISTRY=gpu_resident):
        # 2026-06-07: corrected 40 -> 23. STREAM_OVER=40 was the benchmark's mistaken
        # recommendation that promoted the 39 GB transformer to resident and OOM'd prod.
        # 23 streams Gemma + the transformer, keeps only the small components resident
        # (~8.5 GB), ~48 GB peak, same −30% win. (Inert under the cpu_pinned default.)
        "LTX_GPU_RESIDENT_STREAM_OVER_GB": "23",
        "LTX_GPU_RESIDENT_MAX_GB": "80",
        # 2026-06-06: fp8 base toggle, default OFF (bf16 stays the quality
        # default). Set to "1" on the image env to (a) make download_models()
        # fetch the official fp8 weights at build time AND (b) make the runtime
        # pipeline ctors load fp8 + QuantizationPolicy.fp8_cast(). Flipping this
        # forces a full image rebuild (re-runs download_models) by design — fp8
        # is a deliberate, infrequent build-time choice, not a per-request lever.
        "LTX_FP8": "0",
        # 2026-06-07 zero-compromise optimization stack defaults (verified
        # bit-identical + 1.96x, bench/STACK_VERIFIED.md):
        #   emb-cache ON (bit-identical, eliminates the 3.6-3.8s prompt_encoder
        #   block on every warm request after the first);
        #   persist-pipeline `both` (resolution-keyed + LRU-bounded; the ~6s prize,
        #   72GB on the 96GB card with 1 resolution resident, OOM-guard downgrades);
        #   audio-skip default OFF (per-request `skip_audio` opt-in — audio reels
        #   are unaffected).
        "LTX_CACHE_TEXT_EMB": "1",
        "LTX_PERSIST_PIPELINE": "both",
        # cap=2 = ONE resolution-pair resident (~72 GB peak). The 2026-06-07 ship
        # run proved cap=4 (two resolution-pairs) OOMs at 768x1280 portrait
        # (~140 GB > 95 GB). Switching resolution LRU-evicts the old pair first.
        "LTX_PERSIST_LRU_MAX": "2",
        "LTX_SKIP_AUDIO": "0",
    })
    # NOTE 2026-05-23 K5: we DELIBERATELY do not set TORCHINDUCTOR_CACHE_DIR
    # here, because adding env vars on the Image forces a full layer rebuild
    # (including `download_models`, which crashes on cold runs). Instead, the
    # cache dir is set inside `@modal.enter()` below — TorchInductor only
    # reads the env var lazily when a graph is being compiled, which happens
    # inside our pipeline factories AFTER @modal.enter() runs.
    .run_function(clone_and_setup_repo, gpu="any")  # Need GPU for torch installation
    # 2026-05-23 W7 revised: we removed the original `pip_install("flash-attn", ...)`
    # line for two reasons:
    #
    #   1. The PyPI `flash-attn` build fails in this image because the torch
    #      shipped by ltx-core's editable install is compiled against CUDA 13.0
    #      while the container's `nvcc` is 12.4 → `torch.utils.cpp_extension`
    #      raises `RuntimeError: The detected CUDA version (12.4) mismatches the
    #      version that was used to compile PyTorch (13.0)`. Fixing that would
    #      require either pinning a CUDA-12 torch (incompatible with ltx-core)
    #      or rebuilding flash-attn against torch's CUDA 13 headers (requires a
    #      CUDA-13-toolkit image base).
    #
    #   2. The PyPI `flash-attn` package wouldn't actually be USED by upstream
    #      anyway. `ltx_core/model/transformer/attention.py:8-19` only attempts
    #      `import flash_attn_interface` (the SEPARATE Hopper-FA3 distribution),
    #      never `flash_attn`. With neither package installed,
    #      `AttentionFunction.DEFAULT` falls through to `PytorchAttention` which
    #      calls `torch.nn.functional.scaled_dot_product_attention` — and SDPA
    #      on H100 already uses Flash kernels internally since PyTorch 2.0+.
    #
    # The actual speedup win lives in the SDPA backend toggle inside
    # @modal.enter() (torch.backends.cuda.enable_flash_sdp(True) etc.), which
    # we keep — that one DOES require no pip install. The defensive monkey-
    # patch `_ltx_attn.flash_attn_interface = None` in initialize() also stays
    # as a safety net against future env drift dropping FA3 into the env.
    .run_function(
        download_models,
        volumes=volumes,
    )
    # Ship the utils/ package (CPU-pinned + GPU-resident weight registries,
    # First-Block-Cache, custom guiders). Modal only auto-mounts the entry
    # script, so the helper package is declared explicitly; it resolves via the
    # repo-root sys.path bootstrap at the top of this file.
    .add_local_python_source("utils")
)

# =============================================================================
# Modal App
# =============================================================================

app = modal.App(
    name="ltx2-fast-inference",
    image=ltx2_image,
)


@app.cls(
    # 2026-06-06: switched H100 → RTX PRO 6000 (Blackwell, 96GB). Benchmarked
    # fastest of all cards (~30s warm, −29% vs H100), −45% $/clip (~$0.0254,
    # ~tied with L40S), bf16 output eye-verified equivalent to H100, peak 39.5GB
    # (huge headroom on 96GB). Also the only card with native fp8 + nvfp4 — so
    # once the fp8 `fp8_scaled_mm` path is fixed (see FP8_FIX_RESEARCH.md), this
    # same card gets the next speed/VRAM tier free. Modal string verified to provision.
    gpu="RTX-PRO-6000",
    timeout=30 * MINUTES,
    # S1: bump from 5 -> 20 min. Same-container reuse window. Keeps a warm
    # container alive across the gap between bursty requests so we don't
    # eat re-load + re-compile for follow-up prompts arriving 7-10 min apart.
    scaledown_window=20 * MINUTES,
    # S1: snapshot post-`_snap_init` state. Restore is O(seconds) on every
    # subsequent cold container instead of ~10-30s of imports + sys.path
    # munging + class binding. FREE — does not require any idle container.
    enable_memory_snapshot=True,
    # Deliberately NOT setting min_containers / buffer_containers — user
    # rejected the $96/day cost. The compile-cache fix (S2) + StateDictRegistry
    # (S5) handle the warm/cold gap without a permanent idle container.
    #
    # R3 (2026-05-23): cost ceiling — bound the worst-case bill from a runaway
    # retry storm or 15-keyframe burst. 8 × H100 ≈ $32/hr peak burn.
    # Do NOT raise above 16 without an explicit budget call. NOT setting
    # @modal.concurrent: peak VRAM per request is ~75 GB on an 80 GB H100,
    # so max_inputs=2 would OOM (44 GB shared weights + 2 × 31 GB activations
    # = 106 GB). CUDA kernels also serialize across Python threads on the
    # same device (single legacy default stream), so concurrent inputs would
    # 2× per-request latency with no throughput gain — opposite of the vLLM
    # continuous-batching pattern. See research_v2/R3_modal_concurrent/REPORT.md.
    max_containers=8,
    volumes=volumes,
)
class Model:
    """LTX-2.3 video generation with image-to-video and keyframe interpolation (Lightricks/LTX-2.3 weights)."""

    @modal.enter(snap=True)
    def _snap_init(self):
        """S1: CPU-only init captured into the memory snapshot.

        Anything CUDA-touching (torch.backends.cuda.*, building pipelines,
        moving tensors to GPU) MUST live in `_gpu_init` below. Importing
        torch is fine; touching CUDA isn't.
        """
        import sys

        print("=" * 60)
        print("🚀 LTX-2.3 SNAP INIT (will be captured in memory snapshot)")
        print("=" * 60)

        # --- 1. sys.path so ltx_core / ltx_pipelines resolve ---
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        if str(REPO_DIR / "packages" / "ltx-core" / "src") not in sys.path:
            sys.path.insert(0, str(REPO_DIR / "packages" / "ltx-core" / "src"))
        if str(REPO_DIR / "packages" / "ltx-pipelines" / "src") not in sys.path:
            sys.path.insert(0, str(REPO_DIR / "packages" / "ltx-pipelines" / "src"))

        # --- 2. Path bindings (pure Python, safe to snapshot) ---
        self.ltx_model_path = MODEL_DIR / "ltx2"
        self.gemma_path = MODEL_DIR / "gemma"
        # 2026-06-06 fp8 (opt-in): when LTX_FP8=1, point checkpoint_path +
        # distilled_checkpoint_path at the official pre-quantized fp8 files and
        # build a fp8_cast QuantizationPolicy passed to every pipeline ctor.
        # Default (LTX_FP8=0) keeps the bf16 dev/distilled checkpoints + no
        # quantization policy → bf16 full-quality is unchanged. fp8 is
        # near-lossless on H100 but strictly opt-in per the quality rule.
        if LTX_FP8:
            self.checkpoint_path = str(self.ltx_model_path / LTX_FP8_DEV_FILE)
        else:
            self.checkpoint_path = str(self.ltx_model_path / "ltx-2.3-22b-dev.safetensors")
        self.spatial_upsampler_path = str(self.ltx_model_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        # 2026-05-23 quality upgrade — point stage-2 refinement at the v1.1
        # distilled LoRA published by Lightricks (fast-motion + prompt-adherence
        # improvements over v1.0). Filename of the v1.0 file is kept downloaded
        # in `LTX_SINGLE_FILES` so we can flip back via this one-line edit.
        self.distilled_lora_path = str(self.ltx_model_path / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
        # 2026-05-23 W2: standalone distilled checkpoint for mode='preview' via
        # ltx_pipelines.distilled.DistilledPipeline. Self-contained ~46 GB file
        # used INSTEAD of dev base — has its own VAE / text-conditioner / Gemma
        # adaptor / DIT. ~11 denoising ops total (8 stage-1 + 3 stage-2).
        if LTX_FP8:
            self.distilled_checkpoint_path = str(self.ltx_model_path / LTX_FP8_DISTILLED_FILE)
        else:
            self.distilled_checkpoint_path = str(self.ltx_model_path / "ltx-2.3-22b-distilled-1.1.safetensors")
        # 2026-06-06: IC-LoRA weight paths (downloaded into MODEL_DIR/ic_loras by
        # download_models). Loaded by `_get_ic_lora_pipeline` via the upstream
        # ICLoraPipeline (distilled-base in-context LoRA control).
        self.ic_loras_path = MODEL_DIR / "ic_loras"
        self.ic_lora_motion_track_path = str(self.ic_loras_path / IC_LORA_MOTION_TRACK_FILE)
        self.ic_lora_union_path = str(self.ic_loras_path / IC_LORA_UNION_FILE)
        # 2026-06-06: community LoRA paths (downloaded into MODEL_DIR/community_loras
        # by download_models). Selected per-request via the /generate `lora` param
        # and appended to the pipeline's `loras=` list at the recommended strength.
        self.community_loras_path = MODEL_DIR / "community_loras"
        self.community_lora_paths = {
            key: str(self.community_loras_path / cfg["file"])
            for key, cfg in COMMUNITY_LORAS.items()
        }

        # --- 3. Pipeline classes + LTX core helpers ---
        # 2026-05-23 — also import `TI2VidTwoStagesHQPipeline` (Lightricks's
        # documented "highest quality" single-image path, res_2s sampler with
        # the distilled LoRA applied at both stages with separate strengths).
        # Used when /generate?mode=hq is requested. KF interp has no HQ variant
        # upstream yet, so multi-image requests stay on
        # KeyframeInterpolationPipeline.
        print("\n📦 Loading LTX-2 pipeline classes (CPU import)...")
        from functools import partial as _partial
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        # S5: StateDictRegistry cached on the instance so every pipeline ctor
        # shares the same weight cache. First constructor reads from disk; the
        # other 4+ get instant in-memory dict lookups.
        # S5 (gated): StateDictRegistry keeps weights cached across pipeline
        # ctors. Currently DISABLED by default (LTX_USE_REGISTRY=1 to enable)
        # because the cache lives on GPU and OOMs the second pipeline build.
        # When disabled we still pass `registry=None` to the ctors (the
        # upstream default → DummyRegistry → cold disk reload each time).
        from ltx_core.loader.registry import StateDictRegistry
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        from ltx_pipelines.retake import RetakePipeline
        from ltx_pipelines.utils.samplers import (
            gradient_estimating_euler_denoising_loop as _ge_euler_loop,
        )
        # 2026-05-23 W2: distilled preview pipeline (~11 total denoising ops, single-image i2v only).
        from ltx_pipelines.distilled import DistilledPipeline
        # 2026-06-06: IC-LoRA pipeline (verified present at pinned commit 1799988
        # as `ltx_pipelines.ic_lora.ICLoraPipeline`). Built on the standalone
        # distilled checkpoint; takes `video_conditioning: list[(path, strength)]`
        # control/reference videos plus optional image conditioning. Backs the new
        # mode='motion_track' and mode='union' /generate paths.
        from ltx_pipelines.ic_lora import ICLoraPipeline

        self.TI2VidTwoStagesPipeline = TI2VidTwoStagesPipeline
        self.TI2VidTwoStagesHQPipeline = TI2VidTwoStagesHQPipeline
        self.KeyframeInterpolationPipeline = KeyframeInterpolationPipeline
        self.RetakePipeline = RetakePipeline
        self.DistilledPipeline = DistilledPipeline
        self.ICLoraPipeline = ICLoraPipeline
        # 2026-05-23 Tier-1A — pin a gradient-estimating Euler loop at
        # ge_gamma=2.0 (upstream samplers.py:79-147, README:407-422). We
        # inject this onto the stage_1 instance of every default-mode
        # (Euler) pipeline below so CFG stage_1 uses the velocity-corrected
        # update rule. Matches vanilla quality at ~20 steps vs 30, so the
        # /generate endpoint default also drops 30 → 20 below. HQ pipeline
        # (res_2s sampler) untouched. Stage 2 (distilled LoRA, fixed
        # STAGE_2_DISTILLED_SIGMAS) also untouched — no benefit there.
        self._ge_loop = _partial(_ge_euler_loop, ge_gamma=2.0)
        # S5+R2: shared registry instance — single source-of-truth weight cache
        # across all pipeline variants (default, hq, kf, retake, preview, variant).
        # Mode selected via LTX_REGISTRY env var; see top of file for semantics.
        if _LTX_REGISTRY_MODE == "cpu_pinned":
            # Import here so the snapshot ONLY captures this branch when in use.
            from utils.cpu_pinned_registry import CpuPinnedStateDictRegistry
            self._sd_registry = CpuPinnedStateDictRegistry()
            print(f"   Registry mode: cpu_pinned (max={self._sd_registry._max_bytes / (1 << 30):.0f} GB host)")
        elif _LTX_REGISTRY_MODE == "gpu_resident":
            # 2026-06-06: keep transformer state-dicts RESIDENT on GPU (uses the
            # idle VRAM on the 96 GB RTX PRO 6000), stream Gemma from pinned CPU.
            # `.get()` returns zero-copy aliasing views → no per-request H2D copy.
            # Inherits CpuPinnedStateDictRegistry (→ DummyRegistry), so in-place
            # LoRA fusion + GPU _target_device are preserved. Over-budget entries
            # (Gemma ~24 GB) fall back to the inherited pinned-CPU streaming path.
            # OPT-IN ONLY — prod default stays cpu_pinned. See gpu_resident_registry.py.
            from utils.gpu_resident_registry import GpuResidentStateDictRegistry
            self._sd_registry = GpuResidentStateDictRegistry()
            print(
                f"   Registry mode: gpu_resident "
                f"(gpu_max={self._sd_registry._gpu_max_bytes / (1 << 30):.0f} GB resident, "
                f"stream_over={self._sd_registry._stream_over_bytes / (1 << 30):.0f} GB)"
            )
        elif _LTX_REGISTRY_MODE == "gpu":
            self._sd_registry = StateDictRegistry()
            print("   Registry mode: gpu (StateDictRegistry — ⚠️ OOM-prone, forensic use only)")
        else:
            self._sd_registry = None
            print("   Registry mode: off (no caching, every call reads from disk)")

        # --- 3b. fp8 quantization policy (opt-in, default None=bf16) ---
        # 2026-06-06: when LTX_FP8=1, build ONE QuantizationPolicy.fp8_cast()
        # and pass it to every pipeline ctor (all of them accept `quantization=`
        # at the pinned commit 1799988). fp8_cast stores transformer Linear
        # weights in float8_e4m3fn and upcasts during the forward — near-lossless
        # on H100 Hopper fp8. We use fp8_cast (NOT fp8_scaled_mm) because the
        # latter requires `tensorrt_llm`, which is not installed in this image.
        # Default (LTX_FP8=0) → policy stays None → bf16 full quality unchanged.
        if LTX_FP8:
            from ltx_core.quantization import QuantizationPolicy
            self._quantization = QuantizationPolicy.fp8_cast()
            print("   ⚙️  fp8 ENABLED (LTX_FP8=1): QuantizationPolicy.fp8_cast() on all pipeline ctors")
        else:
            self._quantization = None
            print("   ⚙️  fp8 disabled (LTX_FP8=0): bf16 full-quality default")

        # --- 4. LoRA config (pure dataclass) ---
        # Prepare distilled stage-2 LoRA config at upstream's recommended
        # `DEFAULT_LORA_STRENGTH = 1.0` (was previously 0.6 — under-refining
        # the upsampled stage-2 output).
        self.distilled_lora = [
            LoraPathStrengthAndSDOps(
                self.distilled_lora_path,
                1.0,
                LTXV_LORA_COMFY_RENAMING_MAP
            ),
        ]

        # 2026-06-06 — camera/motion-control path UPDATED. Superseding the
        # 2026-05-23 W3 investigation note:
        #
        # The camera/motion-control path is NOW the 22B IC-LoRA Union-Control
        # (`Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` — unified Canny + Depth
        # + Pose control), wired below via `_get_ic_lora_pipeline` and exposed as
        # /generate mode='union'. Motion-Track-Control is exposed as
        # mode='motion_track'. Both are loaded through the upstream
        # `ICLoraPipeline` (ltx_pipelines.ic_lora @ pinned commit 1799988), which
        # is built on the distilled base and accepts a control/reference video via
        # `video_conditioning`.
        #
        # The original 7 dolly/jib 19B camera-control LoRAs
        # (`LTX-2-19b-LoRA-Camera-Control-*`) REMAIN non-loadable on our 22B base:
        # the 19B and 22B DiTs have different per-layer widths, so 19B LoRA tensors
        # would throw in `fuse_loras._aggregate_deltas` on the first `addmm_`
        # shape mismatch. We DO NOT attempt to load any 19B LoRA. As of 2026-06-06
        # Lightricks still ships no `LTX-2.3-22b-LoRA-Camera-Control-*` dolly/jib
        # variants; the IC-LoRA Union-Control is the supported 22B substitute.

        # --- 5. Pipeline instance slots (lazy load in _gpu_init or first request) ---
        self._i2v_pipeline = None
        self._i2v_hq_pipeline = None
        self._kf_pipeline = None
        self._retake_pipeline = None
        self._distilled_preview_pipeline = None
        # 2026-06-06: IC-LoRA pipeline slots — one per control LoRA, lazily
        # built on first request to the corresponding /generate mode.
        self._ic_lora_motion_track_pipeline = None
        self._ic_lora_union_pipeline = None

        # --- 6. W8 variant cache ---
        # W8: runtime A/B variant cache for distilled-LoRA stage-1 / stage-2
        # strengths. Keyed by (pipeline_kind, effective_s1, effective_s2).
        # FIFO eviction at MAX_VARIANTS. Only consulted when a request passes
        # `distilled_lora_strength_stage_{1,2}` overrides; otherwise the
        # existing single-instance lazy loaders are used unchanged.
        self._variant_cache: dict = {}
        self._variant_cache_order: list = []

        # --- 6b. Community-LoRA pipeline cache (2026-06-06) ---
        # Keyed by (lora_key, kind, strength); FIFO eviction at 4. Only consulted
        # when a request passes the `lora` param to /generate; default path
        # (no `lora`) uses the single-instance lazy loaders unchanged.
        self._community_lora_cache: dict = {}
        self._community_lora_cache_order: list = []

        # =============================================================
        # 2026-06-07 OPTIMIZATION STACK — install the three lever wrappers ONCE.
        # All are pure CPU-side monkey-patches on upstream classes (no CUDA),
        # safe to capture in the snapshot. Each is idempotent + non-fatal.
        # =============================================================

        # --- LEVER 1: emb-cache (PromptEncoder.__call__ LRU output-cache) ---
        # Wrap PromptEncoder.__call__: on a cache HIT we skip BOTH the Gemma
        # forward AND the embeddings-processor; only MISSING prompts are forwarded
        # to the original encoder; results are reassembled in the caller's prompt
        # order so downstream is unchanged. Cache key: (prompt_text, encoder_version)
        # where encoder_version pins the loaded-weights identity so distinct
        # pipelines / weight reloads never alias a stale embedding. BIT-IDENTICAL:
        # same text + same deterministic bf16 weights → identical hidden states →
        # identical EmbeddingsProcessorOutput. Tensors are detach+cloned on store AND
        # on return so the pipeline can never mutate/free a buffer we alias. The
        # POSITIVE (idx 0) is cached only when enhancement is OFF (enhance rewrites
        # prompts[0] via a sampled Gemma .generate()); the negative is always safe.
        try:
            from ltx_pipelines.utils import blocks as _emb_blocks
            _PE_EMB = _emb_blocks.PromptEncoder
            if not getattr(_PE_EMB.__call__, "_emb_cache_wrapped", False):
                import torch as _t_ec

                _orig_pe_call_ec = _PE_EMB.__call__

                def _clone_out(out):
                    # EmbeddingsProcessorOutput is a NamedTuple. Clone field-
                    # AGNOSTICALLY via ._replace so we preserve EVERY field (incl.
                    # attention_mask) and stay robust to upstream schema changes.
                    cloned = {
                        name: (val.detach().clone() if _t_ec.is_tensor(val) else val)
                        for name, val in out._asdict().items()
                    }
                    return out._replace(**cloned)

                def _emb_cached_call(self_pe, prompts, **kw):
                    state = _EMB_CACHE_STATE
                    enabled = state["enabled"]
                    enhance = bool(kw.get("enhance_first_prompt", False))
                    ver = state["version"]
                    if (not enabled) or ver is None:
                        return _orig_pe_call_ec(self_pe, prompts, **kw)

                    prompts = list(prompts)
                    n = len(prompts)
                    results = [None] * n
                    miss_idx = []
                    miss_prompts = []
                    for i, p in enumerate(prompts):
                        cacheable = not (enhance and i == 0)
                        key = (p, ver) if cacheable else None
                        if key is not None and key in _EMB_CACHE:
                            try:
                                _EMB_CACHE_ORDER.remove(key)
                            except ValueError:
                                pass
                            _EMB_CACHE_ORDER.append(key)
                            results[i] = _clone_out(_EMB_CACHE[key])
                            state["hits"] += 1
                            print(f"   [EMB-CACHE] HIT  idx={i} chars={len(p)}")
                        else:
                            miss_idx.append(i)
                            miss_prompts.append(p)
                            state["misses"] += 1
                            print(f"   [EMB-CACHE] MISS idx={i} chars={len(p)}")

                    if miss_prompts:
                        sub_kw = dict(kw)
                        sub_kw["enhance_first_prompt"] = enhance and (0 in miss_idx) and (miss_idx[0] == 0)
                        sub_out = _orig_pe_call_ec(self_pe, miss_prompts, **sub_kw)
                        for j, i in enumerate(miss_idx):
                            out = sub_out[j]
                            results[i] = out
                            cacheable = not (enhance and i == 0)
                            if cacheable:
                                key = (prompts[i], ver)
                                _EMB_CACHE[key] = _clone_out(out)
                                _EMB_CACHE_ORDER.append(key)
                                while len(_EMB_CACHE_ORDER) > _EMB_CACHE_MAX:
                                    old = _EMB_CACHE_ORDER.pop(0)
                                    _EMB_CACHE.pop(old, None)
                    return results

                _emb_cached_call._emb_cache_wrapped = True
                _PE_EMB.__call__ = _emb_cached_call
                print("   [EMB-CACHE] PromptEncoder.__call__ cache wrapper installed "
                      f"(default {'ON' if LTX_CACHE_TEXT_EMB else 'OFF'}, maxsize={_EMB_CACHE_MAX})")
        except Exception as _e_emb:
            print(f"   ⚠️ emb-cache wrapper skipped (non-fatal): {_e_emb}")

        # --- LEVER 3: persist-pipeline (RESOLUTION-KEYED + LRU) ---
        # Wrap DiffusionStage._transformer_ctx: when persist is active for THIS
        # stage's role, build the transformer EXACTLY once per (stage, role,
        # height, width) and thereafter yield the cached module via a no-free
        # context manager — skipping the per-request rebuild + LoRA-refuse +
        # weight-stream + free. The resolution in the key prevents reusing a
        # wrong-shape resident across mixed-res requests (768x512 vs 768x1280).
        # An LRU cap (_PERSIST_LRU_MAX) evicts+frees the oldest resident BEFORE a
        # new build so VRAM stays bounded and a cached + rebuilt module never
        # coexist. OOM on a build → free partial + downgrade both→stage2→off +
        # fall back to the original build+free so the request still completes.
        try:
            from ltx_pipelines.utils import blocks as _persist_blocks
            from contextlib import contextmanager as _persist_cm
            _DS = _persist_blocks.DiffusionStage
            if not getattr(_DS._transformer_ctx, "_persist_wrapped", False):
                import torch as _t_p
                _orig_tctx = _DS._transformer_ctx

                def _role_active(role):
                    mode = _PERSIST_STATE["mode"]
                    if mode == "off" or role is None:
                        return False
                    if mode == "both":
                        return True
                    if mode == "stage2":
                        return role == "stage_2"
                    return False

                @_persist_cm
                def _resident_ctx(module):
                    # No-free: yield the cached resident transformer; do NOT move
                    # it to meta on exit (that is the whole point).
                    yield module

                def _streaming_active(self_ds):
                    # Streaming is gated by self._offload_mode (OffloadMode enum).
                    # Persist is incompatible with layer-offload streaming, so when
                    # offload is on we always fall back to the original ctx.
                    try:
                        om = getattr(self_ds, "_offload_mode", None)
                        if om is None:
                            return False
                        none_val = getattr(type(om), "NONE", None)
                        return om != none_val if none_val is not None else False
                    except Exception:
                        return False

                def _evict_one_lru():
                    # Evict + free the OLDEST resident (FIFO order of the
                    # OrderedDict) to bound VRAM before a new build.
                    try:
                        old_key, old_mod = _PERSIST_TRANSFORMERS.popitem(last=False)
                    except KeyError:
                        return
                    try:
                        old_mod.to("meta")
                    except Exception:
                        pass
                    _PERSIST_STATE["evictions"] += 1
                    print(f"   [PERSIST] 🗑️  LRU evicted resident {old_key} "
                          f"(cap={_PERSIST_LRU_MAX}); freeing")
                    try:
                        _persist_blocks.cleanup_memory()
                    except Exception:
                        pass

                def _persist_transformer_ctx(self_ds, **kwargs):
                    # Runtime signature: _transformer_ctx(self, **kwargs), called as
                    # self._transformer_ctx(video_tools=...). Forward kwargs verbatim
                    # to the original on any fallback path.
                    role = getattr(self_ds, "_persist_role", None)
                    if (not _role_active(role)) or _streaming_active(self_ds):
                        return _orig_tctx(self_ds, **kwargs)
                    # ★ Resolution-keyed: include the current request's (h, w) so a
                    # different-shape request never reuses a wrong-shape module.
                    h = _PERSIST_CURRENT_SHAPE.get("height")
                    w = _PERSIST_CURRENT_SHAPE.get("width")
                    key = (id(self_ds), role, h, w)
                    cached = _PERSIST_TRANSFORMERS.get(key)
                    if cached is not None:
                        # LRU touch (move to most-recent).
                        _PERSIST_TRANSFORMERS.move_to_end(key)
                        if role == "stage_1":
                            _PERSIST_STATE["stage1_hits"] += 1
                        else:
                            _PERSIST_STATE["stage2_hits"] += 1
                        _alloc = _t_p.cuda.memory_allocated() / (1 << 30) if _t_p.cuda.is_available() else 0.0
                        print(f"   [PERSIST] {role} REUSE resident transformer "
                              f"({w}x{h}, no rebuild/refuse)  cuda_alloc={_alloc:.2f}GB")
                        return _resident_ctx(cached)
                    # Build ONCE for this (stage, role, resolution). First make
                    # PROACTIVE room: evict LRU residents until LTX_VRAM_HEADROOM_GB
                    # is free, so a different-resolution/mode build never stacks onto
                    # a near-full card and OOMs the forward activations. Then keep the
                    # count cap as defense-in-depth (a cached + rebuilt module can
                    # never coexist).
                    _ensure_vram_headroom(_VRAM_HEADROOM_GB)
                    # Activation-aware cap (set per-request in _process_video):
                    # evict so that resident_count + this build stays within what
                    # the forward working set leaves room for. Falls back to the
                    # static LRU cap if unset.
                    _cap = _PERSIST_STATE.get("lru_max", _PERSIST_LRU_MAX)
                    while len(_PERSIST_TRANSFORMERS) >= _cap:
                        _evict_one_lru()
                    if _t_p.cuda.is_available():
                        _t_p.cuda.synchronize()
                        _before = _t_p.cuda.memory_allocated() / (1 << 30)
                    else:
                        _before = 0.0
                    print(f"   [PERSIST] {role} BUILD resident transformer (once for "
                          f"{w}x{h})  cuda_alloc_before={_before:.2f}GB  residents={len(_PERSIST_TRANSFORMERS)}")
                    try:
                        built = self_ds._build_transformer(**kwargs)
                    except (RuntimeError, _t_p.cuda.OutOfMemoryError) as _oom:
                        if "out of memory" in str(_oom).lower() or isinstance(_oom, _t_p.cuda.OutOfMemoryError):
                            print(f"   [PERSIST] ❌ OOM building resident {role}: {_oom}")
                            # CRITICAL: free EVERY resident (across all resolutions),
                            # not just the partial alloc — otherwise the old-resolution
                            # residents still pin ~70 GB and BOTH the in-request
                            # fallback below AND the _process_video_safe retry OOM
                            # again. Evicting all residents gives the fallback build +
                            # the retry a clean card.
                            for _ek in list(_PERSIST_TRANSFORMERS.keys()):
                                _em = _PERSIST_TRANSFORMERS.pop(_ek, None)
                                if _em is not None:
                                    try:
                                        _em.to("meta")
                                    except Exception:
                                        pass
                                    _PERSIST_STATE["evictions"] += 1
                                    print(f"   [PERSIST] 🧹 OOM-freed resident {_ek}")
                            try:
                                _persist_blocks.cleanup_memory()
                            except Exception:
                                pass
                            if _t_p.cuda.is_available():
                                try:
                                    _t_p.cuda.empty_cache()
                                    _t_p.cuda.synchronize()
                                except Exception:
                                    pass
                            _PERSIST_STATE["oom_fallback"] = True
                            if role == "stage_1" and _PERSIST_STATE["mode"] == "both":
                                _PERSIST_STATE["mode"] = "stage2"
                                print("   [PERSIST] ⬇️  downgraded mode both -> stage2 "
                                      "(stage_1 will rebuild+free per call)")
                            else:
                                _PERSIST_STATE["mode"] = "off"
                                print("   [PERSIST] ⬇️  downgraded mode -> off")
                            return _orig_tctx(self_ds, **kwargs)
                        raise
                    _PERSIST_TRANSFORMERS[key] = built
                    _PERSIST_TRANSFORMERS.move_to_end(key)
                    if role == "stage_1":
                        _PERSIST_STATE["stage1_builds"] += 1
                    else:
                        _PERSIST_STATE["stage2_builds"] += 1
                    if _t_p.cuda.is_available():
                        _t_p.cuda.synchronize()
                        _after = _t_p.cuda.memory_allocated() / (1 << 30)
                    else:
                        _after = 0.0
                    print(f"   [PERSIST] {role} resident transformer built + RETAINED "
                          f"({w}x{h})  cuda_alloc_after={_after:.2f}GB  (+{_after-_before:.2f}GB)")
                    return _resident_ctx(built)

                _persist_transformer_ctx._persist_wrapped = True
                _DS._transformer_ctx = _persist_transformer_ctx
                print(f"   [PERSIST] DiffusionStage._transformer_ctx wrapped "
                      f"(mode={LTX_PERSIST_PIPELINE}, lru_max={_PERSIST_LRU_MAX}, resolution-keyed)")
        except Exception as _e_persist:
            print(f"   ⚠️ persist-pipeline wrapper skipped (non-fatal): {_e_persist}")

        # --- LEVER 2: audio-skip (video-only requests) ---
        # Wrap AudioDecoder.__call__ so that, when the per-request skip flag is
        # set, it returns None WITHOUT building the audio VAE decoder + vocoder
        # (~1.3s). The pipeline's `audio = self.audio_decoder(...)` becomes None →
        # `encode_video(audio=None)` writes a clean video-only mp4. The audio
        # DENOISE context (carried through the transformer) is untouched, so the
        # VIDEO pixels are byte-identical to the audio-on run. Default OFF.
        try:
            from ltx_pipelines.utils import blocks as _audio_blocks
            _AD = _audio_blocks.AudioDecoder
            if not getattr(_AD.__call__, "_audioskip_wrapped", False):
                _orig_ad_call = _AD.__call__

                def _audioskip_call(self_ad, latent):
                    if _AUDIO_SKIP_STATE["enabled"]:
                        _AUDIO_SKIP_STATE["skipped"] += 1
                        print("   [AUDIO-SKIP] video-only request — audio decoder + "
                              "vocoder NOT built (returning None)")
                        return None
                    return _orig_ad_call(self_ad, latent)

                _audioskip_call._audioskip_wrapped = True
                _AD.__call__ = _audioskip_call
                print(f"   [AUDIO-SKIP] AudioDecoder.__call__ wrapped "
                      f"(default {'ON' if LTX_SKIP_AUDIO else 'OFF'})")
        except Exception as _e_audio:
            print(f"   ⚠️ audio-skip wrapper skipped (non-fatal): {_e_audio}")

        print("✅ _snap_init complete (captured in memory snapshot)")

    @modal.enter(snap=False)
    def _gpu_init(self):
        """S1: GPU init that runs on EVERY container restart (after snapshot
        restore). All CUDA work happens here: volume reload + compile-cache
        prep + eager load + warm-up forward + commit. Cold start = these
        seconds (because the snap=True half is restored in O(seconds)).
        """
        import time
        import torch

        start_time = time.time()
        print("=" * 60)
        print("🚀 LTX-2.3 GPU INIT (post-snapshot restore)")
        print("=" * 60)

        # --- 1. SDPA backend toggles (W7) ---
        # W7 (2026-05-23): explicitly enable all three SDPA backends with
        # Flash prioritised. Lets `torch.nn.functional.scaled_dot_product_attention`
        # pick the Flash kernel for unmasked self-attention (the path the
        # transformer's self-attn blocks hit), while preserving math +
        # mem-efficient fallbacks for masked cross-attention (Flash rejects
        # `attn_mask=...` in recent versions, so disabling those would crash
        # the pipeline on every prompt).
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        # --- GPU-RESIDENT registry MUST be (re)built post-snapshot (2026-06-06) ---
        # `_snap_init` runs on a CPU-only snapshot sandbox: a GPU-resident
        # registry can't allocate CUDA tensors there, and the cached snapshot may
        # predate an `LTX_REGISTRY` env change. So when the live env asks for
        # gpu_resident, build it HERE (snap=False, on the GPU), overriding the
        # restored snapshot. Idempotent; opt-in only — does NOT affect the
        # cpu_pinned default path (that registry stays built in _snap_init).
        if _LTX_REGISTRY_MODE == "gpu_resident":
            from utils.gpu_resident_registry import GpuResidentStateDictRegistry
            if not isinstance(getattr(self, "_sd_registry", None), GpuResidentStateDictRegistry):
                self._sd_registry = GpuResidentStateDictRegistry()
                print(
                    f"   🔁 Registry mode: gpu_resident (rebuilt post-restore; "
                    f"gpu_max={self._sd_registry._gpu_max_bytes / (1 << 30):.0f} GB, "
                    f"stream_over={self._sd_registry._stream_over_bytes / (1 << 30):.0f} GB)"
                )

        # --- 2. S2: pull latest volume contents so we see compile artifacts
        # committed by other containers. Modal volumes are eventually-
        # consistent; without an explicit reload, this container may see a
        # stale snapshot. ---
        try:
            model_volume.reload()
            print("   📦 model_volume reloaded (compile-cache freshness)")
        except Exception as e:
            print(f"   ⚠️ model_volume.reload() failed (non-fatal): {e}")

        # --- 3. S2: pin compile + Triton caches to the volume so the NEXT
        # container can read them after we commit at end of init. ---
        # Original K5 rationale (preserved): persisting TorchInductor cache
        # on the models volume skips the 2-5 min per-pipeline compile cost
        # on subsequent cold starts. Setting env vars at runtime (vs Image())
        # avoids forcing a full `download_models` re-run on builder timeouts.
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/models/.torchinductor_cache")
        os.environ.setdefault("TRITON_CACHE_DIR", "/models/.triton_cache")
        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
        os.makedirs("/models/.torchinductor_cache", exist_ok=True)
        os.makedirs("/models/.triton_cache", exist_ok=True)
        print(f"   TORCHINDUCTOR_CACHE_DIR     = {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
        print(f"   TRITON_CACHE_DIR            = {os.environ['TRITON_CACHE_DIR']}")
        print(f"   TORCHINDUCTOR_FX_GRAPH_CACHE= {os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE']}")
        print(f"   TORCHINDUCTOR_AUTOGRAD_CACHE= {os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE']}")
        print(f"   ENABLE_TORCH_COMPILE        = {ENABLE_TORCH_COMPILE}")

        # --- 4. Defensive FA3 clamp (W7) ---
        # W7 (2026-05-23): even though the PyPI `flash-attn` package does
        # NOT expose `flash_attn_interface` (the Hopper-FA3 module name),
        # we explicitly clamp it to None on the ltx_core attention module
        # so a future env drift cannot silently flip the upstream import
        # guard at attention.py:14-19. `DEFAULT` resolves on
        # `memory_efficient_attention`, but this protects against any
        # direct `AttentionFunction.FLASH_ATTENTION_3` request being
        # satisfied — `FlashAttention3.__call__` raises NotImplementedError
        # on masked cross-attn (attention.py:111-112), which would break us.
        try:
            import ltx_core.model.transformer.attention as _ltx_attn
            _ltx_attn.flash_attn_interface = None
        except Exception:
            pass

        # --- 4b. SageAttention-3 monkey-patch (2026-06-06, default OFF) ---
        # When LTX_SAGE_ATTN=1, route LTX's UNMASKED self-attention through
        # SageAttention-3 (Blackwell FP4). We patch `Attention.forward` rather
        # than `PytorchAttention.__call__` because the self-vs-cross signal we
        # need (`context is None`) is only visible at the module level — the
        # AttentionCallable only receives (q, k, v, heads, mask).
        #
        # Routing rule (quality-safe):
        #   * self-attn (attn1 / audio_attn1): forward() is called with
        #     context=None AND mask=None → q,k,v share one tensor, bidirectional,
        #     unmasked → SageAttention-3 path.
        #   * everything else (attn2 text cross-attn with a context_mask; the
        #     audio<->video cross-attn with differing seq lengths / separate rope)
        #     → original attention_function (SDPA). SageAttention rejects
        #     arbitrary attn_mask, so wiring cross-attn there would crash.
        #
        # API verified 2026-06-06: `from sageattn3 import sageattn3_blackwell`;
        # `sageattn3_blackwell(q, k, v, is_causal=False)` expects q/k/v as
        # (batch, heads, seq, dim) fp16/bf16. LTX's `Attention.forward` produces
        # q,k,v as (b, seq, heads*dim_head); we reshape to BHSD, call Sage3,
        # then reshape back to (b, seq, heads*dim_head) exactly like
        # PytorchAttention does — so `to_out` downstream is unchanged.
        self._sage_attn_active = False
        if LTX_SAGE_ATTN:
            try:
                from sageattn3 import sageattn3_blackwell as _sage3
                import ltx_core.model.transformer.attention as _ltx_attn2
                from ltx_core.model.transformer.rope import apply_rotary_emb as _apply_rope

                _orig_attn_forward = _ltx_attn2.Attention.forward

                # Faithful mirror of the pinned-commit Attention.forward
                # (attention.py:180-249 @ 1799988): same signature, same
                # perturbation_mask blend, same per-head gating, same
                # all_perturbed short-circuit. The ONLY change is that the
                # unmasked self-attention kernel call
                # `self.attention_function(q, k, v, heads, mask)` is replaced by
                # `sageattn3_blackwell(...)`. Sage3 runs ONLY on a pure self-attn
                # step (context is None AND mask is None AND attention actually
                # runs). Every other case delegates verbatim to the original
                # forward — so Sage never sees a mask or a cross-attn context.
                def _sage_attn_forward(
                    self, x, context=None, mask=None, pe=None, k_pe=None,
                    perturbation_mask=None, all_perturbed=False,
                ):
                    is_self_attn = (context is None and mask is None and not all_perturbed)
                    if not is_self_attn:
                        return _orig_attn_forward(
                            self, x, context=context, mask=mask, pe=pe, k_pe=k_pe,
                            perturbation_mask=perturbation_mask, all_perturbed=all_perturbed,
                        )

                    v = self.to_v(x)
                    q = self.q_norm(self.to_q(x))
                    k = self.k_norm(self.to_k(x))
                    if pe is not None:
                        q = _apply_rope(q, pe, self.rope_type)
                        k = _apply_rope(k, pe if k_pe is None else k_pe, self.rope_type)

                    b, seq, _ = q.shape
                    h = self.heads
                    d = q.shape[-1] // h
                    qb = q.view(b, seq, h, d).transpose(1, 2)
                    kb = k.view(b, seq, h, d).transpose(1, 2)
                    vb = v.view(b, seq, h, d).transpose(1, 2).to(qb.dtype)
                    out = _sage3(qb, kb, vb, is_causal=False)
                    out = out.transpose(1, 2).reshape(b, seq, h * d)

                    if perturbation_mask is not None:
                        out = out * perturbation_mask + v * (1 - perturbation_mask)

                    if getattr(self, "to_gate_logits", None) is not None:
                        gate_logits = self.to_gate_logits(x)
                        bb, tt, _ = out.shape
                        out = out.view(bb, tt, self.heads, self.dim_head)
                        gates = 2.0 * torch.sigmoid(gate_logits)
                        out = out * gates.unsqueeze(-1)
                        out = out.view(bb, tt, self.heads * self.dim_head)

                    return self.to_out(out)

                _ltx_attn2.Attention.forward = _sage_attn_forward
                self._sage_attn_active = True
                print("   ⚡ SageAttention-3 ENABLED (LTX_SAGE_ATTN=1): self-attn → "
                      "sageattn3_blackwell, cross-attn → SDPA (mask-safe)")
            except Exception as e:
                # Never let a Sage3 import/patch failure take down the container —
                # fall back to bf16-SDPA and surface the exact error in logs.
                print(f"   ⚠️ SageAttention-3 patch FAILED — falling back to SDPA: "
                      f"{type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        # --- 4c. SageAttention-2.2 monkey-patch (2026-06-06, opt-in, default OFF) ---
        # When LTX_SAGE_ATTN=2, route LTX's UNMASKED self-attention through
        # SageAttention-2.2.0 (`sageattn_qk_int8_pv_fp16_cuda`, INT8-QK + FP16-PV)
        # — the PROVEN Blackwell sm_120 path (~30-35% faster diffusion on RTX
        # 5090, measured), unlike SA3 FP4 (value "1") which is build-blocked on
        # the public sm_120 toolchain. Same routing + reshape contract as the SA3
        # block above; the ONLY differences are the kernel call
        # (`sageattn_qk_int8_pv_fp16_cuda(q,k,v, tensor_layout="HND",
        # is_causal=False)`) and a per-call try/except → SDPA fallback (mask
        # preserved) so a single bad kernel call can never crash a request.
        #
        # NOTE: this branch is INERT on the prod image until two things change:
        #   (a) prod env `LTX_SAGE_ATTN` is flipped to "2", AND
        #   (b) the prod image is rebuilt on a CUDA-13 base with
        #       `sageattention==2.2.0` installed (the 12.4 base cannot build it).
        # With neither done, `from sageattention import ...` raises ImportError
        # here and we log + fall back to bf16-SDPA. The bench app
        # `ltx2-sage22-bench` is where SA2.2 is actually exercised + measured.
        elif LTX_SAGE22:
            try:
                from sageattention import sageattn_qk_int8_pv_fp16_cuda as _sage22
                import ltx_core.model.transformer.attention as _ltx_attn3
                from ltx_core.model.transformer.rope import apply_rotary_emb as _apply_rope22

                _orig_attn_forward22 = _ltx_attn3.Attention.forward

                def _sage22_attn_forward(
                    self, x, context=None, mask=None, pe=None, k_pe=None,
                    perturbation_mask=None, all_perturbed=False,
                ):
                    is_self_attn = (context is None and mask is None and not all_perturbed)
                    if not is_self_attn:
                        return _orig_attn_forward22(
                            self, x, context=context, mask=mask, pe=pe, k_pe=k_pe,
                            perturbation_mask=perturbation_mask, all_perturbed=all_perturbed,
                        )

                    v = self.to_v(x)
                    q = self.q_norm(self.to_q(x))
                    k = self.k_norm(self.to_k(x))
                    if pe is not None:
                        q = _apply_rope22(q, pe, self.rope_type)
                        k = _apply_rope22(k, pe if k_pe is None else k_pe, self.rope_type)

                    b, seq, _ = q.shape
                    h = self.heads
                    d = q.shape[-1] // h
                    # (b, seq, h*d) -> (b, h, seq, d) == HND for SA2.2.
                    qb = q.view(b, seq, h, d).transpose(1, 2)
                    kb = k.view(b, seq, h, d).transpose(1, 2)
                    vb = v.view(b, seq, h, d).transpose(1, 2).to(qb.dtype)
                    try:
                        out = _sage22(qb, kb, vb, tensor_layout="HND", is_causal=False)
                    except Exception as _kerr22:
                        if not getattr(self, "_sage22_warned", False):
                            print(f"   ⚠️ SA2.2 kernel error → SDPA fallback: "
                                  f"{type(_kerr22).__name__}: {_kerr22}")
                            self._sage22_warned = True
                        return _orig_attn_forward22(
                            self, x, context=context, mask=mask, pe=pe, k_pe=k_pe,
                            perturbation_mask=perturbation_mask, all_perturbed=all_perturbed,
                        )
                    out = out.transpose(1, 2).reshape(b, seq, h * d)

                    if perturbation_mask is not None:
                        out = out * perturbation_mask + v * (1 - perturbation_mask)

                    if getattr(self, "to_gate_logits", None) is not None:
                        gate_logits = self.to_gate_logits(x)
                        bb, tt, _ = out.shape
                        out = out.view(bb, tt, self.heads, self.dim_head)
                        gates = 2.0 * torch.sigmoid(gate_logits)
                        out = out * gates.unsqueeze(-1)
                        out = out.view(bb, tt, self.heads * self.dim_head)

                    return self.to_out(out)

                _ltx_attn3.Attention.forward = _sage22_attn_forward
                self._sage_attn_active = True
                print("   ⚡ SageAttention-2.2 ENABLED (LTX_SAGE_ATTN=2): self-attn → "
                      "sageattn_qk_int8_pv_fp16_cuda (HND), cross-attn → SDPA (mask-safe)")
            except Exception as e:
                print(f"   ⚠️ SageAttention-2.2 patch FAILED — falling back to SDPA: "
                      f"{type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⏭️  SageAttention disabled (LTX_SAGE_ATTN={_LTX_SAGE_ATTN_MODE!r}): "
                  "bf16-SDPA default")

        # --- 5. FBCache hooks (W5) — constructed here because their stored
        # `prev_residual` tensors live on GPU. ---
        # W5: one per Euler-stage-1 we want to cache. HQ (res_2s sampler),
        # stage_2 of two-stage (distilled-LoRA refinement near "every step
        # matters"), and Retake (full CFG path) do NOT get attached. Hooks
        # remain disabled until /generate flips `enabled=True` per request
        # via `enable_step_cache`.
        # NOTE: relative import `from .fbcache` in the original W5 patch was
        # corrected to absolute `from fbcache` because the module is shipped
        # via `add_local_python_source("fbcache")` (top-level, not a package).
        from utils.fbcache import FirstBlockCacheHook
        self._fbcache_i2v_stage1 = FirstBlockCacheHook(threshold=0.05, max_skip_steps=3)
        self._fbcache_kf_stage1 = FirstBlockCacheHook(threshold=0.05, max_skip_steps=3)

        # --- 6. S5: monkey-patch AutoImageProcessor to default use_fast=True.
        # Eliminates "Using a slow image processor" warning + ~1-3s on the
        # `enhance_prompt=True` path (zero impact on default). ---
        try:
            import transformers as _transformers
            _orig_from_pretrained = _transformers.AutoImageProcessor.from_pretrained
            def _patched_from_pretrained(*args, **kwargs):
                kwargs.setdefault("use_fast", True)
                return _orig_from_pretrained(*args, **kwargs)
            _transformers.AutoImageProcessor.from_pretrained = _patched_from_pretrained
            print("   ✅ AutoImageProcessor monkey-patched (use_fast=True default)")
        except Exception as e:
            print(f"   ⚠️ AutoImageProcessor patch failed (non-fatal): {e}")

        # --- 7. Verify model files + CUDA report ---
        print(f"\n📋 Configuration:")
        print(f"   HF_HOME: {os.environ.get('HF_HOME', 'NOT SET')}")
        print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")

        print(f"\n📂 Volume mounted at {MODEL_DIR}:")
        if MODEL_DIR.exists():
            for item in MODEL_DIR.iterdir():
                if item.is_dir():
                    print(f"   📁 {item.name}/")
                else:
                    size_mb = item.stat().st_size / 1024 / 1024
                    print(f"   📄 {item.name} ({size_mb:.1f} MB)")

        print(f"\n🔧 PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        print("\n🔍 Verifying model files:")
        for name, path in [
            ("Checkpoint", self.checkpoint_path),
            ("Spatial Upsampler", self.spatial_upsampler_path),
            ("Distilled LoRA", self.distilled_lora_path),
            ("Distilled Checkpoint (preview)", self.distilled_checkpoint_path),
            ("Gemma", str(self.gemma_path)),
        ]:
            exists = Path(path).exists()
            status_marker = "✅" if exists else "❌"
            print(f"   {status_marker} {name}: {path}")

        # --- 8. Eagerly load default pipeline (gated). When the registry is
        # enabled (LTX_USE_REGISTRY=1) this preloads weights into the cache
        # so the first user request is warm. When the registry is disabled
        # (default), eager-load just constructs the pipeline object — still
        # useful because the upstream pipeline ctor pulls upsampler weights
        # and runs other one-time setup. We keep it on by default. ---
        print("\n📦 Eagerly loading default i2v pipeline...")
        try:
            self._get_i2v_pipeline()
        except Exception as e:
            print(f"⚠️ Eager load failed (non-fatal — will retry on request): {e}")

        # --- 9. Pre-warm forward pass (gated, OFF by default). When enabled
        # (LTX_PREWARM_COMPILE=1) this runs a tiny forward pass to populate
        # the torch.compile / TorchInductor / Triton caches. CAVEAT: the
        # forward leaves ~78GB allocated by PyTorch's caching allocator on
        # H100, causing the NEXT user request to fail with
        # CUBLAS_STATUS_ALLOC_FAILED. Disabled until we implement either
        # a cuda.empty_cache+reset after pre-warm OR move the registry's
        # cached weights to pinned CPU memory. ---
        if ENABLE_PREWARM_COMPILE:
            self._prewarm_compile_cache()
        else:
            print("\n⏭️  Pre-warm skipped (LTX_PREWARM_COMPILE=0). First user "
                  "request will pay the torch.compile cold-start cost.")

        # --- 10. S2: persist the freshly-populated cache to volume. ---
        try:
            model_volume.commit()
            print("   ✅ model_volume.commit() — compile + triton cache persisted")
        except Exception as e:
            print(f"   ⚠️ model_volume.commit() failed (non-fatal): {e}")

        print("\n" + "=" * 60)
        print(f"✅ GPU INIT COMPLETE in {time.time() - start_time:.1f}s")
        print("=" * 60)

    @modal.exit()
    def _on_shutdown(self):
        """R3 (2026-05-23): flush inductor + triton artifacts compiled
        mid-session (e.g. first ever mode=hq, distilled-LoRA variant, or a
        new (resolution, frame_count) shape on a long-lived container) so
        the NEXT cold container's volume.reload() in `_gpu_init` sees them.
        Without this, mid-session compiles only land on persistent storage
        opportunistically and the next cold start eats the 50-80s compile
        cost again. See research_v2/R3_modal_concurrent/REPORT.md.
        """
        try:
            model_volume.commit()
            print("✅ Volume committed on shutdown (inductor + triton flush)")
        except Exception as e:
            print(f"⚠️ Volume commit on shutdown failed (non-fatal): {e}")

    class _Stage1GELoopWrapper:
        """Thin forwarder around a `DiffusionStage` instance that injects
        `loop=<gradient_estimating_euler_denoising_loop>` on every call.

        Why a wrapper object instead of monkey-patching `__call__`:
        Python dunder lookup goes through the **type**, so setting
        ``instance.__call__ = ...`` is ignored when the pipeline does
        ``self.stage_1(...)``. A wrapper object lets us intercept the
        call cleanly. `__getattr__` forwards any non-call access
        (e.g. ``model_context``, ``run``) to the underlying stage so
        nothing else breaks.
        """

        __slots__ = ("_inner", "_loop")

        def __init__(self, inner, loop):
            self._inner = inner
            self._loop = loop

        def __call__(self, *args, **kwargs):
            kwargs.setdefault("loop", self._loop)
            return self._inner(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _install_ge_loop_on_stage_1(self, pipeline):
        """Wrap `pipeline.stage_1` so it routes through the gradient-
        estimating Euler loop. Idempotent — re-wrap is a no-op."""
        if isinstance(pipeline.stage_1, Model._Stage1GELoopWrapper):
            return pipeline
        pipeline.stage_1 = Model._Stage1GELoopWrapper(pipeline.stage_1, self._ge_loop)
        print("   ↻ stage_1 wrapped with gradient_estimating_euler_denoising_loop (ge_gamma=2.0)")
        return pipeline

    def _prewarm_compile_cache(self):
        """S2: run one tiny forward pass through the default i2v pipeline so
        the first real user request hits an already-populated cache.

        Shape rationale (17 frames @ 320x256, 4 steps):
          - 320x256 — smallest resolution satisfying upstream's two-stage
            divisibility checks. Cheapest shape that still exercises the
            Stage-1 + upsampler + Stage-2 path.
          - 4 steps — minimum that produces non-zero loss with the default
            sigma schedule. Output is discarded.
          - mode='default' (Euler) — identical compile graph as HQ Stage-2
            and KF interp Stage-1, so warming this also seeds the
            FX/inductor/Triton caches for the other two pipelines.

        Triggers ONE compile per unique compiled subgraph (stage_1, stage_2).
        Non-fatal on failure — first real request will pay compile cost.
        """
        if not ENABLE_TORCH_COMPILE:
            print("   ⏭️ Skipping pre-warm (ENABLE_TORCH_COMPILE=0)")
            return

        import time
        from PIL import Image

        print("\n" + "=" * 60)
        print("🔥 Pre-warming torch.compile cache (one-time per container)")
        print("=" * 60)

        warmup_img = Image.new("RGB", (320, 256), color=(128, 128, 128))
        warmup_path = "/tmp/_ltx2_warmup.png"
        warmup_img.save(warmup_path)

        # S2 Patch 4 (defensive): surface compile config to the inductor in
        # case upstream's context-patch in compiling.py is scoped narrower
        # than expected on PyTorch 2.12. Belt-and-braces with env vars.
        try:
            import torch._inductor.config as _ic
            import torch._functorch.config as _fc
            _ic.fx_graph_cache = True
            _ic.unsafe_skip_cache_dynamic_shape_guards = True
            if hasattr(_fc, "enable_autograd_cache"):
                _fc.enable_autograd_cache = True
        except Exception as e:
            print(f"   ⚠️ Inductor in-process config patch failed (non-fatal): {e}")

        warm_start = time.time()
        try:
            _ = self._process_video(
                image_urls=[f"file://{warmup_path}"],
                prompt="a calm scene, neutral lighting",
                negative_prompt="",
                num_frames=17,
                height=256,
                width=320,
                frame_rate=24.0,
                num_inference_steps=4,
                cfg_guidance_scale=3.0,
                seed=0,
                enhance_prompt=False,
                mode="default",
                enable_step_cache=False,
                guider="cfg",
            )
            elapsed = time.time() - warm_start
            print(f"✅ Pre-warm forward pass complete in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - warm_start
            print(f"⚠️ Pre-warm failed after {elapsed:.1f}s "
                  f"(first real request will pay compile cost): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # =====================================================================
    # GPU memory hygiene (2026-05-23)
    # =====================================================================
    # Defensive layer that ensures we don't carry GPU memory pressure across
    # requests. Cheap (~50-200ms total) — well-amortised vs the multi-second
    # cost of an OOM-induced container restart.
    #
    # Three pieces:
    #   1. `_cleanup_gpu_memory()` — fast end-of-request GC + cache flush.
    #   2. `_log_gpu_memory()`     — one-line peak/current snapshot.
    #   3. `_aggressive_recover()` — for one-shot OOM retry: drops cached
    #                                 pipelines so they re-load lazily on the
    #                                 next request with a clean GPU.

    def _cleanup_gpu_memory(self) -> None:
        """Fast GC + CUDA cache empty. Safe to call after EVERY request.

        Cost: ~50-200ms (gc.collect ~10-100ms, empty_cache ~10-50ms,
        synchronize ~1-10ms). Worth it because:
          - Drops any tensors still held only by reference cycles (PyTorch
            autograd graphs are a common source of these).
          - Returns the caching-allocator's reserved-but-unused segments to
            the driver, so the next request doesn't see fragmentation.
          - Synchronizes the default stream so deferred allocations on copy
            streams (e.g. CpuPinnedStateDictRegistry) are visible to the
            allocator's accounting.
        """
        import gc
        import torch as _t
        try:
            gc.collect()
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
                _t.cuda.synchronize()
        except Exception as e:
            # Never let cleanup itself crash the request response path.
            print(f"   ⚠️ _cleanup_gpu_memory non-fatal: {type(e).__name__}: {e}")

    def _log_gpu_memory(self, tag: str) -> None:
        """One-line GPU memory snapshot for post-request observability."""
        import torch as _t
        if not _t.cuda.is_available():
            return
        try:
            alloc = _t.cuda.memory_allocated() / (1 << 30)
            reserved = _t.cuda.memory_reserved() / (1 << 30)
            peak = _t.cuda.max_memory_allocated() / (1 << 30)
            print(f"   💾 GPU {tag}: alloc={alloc:5.1f} GB  "
                  f"reserved={reserved:5.1f} GB  peak={peak:5.1f} GB")
        except Exception:
            pass

    def _aggressive_recover(self) -> None:
        """Heavyweight recovery — invoked ONLY after an OOM. Drops every
        cached pipeline + LoRA-variant cache so subsequent requests rebuild
        from a known-clean GPU state. ~1-2s. Acceptable for an OOM retry.
        """
        import torch as _t
        print("   🚨 OOM recovery: dropping cached pipelines + variant caches")
        # Cached pipelines (per _gpu_init lazy-load contract).
        for attr in ("_i2v_pipeline", "_i2v_hq_pipeline", "_kf_pipeline",
                     "_retake_pipeline", "_distilled_preview_pipeline",
                     "_ic_lora_motion_track_pipeline", "_ic_lora_union_pipeline"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)
        # Variant LoRA cache (LRU; safe to nuke).
        if hasattr(self, "_variant_pipeline_cache"):
            try:
                self._variant_pipeline_cache.clear()
            except Exception:
                pass
        # Variant distilled-strength cache + community-LoRA cache (safe to nuke;
        # both rebuild lazily on the next matching request).
        for cache_attr, order_attr in (
            ("_variant_cache", "_variant_cache_order"),
            ("_community_lora_cache", "_community_lora_cache_order"),
        ):
            if hasattr(self, cache_attr):
                try:
                    getattr(self, cache_attr).clear()
                    getattr(self, order_attr).clear()
                except Exception:
                    pass
        # State-dict registry: clear if present so the pinned-CPU pages are
        # released back to the OS. Next request re-pins lazily.
        if getattr(self, "_sd_registry", None) is not None:
            try:
                self._sd_registry.clear()
            except Exception:
                pass
        if _t.cuda.is_available():
            try:
                _t.cuda.empty_cache()
                _t.cuda.synchronize()
                _t.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        import gc
        gc.collect()
        if _t.cuda.is_available():
            alloc = _t.cuda.memory_allocated() / (1 << 30)
            print(f"   ✅ Post-recovery GPU alloc: {alloc:.1f} GB")

    def _install_phase_timers(self, pipeline, *, kind: str) -> None:
        """S5: instrument each pipeline block with per-call timing.

        We REPLACE the block attribute on the pipeline with a thin
        `_TimedBlock` forwarder (defined below). We do NOT monkey-patch
        `__call__` because:
          1. Python dunder lookup goes through `type(obj).__call__`, NOT
             the instance — setting `block.__call__ = ...` is silently
             ignored for `block(...)` invocations.
          2. Swapping `type(block).__call__` works but pollutes the class
             globally and causes a `denoiser` arg-binding collision when
             the block has __slots__ (the previous v1 of this method).

        The wrapper forwards `__call__` (with timing) and `__getattr__`
        (transparent passthrough for `.model_context`, `.run`, etc.) and
        is idempotent — `_install_phase_timers` can be called multiple
        times without re-wrapping.

        Composition with `_install_ge_loop_on_stage_1`: this method runs
        AFTER, so `pipeline.stage_1` is already `_Stage1GELoopWrapper`.
        Wrapping that wrapper in `_TimedBlock` gives us:
            pipeline.stage_1 = _TimedBlock(name='stage_1') ↓
                self._inner = _Stage1GELoopWrapper ↓
                    self._inner = DiffusionStage (real)
        and each call cleanly chains through both wrappers.
        """
        class _TimedBlock:
            __slots__ = ("_inner", "_name", "_kind", "_is_timed_block")
            def __init__(self_, inner, name, kind_):
                import time as _time
                self_._inner = inner
                self_._name = name
                self_._kind = kind_
                self_._is_timed_block = True
            def __call__(self_, *args, **kwargs):
                import time as _time
                t0 = _time.perf_counter()
                try:
                    return self_._inner(*args, **kwargs)
                finally:
                    print(f"   [TIMER] {self_._kind}/{self_._name:24s} "
                          f"{_time.perf_counter()-t0:7.2f}s")
            def __getattr__(self_, item):
                # Forward any other attribute access to the wrapped block.
                # Skip our own slots (handled by __slots__ descriptors).
                return getattr(self_._inner, item)

        wrapped_any = False
        # Includes Retake-specific attrs: `stage` (singular DiffusionStage),
        # `audio_conditioner` (unique to Retake). Other pipelines just skip
        # attrs they don't have via the hasattr() check below.
        for attr in ("prompt_encoder", "image_conditioner", "audio_conditioner",
                     "upsampler", "video_decoder", "audio_decoder",
                     "stage_1", "stage_2", "stage"):
            if not hasattr(pipeline, attr):
                continue
            inner = getattr(pipeline, attr)
            if getattr(inner, "_is_timed_block", False):
                continue  # already wrapped, idempotent
            try:
                setattr(pipeline, attr, _TimedBlock(inner, attr, kind))
                wrapped_any = True
            except (AttributeError, TypeError) as e:
                # Slotted pipeline class — can't replace the attribute.
                # Non-fatal: just log and move on.
                print(f"   ⚠️ phase timer for {kind}/{attr} skipped: {e}")
        if wrapped_any:
            print(f"   [TIMER] phase timers installed on {kind} pipeline")

    def _get_i2v_pipeline(self):
        """Lazy load standard two-stage Image-to-Video pipeline (Euler sampler).

        Stage 1 — base dev checkpoint at half resolution with CFG guidance.
        Stage 2 — upsample 2x, refine with the distilled stage-2 LoRA (now
        v1.1 at strength 1.0).
        """
        if self._i2v_pipeline is None:
            print(f"🎬 Loading TI2VidTwoStagesPipeline (torch_compile={ENABLE_TORCH_COMPILE})...")
            self._i2v_pipeline = self.TI2VidTwoStagesPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=self.distilled_lora,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=[],
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
            print(f"✅ TI2VidTwoStagesPipeline loaded ({'fp8' if LTX_FP8 else 'bf16'})")
            # 2026-06-07 ★ PERSIST: tag the RAW DiffusionStage instances with their
            # role so the _transformer_ctx persist wrapper (installed in _snap_init)
            # knows which stage it is building/reusing. Set on the raw objects BEFORE
            # the FBCache / GE-loop / timer wrapping below (those forward __getattr__,
            # so the tag stays readable; the wrapped `_transformer_ctx` still binds to
            # the raw stage as `self`, so id(self_ds) keys the raw object).
            try:
                self._i2v_pipeline.stage_1._persist_role = "stage_1"
                self._i2v_pipeline.stage_2._persist_role = "stage_2"
                print(f"   [PERSIST] tagged i2v stage_1/stage_2 roles (mode={_PERSIST_STATE['mode']})")
            except Exception as _e_tag:
                print(f"   ⚠️ persist role tag (i2v) skipped (non-fatal): {_e_tag}")
            # 2026-05-23 ORDER MATTERS: attach FBCache to the RAW DiffusionStage
            # FIRST (it sets `stage._fbcache_hook` + wraps `stage._build_transformer`),
            # THEN wrap stage_1 with the GE-loop forwarder. The wrapper has
            # `__slots__ = ("_inner", "_loop")` so it can't carry extra attributes.
            # Order is fine because the wrapper's `__call__` forwards to
            # `self._inner(*args, loop=...)` which still hits the FBCache-wrapped
            # `_build_transformer` on the inner DiffusionStage.
            from utils.fbcache import attach_to_diffusion_stage
            attach_to_diffusion_stage(self._i2v_pipeline.stage_1, self._fbcache_i2v_stage1)
            print("   FBCache: wrapped i2v stage_1._build_transformer (disabled by default)")
            self._install_ge_loop_on_stage_1(self._i2v_pipeline)
            self._install_phase_timers(self._i2v_pipeline, kind="default")
        return self._i2v_pipeline

    def _get_i2v_hq_pipeline(self):
        """Lazy load Lightricks's documented "highest quality" two-stage i2v.

        Single-image i2v only. Same dev checkpoint, but uses the res_2s
        second-order sampler with the distilled stage-2 LoRA applied at BOTH
        stages with separate strengths (stage-1: 0.25, stage-2: 0.5) and
        `LTX_2_3_HQ_PARAMS.video_guider_params` (stg_scale=0.0, rescale=0.45).
        Slower than the Euler path; visibly sharper detail and steadier motion.
        Not available for keyframe interpolation upstream yet.
        """
        if self._i2v_hq_pipeline is None:
            print(f"🎬 Loading TI2VidTwoStagesHQPipeline (res_2s sampler, torch_compile={ENABLE_TORCH_COMPILE})...")
            self._i2v_hq_pipeline = self.TI2VidTwoStagesHQPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=self.distilled_lora,
                distilled_lora_strength_stage_1=0.25,
                distilled_lora_strength_stage_2=0.5,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=(),
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
            print(f"✅ TI2VidTwoStagesHQPipeline loaded ({'fp8' if LTX_FP8 else 'bf16'}, res_2s)")
            self._install_phase_timers(self._i2v_hq_pipeline, kind="hq")
        return self._i2v_hq_pipeline

    def _get_kf_pipeline(self):
        """Lazy load Keyframe Interpolation pipeline."""
        if self._kf_pipeline is None:
            print(f"🎬 Loading KeyframeInterpolationPipeline (torch_compile={ENABLE_TORCH_COMPILE})...")
            self._kf_pipeline = self.KeyframeInterpolationPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=self.distilled_lora,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=[],
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
            print(f"✅ KeyframeInterpolationPipeline loaded ({'fp8' if LTX_FP8 else 'bf16'})")
            # 2026-06-07 ★ PERSIST: tag raw stage roles (see _get_i2v_pipeline).
            try:
                self._kf_pipeline.stage_1._persist_role = "stage_1"
                self._kf_pipeline.stage_2._persist_role = "stage_2"
                print(f"   [PERSIST] tagged kf stage_1/stage_2 roles (mode={_PERSIST_STATE['mode']})")
            except Exception as _e_tag:
                print(f"   ⚠️ persist role tag (kf) skipped (non-fatal): {_e_tag}")
            # See _get_i2v_pipeline for order rationale: FBCache attach before GE-loop wrap.
            from utils.fbcache import attach_to_diffusion_stage
            attach_to_diffusion_stage(self._kf_pipeline.stage_1, self._fbcache_kf_stage1)
            print("   FBCache: wrapped kf stage_1._build_transformer (disabled by default)")
            self._install_ge_loop_on_stage_1(self._kf_pipeline)
            self._install_phase_timers(self._kf_pipeline, kind="kf")
        return self._kf_pipeline

    def _get_retake_pipeline(self):
        """Lazy load Retake (video-to-video) pipeline."""
        if self._retake_pipeline is None:
            print(f"🎬 Loading RetakePipeline (torch_compile={ENABLE_TORCH_COMPILE})...")
            self._retake_pipeline = self.RetakePipeline(
                checkpoint_path=self.checkpoint_path,
                gemma_root=str(self.gemma_path),
                loras=[],
                distilled=False,
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
            print(f"✅ RetakePipeline loaded ({'fp8' if LTX_FP8 else 'bf16'}, full CFG)")
            self._install_phase_timers(self._retake_pipeline, kind="retake")
        return self._retake_pipeline

    def _get_distilled_preview_pipeline(self):
        """Lazy load the standalone distilled two-stage i2v pipeline.

        Backs `mode='preview'`. Trades quality for speed:
          * stage-1 sigmas = DISTILLED_SIGMAS (9 values → 8 denoising steps)
          * stage-2 sigmas = STAGE_2_DISTILLED_SIGMAS (4 values → 3 steps)
          * no CFG branch, no guider params, no negative prompt
          * single-image i2v only (no keyframe interpolation upstream)

        Uses a SEPARATE standalone distilled checkpoint
        (`ltx-2.3-22b-distilled-1.1.safetensors`) which embeds the dev
        transformer + audio adaptor + VAE distilled into one ~46 GB file.
        Does NOT compose with `self.checkpoint_path` (dev base).

        NOTE: per merge policy, the 46 GB safetensors download was NOT added
        to `LTX_SINGLE_FILES` in this build. First call will return 503 with
        instructions to add the file and redeploy.
        """
        if self._distilled_preview_pipeline is None:
            if not Path(self.distilled_checkpoint_path).exists():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "DistilledPipeline file not yet on volume — add "
                        "ltx-2.3-22b-distilled-1.1.safetensors to "
                        "LTX_SINGLE_FILES and redeploy"
                    ),
                )
            print(f"🎬 Loading DistilledPipeline (preview, ~11 step, torch_compile={ENABLE_TORCH_COMPILE})...")
            self._distilled_preview_pipeline = self.DistilledPipeline(
                distilled_checkpoint_path=self.distilled_checkpoint_path,
                gemma_root=str(self.gemma_path),
                spatial_upsampler_path=self.spatial_upsampler_path,
                loras=[],
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
            print(f"✅ DistilledPipeline loaded ({'fp8' if LTX_FP8 else 'bf16'}, distilled sigmas)")
            self._install_phase_timers(self._distilled_preview_pipeline, kind="preview")
        return self._distilled_preview_pipeline

    def _get_ic_lora_pipeline(self, control: str):
        """Lazy load an In-Context (IC) LoRA control pipeline.

        `control`:
          - "motion_track" → Motion-Track-Control IC-LoRA (trajectory/point motion).
          - "union"        → Union-Control IC-LoRA (Canny + Depth + Pose). This is
                             the 22B camera/motion-control path (see _snap_init note;
                             supersedes the non-loadable 19B dolly/jib LoRAs).

        Both wrap upstream `ICLoraPipeline` (ltx_pipelines.ic_lora @ pinned commit
        1799988). Key differences from the Euler i2v factories:
          * ICLoraPipeline is built on the standalone DISTILLED checkpoint
            (`distilled_checkpoint_path`), NOT the dev base — same file mode='preview'
            uses.
          * The control LoRA is passed via `loras=[LoraPathStrengthAndSDOps(...)]`
            (the upstream-canonical way the --lora CLI arg builds an entry, using
            LTXV_LORA_COMFY_RENAMING_MAP). The pipeline reads the reference
            downscale factor out of the LoRA metadata at construction time.
          * No FBCache + no GE-loop wrap (distilled sigmas; "every step matters"
            on the short distilled schedule). Phase timers are installed for parity.
        """
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

        if control == "motion_track":
            slot = "_ic_lora_motion_track_pipeline"
            lora_path = self.ic_lora_motion_track_path
            kind = "ic_motion_track"
        elif control == "union":
            slot = "_ic_lora_union_pipeline"
            lora_path = self.ic_lora_union_path
            kind = "ic_union"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown IC-LoRA control: {control!r}. Use 'motion_track' or 'union'.",
            )

        if getattr(self, slot) is None:
            if not Path(self.distilled_checkpoint_path).exists():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "ICLoraPipeline needs the distilled base checkpoint "
                        "ltx-2.3-22b-distilled-1.1.safetensors on the volume. It is in "
                        "LTX_SINGLE_FILES now — redeploy so download_models() fetches it."
                    ),
                )
            if not Path(lora_path).exists():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        f"IC-LoRA weight not on volume: {lora_path}. It is in "
                        "download_models() now — redeploy to fetch it."
                    ),
                )
            print(f"🎬 Loading ICLoraPipeline[{control}] (distilled base, torch_compile={ENABLE_TORCH_COMPILE})...")
            ic_lora = [
                LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),
            ]
            pipeline = self.ICLoraPipeline(
                distilled_checkpoint_path=self.distilled_checkpoint_path,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=ic_lora,
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
            print(f"✅ ICLoraPipeline[{control}] loaded (bf16, ref_downscale={getattr(pipeline, 'reference_downscale_factor', '?')})")
            self._install_phase_timers(pipeline, kind=kind)
            setattr(self, slot, pipeline)
        return getattr(self, slot)

    def _process_ic_lora_safe(self, **kwargs) -> bytes:
        """OOM-aware wrapper for `_process_ic_lora` (mirrors `_process_video_safe`)."""
        import torch as _t
        if _t.cuda.is_available():
            try:
                _t.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        try:
            try:
                video_bytes = self._process_ic_lora(**kwargs)
            except _t.cuda.OutOfMemoryError as oom:
                self._log_gpu_memory("at OOM")
                logger.error(f"OOM during _process_ic_lora: {oom}. Recovering and retrying once.")
                self._aggressive_recover()
                video_bytes = self._process_ic_lora(**kwargs)
                logger.info("✅ Retry after OOM recovery succeeded.")
            return video_bytes
        finally:
            self._log_gpu_memory("post-ic-lora")
            self._cleanup_gpu_memory()
            self._log_gpu_memory("post-cleanup")

    def _process_ic_lora(
        self,
        control: str,
        control_video_url: str,
        prompt: str,
        num_frames: int,
        height: int,
        width: int,
        frame_rate: float,
        seed: int,
        image_urls: list[str] | None = None,
        control_strength: float = 1.0,
        conditioning_attention_strength: float = 1.0,
        enhance_prompt: bool = False,
    ) -> bytes:
        """Run an IC-LoRA control generation.

        `control_video_url` is the positionally-aligned reference/control video
        (e.g. a Canny/Depth/Pose render for union, or a motion-track render).
        Optional `image_urls[0]` supplies an initial-frame image for i2v-style
        conditioning (IC-LoRA supports an optional init image).
        """
        import torch
        import time
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.media_io import encode_video

        total_start = time.time()
        # ICLoraPipeline (distilled base) is NOT persist-tagged and can't reuse the
        # resident two-stage transformers — free them so control gen has the card.
        _free_all_residents()
        pipeline = self._get_ic_lora_pipeline(control)

        # Resolve control/reference video (URL or file://).
        control_path = self._download_video(control_video_url)

        # Optional init image(s) — IC-LoRA places the first on frame 0.
        images: list = []
        temp_files: list[str] = []
        if image_urls:
            img_path = self._download_image(image_urls[0])
            temp_files.append(img_path)
            images = [ImageConditioningInput(path=img_path, frame_idx=0, strength=1.0)]
            print(f"🎬 IC-LoRA[{control}] with init image conditioning")
        else:
            print(f"🎬 IC-LoRA[{control}] text+control only (no init image)")

        # video_conditioning is the upstream control input: list[(path, strength)].
        video_conditioning = [(control_path, float(control_strength))]

        print(f"\n🧠 Running ICLoraPipeline[{control}] (mode={control})")
        print(f"   Prompt: {prompt[:100]}...")
        print(f"   Control video: {control_video_url[:80]} (strength={control_strength})")
        print(f"   Resolution: {width}x{height}, Frames: {num_frames}, FPS: {frame_rate}")

        tiling_config = _make_tiling_config()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        inference_start = time.time()
        with torch.inference_mode():
            video, audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                video_conditioning=video_conditioning,
                enhance_prompt=enhance_prompt,
                tiling_config=tiling_config,
                conditioning_attention_strength=conditioning_attention_strength,
            )
        print(f"   Inference completed in {time.time() - inference_start:.1f}s")

        print("\n📹 Encoding video...")
        encode_start = time.time()
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        with torch.no_grad():
            encode_video(
                video=video,
                fps=int(frame_rate),
                audio=audio,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )
        print(f"   Encoding completed in {time.time() - encode_start:.1f}s")

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        for path in temp_files:
            try:
                os.unlink(path)
            except Exception:
                pass
        try:
            os.unlink(output_path)
        except Exception:
            pass

        print(f"\n✅ IC-LoRA total time: {time.time() - total_start:.1f}s")
        print(f"   Output size: {len(video_bytes) / 1024 / 1024:.1f} MB")
        return video_bytes

    def _get_variant_pipeline(
        self,
        *,
        mode_norm: str,
        use_keyframe_interpolation: bool,
        stage_1_strength: float | None,
        stage_2_strength: float | None,
    ):
        """Build/fetch a pipeline variant with overridden distilled-LoRA strengths.

        Only called when a request supplies `distilled_lora_strength_stage_1`
        or `distilled_lora_strength_stage_2` (W8 Tier 1C/1D runtime A/B).
        Requests that pass neither override continue to use the existing
        `_get_i2v_pipeline` / `_get_i2v_hq_pipeline` / `_get_kf_pipeline`
        single-instance caches at the baked-in defaults.

        Cache is keyed by `(pipeline_kind, effective_s1, effective_s2)` with
        FIFO eviction at MAX_VARIANTS. Eviction triggers `gc.collect()` +
        `torch.cuda.empty_cache()` so released DiffusionStage transformer
        copies actually return GPU memory.

        Notes on per-stage strength semantics (from upstream source):
          - `TI2VidTwoStagesPipeline` (default Euler): applies the distilled
            LoRA only at stage 2, using `distilled_lora[0].strength`.
            `stage_1_strength` is ignored.
          - `TI2VidTwoStagesHQPipeline`: applies the distilled LoRA at both
            stages with separate strengths passed to the constructor.
          - `KeyframeInterpolationPipeline`: applies the distilled LoRA only
            at stage 2 via `distilled_lora[0].strength`. `stage_1_strength`
            ignored.
        """
        import gc
        import torch
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

        MAX_VARIANTS = 4

        if use_keyframe_interpolation:
            kind = "kf"
            effective_s1 = None
            # KF interp baked-in stage-2 strength = 1.0 (matches self.distilled_lora init)
            effective_s2 = stage_2_strength if stage_2_strength is not None else 1.0
        elif mode_norm == "hq":
            kind = "hq"
            # HQ baked-in stage_1=0.25, stage_2=0.5 (see _get_i2v_hq_pipeline)
            effective_s1 = stage_1_strength if stage_1_strength is not None else 0.25
            effective_s2 = stage_2_strength if stage_2_strength is not None else 0.5
        else:
            kind = "default"
            effective_s1 = None
            # Default Euler baked-in stage-2 strength = 1.0
            effective_s2 = stage_2_strength if stage_2_strength is not None else 1.0

        key = (kind, effective_s1, effective_s2)
        if key in self._variant_cache:
            # Move to MRU position
            self._variant_cache_order.remove(key)
            self._variant_cache_order.append(key)
            print(f"♻️ Reusing cached variant pipeline {key}")
            return self._variant_cache[key]

        print(f"🔧 Building variant pipeline kind={kind} s1={effective_s1} s2={effective_s2}")
        distilled_lora_variant = [
            LoraPathStrengthAndSDOps(
                self.distilled_lora_path,
                effective_s2,
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]
        if kind == "hq":
            pipeline = self.TI2VidTwoStagesHQPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=distilled_lora_variant,
                distilled_lora_strength_stage_1=effective_s1,
                distilled_lora_strength_stage_2=effective_s2,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=(),
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
        elif kind == "kf":
            pipeline = self.KeyframeInterpolationPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=distilled_lora_variant,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=[],
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
        else:
            pipeline = self.TI2VidTwoStagesPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=distilled_lora_variant,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=[],
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
        self._install_phase_timers(pipeline, kind=f"variant-{kind}")

        if len(self._variant_cache_order) >= MAX_VARIANTS:
            evicted_key = self._variant_cache_order.pop(0)
            evicted_pipeline = self._variant_cache.pop(evicted_key)
            print(f"🗑️ Evicting variant pipeline {evicted_key} (cache full at {MAX_VARIANTS})")
            del evicted_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._variant_cache[key] = pipeline
        self._variant_cache_order.append(key)
        print(f"✅ Variant pipeline cached (size={len(self._variant_cache_order)}/{MAX_VARIANTS})")
        return pipeline

    def _get_community_lora_pipeline(
        self,
        *,
        lora_key: str,
        use_keyframe_interpolation: bool,
        lora_strength: float,
    ):
        """Build/fetch an i2v (or KF-interp) pipeline with a community LoRA applied.

        Selected per-request via the /generate `lora` param. The community LoRA
        is appended to the pipeline's `loras=` list using the SAME native
        `LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP)`
        pattern already used for distilled_lora + the IC-LoRA control path. Both
        wired LoRAs (transition, vbvr) are trained on the LTX-2.3-22b-dev base,
        so they ride the standard Euler `TI2VidTwoStagesPipeline` (single image)
        or `KeyframeInterpolationPipeline` (2+ images), at the same dev
        `checkpoint_path` / spatial upsampler / distilled stage-2 LoRA as the
        default-mode path. quantization (fp8) is honored when LTX_FP8=1.

        Cache keyed by (lora_key, kf, strength) with FIFO eviction at
        MAX_COMMUNITY_VARIANTS. Mutually exclusive with mode='hq'/'preview' and
        with the distilled-strength variant cache (caller enforces).
        """
        import gc
        import torch
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

        MAX_COMMUNITY_VARIANTS = 4

        if lora_key not in self.community_lora_paths:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unknown lora: {lora_key!r}. Available: "
                    f"{sorted(self.community_lora_paths)}."
                ),
            )
        lora_path = self.community_lora_paths[lora_key]
        if not Path(lora_path).exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Community LoRA weight not on volume: {lora_path}. It is in "
                    "download_models() now — redeploy to fetch it."
                ),
            )

        kind = "kf" if use_keyframe_interpolation else "default"
        key = (lora_key, kind, round(float(lora_strength), 3))
        if key in self._community_lora_cache:
            self._community_lora_cache_order.remove(key)
            self._community_lora_cache_order.append(key)
            print(f"♻️ Reusing cached community-LoRA pipeline {key}")
            return self._community_lora_cache[key]

        print(f"🔧 Building community-LoRA pipeline {key}")
        community_lora = [
            LoraPathStrengthAndSDOps(lora_path, float(lora_strength), LTXV_LORA_COMFY_RENAMING_MAP),
        ]
        if use_keyframe_interpolation:
            pipeline = self.KeyframeInterpolationPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=self.distilled_lora,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=community_lora,
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
        else:
            pipeline = self.TI2VidTwoStagesPipeline(
                checkpoint_path=self.checkpoint_path,
                distilled_lora=self.distilled_lora,
                spatial_upsampler_path=self.spatial_upsampler_path,
                gemma_root=str(self.gemma_path),
                loras=community_lora,
                torch_compile=ENABLE_TORCH_COMPILE,
                registry=self._sd_registry,
                quantization=self._quantization,
            )
        print(f"✅ Community-LoRA pipeline loaded ({'fp8' if LTX_FP8 else 'bf16'}, "
              f"lora={lora_key} strength={lora_strength})")
        self._install_phase_timers(pipeline, kind=f"community-{lora_key}-{kind}")

        if len(self._community_lora_cache_order) >= MAX_COMMUNITY_VARIANTS:
            evicted_key = self._community_lora_cache_order.pop(0)
            evicted_pipeline = self._community_lora_cache.pop(evicted_key)
            print(f"🗑️ Evicting community-LoRA pipeline {evicted_key} (cache full at {MAX_COMMUNITY_VARIANTS})")
            del evicted_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._community_lora_cache[key] = pipeline
        self._community_lora_cache_order.append(key)
        print(f"✅ Community-LoRA pipeline cached (size={len(self._community_lora_cache_order)}/{MAX_COMMUNITY_VARIANTS})")
        return pipeline

    def _download_video(self, url: str) -> str:
        """Resolve a video source (URL or file:// path) to a local temp file."""
        import requests

        if url.startswith("file://"):
            local_path = url[len("file://"):]
            print(f"📁 Using local video: {local_path}")
            return local_path

        print(f"⬇️ Downloading video: {url[:80]}...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(response.content)
        temp_file.close()
        size_mb = len(response.content) / 1024 / 1024
        print(f"   Saved to {temp_file.name} ({size_mb:.1f} MB)")
        return temp_file.name

    def _save_base64_video(self, data: str) -> str:
        """Decode a base64 video string and save to a temp file."""
        import base64

        if "," in data:
            data = data.split(",", 1)[1]

        raw = base64.b64decode(data)
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(raw)
        temp_file.close()
        size_mb = len(raw) / 1024 / 1024
        print(f"   Saved base64 video to {temp_file.name} ({size_mb:.1f} MB)")
        return temp_file.name

    def _download_image(self, url: str) -> str:
        """Resolve an image source (URL or file:// path) to a local temp file."""
        import requests
        from PIL import Image
        import io

        if url.startswith("file://"):
            local_path = url[len("file://"):]
            print(f"📁 Using local image: {local_path}")
            return local_path

        print(f"⬇️ Downloading image: {url[:80]}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content))
        suffix = ".png" if image.mode == "RGBA" else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        image.save(temp_file.name)
        print(f"   Saved to {temp_file.name} (size: {image.size}, mode: {image.mode})")

        return temp_file.name

    def _process_video(
        self,
        image_urls: list[str],
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        height: int,
        width: int,
        frame_rate: float,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        seed: int,
        enhance_prompt: bool,
        mode: str = "default",
        enable_step_cache: bool = False,
        step_cache_threshold: float = 0.05,
        guider: str = "cfg",
        apg_eta: float = 1.0,
        apg_norm_threshold: float = 0.0,
        stg_scale: float | None = None,
        distilled_lora_strength_stage_1: float | None = None,
        distilled_lora_strength_stage_2: float | None = None,
        lora: str | None = None,
        lora_strength: float | None = None,
        skip_audio: bool | None = None,
        emb_cache: bool | None = None,
        persist_mode: str | None = None,
        tile_px: int | None = None,
        temporal_frames: int | None = None,
        force_lru_max: int | None = None,
    ) -> bytes:
        """Generate video from images.

        `mode`:
          - "default": standard Euler two-stage pipeline (TI2Vid for 1 image,
            KeyframeInterpolation for 2+ images). Stage-2 LoRA at strength 1.0
            using v1.1 weights.
          - "hq": single-image i2v only. Uses Lightricks's TI2VidTwoStagesHQ
            pipeline (res_2s sampler, LTX_2_3_HQ_PARAMS guider config). Fails
            with a clear error if more than 1 image is supplied since upstream
            has no `KeyframeInterpolationHQPipeline` yet.
          - "preview": single-image i2v only. Uses DistilledPipeline (~11
            total denoising ops). Ignores negative_prompt, cfg_guidance_scale,
            num_inference_steps.
        """
        import torch
        import time
        from dataclasses import replace as dc_replace
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, detect_params
        from ltx_pipelines.utils.media_io import encode_video

        total_start = time.time()

        # =============================================================
        # 2026-06-07 OPTIMIZATION STACK — per-request lever wiring.
        # The wrappers themselves are installed once in _snap_init; here we set
        # the per-request state they read. All three preserve prod behavior when
        # the request leaves the params at their defaults.
        # =============================================================

        # --- LEVER 1: emb-cache per-request toggle + version pin ---
        # encoder_version pins the loaded-weights identity (gemma_root +
        # checkpoint + fp8 flag) so a weight swap can never alias a stale embedding.
        use_emb_cache = LTX_CACHE_TEXT_EMB if emb_cache is None else bool(emb_cache)
        _EMB_CACHE_STATE["enabled"] = use_emb_cache
        _EMB_CACHE_STATE["version"] = (
            str(getattr(self, "gemma_path", "")),
            str(getattr(self, "checkpoint_path", "")),
            bool(LTX_FP8),
        )
        print(f"   [EMB-CACHE] request emb_cache={use_emb_cache} "
              f"(entries cached={len(_EMB_CACHE)})")

        # --- LEVER 2: audio-skip per-request toggle (default FALSE = audio KEPT) ---
        use_skip_audio = LTX_SKIP_AUDIO if skip_audio is None else bool(skip_audio)
        _AUDIO_SKIP_STATE["enabled"] = use_skip_audio
        print(f"   [AUDIO-SKIP] request skip_audio={use_skip_audio}")

        # --- LEVER 3: persist-pipeline shape pin + optional per-request mode ---
        # ★ Pin the CURRENT request's (height, width) so the resolution-keyed
        # _transformer_ctx wrapper builds/reuses the correct-shape resident. This
        # is what makes mixed-resolution serving (768x512 + 768x1280) safe.
        _PERSIST_CURRENT_SHAPE["height"] = height
        _PERSIST_CURRENT_SHAPE["width"] = width
        # Activation-aware resident cap for THIS request's resolution/frames, so a
        # high-res forward never collides with two resident stage transformers.
        _PERSIST_STATE["lru_max"] = (int(force_lru_max) if force_lru_max
                                     else _max_residents_for(height, width, num_frames))
        # Proactively drop residents from a DIFFERENT resolution (and any excess
        # over the cap) BEFORE the forward — stops a warm container loaded by a
        # prior different-resolution request from OOMing this one's first call.
        _purge_stale_residents(height, width, _PERSIST_STATE["lru_max"])
        # `persist_mode` (None → keep current/env default) lets a request step the
        # mode at runtime. When the mode CHANGES we evict every resident the new
        # mode would no longer use, so a cached + a rebuilt module never coexist.
        if persist_mode is not None:
            _new_raw = str(persist_mode).lower().strip()
            if _new_raw in ("1", "on", "true", "yes", "both"):
                _new_mode = "both"
            elif _new_raw in ("stage2", "stage_2", "2"):
                _new_mode = "stage2"
            else:
                _new_mode = "off"
            if _new_mode != _PERSIST_STATE["mode"]:
                print(f"   [PERSIST] mode change {_PERSIST_STATE['mode']} -> {_new_mode}; "
                      f"evicting residents the new mode won't reuse")
                keep_roles = set()
                if _new_mode == "both":
                    keep_roles = {"stage_1", "stage_2"}
                elif _new_mode == "stage2":
                    keep_roles = {"stage_2"}
                # Evict by role across ALL resolutions (key = (sid, role, h, w)).
                for _k in list(_PERSIST_TRANSFORMERS.keys()):
                    _role = _k[1]
                    if _role not in keep_roles:
                        _mod = _PERSIST_TRANSFORMERS.pop(_k, None)
                        if _mod is not None:
                            try:
                                _mod.to("meta")
                            except Exception:
                                pass
                            _PERSIST_STATE["evictions"] += 1
                            print(f"   [PERSIST] evicted resident {_k}")
                try:
                    from ltx_pipelines.utils.blocks import cleanup_memory as _cm
                    _cm()
                except Exception:
                    pass
                _PERSIST_STATE["mode"] = _new_mode
        print(f"   [PERSIST] mode={_PERSIST_STATE['mode']} "
              f"resident_count={len(_PERSIST_TRANSFORMERS)} (resolution={width}x{height})")

        # Download images
        print(f"\n📥 Processing {len(image_urls)} image(s)...")
        image_paths = []
        temp_files = []

        for i, url in enumerate(image_urls):
            path = self._download_image(url)
            temp_files.append(path)
            image_paths.append(path)

        # Prepare image conditioning
        # Format: list of (path, frame_idx, strength)
        if len(image_paths) == 1:
            # Single image: condition on first frame
            images = [ImageConditioningInput(path=image_paths[0], frame_idx=0, strength=1.0)]
            use_keyframe_interpolation = False
            print("🎬 Single image conditioning")
        else:
            # Multiple images: keyframe interpolation
            # Place images evenly across frames
            images = []
            for i, path in enumerate(image_paths):
                if i == 0:
                    frame_idx = 0
                elif i == len(image_paths) - 1:
                    frame_idx = num_frames - 1
                else:
                    # Distribute evenly
                    frame_idx = int((i / (len(image_paths) - 1)) * (num_frames - 1))
                images.append(ImageConditioningInput(path=path, frame_idx=frame_idx, strength=1.0))
            use_keyframe_interpolation = True
            print(f"🎬 Keyframe-pair conditioning ({len(images)} keyframes)")

        print(f"   Image conditioning: {[(img.path.split('/')[-1], img.frame_idx, img.strength) for img in images]}")

        # Select pipeline based on mode + image count.
        # HQ + preview paths are single-image only — upstream ships no KF interp
        # variant for either.
        mode_norm = (mode or "default").lower()
        if mode_norm not in ("default", "hq", "preview"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown mode: {mode!r}. Use 'default', 'hq', or 'preview'.",
            )
        if mode_norm in ("hq", "preview") and use_keyframe_interpolation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"mode={mode_norm!r} supports single-image i2v only "
                    "(upstream has no KF interp variant for hq/preview). "
                    "Pass mode='default' for keyframe interpolation."
                ),
            )

        # 2026-06-06: community-LoRA selection. When `lora` is supplied, build/
        # fetch a dedicated cached pipeline with that LoRA appended to `loras=`.
        # Community LoRAs (transition, vbvr) are dev-base i2v/KF LoRAs, so they
        # only ride mode='default' (Euler i2v / KF interp). Reject the combos
        # they cannot serve. Mutually exclusive with the distilled-strength
        # variant cache to keep cache keys + pipeline identity unambiguous.
        lora_key = (lora or "").lower().strip() or None
        if lora_key is not None:
            if mode_norm in ("hq", "preview"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"lora={lora_key!r} is only supported with mode='default' "
                        "(community LoRAs are dev-base i2v/KF LoRAs; mode='hq'/'preview' "
                        "use different samplers/checkpoints). Drop mode or use 'default'."
                    ),
                )
            if (
                distilled_lora_strength_stage_1 is not None
                or distilled_lora_strength_stage_2 is not None
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "lora cannot be combined with distilled_lora_strength_stage_{1,2} "
                        "overrides — pick one pipeline-customization path per request."
                    ),
                )

        # W8: route through the variant cache when either LoRA-strength
        # override is supplied. Otherwise use the existing single-instance
        # lazy loaders (default-path unchanged). Variant cache does not
        # support mode='preview' (DistilledPipeline embeds its own weights
        # — no detachable distilled LoRA to override).
        use_variant_cache = (
            distilled_lora_strength_stage_1 is not None
            or distilled_lora_strength_stage_2 is not None
        )
        if use_variant_cache and mode_norm == "preview":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "distilled_lora_strength_stage_{1,2} overrides are not "
                    "supported with mode='preview' — DistilledPipeline embeds "
                    "its own weights and has no detachable distilled LoRA."
                ),
            )
        if lora_key is not None:
            # Recommended strength ~0.8-1.0; per-repo default from COMMUNITY_LORAS.
            effective_lora_strength = (
                lora_strength if lora_strength is not None
                else COMMUNITY_LORAS[lora_key]["strength"]
            )
            pipeline = self._get_community_lora_pipeline(
                lora_key=lora_key,
                use_keyframe_interpolation=use_keyframe_interpolation,
                lora_strength=effective_lora_strength,
            )
            pipeline_name = (
                f"{'KeyframeInterpolationPipeline' if use_keyframe_interpolation else 'TI2VidTwoStagesPipeline'}"
                f"(lora={lora_key})"
            )
        elif use_variant_cache:
            pipeline = self._get_variant_pipeline(
                mode_norm=mode_norm,
                use_keyframe_interpolation=use_keyframe_interpolation,
                stage_1_strength=distilled_lora_strength_stage_1,
                stage_2_strength=distilled_lora_strength_stage_2,
            )
            if use_keyframe_interpolation:
                pipeline_name = "KeyframeInterpolationPipeline(variant)"
            elif mode_norm == "hq":
                pipeline_name = "TI2VidTwoStagesHQPipeline(variant)"
            else:
                pipeline_name = "TI2VidTwoStagesPipeline(variant)"
        elif use_keyframe_interpolation:
            pipeline = self._get_kf_pipeline()
            pipeline_name = "KeyframeInterpolationPipeline"
        elif mode_norm == "hq":
            pipeline = self._get_i2v_hq_pipeline()
            pipeline_name = "TI2VidTwoStagesHQPipeline"
        elif mode_norm == "preview":
            pipeline = self._get_distilled_preview_pipeline()
            pipeline_name = "DistilledPipeline"
        else:
            pipeline = self._get_i2v_pipeline()
            pipeline_name = "TI2VidTwoStagesPipeline"

        # W5: FBCache wiring — Euler stage_1 only.
        # HQ (res_2s), preview (DistilledPipeline), Retake, and variant-cache
        # pipelines bypass entirely. Stage_2 of any pipeline is never wrapped,
        # so caching is automatically off there. Stale state on the unselected
        # hook does not matter — only the selected hook is invoked via the
        # wrapped DiffusionStage we picked above.
        active_fbcache = None
        if (
            enable_step_cache
            and mode_norm not in ("hq", "preview")
            and not use_variant_cache
            and lora_key is None
        ):
            active_fbcache = (
                self._fbcache_kf_stage1 if use_keyframe_interpolation
                else self._fbcache_i2v_stage1
            )
            active_fbcache.enabled = True
            active_fbcache.threshold = step_cache_threshold
            active_fbcache.reset()
        else:
            # Defensive: explicitly disable both hooks so a prior request that
            # set enabled=True cannot leak into this one.
            self._fbcache_i2v_stage1.enabled = False
            self._fbcache_kf_stage1.enabled = False

        # Generate video
        print(f"\n🧠 Running inference via {pipeline_name} (mode={mode_norm})")
        print(f"   Prompt: {prompt[:100]}...")
        print(f"   Resolution: {width}x{height}, Frames: {num_frames}, FPS: {frame_rate}")
        print(f"   Steps: {num_inference_steps}, CFG: {cfg_guidance_scale}, Seed: {seed}")

        inference_start = time.time()

        tiling_config = _make_tiling_config(tile_px, temporal_frames)
        _sp = tiling_config.spatial_config
        _tp = tiling_config.temporal_config
        print(f"   [VAE-TILE] spatial={getattr(_sp,'tile_size_in_pixels',None)}/{getattr(_sp,'tile_overlap_in_pixels',None)} "
              f"temporal={getattr(_tp,'tile_size_in_frames',None)}/{getattr(_tp,'tile_overlap_in_frames',None)} "
              f"lru_max={_PERSIST_STATE['lru_max']}")
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        # HQ pipeline uses LTX_2_3_HQ_PARAMS (stg_scale=0, rescale=0.45);
        # the Euler path uses LTX_2_3_PARAMS via detect_params. The user
        # `cfg_guidance_scale` overrides whichever variant was selected.
        # Preview (DistilledPipeline) has NO guider / CFG / negative prompt:
        # its sigmas are baked in and it does no classifier-free pass.
        if mode_norm == "preview":
            if negative_prompt and negative_prompt != DEFAULT_NEGATIVE_PROMPT:
                print("   ⚠️  mode='preview' ignores negative_prompt (DistilledPipeline has no CFG branch)")
            if num_inference_steps != 30:
                print(f"   ⚠️  mode='preview' ignores num_inference_steps={num_inference_steps} (fixed by DISTILLED_SIGMAS = ~11 total ops)")
            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=enhance_prompt,
                )
        else:
            if mode_norm == "hq":
                base_params = LTX_2_3_HQ_PARAMS
            else:
                base_params = detect_params(self.checkpoint_path)
            # 2026-05-23 Tier-0B: LTX_2_3_PARAMS / LTX_2_3_HQ_PARAMS ship modality_scale=3.0,
            # which adds an audio-conditioned forward per step. Upstream README:330-335
            # ("ltx-pipelines/README.md") says "If generating video-only, set it to 1.0
            # to disable." We don't read the audio output anywhere downstream, so this
            # forward is pure waste. Override on both guider param dicts.
            # W8 Tier-1C: optional per-request override of `stg_scale` for runtime A/B.
            video_guider_kwargs = dict(cfg_scale=cfg_guidance_scale, modality_scale=1.0)
            audio_guider_kwargs = dict(modality_scale=1.0)
            if stg_scale is not None:
                video_guider_kwargs["stg_scale"] = stg_scale
                audio_guider_kwargs["stg_scale"] = stg_scale
                print(f"   stg_scale override: {stg_scale}")
            video_guider = dc_replace(base_params.video_guider_params, **video_guider_kwargs)
            audio_guider = dc_replace(base_params.audio_guider_params, **audio_guider_kwargs)

            # W6: APG / CFG★ guider port. When guider != 'cfg', swap the params with a
            # custom MultiModalGuiderFactory subclass so the pipeline's
            # `create_multimodal_guider_factory` path yields our subclass guider for
            # stage-1 denoising. HQ pipeline bypasses that factory (constructs
            # MultiModalGuider directly in ti2vid_two_stages_hq.py:177), so reject the
            # combo there.
            guider_norm = (guider or "cfg").lower()
            if guider_norm not in ("cfg", "apg", "cfg_star"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown guider: {guider!r}. Use 'cfg', 'apg', or 'cfg_star'.",
                )
            if guider_norm != "cfg":
                if mode_norm == "hq":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"guider={guider_norm!r} is not supported with mode='hq' "
                            "(TI2VidTwoStagesHQPipeline constructs MultiModalGuider "
                            "directly and bypasses the factory swap). Use mode='default'."
                        ),
                    )
                from utils.custom_guiders import (
                    install_factory_preserving_patch,
                    wrap_params_with_guider,
                )
                install_factory_preserving_patch()
                video_guider = wrap_params_with_guider(
                    video_guider,
                    guider_norm,
                    apg_eta=apg_eta,
                    apg_norm_threshold=apg_norm_threshold,
                )
                audio_guider = wrap_params_with_guider(
                    audio_guider,
                    guider_norm,
                    apg_eta=apg_eta,
                    apg_norm_threshold=apg_norm_threshold,
                )
                print(
                    f"   Guider override: {guider_norm} "
                    f"(apg_eta={apg_eta}, apg_norm_threshold={apg_norm_threshold})"
                )

            with torch.inference_mode():
                video, audio = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    num_inference_steps=num_inference_steps,
                    video_guider_params=video_guider,
                    audio_guider_params=audio_guider,
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=enhance_prompt,
                )

        print(f"   Inference completed in {time.time() - inference_start:.1f}s")

        if active_fbcache is not None:
            print(f"   {active_fbcache.stats_str()}")
            active_fbcache.enabled = False  # leave the hook off after we're done

        # Encode video
        print("\n📹 Encoding video...")
        encode_start = time.time()

        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        # Use torch.no_grad() to avoid inference mode issues with VAE decoder
        with torch.no_grad():
            encode_video(
                video=video,
                fps=int(frame_rate),
                audio=audio,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )

        print(f"   Encoding completed in {time.time() - encode_start:.1f}s")

        # Read video bytes
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        # Cleanup temp files
        for path in temp_files:
            try:
                os.unlink(path)
            except:
                pass
        try:
            os.unlink(output_path)
        except:
            pass

        print(f"\n✅ Total processing time: {time.time() - total_start:.1f}s")
        print(f"   Output size: {len(video_bytes) / 1024 / 1024:.1f} MB")

        return video_bytes

    def _process_video_safe(self, **kwargs) -> bytes:
        """Wrap `_process_video` with:
           - peak-memory tracking reset before the call
           - always-runs cleanup_gpu_memory() in finally
           - one-shot OOM recovery + retry

        Cost on the happy path: ~50-200ms (one GC + empty_cache + sync at end).
        Cost on the OOM path:   ~1-2s recovery + full retry (acceptable
        compared to a 500 response forcing the client to reissue).
        """
        import torch as _t
        if _t.cuda.is_available():
            try:
                _t.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        try:
            try:
                video_bytes = self._process_video(**kwargs)
            except _t.cuda.OutOfMemoryError as oom:
                # Log what we saw, do an aggressive recover, retry ONCE.
                self._log_gpu_memory("at OOM")
                logger.error(f"OOM during _process_video: {oom}. Recovering and retrying once.")
                self._aggressive_recover()
                # Second attempt — if this OOMs too, propagate so the client
                # sees a 500 instead of looping.
                video_bytes = self._process_video(**kwargs)
                logger.info("✅ Retry after OOM recovery succeeded.")
            return video_bytes
        finally:
            # ALWAYS log + cleanup so the next request starts from a clean GPU.
            self._log_gpu_memory("post-request")
            self._cleanup_gpu_memory()
            self._log_gpu_memory("post-cleanup")

    def _save_base64_image(self, data: str) -> str:
        """Decode a base64 image string and save to a temp file. Returns the file path."""
        import base64
        from PIL import Image
        import io

        if "," in data:
            data = data.split(",", 1)[1]

        raw = base64.b64decode(data)
        image = Image.open(io.BytesIO(raw))
        suffix = ".png" if image.mode == "RGBA" else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        image.save(temp_file.name)
        print(f"   Saved base64 image to {temp_file.name} (size: {image.size}, mode: {image.mode})")
        return temp_file.name

    @modal.method()
    def smoke_modes(self) -> dict:
        """Auth-free: validate i2v / keyframe / t2v paths in ONE warm container.
        Tiny 320x256 / 17-frame / 4-step gens — confirms each mode's code path
        runs (not a quality test). v2v/retake is tested via smoke_retake."""
        import tempfile as _tf
        import time as _t
        from PIL import Image as _Image

        def _mk(color):
            p = _tf.mktemp(suffix=".png")
            _Image.new("RGB", (320, 256), color).save(p)
            return f"file://{p}"

        cases = {
            "i2v (1 image)": [_mk((128, 128, 128))],
            "keyframe (2 images)": [_mk((40, 40, 90)), _mk((200, 120, 60))],
            "t2v (0 images)": [],
        }
        results = {}
        for name, urls in cases.items():
            t0 = _t.time()
            try:
                vb = self._process_video_safe(
                    image_urls=urls, prompt="a calm cinematic scene, soft light",
                    negative_prompt="", num_frames=17, height=256, width=320,
                    frame_rate=24.0, num_inference_steps=4, cfg_guidance_scale=3.0,
                    seed=0, enhance_prompt=False, mode="default",
                )
                import base64 as _b64
                results[name] = {"status": "ok", "video_bytes": len(vb) if vb else 0,
                                 "latency_s": round(_t.time() - t0, 1),
                                 "video_b64": _b64.b64encode(vb).decode("ascii") if vb else None}
            except Exception as e:  # noqa: BLE001
                results[name] = {"status": "UNSUPPORTED", "error": f"{type(e).__name__}: {str(e)[:140]}"}
            print(f"   [MODE] {name}: {results[name].get('status')}")
        return results

    @modal.method()
    def smoke_retake(
        self, video_b64: str,
        prompt: str = "same scene, vivid neon lighting, cinematic color grade",
        start_time: float = 2.0, end_time: float = 5.0,
        num_inference_steps: int = 20, seed: int = 0,
    ) -> dict:
        """Auth-free v2v retake: regenerate a time window of a source video."""
        import base64 as _b64
        import tempfile as _tf
        import time as _t

        src = _tf.mktemp(suffix=".mp4")
        with open(src, "wb") as f:
            f.write(_b64.b64decode(video_b64))
        t0 = _t.time()
        vb = self._process_retake_safe(
            video_path=src, prompt=prompt, start_time=start_time, end_time=end_time,
            negative_prompt="", num_inference_steps=num_inference_steps,
            cfg_guidance_scale=3.0, seed=seed, regenerate_video=True,
            regenerate_audio=False, enhance_prompt=False,
        )
        return {"status": "ok", "video_bytes": len(vb) if vb else 0,
                "latency_s": round(_t.time() - t0, 1),
                "video_b64": _b64.b64encode(vb).decode("ascii") if vb else None}

    @modal.method()
    def smoke_control(
        self,
        control: str = "union",                 # "union" (Canny/Depth/Pose) | "motion_track"
        control_video_b64: str = "",            # the positionally-aligned control render (canny/depth/pose/track) as mp4
        prompt: str = "a cinematic scene, photorealistic",
        image_b64: str | None = None,           # optional init frame (i2v-style conditioning)
        num_frames: int = 97,
        height: int = 768,
        width: int = 768,
        seed: int = 0,
        control_strength: float = 1.0,
    ) -> dict:
        """Auth-free IC-LoRA control smoke: drives mode='union' (Canny+Depth+Pose)
        or 'motion_track'. Returns a diagnostic dict (never raises) with VRAM +
        persist counters so the client can read failures without torch locally."""
        import base64 as _b64
        import tempfile as _tf
        import time as _t
        import torch as _t_sm

        if not control_video_b64:
            return {"status": "ERROR", "error": "control_video_b64 required", "is_oom": False}
        cv = _tf.mktemp(suffix=".mp4")
        with open(cv, "wb") as f:
            f.write(_b64.b64decode(control_video_b64))
        image_urls = None
        if image_b64:
            ip = _tf.mktemp(suffix=".png")
            with open(ip, "wb") as f:
                f.write(_b64.b64decode(image_b64))
            image_urls = [f"file://{ip}"]
        if _t_sm.cuda.is_available():
            try:
                _t_sm.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        t0 = _t.time()
        try:
            vb = self._process_ic_lora_safe(
                control=control,
                control_video_url=f"file://{cv}",
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                frame_rate=24.0,
                seed=seed,
                image_urls=image_urls,
                control_strength=control_strength,
                enhance_prompt=False,
            )
        except Exception as _e:
            _free = -1.0
            if _t_sm.cuda.is_available():
                try:
                    _free = round(_t_sm.cuda.mem_get_info()[0] / (1 << 30), 2)
                except Exception:
                    pass
            return {"status": "ERROR", "control": control,
                    "error": f"{type(_e).__name__}: {str(_e)[:400]}",
                    "is_oom": ("out of memory" in str(_e).lower() or type(_e).__name__ == "OutOfMemoryError"),
                    "latency_s": round(_t.time() - t0, 2), "free_vram_gb": _free,
                    "persist_resident_count": len(_PERSIST_TRANSFORMERS)}
        peak = 0.0
        if _t_sm.cuda.is_available():
            try:
                peak = round(_t_sm.cuda.max_memory_allocated() / (1 << 30), 2)
            except Exception:
                pass
        return {"status": "ok", "control": control,
                "video_bytes": len(vb) if vb else 0,
                "latency_s": round(_t.time() - t0, 2), "peak_vram_gb": peak,
                "height": height, "width": width, "num_frames": num_frames,
                "persist_resident_count": len(_PERSIST_TRANSFORMERS),
                "video_b64": _b64.b64encode(vb).decode("ascii") if vb else None}

    @modal.method()
    def smoke_generate(
        self,
        height: int = 256,
        width: int = 320,
        num_frames: int = 17,
        num_inference_steps: int = 4,
        prompt: str = "a calm scene, neutral lighting",
        seed: int = 0,
        skip_audio: bool | None = None,
        emb_cache: bool | None = None,
        persist_mode: str | None = None,
        return_video_b64: bool = False,
        image_b64: str | None = None,
        tile_px: int | None = None,
        temporal_frames: int | None = None,
        force_lru_max: int | None = None,
    ) -> dict:
        """Auth-free smoke entrypoint for reorg + optimization-stack verification.

        Generates a tiny default-mode i2v clip from an in-container solid-color
        image (no URL fetch, no JWT, no api-key secret). Returns a status dict
        with the produced byte count, wall-clock latency, peak VRAM, and the
        per-request optimization-stack counters (emb-cache hits/misses, audio-skip,
        persist mode + resident count + builds/hits + oom_fallback) so the smoke
        driver can verify the levers end-to-end. Stays cheap to invoke via
        `Model().smoke_generate.remote()`.
        """
        import time as _time
        import torch as _t_sm
        from PIL import Image as _Image

        smoke_img_path = "/tmp/_ltx2_smoke.png"
        import base64 as _b64_sm
        import io as _io_sm
        if image_b64:
            # One image -> i2v; a list of 2+ -> keyframe interpolation.
            _items = image_b64 if isinstance(image_b64, list) else [image_b64]
            smoke_img_urls = []
            for _i, _b in enumerate(_items):
                _src = _Image.open(_io_sm.BytesIO(_b64_sm.b64decode(_b))).convert("RGB")
                _src = _src.resize((width, height), _Image.LANCZOS)
                _p = f"/tmp/_ltx2_smoke_{_i}.png"
                _src.save(_p)
                smoke_img_urls.append(f"file://{_p}")
        else:
            _Image.new("RGB", (width, height), color=(128, 128, 128)).save(smoke_img_path)
            smoke_img_urls = [f"file://{smoke_img_path}"]

        if _t_sm.cuda.is_available():
            try:
                _t_sm.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        _emb_h0 = _EMB_CACHE_STATE["hits"]
        _emb_m0 = _EMB_CACHE_STATE["misses"]
        _aud0 = _AUDIO_SKIP_STATE["skipped"]
        _s1b0 = _PERSIST_STATE["stage1_builds"]
        _s2b0 = _PERSIST_STATE["stage2_builds"]
        _s1h0 = _PERSIST_STATE["stage1_hits"]
        _s2h0 = _PERSIST_STATE["stage2_hits"]
        _ev0 = _PERSIST_STATE["evictions"]

        t0 = _time.time()
        try:
            video_bytes = self._process_video_safe(
                image_urls=smoke_img_urls,
                prompt=prompt,
                negative_prompt="",
                num_frames=num_frames,
                height=height,
                width=width,
                frame_rate=24.0,
                num_inference_steps=num_inference_steps,
                cfg_guidance_scale=3.0,
                seed=seed,
                enhance_prompt=False,
                mode="default",
                enable_step_cache=False,
                guider="cfg",
                skip_audio=skip_audio,
                emb_cache=emb_cache,
                persist_mode=persist_mode,
                tile_px=tile_px,
                temporal_frames=temporal_frames,
                force_lru_max=force_lru_max,
            )
        except Exception as _e:
            # Return the failure as a DIAGNOSTIC dict (don't raise) so the client
            # — which may not have torch installed to deserialize a remote
            # torch.OutOfMemoryError — can see the VRAM state + persist counters.
            _free = _resv = _alloc = -1.0
            if _t_sm.cuda.is_available():
                try:
                    _fb, _tb = _t_sm.cuda.mem_get_info()
                    _free = round(_fb / (1 << 30), 2)
                    _resv = round(_t_sm.cuda.memory_reserved() / (1 << 30), 2)
                    _alloc = round(_t_sm.cuda.memory_allocated() / (1 << 30), 2)
                except Exception:
                    pass
            return {
                "status": "ERROR",
                "error": f"{type(_e).__name__}: {str(_e)[:400]}",
                "is_oom": ("out of memory" in str(_e).lower()
                           or type(_e).__name__ == "OutOfMemoryError"),
                "latency_s": round(_time.time() - t0, 2),
                "free_vram_gb": _free, "reserved_vram_gb": _resv, "allocated_vram_gb": _alloc,
                "height": height, "width": width, "num_frames": num_frames,
                "persist_mode": _PERSIST_STATE["mode"],
                "persist_resident_count": len(_PERSIST_TRANSFORMERS),
                "persist_evictions": _PERSIST_STATE["evictions"],
                "persist_oom_fallback": _PERSIST_STATE["oom_fallback"],
            }
        elapsed = _time.time() - t0

        peak_gb = 0.0
        free_gb = resv_gb = -1.0
        if _t_sm.cuda.is_available():
            try:
                peak_gb = _t_sm.cuda.max_memory_allocated() / (1 << 30)
                _fb, _tb = _t_sm.cuda.mem_get_info()
                free_gb = round(_fb / (1 << 30), 2)
                resv_gb = round(_t_sm.cuda.memory_reserved() / (1 << 30), 2)
            except Exception:
                pass

        return {
            "status": "ok",
            "video_bytes": len(video_bytes) if video_bytes else 0,
            "latency_s": round(elapsed, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "free_vram_gb": free_gb,
            "reserved_vram_gb": resv_gb,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            # optimization-stack per-request counters
            "emb_cache_hits": _EMB_CACHE_STATE["hits"] - _emb_h0,
            "emb_cache_misses": _EMB_CACHE_STATE["misses"] - _emb_m0,
            "emb_cache_entries": len(_EMB_CACHE),
            "audio_skipped": _AUDIO_SKIP_STATE["skipped"] - _aud0,
            "persist_mode": _PERSIST_STATE["mode"],
            "persist_stage1_builds": _PERSIST_STATE["stage1_builds"] - _s1b0,
            "persist_stage2_builds": _PERSIST_STATE["stage2_builds"] - _s2b0,
            "persist_stage1_hits": _PERSIST_STATE["stage1_hits"] - _s1h0,
            "persist_stage2_hits": _PERSIST_STATE["stage2_hits"] - _s2h0,
            "persist_evictions": _PERSIST_STATE["evictions"] - _ev0,
            "persist_resident_count": len(_PERSIST_TRANSFORMERS),
            "persist_oom_fallback": _PERSIST_STATE["oom_fallback"],
            # Optional raw mp4 (base64) for the bit-identical RGB comparison. Only
            # set when explicitly requested (keeps the default smoke payload tiny).
            "video_b64": (
                __import__("base64").b64encode(video_bytes).decode("ascii")
                if (return_video_b64 and video_bytes) else None
            ),
        }

    def _process_retake_safe(self, **kwargs) -> bytes:
        """Retake counterpart to `_process_video_safe`. See that method's
        docstring; behaviour is identical (peak reset, OOM retry, finally
        cleanup + memory log).
        """
        import torch as _t
        if _t.cuda.is_available():
            try:
                _t.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        try:
            try:
                video_bytes = self._process_retake(**kwargs)
            except _t.cuda.OutOfMemoryError as oom:
                self._log_gpu_memory("at OOM")
                logger.error(f"OOM during _process_retake: {oom}. Recovering and retrying once.")
                self._aggressive_recover()
                video_bytes = self._process_retake(**kwargs)
                logger.info("✅ Retry after OOM recovery succeeded.")
            return video_bytes
        finally:
            self._log_gpu_memory("post-retake")
            self._cleanup_gpu_memory()
            self._log_gpu_memory("post-cleanup")

    def _process_retake(
        self,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        negative_prompt: str,
        num_inference_steps: int,
        cfg_guidance_scale: float,
        seed: int,
        regenerate_video: bool,
        regenerate_audio: bool,
        enhance_prompt: bool,
    ) -> bytes:
        """Regenerate a time region of an existing video."""
        import torch
        import time
        from dataclasses import replace as dc_replace
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.constants import detect_params
        from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata

        total_start = time.time()

        # RetakePipeline is NOT persist-tagged, so it can't reuse the resident
        # keyframe/i2v stage transformers — it would load on TOP of them and OOM
        # a near-full card. Free ALL residents so v2v has the full card for its
        # own ~35 GB weights + forward (the #1 back-to-back OOM fix).
        _free_all_residents()
        pipeline = self._get_retake_pipeline()

        src = get_videostream_metadata(video_path)
        print(f"   Source video: {src.width}x{src.height}, {src.frames} frames @ {src.fps}fps")
        print(f"   Retake window: {start_time:.1f}s - {end_time:.1f}s")

        params = detect_params(self.checkpoint_path)
        video_guider = dc_replace(params.video_guider_params, cfg_scale=cfg_guidance_scale)
        audio_guider = params.audio_guider_params

        tiling_config = _make_tiling_config()
        video_chunks_number = get_video_chunks_number(src.frames, tiling_config)

        print(f"\n🧠 Running retake inference...")
        inference_start = time.time()

        with torch.inference_mode():
            video_iter, audio = pipeline(
                video_path=video_path,
                prompt=prompt,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                video_guider_params=video_guider,
                audio_guider_params=audio_guider,
                regenerate_video=regenerate_video,
                regenerate_audio=regenerate_audio,
                enhance_prompt=enhance_prompt,
                tiling_config=tiling_config,
            )

        print(f"   Inference completed in {time.time() - inference_start:.1f}s")

        print("\n📹 Encoding video...")
        encode_start = time.time()
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        with torch.no_grad():
            encode_video(
                video=video_iter,
                fps=int(src.fps),
                audio=audio,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )

        print(f"   Encoding completed in {time.time() - encode_start:.1f}s")

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        try:
            os.unlink(output_path)
        except:
            pass

        print(f"\n✅ Retake total time: {time.time() - total_start:.1f}s")
        print(f"   Output size: {len(video_bytes) / 1024 / 1024:.1f} MB")
        return video_bytes

@app.local_entrypoint()
def smoke_real(
    image_path: str = "",
    height: int = 1280,
    width: int = 768,
    num_frames: int = 49,
    num_inference_steps: int = 20,
    prompt: str = "cinematic portrait, subtle natural head and eye movement, soft window light, photographic, shallow depth of field",
) -> None:
    """Real-input i2v smoke: a real image -> LTX-2.3 video clip, saved locally.

    Mirrors deploy/avatar/longcat_avatar_model.py::smoke_real for a side-by-side
    file comparison (LongCat = portrait+audio talking-head; LTX = image->motion).
    Defaults to the same deploy/avatar/test_inputs/portrait.jpg. Native 9:16.
    """
    import base64 as _b64
    import datetime as _dt
    import time as _time
    from pathlib import Path as _P

    if not image_path:
        image_path = str(_P(__file__).resolve().parents[1] / "avatar" / "test_inputs" / "portrait.jpg")
    img_bytes = _P(image_path).read_bytes()
    image_b64 = _b64.b64encode(img_bytes).decode("ascii")
    print("=== LTX-2.3 i2v REAL-input smoke ===")
    print(f"image={image_path} ({len(img_bytes)} B)  res={width}x{height}  frames={num_frames}")

    t0 = _time.time()
    res = Model().smoke_generate.remote(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        prompt=prompt,
        image_b64=image_b64,
        return_video_b64=True,
        seed=42,
    )
    elapsed = _time.time() - t0

    out_dir = _P(__file__).parent / "smoke_outputs"
    out_dir.mkdir(exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"real_{ts}_i2v.mp4"
    vb64 = res.get("video_b64")
    if vb64:
        out_path.write_bytes(_b64.b64decode(vb64))

    print("=== LTX i2v REAL SMOKE RESULT ===")
    print(f"latency_s={res.get('latency_s')}  peak_vram_gb={res.get('peak_vram_gb')}  bytes={res.get('video_bytes')}")
    print(f"wall(incl cold start)={elapsed:.1f}s  res={res.get('width')}x{res.get('height')}  frames={res.get('num_frames')}")
    print(f"Saved to: {out_path}" if vb64 else "NO video bytes returned")


@app.local_entrypoint()
def run_modes() -> None:
    """Validate + SAVE t2v / i2v / keyframe in one warm container -> mode_clips/."""
    import base64
    import pathlib

    out = pathlib.Path(__file__).parent / "mode_clips"
    out.mkdir(parents=True, exist_ok=True)
    print("\n=== LTX-2.3 MODE COVERAGE ===")
    res = Model().smoke_modes.remote()  # type: ignore[union-attr]
    for mode, r in res.items():
        b64 = r.pop("video_b64", None)
        if b64:
            p = out / f"{mode.split()[0]}.mp4"
            p.write_bytes(base64.b64decode(b64))
            r["saved"] = str(p)
        print(f"  {mode:24s} -> {r}")


@app.local_entrypoint()
def run_retake() -> None:
    """v2v retake in its OWN fresh container (avoids VRAM contention with the
    pipelines run_modes loads). Source = a real i2v clip in smoke_outputs/.
    Saves mode_clips/v2v_retake.mp4."""
    import base64
    import pathlib

    out = pathlib.Path(__file__).parent / "mode_clips"
    out.mkdir(parents=True, exist_ok=True)
    srcs = sorted((pathlib.Path(__file__).parent / "smoke_outputs").glob("real_*_i2v.mp4"))
    if not srcs:
        print("no source clip in smoke_outputs/ — run smoke_real first")
        return
    print(f"retake source: {srcs[-1].name}")
    rr = Model().smoke_retake.remote(  # type: ignore[union-attr]
        video_b64=base64.b64encode(srcs[-1].read_bytes()).decode("ascii"),
        start_time=2.0, end_time=5.0, num_inference_steps=20,
    )
    b64 = rr.pop("video_b64", None)
    if b64:
        p = out / "v2v_retake.mp4"
        p.write_bytes(base64.b64decode(b64))
        rr["saved"] = str(p)
    print(f"v2v (retake) -> {rr}")


@app.local_entrypoint()
def kf_real(
    image_a: str, image_b: str,
    prompt: str = "smooth cinematic transition between the two scenes, natural motion",
    num_frames: int = 97, height: int = 1280, width: int = 768,
    num_inference_steps: int = 25,
) -> None:
    """Real-image keyframe interpolation: 2 images -> interpolated video.
    Saves mode_clips/kf_real.mp4."""
    import base64
    import pathlib

    out = pathlib.Path(__file__).parent / "mode_clips"
    out.mkdir(parents=True, exist_ok=True)

    def _b64img(path):
        return base64.b64encode(pathlib.Path(path).read_bytes()).decode("ascii")

    res = Model().smoke_generate.remote(  # type: ignore[union-attr]
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=num_inference_steps, prompt=prompt,
        image_b64=[_b64img(image_a), _b64img(image_b)], return_video_b64=True, seed=42,
    )
    b64 = res.get("video_b64")
    if b64:
        p = out / "kf_real.mp4"
        p.write_bytes(base64.b64decode(b64))
        print(f"keyframe saved: {p}  ({res.get('video_bytes')} B, {res.get('latency_s')}s)")
    else:
        print(f"NO video bytes -> {res}")
