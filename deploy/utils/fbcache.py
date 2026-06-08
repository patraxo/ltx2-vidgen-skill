"""First-Block Cache (FBCache) for Lightricks LTX-2.3 transformer.

Port of the diffusers `apply_first_block_cache` (FLUX / HunyuanVideo / Wan2.1)
adapted for LTX-2.3's dual-modality (video, audio) -> TransformerArgs block I/O
signature in `LTXModel._process_transformer_blocks`
(ltx_core/model/transformer/model.py:339-367, 48 BasicAVTransformerBlock stack).

Mechanism: at each denoising step, run block 0 once. Compare its output against
the block-0 output cached from the previous step via relative-L1 of the video
stream. If the delta is below `threshold`, short-circuit blocks 1..47 by
replaying the cached cumulative residual; otherwise run the full stack and
capture a fresh tail residual. A `max_skip_steps` ceiling forces a recompute
after N consecutive hits to bound drift.

NOT a global LTXModel patch. We swap `_process_transformer_blocks` as a bound
method on the specific transformer instance built inside one DiffusionStage,
by wrapping that stage's `_build_transformer`. Stage 2 + HQ + Retake are left
untouched (see attach_to_diffusion_stage caller in ltx2_model.py).
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class _FBCacheState:
    """Internal per-hook tensor + counter slot. Reset between requests."""

    prev_video_block0_out: Optional[torch.Tensor] = None
    prev_audio_block0_out: Optional[torch.Tensor] = None
    prev_video_tail_residual: Optional[torch.Tensor] = None
    prev_audio_tail_residual: Optional[torch.Tensor] = None
    consecutive_hits: int = 0
    hits: int = 0
    misses: int = 0

    def clear(self) -> None:
        self.prev_video_block0_out = None
        self.prev_audio_block0_out = None
        self.prev_video_tail_residual = None
        self.prev_audio_tail_residual = None
        self.consecutive_hits = 0
        self.hits = 0
        self.misses = 0


def _rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L1 of (a - b) vs b, computed in fp32 on the same device."""
    num = (a - b).abs().mean(dtype=torch.float32)
    den = b.abs().mean(dtype=torch.float32).clamp_min(1e-8)
    return float((num / den).item())


class FirstBlockCacheHook:
    """Per-request FBCache controller for one LTX DiffusionStage.

    Lifecycle:
      1. Constructed at @modal.enter() time, one per stage we cache.
      2. `attach_to_diffusion_stage(stage, hook)` wraps `stage._build_transformer`
         so each newly-built transformer gets the FBCache method swap.
      3. Before each request, the caller flips `hook.enabled` and (optionally)
         updates `hook.threshold` / `hook.max_skip_steps`.
      4. Caller invokes `hook.reset()` before the pipeline call. State is also
         auto-reset inside the wrapped `_build_transformer` (defence in depth).
      5. After the pipeline call, caller reads `hook.hits` / `hook.misses` for
         logging.
    """

    def __init__(self, threshold: float = 0.05, max_skip_steps: int = 3) -> None:
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0; got {threshold}")
        if max_skip_steps < 0:
            raise ValueError(f"max_skip_steps must be >= 0; got {max_skip_steps}")
        self.threshold: float = float(threshold)
        self.max_skip_steps: int = int(max_skip_steps)
        self.enabled: bool = False
        self._state: _FBCacheState = _FBCacheState()

    # ------------------------------------------------------------------ API

    def reset(self) -> None:
        """Clear all cached tensors + stats. Call at start of every request."""
        self._state.clear()

    @property
    def hits(self) -> int:
        return self._state.hits

    @property
    def misses(self) -> int:
        return self._state.misses

    def stats_str(self) -> str:
        total = self.hits + self.misses
        if total == 0:
            return "FBCache: no steps"
        pct = 100.0 * self.hits / total
        return (
            f"FBCache: {self.hits}/{total} hits ({pct:.1f}%) "
            f"@ tau={self.threshold} max_skip={self.max_skip_steps}"
        )

    # ----------------------------------------------------- block-loop logic

    def _run_blocks(self, blocks, video, audio, perturbations):
        """Run the patched block loop. Returns (video, audio).

        Inputs are TransformerArgs dataclasses or None. The block returns
        `(TransformerArgs', TransformerArgs')` where only the `.x` tensor is
        replaced (see transformer.py:376 `dataclasses.replace(video, x=vx)`).
        """
        from dataclasses import replace as dc_replace

        st = self._state

        if not self.enabled or len(blocks) < 2:
            for block in blocks:
                video, audio = block(
                    video=video, audio=audio, perturbations=perturbations
                )
            return video, audio

        # Run block 0 (the tripwire).
        video, audio = blocks[0](
            video=video, audio=audio, perturbations=perturbations
        )
        vx_after0 = video.x if video is not None else None
        ax_after0 = audio.x if audio is not None else None

        # Decide cache hit/miss using the video stream only (audio dynamics
        # piggyback the video schedule; using both adds bookkeeping with no
        # signal gain per FBCache benchmarks).
        hit = False
        if (
            vx_after0 is not None
            and st.prev_video_block0_out is not None
            and st.prev_video_tail_residual is not None
            and st.consecutive_hits < self.max_skip_steps
        ):
            delta = _rel_l1(vx_after0, st.prev_video_block0_out)
            hit = delta < self.threshold

        if hit:
            # Short-circuit blocks 1..N-1 by replaying the cached tail residual.
            if video is not None:
                video = dc_replace(
                    video, x=vx_after0 + st.prev_video_tail_residual
                )
            if (
                audio is not None
                and ax_after0 is not None
                and st.prev_audio_tail_residual is not None
            ):
                audio = dc_replace(
                    audio, x=ax_after0 + st.prev_audio_tail_residual
                )
            st.hits += 1
            st.consecutive_hits += 1
        else:
            for block in blocks[1:]:
                video, audio = block(
                    video=video, audio=audio, perturbations=perturbations
                )
            # Capture the tail residual for the NEXT step's potential hit.
            if video is not None and vx_after0 is not None:
                st.prev_video_tail_residual = (video.x - vx_after0).detach()
            if audio is not None and ax_after0 is not None:
                st.prev_audio_tail_residual = (audio.x - ax_after0).detach()
            st.misses += 1
            st.consecutive_hits = 0

        # Always refresh the block-0 cache for next step's delta check.
        if vx_after0 is not None:
            st.prev_video_block0_out = vx_after0.detach()
        if ax_after0 is not None:
            st.prev_audio_block0_out = ax_after0.detach()

        return video, audio

    # --------------------------------------------------- transformer attach

    def attach_to_ltx_model(self, ltx_model: torch.nn.Module) -> None:
        """Swap `_process_transformer_blocks` on this LTXModel instance only.

        The bound-method swap means no other LTXModel in the process is
        affected. Re-attach is idempotent (overwrites a prior bind to this hook).
        """
        hook = self
        blocks_ref = ltx_model.transformer_blocks

        def _process_transformer_blocks_fbc(self, video, audio, perturbations):
            return hook._run_blocks(blocks_ref, video, audio, perturbations)

        ltx_model._process_transformer_blocks = types.MethodType(
            _process_transformer_blocks_fbc, ltx_model
        )
        ltx_model._fbcache_hook = hook  # back-reference for debugging


def attach_to_diffusion_stage(stage, hook: FirstBlockCacheHook) -> None:
    """Wrap `stage._build_transformer` so every transformer built inside this
    DiffusionStage gets `hook` installed on its inner LTXModel.

    DiffusionStage builds the transformer lazily inside `model_context()` and
    frees it on exit (see ltx_pipelines/utils/blocks.py:159+). The wrapper:
      1. delegates to the original builder,
      2. unwraps `X0Model.velocity_model` to reach the LTXModel,
      3. attaches the hook to that instance,
      4. resets the hook so we start each pipeline call with a clean cache.
    """
    if getattr(stage, "_fbcache_wrapped", False):
        stage._fbcache_hook = hook  # rebind state holder if re-attaching
        return

    original_build = stage._build_transformer

    def _build_with_fbcache(*args, **kwargs):
        x0_model = original_build(*args, **kwargs)
        ltx_model = getattr(x0_model, "velocity_model", x0_model)
        if not hasattr(ltx_model, "transformer_blocks"):
            # Defensive: if upstream restructures and velocity_model is gone,
            # just no-op the cache rather than crash the pipeline.
            return x0_model
        stage._fbcache_hook.attach_to_ltx_model(ltx_model)
        stage._fbcache_hook.reset()
        return x0_model

    stage._fbcache_hook = hook
    stage._build_transformer = _build_with_fbcache
    stage._fbcache_wrapped = True
