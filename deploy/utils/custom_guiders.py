"""Custom guider port for LTX-2.3 Modal deploy.

W6 — Tier 3D + 2C (research_v2/T3.md finding 1).

Upstream `TI2VidTwoStagesPipeline` / `TI2VidTwoStagesHQPipeline` / `KeyframeInterpolationPipeline`
use `MultiModalGuider` (ltx_core.components.guiders.MultiModalGuider) for stage-1
guidance. CFG++/APG/CFG★ are NOT constructor args on these pipelines — production
guidance lives in `MultiModalGuider.calculate()` and is the only override point.

This module ports the standalone `LtxAPGGuider` and `CFGStarRescalingGuider`
formulas onto the MultiModal interface so they can be swapped into the Euler
two-stage and keyframe-interpolation pipelines via a custom
`MultiModalGuiderFactory` subclass.

Upstream references (commit 76730e6, /tmp/ltx2-research/ltx2):
  - LtxAPGGuider.delta              : guiders.py:110-122
  - CFGStarRescalingGuider.delta    : guiders.py:47-49
  - MultiModalGuider.calculate      : guiders.py:244-268 (the formula we replace)
  - create_multimodal_guider_factory: guiders.py:340-355  (the rebuild path we patch)

Design:
  1. `MultiModalAPGGuider` / `MultiModalCFGStarGuider` are frozen-dataclass
     subclasses of `MultiModalGuider` with overridden `.calculate()`. They share
     the parent's MultiModalGuiderParams (cfg_scale, stg_scale, modality_scale,
     rescale_scale, skip_step) so STG and modality CFG remain available and
     post-rescale still applies.
  2. `MultiModalAPGGuiderFactory` / `MultiModalCFGStarGuiderFactory` subclass
     `MultiModalGuiderFactory` and override `build_from_sigma` to instantiate
     the subclass guider. They also expose `_rebind_negative_context`, used by
     the patched `create_multimodal_guider_factory` to survive the upstream
     negative-context rebuild path WITHOUT downcasting back to the base factory.
  3. `install_factory_preserving_patch()` monkey-patches
     `create_multimodal_guider_factory` in `ltx_core.components.guiders` AND in
     every consumer module that imported it by name (ti2vid_two_stages,
     ti2vid_one_stage, keyframe_interpolation). The replacement preserves the
     caller's factory subclass via `type(params)._rebind_negative_context` when
     available, falling back to the upstream rebuild otherwise. Idempotent.

HQ caveat: `TI2VidTwoStagesHQPipeline.__call__` directly constructs
`MultiModalGuider(params=..., negative_context=...)` (ti2vid_two_stages_hq.py:177)
instead of going through `create_multimodal_guider_factory`, so this factory swap
cannot reach HQ mode. Callers selecting `guider != 'cfg'` together with
`mode='hq'` must be rejected by the endpoint.

Calibration guidance (document in README, not enforced here):
  - APG  : typically wants lower cfg_scale (e.g. 1.5 instead of 3.0); start with
           apg_eta=0.0 (drop the parallel component entirely — most aggressive
           projection) or 1.0 (preserve full parallel, mildest variant). Norm
           threshold 0.0 disables clipping; 5.0-15.0 is the LegacyStatefulAPGGuider
           default range when used.
  - CFG★ : same cfg_scale as vanilla CFG (no calibration shift needed).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace as dc_replace

import torch

from ltx_core.components.guiders import (
    MultiModalGuider,
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    projection_coef,
)


# ---------------------------------------------------------------------------
# Guider subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiModalAPGGuider(MultiModalGuider):
    """MultiModalGuider variant that applies Adaptive Projected Guidance (APG)
    to the text-conditioning branch.

    Replaces the linear CFG combo
        (cfg_scale - 1) * (cond - uncond_text)
    with the APG combo (cf. LtxAPGGuider.delta in upstream guiders.py:110-122):
        guidance = cond - uncond_text
        (optional) clip ||guidance|| to apg_norm_threshold
        g_parallel = projection_coef(guidance, cond) * cond
        g_orth     = guidance - g_parallel
        g_apg      = apg_eta * g_parallel + g_orth
        delta      = (cfg_scale - 1) * g_apg

    STG, modality CFG, and rescale_scale terms are preserved verbatim from
    `MultiModalGuider.calculate`.
    """

    apg_eta: float = 1.0
    apg_norm_threshold: float = 0.0

    def calculate(
        self,
        cond: torch.Tensor,
        uncond_text: "torch.Tensor | float",
        uncond_perturbed: "torch.Tensor | float",
        uncond_modality: "torch.Tensor | float",
    ) -> torch.Tensor:
        p = self.params
        pred = cond

        if p.cfg_scale != 1.0:
            guidance = cond - uncond_text
            if self.apg_norm_threshold > 0:
                ones = torch.ones_like(guidance)
                guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale_factor = torch.minimum(ones, self.apg_norm_threshold / guidance_norm)
                guidance = guidance * scale_factor
            proj = projection_coef(guidance, cond)
            g_parallel = proj * cond
            g_orth = guidance - g_parallel
            g_apg = g_parallel * self.apg_eta + g_orth
            pred = pred + (p.cfg_scale - 1) * g_apg

        if p.stg_scale != 0:
            pred = pred + p.stg_scale * (cond - uncond_perturbed)
        if p.modality_scale != 1.0:
            pred = pred + (p.modality_scale - 1) * (cond - uncond_modality)
        if p.rescale_scale != 0:
            factor = cond.std() / pred.std()
            factor = p.rescale_scale * factor + (1 - p.rescale_scale)
            pred = pred * factor
        return pred


@dataclass(frozen=True)
class MultiModalCFGStarGuider(MultiModalGuider):
    """MultiModalGuider variant that applies CFG★ rescaling to the text branch.

    Replaces the linear CFG combo with CFG★ (cf. CFGStarRescalingGuider.delta
    in upstream guiders.py:47-49):
        rescaled_uncond = projection_coef(cond, uncond_text) * uncond_text
        delta = (cfg_scale - 1) * (cond - rescaled_uncond)

    The unconditioned sample is rescaled to match the conditioned sample's norm,
    so the guidance step stays mostly along the conditioning axis. STG, modality
    CFG, and rescale_scale are preserved.
    """

    def calculate(
        self,
        cond: torch.Tensor,
        uncond_text: "torch.Tensor | float",
        uncond_perturbed: "torch.Tensor | float",
        uncond_modality: "torch.Tensor | float",
    ) -> torch.Tensor:
        p = self.params
        pred = cond

        if p.cfg_scale != 1.0:
            rescaled = projection_coef(cond, uncond_text) * uncond_text
            pred = pred + (p.cfg_scale - 1) * (cond - rescaled)

        if p.stg_scale != 0:
            pred = pred + p.stg_scale * (cond - uncond_perturbed)
        if p.modality_scale != 1.0:
            pred = pred + (p.modality_scale - 1) * (cond - uncond_modality)
        if p.rescale_scale != 0:
            factor = cond.std() / pred.std()
            factor = p.rescale_scale * factor + (1 - p.rescale_scale)
            pred = pred * factor
        return pred


# ---------------------------------------------------------------------------
# Factory subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiModalAPGGuiderFactory(MultiModalGuiderFactory):
    """Factory that builds `MultiModalAPGGuider` for each sigma.

    Mirrors `MultiModalGuiderFactory` ctor/classmethods, but every guider it
    yields carries `apg_eta` / `apg_norm_threshold` and routes through the APG
    calculate path.
    """

    apg_eta: float = 1.0
    apg_norm_threshold: float = 0.0

    @classmethod
    def constant(
        cls,
        params: MultiModalGuiderParams,
        negative_context: "torch.Tensor | None" = None,
        apg_eta: float = 1.0,
        apg_norm_threshold: float = 0.0,
    ) -> "MultiModalAPGGuiderFactory":
        return cls(
            negative_context=negative_context,
            _params_by_sigma=((float("inf"), params),),
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
        )

    def build_from_sigma(self, sigma) -> MultiModalAPGGuider:
        return MultiModalAPGGuider(
            params=self.params(sigma),
            negative_context=self.negative_context,
            apg_eta=self.apg_eta,
            apg_norm_threshold=self.apg_norm_threshold,
        )

    def _rebind_negative_context(self, negative_context) -> "MultiModalAPGGuiderFactory":
        return dc_replace(self, negative_context=negative_context)


@dataclass(frozen=True)
class MultiModalCFGStarGuiderFactory(MultiModalGuiderFactory):
    """Factory that builds `MultiModalCFGStarGuider` for each sigma."""

    @classmethod
    def constant(
        cls,
        params: MultiModalGuiderParams,
        negative_context: "torch.Tensor | None" = None,
    ) -> "MultiModalCFGStarGuiderFactory":
        return cls(
            negative_context=negative_context,
            _params_by_sigma=((float("inf"), params),),
        )

    def build_from_sigma(self, sigma) -> MultiModalCFGStarGuider:
        return MultiModalCFGStarGuider(
            params=self.params(sigma),
            negative_context=self.negative_context,
        )

    def _rebind_negative_context(self, negative_context) -> "MultiModalCFGStarGuiderFactory":
        return dc_replace(self, negative_context=negative_context)


# ---------------------------------------------------------------------------
# Factory-preserving monkey-patch
# ---------------------------------------------------------------------------


def install_factory_preserving_patch() -> None:
    """Monkey-patch `create_multimodal_guider_factory` in upstream pipeline
    modules so that custom `MultiModalGuiderFactory` subclasses survive the
    `negative_context` rebinding the pipelines perform inside `__call__`.

    Upstream behaviour (ltx_core.components.guiders.create_multimodal_guider_factory):
      if isinstance(params, MultiModalGuiderFactory):
          if negative_context is not None and params.negative_context is not negative_context:
              return MultiModalGuiderFactory.from_dict(
                  dict(params._params_by_sigma), negative_context=negative_context,
              )
          return params
      return MultiModalGuiderFactory.constant(params, negative_context=negative_context)

    The `from_dict` rebuild is called on the BASE class, so any subclass type
    is lost. That makes pure-subclass swapping unworkable.

    Replacement: when the caller passes a factory subclass that defines
    `_rebind_negative_context`, route through it instead of `from_dict`. Pure
    base-class factory and raw-params paths are unchanged.

    Patched call sites (re-bound module-local references because each module
    did `from ltx_core.components.guiders import create_multimodal_guider_factory`):
      - ltx_core.components.guiders
      - ltx_pipelines.ti2vid_two_stages
      - ltx_pipelines.ti2vid_one_stage
      - ltx_pipelines.keyframe_interpolation

    Idempotent: returns immediately on re-call.
    """
    import ltx_core.components.guiders as _g_mod

    if getattr(_g_mod.create_multimodal_guider_factory, "_ltx2_w6_patched", False):
        return

    def _preserving_create(
        params,
        negative_context=None,
    ):
        if isinstance(params, MultiModalGuiderFactory):
            if (
                negative_context is not None
                and params.negative_context is not negative_context
            ):
                rebind = getattr(params, "_rebind_negative_context", None)
                if rebind is not None:
                    return rebind(negative_context)
                return MultiModalGuiderFactory.from_dict(
                    dict(params._params_by_sigma),
                    negative_context=negative_context,
                )
            return params
        return MultiModalGuiderFactory.constant(
            params, negative_context=negative_context
        )

    _preserving_create._ltx2_w6_patched = True

    _g_mod.create_multimodal_guider_factory = _preserving_create

    # Re-bind in every module that imported the symbol by name.
    for _mod_name in (
        "ltx_pipelines.ti2vid_two_stages",
        "ltx_pipelines.ti2vid_one_stage",
        "ltx_pipelines.keyframe_interpolation",
    ):
        try:
            _mod = __import__(_mod_name, fromlist=["create_multimodal_guider_factory"])
        except Exception:
            continue
        if hasattr(_mod, "create_multimodal_guider_factory"):
            _mod.create_multimodal_guider_factory = _preserving_create


# ---------------------------------------------------------------------------
# Convenience builders for ltx2_model.py
# ---------------------------------------------------------------------------


def wrap_params_with_guider(
    params: MultiModalGuiderParams,
    guider: str,
    apg_eta: float = 1.0,
    apg_norm_threshold: float = 0.0,
):
    """Return either the raw params (cfg path) or a factory subclass instance
    that the pipeline can pass through `create_multimodal_guider_factory` and
    end up with our custom guider for stage-1 denoising.

    `guider` must be one of: 'cfg' | 'apg' | 'cfg_star'.
    """
    if guider == "cfg":
        return params
    if guider == "apg":
        return MultiModalAPGGuiderFactory.constant(
            params,
            negative_context=None,
            apg_eta=apg_eta,
            apg_norm_threshold=apg_norm_threshold,
        )
    if guider == "cfg_star":
        return MultiModalCFGStarGuiderFactory.constant(
            params, negative_context=None,
        )
    raise ValueError(
        f"Unknown guider {guider!r}. Use 'cfg', 'apg', or 'cfg_star'."
    )
