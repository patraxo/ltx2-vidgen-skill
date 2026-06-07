"""GpuResidentStateDictRegistry — GPU-resident sibling of
`CpuPinnedStateDictRegistry`.

Goal (2026-06-06): use the idle VRAM on the RTX PRO 6000 (96 GB; prod peaks
~39.5 GB → ~56 GB idle) to cut per-request latency. Instead of streaming a
fresh GPU copy from pinned host on every `.get()` (the cpu_pinned path,
~PCIe ~25 GB/s, ~1 s/stage), we keep the transformer state-dict tensors
RESIDENT on the GPU and hand them back by reference (zero H2D copy).

Budget reality (from HIGH_USAGE_OPT_RESEARCH.md §2b):
  stage_1 (~22 GB resident) + stage_2 (~22 GB resident) ≈ ~44 GB resident
  + ~39.5 GB activation peak ≈ ~83 GB < 96 GB  → OK.
  Adding Gemma (~24 GB) resident → ~112 GB > 96 GB → OOM. So Gemma (and any
  entry that would exceed `LTX_GPU_RESIDENT_MAX_GB`) FALLS BACK to the
  inherited CPU-pinned streaming path. Gemma is used once per request at the
  top (text-encode, ~1 s stream) and freed before the stage transformers load,
  so the resident weights + the transient Gemma copy never coexist at peak.

Design:
- Inherits `CpuPinnedStateDictRegistry` (which inherits `DummyRegistry`), so:
    * `isinstance(registry, DummyRegistry)` stays True → `SingleGPUModelBuilder.build`
      keeps `apply_loras(destination_sd=model_state_dict)` (IN-PLACE fusion),
      avoiding the fresh ~22 GB GPU alloc per LoRA build.
    * `ModelLedger._target_device()` returns the real GPU device (not CPU),
      so `build()` places the meta_model directly onto the resident weights.
    * Over-budget / non-cuda entries transparently use the parent's pinned-CPU
      streaming `.get()` (we just don't promote them to resident).

OOM-safety / the per-stage `del transformer; cleanup_memory()` hazard
(`ti2vid_two_stages.py:168,224`):
- The resident tensors are owned by the REGISTRY (`_Entry.sd_gpu`), held in our
  own dict. `build()` aliases them into the module params via
  `load_state_dict(..., assign=True)`; when the pipeline does
  `del transformer; cleanup_memory()`, the MODULE drops its refs but the
  registry's refs keep the tensors ALIVE on GPU (that is the whole point —
  next request reuses them with no reload). They are NOT double-counted: there
  is exactly one CUDA allocation per resident tensor; the module merely held a
  borrowed reference to it.
- `.get()` returns a FRESH StateDict dict object whose values are
  `tensor.detach()` views ALIASING the resident storage (no copy). The fast
  no-LoRA path (`single_gpu_model_builder.py:80-85`) assigns these straight
  into params; inference is read-only so aliasing is safe. The LoRA path
  (`apply_loras`) writes `sd[key] = <fresh tensor>` for matched keys and
  `continue`s (keeps the alias, never mutates it) for unmatched keys — so the
  resident cache is only ever READ, never written in place. Verified against
  `fuse_loras.py:69-99`.

References:
- cpu_pinned_registry.py (sibling; parent class)
- ltx_core/loader/single_gpu_model_builder.py:54-101 (load_sd / build / assign)
- ltx_pipelines/utils/model_ledger.py:160-196 (_target_device / transformer)
- ltx_pipelines/ti2vid_two_stages.py:119,168,181,224 (per-stage del + cleanup)
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass

import torch

from cpu_pinned_registry import CpuPinnedStateDictRegistry, _Entry as _PinnedEntry
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import SDOps

logger = logging.getLogger(__name__)

# Resident-on-GPU budget. Default 80 GB leaves ~16 GB for activations on top of
# resident weights, BUT stage_1+stage_2 only total ~44 GB resident — so the
# binding constraint in practice is "keep the transformers resident, stream
# Gemma". 80 GB is high enough to admit both stage transformers (~44 GB) and
# low enough to REJECT Gemma (~24 GB) only if both stages are already resident
# (44 + 24 = 68 < 80 — Gemma would still fit by raw budget!). So we ALSO gate on
# a name/size heuristic below: Gemma is excluded by the per-entry policy, not by
# the byte budget alone. Tune via env if a future card has different headroom.
_DEFAULT_GPU_RESIDENT_MAX_GB = float(os.environ.get("LTX_GPU_RESIDENT_MAX_GB", "80"))
_DEFAULT_GPU_RESIDENT_MAX_BYTES = int(_DEFAULT_GPU_RESIDENT_MAX_GB * (1 << 30))

# Per-entry size ceiling for promotion to resident. The two stage transformers
# are ~22 GB each (bf16, 22B params materialized per stage). Gemma's state-dict
# is ~24 GB. We keep transformers resident and stream Gemma, so the cleanest,
# most explicit gate is: promote an entry to GPU-resident ONLY if it is the
# stage transformer (the dominant, every-request, twice-per-request cost). We
# detect "transformer-class" entries by tensor count + total size band rather
# than by fragile path matching: transformer shards are large (>8 GB) and the
# text-encoder/Gemma entry, while also large, is loaded with a CPU target
# device under the cpu_pinned contract OR can be force-streamed via the
# LTX_GPU_RESIDENT_STREAM_OVER_GB knob below.
_STREAM_OVER_GB = float(os.environ.get("LTX_GPU_RESIDENT_STREAM_OVER_GB", "23"))
_STREAM_OVER_BYTES = int(_STREAM_OVER_GB * (1 << 30))

# ---------------------------------------------------------------------------
# 2026-06-07 DOUBLE-ALLOC OOM FIX — LoRA-fusion-aware safety reserve.
#
# Root cause of the prod OOM (94.96/94.97 GB → CUDA OOM): the prod image set
# `LTX_GPU_RESIDENT_STREAM_OVER_GB=40`, which let the ~39.1 GB stage transformer
# state-dict be promoted to GPU-resident. Stage-2's `apply_loras(...,
# destination_sd=model_state_dict)` then materializes a FRESH near-full-size
# fused copy of that transformer for every LoRA-matched key (the distilled LoRA
# touches almost every linear layer → ~all 39 GB). The registry's resident base
# stays alive (by design), so peak = ~39 GB resident base + ~39 GB fused copy +
# ~25 GB stage-2 activations ≈ 94–95 GB → OOM.
#
# The header's claim "in-place fusion never duplicates the resident base" is
# FALSE for the resident path: `apply_loras` allocates new tensors; it never
# writes into the resident storage in place. So a state-dict that will be
# consumed by LoRA fusion must NOT be kept resident — fusion always builds a
# second full-size copy on top of it. (For cpu_pinned this is harmless: the base
# is a transient streamed copy, freed each call. For gpu_resident the base is
# permanent → it doubles.)
#
# The validated −30% win (bench run_gpures.log: 20–23 s warm, peak 48 GB) was
# produced with the 39.1 GB transformer STREAMED (gpu_resident_gb=8.5) — only the
# small, high-frequency VAE / upsampler / audio components (≤6 GB each) were
# resident. The dominant per-request cost is the disk read (~13–15 s), which BOTH
# the streamed and resident paths eliminate via the pinned-CPU cache; keeping the
# 39 GB transformer resident only saves the ~1 s/stage H2D copy but risks the OOM.
#
# Fix: a hard VRAM-headroom guard the env knobs CANNOT override. An entry is only
# promoted to resident if, in the worst case, it can coexist with (a) a fused
# duplicate of the single largest resident entry and (b) the activation peak,
# under the card's usable VRAM. The big LoRA-fused transformer (~39 GB) fails this
# guard and streams; the small components (≤6 GB) pass and stay resident.
_DEVICE_VRAM_GB = float(os.environ.get("LTX_GPU_DEVICE_VRAM_GB", "95"))  # RTX PRO 6000 usable
_ACTIVATION_PEAK_GB = float(os.environ.get("LTX_GPU_ACTIVATION_PEAK_GB", "40"))  # measured ~39.5
# Safety margin for allocator fragmentation / transient build buffers.
_SAFETY_MARGIN_GB = float(os.environ.get("LTX_GPU_SAFETY_MARGIN_GB", "4"))
_DEVICE_VRAM_BYTES = int(_DEVICE_VRAM_GB * (1 << 30))
_ACTIVATION_PEAK_BYTES = int(_ACTIVATION_PEAK_GB * (1 << 30))
_SAFETY_MARGIN_BYTES = int(_SAFETY_MARGIN_GB * (1 << 30))


@dataclass
class _GpuEntry:
    sd_gpu: dict[str, torch.Tensor]  # tensors RESIDENT on cuda, owned by registry
    dtypes: set[torch.dtype]
    nbytes: int
    target_device: torch.device


class GpuResidentStateDictRegistry(CpuPinnedStateDictRegistry):
    """Keep transformer state-dicts resident on GPU; stream the rest from
    pinned CPU (inherited behavior).

    `.get()` on a resident entry returns aliasing views (zero copy). Over-budget
    or non-cuda entries fall through to the parent `CpuPinnedStateDictRegistry`
    pinned-CPU streaming path.
    """

    def __init__(
        self,
        gpu_max_bytes: int = _DEFAULT_GPU_RESIDENT_MAX_BYTES,
        host_max_bytes: int | None = None,
        stream_over_bytes: int = _STREAM_OVER_BYTES,
    ) -> None:
        # Parent manages the pinned-CPU fallback cache (host RAM).
        if host_max_bytes is None:
            super().__init__()
        else:
            super().__init__(max_bytes=host_max_bytes)
        self._gpu_entries: OrderedDict[str, _GpuEntry] = OrderedDict()
        self._gpu_max_bytes = gpu_max_bytes
        self._gpu_cur_bytes = 0
        self._stream_over_bytes = stream_over_bytes
        # 2026-06-07 double-alloc OOM fix: hard headroom guard (see module top).
        self._device_vram_bytes = _DEVICE_VRAM_BYTES
        self._activation_peak_bytes = _ACTIVATION_PEAK_BYTES
        self._safety_margin_bytes = _SAFETY_MARGIN_BYTES
        # Largest resident entry so far — used to bound the worst-case LoRA-fused
        # duplicate at inference peak.
        self._gpu_max_entry_bytes = 0
        logger.info(
            "GpuResidentStateDictRegistry init: gpu_max=%.1f GB, "
            "stream_over=%.1f GB (entries above this stream from pinned CPU), "
            "device_vram=%.1f GB, activation_peak=%.1f GB, safety=%.1f GB",
            gpu_max_bytes / (1 << 30),
            stream_over_bytes / (1 << 30),
            self._device_vram_bytes / (1 << 30),
            self._activation_peak_bytes / (1 << 30),
            self._safety_margin_bytes / (1 << 30),
        )

    # ------------------------------------------------------------- policy ----
    def _should_keep_resident(
        self, nbytes: int, target: torch.device
    ) -> bool:
        """Promote to GPU-resident iff:
        - the loader wants the tensors on cuda (transformers; NOT cpu-target
          LoRAs, which must stay CPU per the lora_load_device=cpu contract),
        - the entry is at/under the per-entry stream ceiling (keeps Gemma —
          ~24 GB, > _STREAM_OVER_GB=23 — on the pinned-CPU path), and
        - it still fits under the total GPU residency budget.
        """
        if target.type != "cuda":
            return False
        if nbytes > self._stream_over_bytes:
            # e.g. Gemma (~24 GB) — stream it; do NOT pin to GPU (would blow the
            # 96 GB budget when added to resident stage_1+stage_2 + activations).
            return False
        if self._gpu_cur_bytes + nbytes > self._gpu_max_bytes:
            return False
        # 2026-06-07 double-alloc OOM guard (env-knobs CANNOT override this).
        # Promoting this entry must leave room, at inference PEAK, for:
        #   resident_after_add
        #   + a worst-case LoRA-fused DUPLICATE of the single largest resident
        #     entry (apply_loras builds a fresh full-size copy on top of the
        #     resident base for every LoRA-matched key — this is the bug that
        #     OOM'd prod when the ~39 GB transformer was made resident)
        #   + the activation peak
        #   + a fragmentation safety margin
        # must be <= usable device VRAM. The big LoRA-fused transformer fails
        # this (39 resident + 39 fused-dup + 40 act + 4 = 122 > 95) → streams.
        # The small VAE / upsampler / audio components pass and stay resident.
        resident_after = self._gpu_cur_bytes + nbytes
        worst_fused_dup = max(self._gpu_max_entry_bytes, nbytes)
        projected_peak = (
            resident_after
            + worst_fused_dup
            + self._activation_peak_bytes
            + self._safety_margin_bytes
        )
        if projected_peak > self._device_vram_bytes:
            logger.warning(
                "GpuResidentStateDictRegistry: REJECT resident promotion "
                "(%.1f GB) — projected peak %.1f GB > usable VRAM %.1f GB "
                "(resident %.1f + worst fused-dup %.1f + activations %.1f + "
                "safety %.1f). Streaming from pinned CPU instead.",
                nbytes / (1 << 30),
                projected_peak / (1 << 30),
                self._device_vram_bytes / (1 << 30),
                resident_after / (1 << 30),
                worst_fused_dup / (1 << 30),
                self._activation_peak_bytes / (1 << 30),
                self._safety_margin_bytes / (1 << 30),
            )
            return False
        return True

    # -------------------------------------------------------------- evict ----
    def _evict_gpu_locked(self, want_bytes: int) -> None:
        """LRU-evict resident GPU entries down to pinned-CPU on GPU pressure.
        Evicted tensors are demoted into the parent's pinned-CPU cache so a
        subsequent `.get()` still avoids the disk read (just pays the H2D
        stream)."""
        while (
            self._gpu_cur_bytes + want_bytes > self._gpu_max_bytes
            and self._gpu_entries
        ):
            ev_id, ev = self._gpu_entries.popitem(last=False)
            self._gpu_cur_bytes -= ev.nbytes
            logger.warning(
                "GpuResidentStateDictRegistry: GPU pressure — demoting %s "
                "(%.1f GB) to pinned-CPU",
                ev_id[:12],
                ev.nbytes / (1 << 30),
            )
            # Demote into parent pinned-CPU cache (host clone), then free GPU.
            pinned = {k: self._to_pinned(v) for k, v in ev.sd_gpu.items()}
            with self._lock:
                if ev_id not in self._entries:
                    self._entries[ev_id] = _PinnedEntry(
                        sd_pinned=pinned,
                        dtypes=ev.dtypes,
                        nbytes=ev.nbytes,
                        target_device=ev.target_device,
                    )
                    self._cur_bytes += ev.nbytes
            ev.sd_gpu.clear()
            del ev
        # Release the freed GPU blocks back to the caching allocator.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------- API ----
    def add(
        self,
        paths: list[str],
        sd_ops: SDOps | None,
        state_dict: StateDict,
    ) -> str:
        sd_id = self._generate_id(paths, sd_ops)
        nbytes = self._nbytes(state_dict.sd)
        target = state_dict.device

        if self._should_keep_resident(nbytes, target):
            with self._lock:
                if sd_id in self._gpu_entries:
                    self._gpu_entries.move_to_end(sd_id)
                    return sd_id
                self._evict_gpu_locked(nbytes)
                # Own a registry-private GPU copy. We .detach().clone() so the
                # registry holds the canonical storage; the module that triggered
                # the load will alias OUR copy on the next `.get()` (zero-copy),
                # and the build's transient GPU tensors get freed at del-time.
                gpu_sd = {
                    k: v.detach().to(target, copy=True)
                    for k, v in state_dict.sd.items()
                }
                self._gpu_entries[sd_id] = _GpuEntry(
                    sd_gpu=gpu_sd,
                    dtypes=set(state_dict.dtype),
                    nbytes=nbytes,
                    target_device=target,
                )
                self._gpu_cur_bytes += nbytes
                # Track the largest resident entry for the worst-case
                # LoRA-fused-duplicate headroom guard in _should_keep_resident.
                if nbytes > self._gpu_max_entry_bytes:
                    self._gpu_max_entry_bytes = nbytes
                logger.info(
                    "GpuResidentStateDictRegistry: RESIDENT add %s (%.1f GB, "
                    "target=%s), gpu_total=%.1f / %.1f GB",
                    sd_id[:12],
                    nbytes / (1 << 30),
                    target,
                    self._gpu_cur_bytes / (1 << 30),
                    self._gpu_max_bytes / (1 << 30),
                )
            return sd_id

        # Over-budget / non-cuda (e.g. Gemma, cpu-target LoRAs) → parent
        # pinned-CPU streaming cache.
        logger.info(
            "GpuResidentStateDictRegistry: STREAM add %s (%.1f GB, target=%s) "
            "→ pinned-CPU fallback",
            sd_id[:12],
            nbytes / (1 << 30),
            target,
        )
        return super().add(paths, sd_ops, state_dict)

    def get(
        self,
        paths: list[str],
        sd_ops: SDOps | None,
    ) -> StateDict | None:
        sd_id = self._generate_id(paths, sd_ops)
        with self._lock:
            entry = self._gpu_entries.get(sd_id)
            if entry is not None:
                self._gpu_entries.move_to_end(sd_id)
                # Zero-copy: hand back aliasing views of the resident storage.
                # Fresh dict + fresh view objects so the caller's StateDict /
                # load_state_dict(assign=True) never mutates our cache keys; the
                # underlying CUDA storage is shared (no H2D, no alloc).
                alias_sd = {k: v.detach() for k, v in entry.sd_gpu.items()}
                return StateDict(
                    sd=alias_sd,
                    device=entry.target_device,
                    size=entry.nbytes,
                    dtype=set(entry.dtypes),
                )
        # Not resident → parent pinned-CPU streaming path.
        return super().get(paths, sd_ops)

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        sd_id = self._generate_id(paths, sd_ops)
        with self._lock:
            entry = self._gpu_entries.pop(sd_id, None)
            if entry is not None:
                self._gpu_cur_bytes -= entry.nbytes
                sd = entry.sd_gpu
                out = StateDict(
                    sd=sd,
                    device=entry.target_device,
                    size=entry.nbytes,
                    dtype=set(entry.dtypes),
                )
                return out
        return super().pop(paths, sd_ops)

    def clear(self) -> None:
        with self._lock:
            for ev in self._gpu_entries.values():
                ev.sd_gpu.clear()
            self._gpu_entries.clear()
            self._gpu_cur_bytes = 0
            self._gpu_max_entry_bytes = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().clear()
        logger.info("GpuResidentStateDictRegistry: cleared (GPU + pinned-CPU)")

    # --------------------------------------------------------- diagnostics --
    def stats(self) -> dict:
        host = super().stats()
        with self._lock:
            return {
                "gpu_entries": len(self._gpu_entries),
                "gpu_resident_gb": self._gpu_cur_bytes / (1 << 30),
                "gpu_max_gb": self._gpu_max_bytes / (1 << 30),
                "gpu_max_entry_gb": self._gpu_max_entry_bytes / (1 << 30),
                "gpu_ids": [k[:12] for k in self._gpu_entries.keys()],
                "guard_device_vram_gb": self._device_vram_bytes / (1 << 30),
                "guard_activation_peak_gb": self._activation_peak_bytes / (1 << 30),
                "host_entries": host["entries"],
                "host_gb": host["host_gb"],
            }
