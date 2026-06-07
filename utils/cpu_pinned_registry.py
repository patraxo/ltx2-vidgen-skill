"""CpuPinnedStateDictRegistry — host-pinned drop-in for
`ltx_core.loader.registry.StateDictRegistry`.

Holds StateDict tensors in CPU pinned memory. On `.get()`, streams a fresh
GPU copy via a dedicated CUDA stream. Never leaves a GPU reference behind
after the call returns, so the registry itself contributes ZERO bytes of
permanent GPU pressure.

Replaces the GPU-resident `StateDictRegistry` from
`packages/ltx-core/src/ltx_core/loader/registry.py` which OOMs the H100 80GB
when it caches stage_1 (~44GB) + stage_2 (~44GB) + Gemma (~24GB) + VAE
state-dicts simultaneously (~150GB+).

References:
- PyTorch pinned-memory + non_blocking guide:
  https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.md
- Cross-stream caching-allocator hazard: https://github.com/pytorch/pytorch/issues/113622
- Pinned-memory leak fix in torch 2.5+: https://github.com/pytorch/pytorch/pull/131270
- Diffusers group offloading pattern: https://github.com/huggingface/diffusers/pull/11682
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from ltx_core.loader.primitives import StateDict
from ltx_core.loader.registry import DummyRegistry  # base class — see CpuPinnedStateDictRegistry below
from ltx_core.loader.sd_ops import SDOps

logger = logging.getLogger(__name__)

_DEFAULT_MAX_GB = float(os.environ.get("LTX_REGISTRY_MAX_GB", "80"))
_DEFAULT_MAX_BYTES = int(_DEFAULT_MAX_GB * (1 << 30))


@dataclass
class _Entry:
    sd_pinned: dict[str, torch.Tensor]
    dtypes: set[torch.dtype]
    nbytes: int
    # 2026-05-23 fix: remember the device the loader originally produced
    # tensors on. LoRA loads use device=cpu; transformer loads use device=cuda.
    # We always cache CPU-pinned but stream back to whichever device the
    # loader wants. Without this, LoRA .get() returns GPU tensors and we
    # waste 1-5 GB GPU per LoRA build.
    target_device: torch.device


# 2026-05-23 fix: inherit from DummyRegistry so SingleGPUModelBuilder.build's
# `isinstance(self.registry, DummyRegistry)` check passes, which keeps
# `apply_loras(destination_sd=model_state_dict)` (in-place fusion) instead of
# `apply_loras(destination_sd=None)` (allocates a FRESH ~44GB GPU state-dict
# for the fused output — doubled GPU pressure during every LoRA build).
# In-place is safe for us because `.get()` always returns a FRESH StateDict
# (fresh CPU clones or fresh GPU tensors), never the cached internal storage.
class CpuPinnedStateDictRegistry(DummyRegistry):
    """Drop-in for ltx_core.loader.registry.StateDictRegistry.

    Cache lives in pinned host memory; `.get()` returns a fresh StateDict on
    the device the original loader used, so the registry holds zero GPU
    references and never blocks in-place LoRA fusion.

    Inherits from `DummyRegistry` so that
    `SingleGPUModelBuilder.build()`'s `isinstance(self.registry, DummyRegistry)`
    check passes — which makes `apply_loras` use in-place fusion
    (`destination_sd=model_state_dict`) instead of allocating a fresh ~44GB
    fused state-dict on GPU.
    """

    def __init__(self, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._max_bytes = max_bytes
        self._cur_bytes = 0
        self._copy_stream: torch.cuda.Stream | None = None
        logger.info(
            "CpuPinnedStateDictRegistry init: max_bytes=%.1f GB",
            max_bytes / (1 << 30),
        )

    # ---------------------------------------------------------------- id ----
    def _generate_id(self, paths: list[str], sd_ops: SDOps | None) -> str:
        m = hashlib.sha256()
        parts = [str(Path(p).resolve()) for p in paths]
        if sd_ops is not None:
            parts.append(sd_ops.name)
        m.update("\0".join(parts).encode("utf-8"))
        return m.hexdigest()

    # --------------------------------------------------------------- pin ----
    @staticmethod
    def _to_pinned(t: torch.Tensor) -> torch.Tensor:
        if t.device.type == "cuda":
            t = t.detach().to("cpu", copy=True)
        elif not t.is_pinned():
            t = t.detach().clone()
        else:
            return t.detach()
        try:
            return t.pin_memory()
        except RuntimeError as e:
            logger.warning(
                "pin_memory() failed (%s) — falling back to non-pinned host",
                e,
            )
            return t  # graceful degradation: still cached, just slower H2D

    @staticmethod
    def _nbytes(sd: dict[str, torch.Tensor]) -> int:
        return sum(v.numel() * v.element_size() for v in sd.values())

    # ------------------------------------------------------------- stream ---
    def _get_copy_stream(self) -> torch.cuda.Stream:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    # ------------------------------------------------------------- evict ----
    def _evict_locked(self, want_bytes: int) -> None:
        while self._cur_bytes + want_bytes > self._max_bytes and self._entries:
            ev_id, ev = self._entries.popitem(last=False)
            self._cur_bytes -= ev.nbytes
            logger.warning(
                "CpuPinnedStateDictRegistry: LRU-evict %s (%.1f GB) — host pressure",
                ev_id[:12],
                ev.nbytes / (1 << 30),
            )
            del ev

    # --------------------------------------------------------------- API ----
    def add(
        self,
        paths: list[str],
        sd_ops: SDOps | None,
        state_dict: StateDict,
    ) -> str:
        sd_id = self._generate_id(paths, sd_ops)
        with self._lock:
            if sd_id in self._entries:
                self._entries.move_to_end(sd_id)
                return sd_id

            pinned = {k: self._to_pinned(v) for k, v in state_dict.sd.items()}
            nbytes = self._nbytes(pinned)
            self._evict_locked(nbytes)
            self._entries[sd_id] = _Entry(
                sd_pinned=pinned,
                dtypes=set(state_dict.dtype),
                nbytes=nbytes,
                target_device=state_dict.device,
            )
            self._cur_bytes += nbytes
            logger.info(
                "CpuPinnedStateDictRegistry: add %s (%.1f GB, target=%s), "
                "total=%.1f / %.1f GB",
                sd_id[:12],
                nbytes / (1 << 30),
                state_dict.device,
                self._cur_bytes / (1 << 30),
                self._max_bytes / (1 << 30),
            )
        return sd_id

    def get(
        self,
        paths: list[str],
        sd_ops: SDOps | None,
    ) -> StateDict | None:
        sd_id = self._generate_id(paths, sd_ops)
        with self._lock:
            entry = self._entries.get(sd_id)
            if entry is None:
                return None
            self._entries.move_to_end(sd_id)
            pinned = entry.sd_pinned
            dtypes = entry.dtypes
            target = entry.target_device

        # 2026-05-23 fix: honor the device the loader originally produced.
        # LoRAs are loaded with device=cpu — returning GPU tensors here would
        # waste 1-5 GB GPU per LoRA build and break the lora_load_device=cpu
        # contract upstream uses to bound peak GPU memory during fusion.
        if target.type != "cuda":
            # Return CPU clones (not the pinned originals — caller's assign=True
            # would hand model.params a ref to our cache, then model.to("meta")
            # would NOT release them since we still hold the ref → memory leak
            # in the form of "phantom" CPU tensors that never get unpinned).
            cpu_sd = {k: v.clone() for k, v in pinned.items()}
            return StateDict(
                sd=cpu_sd,
                device=target,
                size=entry.nbytes,
                dtype=set(dtypes),
            )

        copy_stream = self._get_copy_stream()
        gpu_sd: dict[str, torch.Tensor] = {}
        with torch.cuda.stream(copy_stream):
            for k, src in pinned.items():
                # Use torch.empty(shape,...) instead of empty_like to be
                # explicit about device override regardless of `src`'s device.
                dst = torch.empty(
                    src.shape,
                    dtype=src.dtype,
                    device=target,
                )
                dst.copy_(src, non_blocking=True)
                gpu_sd[k] = dst
        # Make the current (default) stream wait for the copy to complete BEFORE
        # the caller can read gpu_sd. Prevents pytorch#113622 cross-stream
        # caching-allocator use-after-free.
        torch.cuda.current_stream().wait_stream(copy_stream)
        return StateDict(
            sd=gpu_sd,
            device=target,
            size=entry.nbytes,
            dtype=set(dtypes),
        )

    def pop(self, paths: list[str], sd_ops: SDOps | None) -> StateDict | None:
        sd_id = self._generate_id(paths, sd_ops)
        with self._lock:
            entry = self._entries.pop(sd_id, None)
            if entry is None:
                return None
            self._cur_bytes -= entry.nbytes
        return StateDict(
            sd=entry.sd_pinned,
            device=torch.device("cpu"),
            size=entry.nbytes,
            dtype=set(entry.dtypes),
        )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._cur_bytes = 0
        logger.info("CpuPinnedStateDictRegistry: cleared")

    # --------------------------------------------------------- diagnostics --
    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "host_gb": self._cur_bytes / (1 << 30),
                "max_gb": self._max_bytes / (1 << 30),
                "ids": [k[:12] for k in self._entries.keys()],
            }
