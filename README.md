# ltx2-fast

Fast, self-hosted **LTX-2.3** (Lightricks, 22B video DiT) on [Modal](https://modal.com). An optimization stack that turns a 10-second 9:16 reel into **~7–9s of warm generation for a few cents**, on a single **RTX PRO 6000 (Blackwell)** — output verified **bit-identical** to the slow path.

> Text-to-video, image-to-video, and keyframe interpolation. bf16 only — no quality-degrading quantization.

## Why

Hosted video APIs are excellent but charge per clip and rate-limit you. When you iterate — generate 100 takes, keep 3 — the 97 you discard cost exactly as much as the keepers. Running the open model yourself removes the per-clip meter and the rate limit, and lets you batch.

## Measured

| Resolution | Cold (once) | Warm (every clip) | VRAM |
|---|---|---|---|
| 768×512 | 93.6s | **7.3s** | ~40 → ~72 GB |
| 768×1280 (reel) | — | **9.3s** | ~72 GB |

| Optimization | Gain (all bit-identical) |
|---|---|
| Weight cache (CPU-pinned) | −13–15s per cold clip |
| First-Block-Cache | −17%, lossless |
| Text-embedding cache | −4s |
| Pre-fuse + persist pipeline | kills the per-clip rebuild (biggest win) |
| Batch 4 → 32 clips | 3.6× → 6.2× throughput, −84% $/clip |
| **Net** | **1.96× faster, pixel-identical** |

GPU: RTX PRO 6000 (96GB) over H100 → ~45% lower $/clip. Same model in bf16 = same-fidelity frames on any card; the GPU only moves speed/cost (it does, however, gate which precision/attention kernels are *available* — this runs SDPA + bf16 by design).

## Layout

```
deploy/
  ltx2_model.py          # the Modal app (@app.cls Model): t2v / i2v / keyframe + the opt stack
utils/                   # helper package (mounted into the container)
  cpu_pinned_registry.py   # CPU-pinned weight cache (bit-identical, fast warm load)
  gpu_resident_registry.py # resident-weights registry (VRAM-headroom guarded)
  fbcache.py               # First-Block-Cache hook (near-lossless step skipping)
  custom_guiders.py        # guidance helpers (APG / CFG variants)
tests/
  smoke_test.py          # auth-free functional smoke (tiny i2v, verifies the opt stack)
verification/
  ship_verify.py         # prod ship-verification: latency/VRAM, bit-identical, audio-skip
scripts/
  deploy.sh              # one-command deploy (+ api-key secret setup)
requirements.txt         # client-side: just the Modal SDK
```

## Deploy

```bash
pip install modal && modal token new          # one-time
./scripts/deploy.sh                            # no secrets to set up — just deploys
```

### Test it

```bash
PYTHONPATH=. modal run tests/smoke_test.py                                   # tiny smoke
PYTHONPATH=. modal run deploy/ltx2_model.py::smoke_real --image-path pic.jpg  # real-image i2v -> mp4
PYTHONPATH=. modal run verification/ship_verify.py                            # full verification
```

## Secrets & weights

- **No secrets required.** The endpoints are open (no JWT / no api-key) and no HuggingFace key is used. Nothing sensitive lives in this repo.
- **Weights:** provision the LTX-2.3 weights (and the Gemma text encoder) to the Modal volume the app references (see the `download_models` step + volume name in `ltx2_model.py`). The build pulls only public components; if you ever need a *gated* model cold, add an HF token as a Modal secret and wire it into that step — otherwise pre-stage the weights in the volume.

## Notes

- **bf16 only.** fp8 / fp4 / SageAttention were all tested and rejected (noise / black frames / won't build on Blackwell sm_120). Speed comes from caching + residency, not from cheapening the math.
- **Blackwell (sm_120):** flash-attention has no kernel for it yet — this runs PyTorch SDPA, which is exact.
- **Native 9:16:** generate vertical directly; don't render wide and crop (wastes ~66% of paid pixels).

## License

Code: private (personal). LTX-2.3 weights: per Lightricks' license — review before any commercial use.
