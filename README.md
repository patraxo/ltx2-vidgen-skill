# ltx2-fast-inference

Fast, self-hosted **LTX-2.3** (Lightricks, 22B video DiT) on [Modal](https://modal.com) — text-to-video, image-to-video, and keyframe interpolation, with an optimization stack that turns a 10-second 9:16 reel into **~7–9s of warm generation for a few cents** on a single **RTX PRO 6000 (Blackwell)**. Output is verified **bit-identical** to the unoptimized path. bf16 only — no quality-degrading quantization. No secrets, no auth.

---

## Quickstart

```bash
# 1. install the Modal client + authenticate (one-time)
pip install -r requirements.txt        # just the Modal SDK
modal token new

# 2. deploy (no secrets to set up)
./deploy.sh

# 3. test it
PYTHONPATH=. modal run tests/smoke_test.py                                    # tiny auth-free smoke
PYTHONPATH=. modal run deploy/ltx2_model.py::smoke_real --image-path pic.jpg   # real-image i2v -> mp4
PYTHONPATH=. modal run verification/ship_verify.py                             # full verification
```

That's it — `./deploy.sh` pushes the app to Modal and prints the endpoint. The first generation is a cold start (~90s); every clip after that on a warm container is 7–9s.

> **Weights:** the LTX-2.3 weights (+ Gemma text encoder) are pulled into the Modal volume the app references on first build (public components only — no HuggingFace token needed). See the `download_models` step + volume name in `deploy/ltx2_model.py`.

---

## How it works

LTX-2.3 is a 22B video diffusion transformer. Out of the box, serving it has two costs: a slow cold start, and a per-clip cost where the pipeline re-assembles and re-fuses model internals on **every** request. This repo attacks both:

1. **Build once, stay resident.** The fully-assembled, LoRA-fused transformer is kept in memory between clips (resolution-keyed) instead of rebuilt per request. This is the single biggest win — it's what takes a clip from ~90s to ~7s.
2. **Weight caching (CPU-pinned).** Weights are pinned in host RAM and streamed to GPU, skipping disk reads on warm loads. Bit-identical.
3. **First-Block-Cache.** Skips recomputing the early transformer blocks whose features barely change across steps. ~17%, near-lossless.
4. **Embedding cache.** The text encoder isn't re-run for a repeated prompt.
5. **Batching.** Many clips in one warm container amortize the fixed cost — 32 at once ≈ 6.2× throughput.
6. **Blackwell-native attention.** flash-attention has no sm_120 kernel yet, so this runs PyTorch SDPA (exact). fp8/SageAttention were tested and rejected (noise / black frames) — speed comes from architecture, not from cheapening the math.

The Modal app (`@app.cls Model` in `deploy/ltx2_model.py`) exposes text-to-video, image-to-video, and 2-frame keyframe interpolation. The helper modules live in `utils/` and are mounted into the container.

---

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

GPU: RTX PRO 6000 (96GB) over H100 → ~45% lower $/clip. Same model in bf16 = same-fidelity frames on any card; the GPU only moves speed/cost.

---

## Layout

```
deploy.sh                  # one-command deploy (no secrets)
requirements.txt           # client-side: just the Modal SDK
deploy/
  ltx2_model.py            # the Modal app: t2v / i2v / keyframe + the opt stack
utils/                     # helper package (mounted into the container)
  cpu_pinned_registry.py     # CPU-pinned weight cache
  gpu_resident_registry.py   # resident-weights registry (VRAM-headroom guarded)
  fbcache.py                 # First-Block-Cache hook
  custom_guiders.py          # guidance helpers (APG / CFG variants)
tests/
  smoke_test.py            # tiny auth-free functional smoke
verification/
  ship_verify.py           # latency/VRAM + bit-identical + audio-skip verification
```

## Notes

- **bf16 only**, no quantization. **Native 9:16** beats render-wide-then-crop (cropping wastes ~66% of paid pixels). No secrets / no auth — endpoints are open.

## License

Code: private (personal). LTX-2.3 weights: per Lightricks' license — review before any commercial use.
