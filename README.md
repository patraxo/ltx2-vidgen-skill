# ltx2-fast-inference

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Serverless](https://img.shields.io/badge/serverless-Modal-7C3AED)
![Model](https://img.shields.io/badge/model-LTX--2.3%20·%2022B-FF4088)
![GPU](https://img.shields.io/badge/GPU-RTX%20PRO%206000-76B900?logo=nvidia&logoColor=white)
![Precision](https://img.shields.io/badge/precision-bf16-0A7E8C)
![uv](https://img.shields.io/badge/packaging-uv-DE5FE9)

> **Own your AI video pipeline.** Self-hosted, optimized LTX-2.3 on serverless GPU — frontier-quality text-to-video, image-to-video, keyframe interpolation, and video-to-video for cents a clip, with no per-clip API meter and no rate limit.

Fast, self-hosted **LTX-2.3** (Lightricks, 22B video DiT) on [Modal](https://modal.com) — text-to-video, image-to-video, keyframe interpolation, and video-to-video, with an optimization stack that turns a 10-second 9:16 reel into **~7–9s of warm generation for a few cents** on a single **RTX PRO 6000 (Blackwell)**. Output is verified **bit-identical** to the unoptimized path, bf16 throughout.

Generates at **any resolution** whose sides are divisible by 32 — 768×512, 768×1280 (vertical 9:16 reel), 1280×768, 1024×1024, etc. — and clips up to ~10 seconds, with optional generated audio.

---

## Quickstart

```bash
# 1. install uv (https://docs.astral.sh/uv) then sync + authenticate (one-time)
uv sync
uv run modal token new

# 2. deploy
./deploy.sh

# 3. run it
PYTHONPATH=. uv run modal run tests/smoke_test.py                                    # tiny smoke
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::smoke_real --image-path pic.jpg   # real-image i2v -> mp4
PYTHONPATH=. uv run modal run tests/ship_verify.py                                    # full verification
```

`./deploy.sh` syncs the env, checks auth, and pushes the app to Modal. The first generation is a cold start (~90s); every clip after that on a warm container is 7–9s.

> **Weights:** the LTX-2.3 weights (+ Gemma text encoder) are pulled into the Modal volume the app references on first build — public components only, no HuggingFace token needed. See the `download_models` step + volume name in `deploy/ltx2_model.py`.

---

## How it works

LTX-2.3 is a 22B video diffusion transformer. Serving it naively has two costs: a slow cold start, and a per-clip cost where the pipeline re-assembles and re-fuses model internals on **every** request. This repo attacks both:

1. **Build once, stay resident.** The fully-assembled, LoRA-fused transformer is kept in memory between clips (resolution-keyed) instead of rebuilt per request. The single biggest win — it takes a clip from ~90s to ~7s.
2. **Weight caching (CPU-pinned).** Weights are pinned in host RAM and streamed to GPU, skipping disk reads on warm loads. Bit-identical.
3. **First-Block-Cache.** Skips recomputing the early transformer blocks whose features barely change across steps. ~17%, near-lossless.
4. **Embedding cache.** The text encoder isn't re-run for a repeated prompt.
5. **Batching.** Many clips in one warm container amortize the fixed cost — 32 at once ≈ 6.2× throughput.
6. **torch.compile + persisted Inductor cache.** The model is `torch.compile`d (default mode — `max-autotune` was tested and ran *slower* on this GPU), and the compiled artifact is persisted to the Modal volume, so the one-time compile is paid at the first cold start and restored on every later cold container instead of recompiling.
7. **Blackwell-native attention.** flash-attention has no sm_120 kernel yet, so this runs PyTorch SDPA (exact). fp8/SageAttention were tested and rejected (noise / black frames) — speed comes from architecture, not from cheapening the math.

The Modal app (`@app.cls Model` in `deploy/ltx2_model.py`) exposes **text-to-video**, **image-to-video**, **2-frame keyframe interpolation**, and **video-to-video** (retake / control-guided restyle). Helper modules live in `utils/` and are mounted into the container.

---

## Performance

Measured on RTX PRO 6000 (Blackwell, 96 GB), bf16:

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

GPU choice: RTX PRO 6000 (96 GB) over H100 → ~45% lower $/clip. Same model in bf16 = same-fidelity frames on any card; the GPU only moves speed and cost.

**Resolution & aspect:** any resolution whose sides are divisible by 32 — square, landscape, or vertical 9:16 (e.g. 768×1280).

---

## Layout

```
pyproject.toml             # uv project (client-side dep: the Modal SDK)
deploy.sh                  # one-command deploy (uv + modal)
deploy/
  ltx2_model.py            # the Modal app: t2v / i2v / keyframe + the opt stack
utils/                     # helper package (mounted into the container)
  cpu_pinned_registry.py     # CPU-pinned weight cache
  gpu_resident_registry.py   # resident-weights registry (VRAM-headroom guarded)
  fbcache.py                 # First-Block-Cache hook
  custom_guiders.py          # guidance helpers (APG / CFG variants)
tests/
  smoke_test.py            # tiny functional smoke
  ship_verify.py           # latency/VRAM + bit-identical + audio-skip verification
```

## License

Code: private (personal). LTX-2.3 weights: per Lightricks' license — review before any commercial use.
