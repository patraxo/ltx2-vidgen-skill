# ltx2-fast-inference

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Serverless](https://img.shields.io/badge/serverless-Modal-7C3AED)
![Model](https://img.shields.io/badge/model-LTX--2.3%20·%2022B-FF4088)
![GPU](https://img.shields.io/badge/GPU-RTX%20PRO%206000-76B900?logo=nvidia&logoColor=white)
![Precision](https://img.shields.io/badge/precision-bf16-0A7E8C)
![uv](https://img.shields.io/badge/packaging-uv-DE5FE9)

> **Own your AI video pipeline.** Self-hosted, optimized LTX-2.3 (22B) on serverless GPU — text-to-video, image-to-video, keyframe interpolation, and video-to-video for a few cents a clip, no per-clip API meter, no rate limit. Ships as a **Claude Code skill** too: drop a photo in Claude Code → get a video.

![demo](assets/demo.gif)

*Neon UGC clip, image-to-video on RTX PRO 6000 (Blackwell, 96 GB, bf16). Warm generation ~7–9 s.*

---

## What this is

Two ways to use it, one repo:

| Path | What you get |
|---|---|
| **Modal deploy** | A self-hosted LTX-2.3 backend on serverless GPU — `./deploy.sh` and generate clips via `modal run` entrypoints. |
| **Claude Code skill** | Drop a photo in Claude Code and ask for a video — the skill calls your deployed backend and saves the `.mp4`. |

Both share the optimization stack (resident pipeline, weight cache, FBCache, embedding cache, torch.compile) → **~7–9 s warm generation**, output verified **bit-identical** to the unoptimized path, bf16 throughout. Any resolution with sides divisible by 32 (768×512, 768×1280 reel, 1280×768, 1024²…), clips up to ~10 s, optional generated audio.

---

## Quickstart — deploy

```bash
# install uv (https://docs.astral.sh/uv), then:
uv sync
uv run modal token new      # one-time auth
./deploy.sh                 # deploys the app (no secrets to set up)
```

First build downloads the LTX-2.3 weights + Gemma text encoder into a Modal volume (public components only — no HuggingFace token). Cached on every later deploy.

```bash
# image-to-video from a real photo
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::smoke_real --image-path pic.jpg
# t2v + i2v + keyframe (saved to deploy/mode_clips/)
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::run_modes
# video-to-video retake
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::run_retake
# keyframe interpolation between two images
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::kf_real --image-a a.jpg --image-b b.jpg
# tiny smoke + full verification
PYTHONPATH=. uv run modal run tests/smoke_test.py
PYTHONPATH=. uv run modal run tests/ship_verify.py
```

First clip cold-starts ~90 s; every warm clip after is 7–9 s.

---

## Quickstart — Claude Code skill

Drop a photo in Claude Code, ask for a video.

```bash
# 1. deploy the backend (above) — the skill calls it BY APP NAME, no endpoint URL needed
# 2. install the skill (one-time)
cp -R skills/ltx2-video ~/.claude/skills/ltx2-video
pip install modal && modal token new      # the skill's script needs the Modal SDK
```

Then in Claude Code, just say:

- *"turn this photo into a video, subtle natural motion"* (attach an image → **i2v**)
- *"interpolate between these two frames"* (two images → **keyframe**)
- *"retake the middle of this clip with more motion"* (a video → **v2v**)
- *"generate a video of a neon city street, vertical reel"* (no image → **t2v**)

The skill resolves + validates the image, confirms the cost (cold-start/cheap-smoke-first), calls the deployed app, and saves the `.mp4` + a preview frame to `./video_out/`. Under the hood it runs `skills/ltx2-video/scripts/submit_video.py`, which calls `modal.Cls.from_name("ltx2-fast-inference", "Model")` — no HTTP endpoint, no auth.

---

## Modes

| Mode | Input | Underlying pipeline |
|---|---|---|
| **text-to-video** | prompt only | KeyframeInterpolation (0 keyframes) |
| **image-to-video** | 1 image + prompt | TI2VidTwoStages |
| **keyframe interpolation** | 2 images + prompt | KeyframeInterpolation |
| **video-to-video (retake)** | source video + window + prompt | RetakePipeline |

All four are exercised by `run_modes` / `run_retake` / `kf_real` and verified working (frame-inspected). The opt stack applies across every mode.

---

## Performance

Measured on RTX PRO 6000 (Blackwell, 96 GB), bf16, all outputs bit-identical to the unoptimized path:

| Resolution | Cold (once per container) | Warm (every clip) | Peak VRAM |
|---|---|---|---|
| 768×512 | 93.6 s | **7.3 s** | ~72 GB |
| 768×1280 (9:16 reel) | — | **9.3 s** | ~72 GB |

| Optimization | Gain (all bit-identical) |
|---|---|
| CPU-pinned weight cache | −13–15 s per cold clip |
| First-Block-Cache | −17%, lossless |
| Text-embedding cache | −4 s |
| Resident + pre-fused pipeline | kills the per-clip rebuild — biggest win |
| Batch 4 → 32 clips | 3.6× → 6.2× throughput, −84% $/clip |
| **Net** | **1.96× faster, pixel-identical** |

GPU choice: RTX PRO 6000 over H100 → ~45% lower $/clip. Same model in bf16 = same-fidelity frames on any card; the GPU only moves speed and cost.

---

## How it works

LTX-2.3 is a 22B video diffusion transformer. Serving it naively has two costs: a slow cold start, and a per-clip cost where the pipeline re-assembles + re-fuses model internals every request. This repo attacks both:

1. **Build once, stay resident.** The fully-assembled, LoRA-fused transformer stays in GPU memory between clips (resolution-keyed, LRU-bounded) instead of rebuilt per request — the single biggest win, ~90 s → ~7 s.
2. **CPU-pinned weight cache** — weights pinned in host RAM, streamed to GPU, skipping disk reads. Bit-identical.
3. **First-Block-Cache** — skips recomputing early transformer blocks (~17%, near-lossless).
4. **Embedding cache** — the text encoder isn't re-run for a repeated prompt.
5. **Batching** — 32 clips in one warm container ≈ 6.2× throughput.
6. **torch.compile + persisted Inductor cache** — compiled once on the first cold container, restored from the volume on later cold starts (`max-autotune` tested + rejected, slower here).
7. **Blackwell-native attention** — flash-attn has no sm_120 kernel yet, so this runs PyTorch SDPA (exact). fp8/SageAttention tested + rejected (noise / black frames). Speed comes from architecture, not from cheapening the math.

Helper modules live in `utils/` and are mounted into the container.

---

## Optimization flags (env vars, set in the image `.env` block)

| Variable | Default | Description |
|---|---|---|
| `LTX_PERSIST_PIPELINE` | `both` | Keep stage transformers GPU-resident across requests. `off`/`stage2`/`both`. |
| `LTX_PERSIST_LRU_MAX` | `2` | Max resident (stage, resolution) entries. Don't exceed 2 on a 96 GB card. |
| `LTX_REGISTRY` | `cpu_pinned` | Weight cache: `cpu_pinned` (recommended) / `gpu_resident` / `off`. |
| `LTX_CACHE_TEXT_EMB` | `1` | LRU cache on the text encoder output. |
| `LTX_SKIP_AUDIO` | `0` | Per-request default for skipping audio decode (video pixels byte-identical). |
| `LTX_FP8` | `0` | Load official fp8 weights instead of bf16 (off = bf16 quality default). |

---

## Layout

```
pyproject.toml             # uv project (client-side dep: Modal SDK)
deploy.sh                  # one-command deploy
deploy/ltx2_model.py       # the Modal app: t2v / i2v / keyframe / v2v + opt stack
utils/                     # helper package (weight registries, FBCache, guiders)
skills/ltx2-video/         # Claude Code skill (SKILL.md + scripts/submit_video.py)
tests/                     # smoke_test.py + ship_verify.py
assets/demo.gif
```

---

## Acknowledgements & license

- [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) — model weights + upstream `ltx-core`/`ltx-pipelines`. **Review Lightricks' license before any commercial use.**
- [Modal](https://modal.com) — serverless GPU.
- Gemma-3 — text encoder.

Code: private (personal project by [@patraxo](https://github.com/patraxo)). LTX-2.3 weights per Lightricks' license.
