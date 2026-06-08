<h1 align="center">ltx2-vidgen-skill</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/serverless-Modal-7C3AED" alt="Modal">
  <img src="https://img.shields.io/badge/model-LTX--2.3%20·%2022B-FF4088" alt="LTX-2.3">
  <img src="https://img.shields.io/badge/GPU-RTX%20PRO%206000-76B900?logo=nvidia&logoColor=white" alt="GPU">
  <img src="https://img.shields.io/badge/precision-bf16-0A7E8C" alt="bf16">
  <img src="https://img.shields.io/badge/packaging-uv-DE5FE9" alt="uv">
</p>

> **Own your AI video pipeline.** Self-hosted, optimized LTX-2.3 (22B) on serverless GPU — text-to-video, image-to-video, keyframe interpolation, video-to-video, and IC-LoRA (canny/depth/pose) control, with synced audio, for a few cents a clip — no per-clip API meter, no rate limit. Ships as a **Claude Code skill** too: drop a photo in Claude Code → get a video.

<p align="center">
  <img src="assets/demo.gif" width="540" alt="LTX-2.3 image-to-video demo">
</p>

<p align="center"><em>Side-by-side image-to-video — two stills brought to life (slow push-in + wind-blown hair). LTX-2.3 (22B) on RTX PRO 6000 (Blackwell, 96 GB, bf16). <a href="assets/demo.mp4">▶ full-quality mp4</a></em></p>

---

## What this is

Two ways to use it, one repo:

| Path | What you get |
|---|---|
| **Modal deploy** | A self-hosted LTX-2.3 backend on serverless GPU — `./deploy.sh` and generate clips via `modal run` entrypoints. |
| **Claude Code skill** | Drop a photo in Claude Code and ask for a video — the skill calls your deployed backend and saves the `.mp4`. |

Both share the optimization stack (resident pipeline, weight cache, FBCache, embedding cache, torch.compile) → **~7–9 s warm generation**, output verified **bit-identical** to the unoptimized path, bf16 throughout. Any resolution with sides divisible by 32 (768×512, 768×1280 reel, 768×1280, 1024²…), clips up to ~10 s, optional generated audio.

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
# 1. deploy the backend (above) — the skill calls it by Modal app name (`ltx2-fast-inference`), no endpoint URL needed
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

Measured on RTX PRO 6000 (Blackwell, 96 GB), bf16. Warm latency scales with clip length and resolution — short clips keep both stage transformers resident; long/high-res clips hold one alongside the activation working set:

| Config | Warm latency |
|---|---|
| Short clip (≤97 f), low res | **~7–9 s** |
| 4 s clip (97 f) @ 768×1280 | **~42 s** |
| 10 s clip (241 f) @ 768×1280 | **~95–120 s** |
| 10 s video-to-video (retake) | **~470 s** |
| 4 s control (IC-LoRA union) | **~28 s** |

Cold start (first clip on a fresh container): ~90–200 s. Every config is **a few cents** at per-second billing.

| Optimization | Gain (measured on short clips, bit-identical) |
|---|---|
| CPU-pinned weight cache | −13–15 s per cold clip |
| First-Block-Cache | −17%, lossless |
| Text-embedding cache | −4 s |
| Resident + pre-fused pipeline | kills the per-clip rebuild — biggest win *when it fits* |
| Batch 4 → 32 clips | 3.6× → 6.2× throughput, −84% $/clip |
| **Net (short-clip regime)** | **1.96× faster, pixel-identical** |

GPU choice: RTX PRO 6000 over H100 → ~45% lower $/clip. Same model in bf16 = same-fidelity frames on any card; the GPU only moves speed and cost.

---

## How it works

LTX-2.3 is a 22B video diffusion transformer. Serving it naively has two costs: a slow cold start, and a per-clip cost where the pipeline re-assembles + re-fuses model internals every request. This repo attacks both:

1. **Build once, stay resident.** The fully-assembled, LoRA-fused transformer stays in GPU memory between clips (resolution-keyed, LRU-bounded) instead of rebuilt per request — the single biggest win when it fits (short/low-res: ~90 s → ~7 s).
2. **Activation-aware resident cap.** Each stage transformer is ~35 GB; the two-stage forward also needs an activation + audio/VAE/text working set that scales with `height × width × frames` (~24 GB at 768×1280×97). You can't hold *both* 35 GB transformers resident **and** run a high-res forward on a 96 GB card (2×35 + 24 = 94 GB → OOM). So the resident cap is computed per request: keep as many stage transformers resident as fit alongside the projected activation set (2 at low res = warm-fast; 1 at full-res 10 s = rebuild-per-call but never OOM). This is what makes back-to-back mode-switching safe.
3. **CPU-pinned weight cache** — weights pinned in host RAM, streamed to GPU, skipping disk reads. Bit-identical.
4. **First-Block-Cache** — skips recomputing early transformer blocks (~17%, near-lossless).
5. **Embedding cache** — the text encoder isn't re-run for a repeated prompt.
6. **Batching** — 32 clips in one warm container ≈ 6.2× throughput.
7. **torch.compile + persisted Inductor cache** — compiled once on the first cold container, restored from the volume on later cold starts (`max-autotune` tested + rejected, slower here).
8. **Blackwell-native attention** — flash-attn has no sm_120 kernel yet, so this runs PyTorch SDPA (exact). fp8/SageAttention tested + rejected (noise / black frames). Speed comes from architecture, not from cheapening the math.

Helper modules live in `utils/` and are mounted into the container.

---

## Optimization flags (env vars, set in the image `.env` block)

| Variable | Default | Description |
|---|---|---|
| `LTX_PERSIST_PIPELINE` | `both` | Keep stage transformers GPU-resident across requests. `off`/`stage2`/`both`. |
| `LTX_PERSIST_LRU_MAX` | `2` | Upper bound on resident (stage, resolution) entries. The effective cap is computed per request (activation-aware, see below) and never exceeds this. |
| `LTX_VRAM_HEADROOM_GB` | `40` | Free VRAM kept available before building a new resident transformer / loading the v2v/control pipeline. LRU residents are evicted to reach it, so a cross-mode/resolution build never OOMs the forward. |
| `LTX_VRAM_USABLE_GB` | `91` | Usable VRAM for the activation-aware resident-cap math (96 GB card minus a safety margin). |
| `LTX_XFMR_GB` | `35` | Assumed size of one stage transformer, used by the resident-cap math. |
| `LTX_REGISTRY` | `cpu_pinned` | Weight cache: `cpu_pinned` (recommended) / `gpu_resident` / `off`. |
| `LTX_CACHE_TEXT_EMB` | `1` | LRU cache on the text encoder output. |
| `LTX_SKIP_AUDIO` | `0` | Per-request default for skipping audio decode (video pixels byte-identical). |
| `LTX_FP8` | `0` | Load official fp8 weights instead of bf16 (off = bf16 quality default). |
| `LTX_VAE_TILE_PX` | `768` | VAE-decode spatial tile size (px, ≥64 & ÷32). Smaller → smaller decode peak (measured ~1 GB lever; decode is already well-tiled). Overlap blends seams. |
| `LTX_VAE_TILE_OVERLAP` | `64` | Spatial tile overlap (px, ÷32, < tile). |
| `LTX_VAE_TEMPORAL_FRAMES` | `80` | VAE-decode temporal chunk (frames, ≥16 & ÷8). |
| `LTX_VAE_TEMPORAL_OVERLAP` | `24` | Temporal chunk overlap (frames, ÷8, < chunk). |

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
