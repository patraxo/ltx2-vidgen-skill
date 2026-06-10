<h1 align="center">ltx2-vidgen-skill</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/serverless-Modal-7C3AED" alt="Modal">
  <img src="https://img.shields.io/badge/model-LTX--2.3%20·%2022B-FF4088" alt="LTX-2.3">
  <img src="https://img.shields.io/badge/GPU-RTX%20PRO%206000-76B900?logo=nvidia&logoColor=white" alt="GPU">
  <img src="https://img.shields.io/badge/precision-bf16-0A7E8C" alt="bf16">
  <img src="https://img.shields.io/badge/packaging-uv-DE5FE9" alt="uv">
</p>

> **A Claude Code skill that owns your AI video pipeline** — drop a photo in Claude Code → get a video. It deploys your own optimized LTX-2.3 (22B) backend to *your* Modal serverless GPU and drives it: text-to-video, image-to-video, keyframe interpolation, video-to-video, and IC-LoRA (canny/depth/pose) control, with synced audio — a few cents a clip, your GPU, no per-clip API meter, no rate limit.

<p align="center">
  <img src="assets/demo.gif" width="540" alt="LTX-2.3 image-to-video demo">
</p>

<p align="center"><em>Side-by-side image-to-video — two stills brought to life (slow push-in + wind-blown hair). LTX-2.3 (22B) on RTX PRO 6000 (Blackwell, 96 GB, bf16). <a href="assets/demo.mp4">▶ full-quality mp4</a></em></p>

---

## What this is

A **Claude Code skill that owns the whole loop** — it deploys your own LTX-2.3 (22B) video backend to *your* Modal account on first run, then drives it from Claude Code: drop a photo (or a prompt), ask for a video, get the `.mp4`. You never leave Claude Code; the GPU is yours; there's no SaaS in the middle.

- **Modes:** text-to-video, image-to-video, keyframe interpolation, video-to-video (retake/restyle), and IC-LoRA **canny/depth/pose control** — with synced audio.
- **Batch:** `--variations N` (one prompt, N takes) and `--prompts-file` (many prompts) in one warm container.
- **Formats:** reel / TikTok / Shorts (9:16), YouTube (16:9), square — native, no cropping.
- **Optimized:** resident pipeline + caches + torch.compile → **~1.96× faster**, output bit-identical, bf16 throughout. Clips up to ~10 s; any resolution with sides ÷32.

---

## Install

**1. Install the skill** (Claude Code, or any agent that reads `SKILL.md`):

```bash
npx skills add patraxo/ltx2-vidgen-skill        # or: cp -R skills/ltx2-video ~/.claude/skills/
pip install modal && modal token new            # Modal SDK + your account (free — $30/mo credits)
```

…or as a Claude Code **plugin marketplace**:

```text
/plugin marketplace add patraxo/ltx2-vidgen-skill
/plugin install ltx2-vidgen@ltx2-vidgen-skill
```

**2. Deploy your backend once** — into *your* Modal account (downloads the LTX-2.3 weights + Gemma text encoder; public components only, no HuggingFace token):

```bash
git clone https://github.com/patraxo/ltx2-vidgen-skill && cd ltx2-vidgen-skill && ./deploy.sh
```

That's it. Now in Claude Code, drop a photo (or just describe a scene) and ask:

- *"turn this photo into a video, subtle natural motion"* — image → **i2v**
- *"interpolate between these two frames"* — two images → **keyframe**
- *"restyle the middle of this clip"* — a video → **v2v**
- *"generate a neon city street, vertical reel"* — prompt only → **t2v**
- *"follow the edges of this clip"* — a control render → **IC-LoRA control**

The skill validates the input, confirms cost (offers a cheap smoke first), calls your deployed app via `modal.Cls.from_name` (no endpoint, no auth, no secrets), and saves the `.mp4` + a preview frame to `./video_out/`. First clip cold-starts ~90 s; warm ~31 s.

<details><summary>Prefer raw <code>modal run</code>? Drive the backend directly, no skill</summary>

```bash
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::smoke_real --image-path pic.jpg   # i2v
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::run_modes                          # t2v + i2v + keyframe
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::run_retake                         # v2v
PYTHONPATH=. uv run modal run deploy/ltx2_model.py::kf_real --image-a a.jpg --image-b b.jpg
PYTHONPATH=. uv run modal run tests/ship_verify.py                                     # full verify
```
</details>

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

Runs on **[Modal](https://modal.com)** serverless **NVIDIA RTX PRO 6000** (Blackwell, 96 GB), bf16, **billed per-second at $0.000842/s (~$3.03/hr)** — so cost ≈ latency:

| Mode @ 768×1280 (9:16) | Warm latency | ~ $/clip |
|---|---|---|
| image / text / keyframe — 5 s (121 f) | **~23 s** | **~1.9¢** |
| image / text / keyframe — 10 s (241 f) | **~45 s** | ~3.8¢ |
| landscape 1280×768 — 5 s / 10 s | ~24 s / ~47 s | ~2.0¢ / ~4.0¢ |
| 3-clip batch with decode/encode overlap | **~25 s/clip** (−20% vs serial) | ~2.1¢ |
| IC-LoRA control — 4 s | **~28 s** | ~2.4¢ |
| video-to-video (retake) — 10 s | ~470 s | ~40¢ |

(5 s/10 s rows re-measured 2026-06-10, bench config with the first-block cache off — the production path with it on is faster still. First call at a NEW frame count pays a one-time shape compile, e.g. ~86 s at 241 f, then steady.)

Cold start (first clip on a fresh container) ~90–200 s; idle scales to **$0**. Both 22B stage transformers stay resident — peak **~75 GB / 96 GB**, verified zero OOM over a 10-generation run (<1 GB drift).

**Optimization stack** — bf16 throughout, cache paths bit-identical to the unoptimized run, **~1.96× net faster**:

- Resident pre-fused pipeline + **activation-aware resident cap** + **cross-resolution purge** → no per-clip rebuild and no OOM when switching mode/resolution.
- First-block feature cache (~17%), CPU-pinned weight cache, text-embedding cache, torch.compile (persisted Inductor cache), tunable VAE-decode tiling.
- **fp8 / SageAttention / flash-attn all tested and rejected** — fp8: quality rule; flash-attn: no sm_120 kernel; SageAttention 2.2: builds + runs on sm_120 but measured +5% slower here (compile graph breaks — see Research findings). Speed comes from architecture, not from cheapening the math.

### Costs — and what that buys you

New Modal accounts get **$30/month in free credits**. At ~2.6¢ per 4-second clip that's **~1,000 free clips/month** — so you can explore a prompt 20 ways (`--variations 20`) and keep the one that lands, for pennies. It's **your** Modal account and bill (visible in your dashboard), no per-clip meter, no rate limit, idle = $0.

---

## How it works

LTX-2.3 is a 22B video diffusion transformer. Serving it naively has two costs: a slow cold start, and a per-clip cost where the pipeline re-assembles + re-fuses model internals every request. This repo attacks both:

1. **Build once, stay resident.** The fully-assembled, LoRA-fused transformer stays in GPU memory between clips (resolution-keyed, LRU-bounded) instead of rebuilt per request — the single biggest win when it fits (short/low-res: ~90 s → ~7 s).
2. **Both stages stay resident — safely.** Two ~35 GB stage transformers + the (tiled) forward working set peak ~75 GB / 96 GB, so both stay GPU-resident at every supported resolution (measured; zero OOM over a 10-gen run, <1 GB drift). An **activation-aware cap** + a **cross-resolution purge** (drop residents from a prior, different-resolution request before the next forward) keep it safe — back-to-back mode/resolution switching never OOMs. (The earlier 94 GB OOMs were that cross-resolution accumulation, not the resident footprint.)
3. **CPU-pinned weight cache** — weights pinned in host RAM, streamed to GPU, skipping disk reads. Bit-identical.
4. **First-Block-Cache** — skips recomputing early transformer blocks (~17%, near-lossless).
5. **Embedding cache + streaming text-encoder guard** — the text encoder isn't re-run for a repeated prompt. On a cache miss with both stages resident there isn't room for the full ~23 GB Gemma build (measured 93.7 GB → OOM), so the miss automatically switches to the upstream layer-streaming build (~5 GB peak, identical embeddings, ~+10 s once per prompt per container).
6. **Batching + decode/encode overlap** — 32 clips in one warm container ≈ 6.2× throughput; finalize (VAE decode + mp4 mux) of clip N overlaps the denoise of clip N+1 on a worker thread → **−20% wall** measured on a 3-clip batch (92.5 s → 73.7 s).
7. **torch.compile + persisted Inductor cache** — compiled once on the first cold container, restored from the volume on later cold starts (`max-autotune` tested + rejected, slower here).
8. **Blackwell-native attention** — flash-attn has no sm_120 kernel yet, so this runs PyTorch SDPA (exact). fp8 rejected (quality rule); SageAttention 2.2 builds + runs cleanly on sm_120 but measured **+5% slower** here (~770 torch.compile graph breaks per request — the kernel can't be traced). Speed comes from architecture, not from cheapening the math.

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
| `LTX_VAE_TILE_PX` | `768` | VAE-decode spatial tile size (px, ≥64 & ÷32). Smaller → smaller decode peak (measured ~1 GB lever; decode is already well-tiled). Overlap blends seams. **`0` disables tiling entirely** → reference-exact non-tiled decode (PSNR 50–51 dB vs tiled — the delta is the tiled arm's seam blending), latency-neutral, +2.6 GB peak; fits even at 241 f (82.3 GB peak, no OOM). |
| `LTX_VAE_TILE_OVERLAP` | `64` | Spatial tile overlap (px, ÷32, < tile). |
| `LTX_VAE_TEMPORAL_FRAMES` | `80` | VAE-decode temporal chunk (frames, ≥16 & ÷8). |
| `LTX_VAE_TEMPORAL_OVERLAP` | `24` | Temporal chunk overlap (frames, ÷8, < chunk). |
| `LTX_EMB_STREAM_FREE_GB` | `28` | Free-VRAM threshold below which a text-embedding cache MISS builds Gemma via the upstream layer-streaming path (~5 GB peak) instead of the full ~23 GB GPU build. Prevents the warm-container new-prompt OOM; identical embeddings. |
| `LTX_SDPA_PRIORITY` | unset | `cudnn` prefers the cuDNN SDPA backend **at trace time** (inductor captures the backend when compiling — runtime flips are no-ops). Measured 0% here; flag kept for other shapes/stacks. |
| `LTX_CUDNN_BENCH` | `1` | `torch.backends.cudnn.benchmark` autotuning. `0` disables. |
| `LTX_VIDEO_ENCODER` | unset | `nvenc` opts into h264_nvenc via a torch-free subprocess. **Known-dead under `enable_memory_snapshot=True`** (checkpoint-restored containers can't open NVENC sessions — fails loudly back to libx264, root error surfaced in `nvenc_last_error`). Works only if you disable snapshots; not worth the cold-start trade here. |

---

## Research findings (latency fan-out, June 2026)

A systematic hunt for lossless latency wins on this stack — every lever measured A/B on the same warm container, gated on zero visual loss (blackdetect + PSNR/SSIM + frame inspection + audio). Full write-ups in [`references/`](references/):

| Lever | Verdict | Evidence |
|---|---|---|
| Batch decode/encode overlap | ✅ **shipped, −20.4% batch wall** | 3 clips: 92.5 s → 73.7 s, peak 76 GB |
| Streaming-Gemma OOM guard | ✅ **shipped — correctness fix** | warm new-prompt request: OOM @ 93.7 GB → 33 s success @ 78.6 GB |
| Non-tiled VAE decode (`LTX_VAE_TILE_PX=0`) | ✅ shipped, optional | reference decode, 50–51 dB vs tiled, latency-neutral |
| `cudnn.benchmark` | ✅ shipped | free, no regression |
| SageAttention 2.2 (sm_120 source build) | ❌ rejected, **+5.1% slower** | ~770 compile graph breaks/request; LTX's 1:192 compression leaves only ~15 k tokens — attention is just 20–30% of step time. [`references/SAGEATTENTION_SM120.md`](references/SAGEATTENTION_SM120.md) |
| cuDNN SDPA backend | ➖ 0% | inductor captures the SDPA backend at **trace time** — runtime `sdpa_kernel()` contexts are no-ops for compiled blocks (proved byte-identical) |
| NVENC encode | ❌ dead under memory snapshots | works in a bare Modal function, fails (`avcodec_open2` UnknownError) in checkpoint-restored containers — in-process **and** in torch-free subprocesses |
| SpargeAttn, step caches (@8 distilled steps), FA4/xformers, fp8/int8, max-autotune | ❌ rejected | no sm_120 kernels / collapse at 8 steps / quality rule / measured slower |

Floor: a warm 5 s clip is ~19.5 s GPU-bound bf16 DiT + ~2.7 s CPU x264 encode + ~0.5 s overhead. The DiT term only moves with quantization or step cuts — both off the table by the quality rule. Full record: [`references/LATENCY_FANOUT_2026_06_10.md`](references/LATENCY_FANOUT_2026_06_10.md), distilled gotchas in [`references/learnings.md`](references/learnings.md).

Two transferable gotchas worth stealing:
1. **Inductor captures the SDPA backend at trace time** — benchmarking attention backends with runtime context managers on a compiled model measures nothing.
2. **Same-prompt benchmarks never exercise the text-encoder memory path** — always include a warm new-prompt arm, or you'll ship a latent OOM like the one found here.

---

## Layout

```
pyproject.toml             # uv project (client-side dep: Modal SDK)
deploy.sh                  # one-command deploy
deploy/ltx2_model.py       # the Modal app: t2v / i2v / keyframe / v2v + opt stack
deploy/bench_*.py          # same-container A/B bench drivers (sage, cudnn, matrix, fixes)
deploy/probe_nvenc.py      # bare-container NVENC probe (no memory snapshot)
deploy/utils/              # helper package (weight registries, FBCache, guiders)
references/                # research findings: latency fan-out, sage sm_120, learnings
skills/ltx2-video/         # Claude Code skill (SKILL.md + scripts/submit_video.py)
tests/                     # smoke_test.py + ship_verify.py
assets/demo.gif
```

---

## Acknowledgements & references

- **[LTX-Video](https://huggingface.co/Lightricks/LTX-2.3)** (Lightricks) — the 22B model + upstream `ltx-core`/`ltx-pipelines`. Paper: HaCohen et al., *LTX-Video: Realtime Video Latent Diffusion* — [arXiv:2501.00103](https://arxiv.org/abs/2501.00103).
- **[Modal](https://modal.com)** — serverless GPU (RTX PRO 6000, per-second billing).
- **Gemma-3** — text encoder.

Optimization techniques:
- Block-feature caching (the first-block cache here) — cf. *BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching* — [arXiv:2509.13789](https://arxiv.org/abs/2509.13789); broader caching lineage e.g. *SenCache* (LTX-Video) — [arXiv:2602.24208](https://arxiv.org/abs/2602.24208).
- VAE-decode spatial/temporal tiling, CPU-pinned weight cache, persisted `torch.compile` (Inductor) cache, bf16-exact throughout (no fp8 / quantization).

By [@patraxo](https://github.com/patraxo). LTX-2.3 weights are distributed under Lightricks' license.
