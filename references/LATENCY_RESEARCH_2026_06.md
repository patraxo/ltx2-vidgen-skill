# Latency optimization research (June 2026) — quality-preserving levers only (bf16, NO fp8/fp4)

Research sweep across six areas (deploy audit, Lightricks upstream, community
Blackwell reports, Modal platform, torch 2.12 sm_120, arXiv 2024-26) + same-container A/B benches.
Workload: 768×1280, 121f, 8-step distilled, RTX PRO 6000 (sm_120, 96GB), warm.
HARD RULE: bf16 only. No fp8, no fp4, no int8 weights, no step cuts. Every
quantized path in the research = BANNED, none implemented.

## Framing result (literature survey)

**8-step distilled kills step-caching.** Chorus (2604.04451), X-Cache
(2604.20289), DisCa (2602.05449): distillation already removed inter-step
redundancy. Δ-DiT at 4 steps: 1.12× only. TeaCache/MagCache/EasyCache headline
2-6× = 30-50-step models. Live axes at 8 steps: per-step compute (attention
kernels), VAE/decode, system pipelining, serving.

## MEASURED (bench_opts_ab.py + bench_sage_ab.py, same-container A/B)

| lever | latency | quality | verdict |
|---|---|---|---|
| SageAttention 2.2 | **-5.1% (SLOWER)** 23.88 vs 22.72s | no black frames, PSNR 33.7 (trajectory drift), nondeterministic | REJECTED (see SAGEATTENTION_SM120.md) |
| cuDNN SDPA, runtime flip | no-op — mp4 **byte-identical** to base | identical | inductor picks SDPA backend at TRACE time; runtime ctx ignored by compiled blocks |
| cuDNN SDPA, trace-time | **0%**: median 22.735s (22.65-22.88, ±0.5% tight) vs flash-traced 22.72-22.81s on equally-quiet host | artifact-free; PSNR 33.9 vs flash trace (trajectory divergence, same as any kernel change); flip-back byte-identical (capture proof) | **NEUTRAL — no gain, no harm. Flag kept (`LTX_SDPA_PRIORITY=cudnn` at trace time), default stays flash. Inductor-fused blocks + ~15k-token seqs → flash ≈ cuDNN here** |
| VAE no-tiling (`tile_px=0`) | ~neutral (noise ±20%) | **PSNR 50.4 dB / SSIM 0.994 vs tiled; frames visually identical; audio RMS exactly equal; +2.6GB peak (76.9/96)** | **PASS — reference-correct decode. Use for standard reels. Default stays tiled (big-res retake/HQ OOM risk)** |
| cudnn.benchmark=True | folded into all arms (global) | conv algo only, same math | kept, default ON, kill `LTX_CUDNN_BENCH=0` |

Bench noise: same-arm spread ±20% (base 23.18↔28.92s) — single-run deltas
meaningless, use medians + byte-compare. SDPA path deterministic (mp4
byte-identical across runs in same container/config) — strongest quality gate.

Phase split (warm, tiled): inference ~20-25s, encode ~2.8-3.8s. Encode = CPU
libx264 via upstream `encode_video()`. NVENC ceiling ≈ -2s (~8-10%) — worth a
look ONLY if upstream exposes codec; not implemented.

## IMPLEMENTED 2026-06-10 (deploy/ltx2_model.py)

- `_SDPA_CUDNN_RUNTIME` + `_sdpa_priority_ctx()` — cuDNN>FLASH>EFFICIENT>MATH
  via `sdpa_kernel(set_priority=True)` around all 4 pipeline call sites. Env
  `LTX_SDPA_PRIORITY=cudnn`, per-request `smoke_generate(sdpa_cudnn=0|1)`.
  NOTE: affects compiled blocks only at trace time (first call of container).
- `tile_px=0` sentinel → `tiling_config=None` → non-tiled reference decode
  (+ `video_chunks_number=1` guards at 3 call sites).
- `torch.backends.cudnn.benchmark=True` in `_gpu_init` (LTX_CUDNN_BENCH=0 off).
- `phases` dict in smoke result: `inference_s`, `encode_s`, `vae_tiling`,
  `sdpa_cudnn` — phase attribution per request.

## RESEARCHED — NOT implemented (reasons)

- **reduce-overhead/CUDA graphs**: max-autotune already measured -22-34% SLOWER
  on this stack (learnings.md); CPU dispatch ≲2% for 22B steps. Skip.
- **TF32**: only touches fp32 matmuls — pipeline is bf16. No-op. Skip.
- **SpargeAttn**: NO sm_120 kernels (thu-ml/SpargeAttn#76 open since 2025-08,
  setup.py rejects arch 12.0); calibration hours-days; FLUX repro failures
  (#94). Dead until upstream ships. Watch.
- **SVG2 / CalibAtt**: compete with each other; CalibAtt (2603.05503,
  Lightricks-affiliated, few-step-validated) has NO public code yet. Watch.
- **MagCache/EasyCache swap for FBCache**: gains collapse at 8 steps
  (~1.15-1.4× realistic); FBCache default OFF anyway. Skip.
- **Chorus inter-request latent reuse** (2604.04451): only caching proven on
  4-step distilled (45%); fits 12-variant reel batches BUT couples variant
  outputs + serving-layer rework. Future.
- **PipeDiT-style decode/denoise overlap** (CUDA streams, bit-identical): real
  candidate for BATCHED runs (decode clip N while denoising N+1). Engineering
  ~day. Future — biggest remaining lossless lever for reel batches.
- **NVENC encode**: encode_s 2.8-3.8s of ~23s (12-16%); upstream encode_video
  owns the ffmpeg call. Patch = monkeypatch or upstream PR. Future, medium.
- **FlashAttention-4**: Lightricks explicitly excludes on consumer Blackwell
  ("known regressions"). xformers: incompatible sm_120 (LTX-2 #27/#37) — never
  install. FlexAttention: sm_120 falls to Triton path ≈ FA2 at best. Skip all.
- **Modal levers**: scaledown_window already 20 min (max). GPU memory
  snapshots = experimental, Blackwell unverified, helps cold-start init only
  (weights transfer unaffected). `routing_region` immutable post-deploy —
  consider at next fresh app (payloads >2MiB round-trip us-east object store).
  `@modal.concurrent(max_inputs=2, target_inputs=1)` as cold-boot shield —
  optional, queue-vs-second-container tradeoff.
- **Lightricks shipped-but-unused**: `StateDictRegistry` (we have cpu_pinned —
  superior), `model_context()` cross-stage residency (persist-pipeline covers),
  memory-efficient decode channels_last_3d (ON by default upstream),
  precomputed text embeddings (emb-cache covers), batch-split (CFG only — N/A
  distilled), audio=None ModalitySpec (NOT output-preserving: joint AV
  attention changes video — excluded by quality rule).

## Sources

Compiled 2026-06-10. Key URLs: Lightricks/LTX-2
compiling.py + attention.py + memory_efficient_decode.py, issues #208/#27/#37,
PR #215; ComfyUI PR #13618 (block prefetch, 5090 LTX-2.3 ~14%), PR #13768;
thu-ml/SpargeAttn#76/#94/#109; pytorch #176426 (sm_120 Triton miscompile,
OPEN — validate compiled vs eager once); modal.com gpu-mem-snapshots blog;
arxiv IDs inline above.

## Follow-up verdicts (2026-06-10 — NVENC + batch overlap + Gemma OOM)

### NVENC — DEAD on this app (snapshot-restored containers). Evidence chain:
1. Bare probe app (`probe_nvenc.py`, plain `@app.function`, same image, same
   `gpu="RTX-PRO-6000"`): PyAV h264_nvenc real encoder-open **PASSES**.
2. In-process on Model: `avcodec_open2("h264_nvenc")` →
   `UnknownError(1313558101)` with ~21 GB VRAM free.
3. Torch-free CHILD subprocess inside the Model container: **same error** —
   kills the "torch/PyAV lib collision" theory.
4. Only remaining structural difference: `enable_memory_snapshot=True` on the
   Model class — every serving container (including the first) runs a
   checkpoint-RESTORED process; Modal's GPU restore shim covers CUDA compute,
   not the NVENC ioctl/device-fd surface, and the broken state is
   container-wide (new processes inherit it).
   Verdict: NVENC incompatible with snapshot lifecycle. Dropping snapshots
   (10-30 s cold-start win) to gain ~2 s encode = bad trade. Dispatch code
   stays (default OFF, subprocess preflight, loud fail-open to libx264,
   `nvenc_active` reports ENGAGEMENT, `nvenc_last_error` carries root error).
   fx_nvenc2.mp4 byte-identical to fx_base.mp4 (fallback proof).

### Batch overlap — SHIPPED + WORKING (after Gemma fix below)
- `smoke_generate_batch`, 3 clips 768×1280×121f: serial wall **92.49 s**
  (denoise [19.7, 32.4, 30.1] — clips 2-3 include streaming-Gemma MISS),
  overlap wall **73.65 s** = **−20.4 %** (saved 24.4 s), peak 76.0 GB.
- Overlap mechanics: finalize (lazy chunked VAE decode pulled by CPU encoder +
  mux) on a worker thread, DEFAULT CUDA stream; decode FIFO-interleaves with
  next clip's denoise. Second batch on same container = emb-cache HITs →
  denoise back to ~20-23 s/clip.
- Quality gate: batch_overlap_clip3.mp4 — 0 black segments, 121f, audio
  −20.3 dB RMS, frames prompt-faithful (floating-candle library).

### Gemma OOM — ROOT CAUSE FOUND + FIXED (prod-relevant correctness bug)
- Upstream `PromptEncoder.__call__` builds the FULL ~23 GB Gemma on GPU per
  call (frees on exit). On a warm container with both stage transformers
  persisted (~70.8 GB), the FIRST NEW-PROMPT request (emb-cache MISS) hit
  93.75 GB → **OOM = dead request**. Every earlier bench reused one prompt
  (always HIT) — batch clip 2 exposed it.
- Fix: emb-cache wrapper now flips `PromptEncoder._offload_mode` →
  `OffloadMode.CPU` for the MISS call when `_free_vram_gb() <
  LTX_EMB_STREAM_FREE_GB` (default 28): upstream streaming builder, layer-wise
  blocks, ~5 GB peak, same weights/math/dtype → identical embeddings. Restores
  after. Fresh containers keep the fast full-GPU build.
- Measured: warm new-prompt single = **33.0 s, peak 78.6 GB** (was OOM).
  Streaming MISS cost ≈ +10.2 s, once per prompt per container (then cached).
  Quality: fx_newprompt2.mp4 fully prompt-faithful (sorceress + emerald sigil +
  forked lightning) — semantic-fidelity proof through the streamed encoder.
- The earlier "side-stream allocator pool stranding" theory was WRONG (alloc
  was LIVE Gemma weights); batch `_finalize` stays on the default stream
  anyway (cross-stream pools genuinely don't share).
- Future (optional): cut the +10 s by backing the streaming builder with the
  cpu_pinned registry if it isn't already; measure first.

### Follow-up measured table (this deploy, warm, 768×1280×121f unless noted)
| arm | wall | note |
|---|---|---|
| fx_base (libx264 ref) | 22.8-23.0 s | inf 19.4-19.6 + enc 2.7 |
| warm NEW prompt | 33.0 s | streaming Gemma, was OOM; → 22.8 s once cached |
| nvenc arms | 22.8-23.3 s | fell back to libx264 (see verdict) |
| batch 3 serial | 92.5 s | 30.8 s/clip incl. 2 streaming MISSes |
| batch 3 overlap | **73.7 s** | **24.6 s/clip, −20.4 %**, peak 76 GB |
