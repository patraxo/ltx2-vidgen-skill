# Learnings & gotchas — ltx2-clip-skill

Hard-won, measured findings from running LTX-2.3 (22B) self-hosted on one RTX PRO 6000 (Blackwell sm_120, 96 GB, bf16) on Modal. Read this before touching the persist/eviction/tiling paths.

## Memory & OOM

- **2 resident stage transformers FIT at 768×1280×97 on a CLEAN container** — measured peak **~75 GB** (23 GB free), zero OOM. The forward activation is only **~5 GB**, not the ~24 GB once assumed. Tile-size sweep: 768→74.94 GB, ≤512→73.5 GB (a ~1.4 GB lever).
- **The real OOM cause was memory ACCUMULATION over long-lived warm containers + cross-resolution stale residents**, not the inherent 2-resident footprint. A container that had built residents for one resolution (e.g. a 512 batch) then served a different resolution (1280) would stack weights and hit 94 GB.
  - Fix shipped: `_purge_stale_residents()` drops residents whose `(h,w)` ≠ the current request before any allocation (transformer weights are resolution-independent, ~35 GB each). Verified: 512→1280→youtube→1280 switches all pass.
  - `_max_residents_for(h,w,frames)` caps residents by a VRAM estimate. NOTE: it is currently **conservative** (forces 1 resident at high-res); the clean-container sweep shows 2 fit and are *faster* warm (~35 s vs ~43 s). Raising it to 2 is gated on a longevity/creep test (does peak drift back to OOM over many gens?).
- **`torch.cuda.empty_cache()` alone doesn't reclaim** — pipeline ref-cycles need `gc.collect()` *first*, then `empty_cache()` then `synchronize()`. `expandable_segments:True` is set and helps fragmentation but can't free genuinely-live blocks.
- A remote `torch.OutOfMemoryError` **fails to deserialize** in a local env without torch → surfaces as a generic client exception with no "out of memory" string. The smoke methods therefore catch and **return a diagnostic dict** (status/ is_oom/ free_vram_gb/ resident_count) instead of raising.
- **Stale warm containers serve OLD code** after a `modal deploy` until they drain. A new method kwarg → `TypeError: unexpected keyword argument` from the lingering container. Force a clean state with `modal app stop <app> --yes` then redeploy when testing signature changes.

## Precision / attention (all rejected — kept bf16)

- **fp8** (cast / scaled-mm): rejected — quality bar is bf16. (Studied LTX repos lean on fp8 to fit; we don't.)
- **SageAttention-2.2 (INT8-QK + fp16-PV/fp32-accum)**: RE-TESTED 2026-06-10 with correct
  source build (CUDA-13 base, sm_120 kernels — old "black frames" = env failure: nvcc 12.4
  never built it, patch was inert). Verdict: **works, no black frames, audio intact — but
  NET SLOWER: 23.88s vs 22.72s SDPA warm (-5.1%)** at 768×1280×121f/8-step. Why: our
  torch.compile'd flash-SDPA baseline is strong; sage = untraceable external op → graph
  breaks per block; LTX 1:192 compression → short seqs → attention share too small. Also
  nondeterministic run-to-run (SDPA = byte-identical) and trajectory diverges (PSNR 33.7 dB
  vs SDPA, per-frame quality comparable). REJECTED as default; patch stays (default OFF),
  runtime-togglable via `smoke_generate(sage_attn=1)`. First runtime flip costs ~70s recompile.
- **SageAttention-3 (FP4)**: banned — fp4 attention, quality rule bf16-only.
- **flash-attn**: no sm_120 kernel → PyTorch SDPA (exact) is used.
- **max-autotune** (torch.compile): tested, slower here than default compile.
- **SDPA backend is captured at TRACE time** (2026-06-10): inductor lowers
  F.s_d_p_a to a specific backend kernel during compile — a runtime
  `sdpa_kernel(set_priority=True)` context around the call does NOTHING for
  already-compiled blocks (proved: cuDNN-flip arm mp4 byte-identical to base).
  To change backend, the FIRST call of a fresh container must run under the
  desired priority. cuDNN-first measured = 0% vs flash (22.735s vs 22.72s
  medians, quiet hosts) — short LTX seqs + fused blocks equalize them.
- **Bench noise across Modal hosts**: same arm 23.18↔28.92s (±11-20%) on a
  noisy host vs ±0.5% on a quiet one. Never trust single-run cross-arm deltas;
  use byte-compares (SDPA path is deterministic per container+config) + medians.
- **VAE no-tiling (`tile_px=0`, 2026-06-10)**: non-tiled reference decode at
  768×1280×121f costs only +2.6 GB peak (74.3→76.9), latency-neutral, PSNR
  50.4 dB / SSIM 0.994 vs tiled (the delta IS the tiled arm's seam error),
  audio bit-equal. Use for standard reels; default stays tiled (retake/HQ at
  big res = decode-peak risk).

## VAE tiling

- `TilingConfig.default()` = spatial **768 px / 64 overlap**, temporal **80 frames / 24 overlap** (same as Lightricks + community repos). Env-tunable via `LTX_VAE_TILE_PX/OVERLAP/TEMPORAL_FRAMES/TEMPORAL_OVERLAP` + per-request `tile_px`/`temporal_frames`. Smaller tiles = smaller decode peak but only a ~1 GB lever here, and may add overlap-blend seams — fine-tune, not the OOM fix.

## Versions (verified 2026-06-08)

- HF weights `Lightricks/LTX-2.3`: `MODEL_REVISION=76730e6…` = **HF `main` HEAD (latest)**.
- LTX-2 git: pinned `1799988…` (vs latest `d605370…`). Pinned for a deliberate issue-#216 (multigpu import) workaround; `tiling.py`/TilingConfig is byte-identical to latest. Bumping = fresh image rebuild + full retest.
- Gemma `68f7ee4`, IC-LoRA union `b4d1c4d`, motion-track `572bb9c`, fp8 `1d756cd`.

## Behaviour notes

- **Resolution is W×H** in this repo: `768×1280` = vertical 9:16 reel (768 wide, 1280 tall). (Some earlier notes wrote it backwards.)
- **Canny control**: derive a DENSE edge map (`edgedetect=low=0.05:high=0.2`); sparse thresholds on a soft-lit face yield only a hair outline → the model drifts.
- **Keyframe coherence**: A and B must be the same subject/scene or you get a morph/identity-swap, not motion. Derive B by editing A, or use a clip's first/last frames.
- **v2v (retake)** is the slowest mode (~470 s for a 10 s window — single-stage full-CFG).

## NVENC is dead under Modal memory snapshots (2026-06-10)

`enable_memory_snapshot=True` ⇒ every serving container runs a
checkpoint-restored process. NVENC session creation
(`avcodec_open2("h264_nvenc")` → `UnknownError(1313558101)`) fails
container-wide in restored containers — parent process AND torch-free child
subprocesses — while the identical open passes in a plain non-snapshot
function on the same image + GPU class. Modal's GPU restore covers CUDA
compute, not the NVENC device-fd/ioctl surface. Don't burn time on encoder
swaps inside snapshot apps; test NVENC in a bare function first AND in the
real lifecycle before building anything.

## Warm-container OOM on first NEW prompt — Gemma full-GPU build (2026-06-10)

Upstream `PromptEncoder.__call__` builds the full ~23 GB Gemma per call. With
both stage transformers persisted (~70.8 GB), the first emb-cache MISS on a
warm container = 93.75 GB → OOM = dead request. Hidden by every same-prompt
bench (always HIT). Fix shipped: emb-cache wrapper flips the MISS call to
upstream `OffloadMode.CPU` streaming (layer-wise, ~5 GB peak, identical
embeddings) when free VRAM < `LTX_EMB_STREAM_FREE_GB` (28). Cost +10.2 s once
per prompt per container. LESSON: benches that reuse one prompt never
exercise the text-encoder memory path — always include a new-prompt warm arm.

## Batch overlap worth it; side-stream theory was wrong (2026-06-10)

3-clip batch: overlap (worker-thread finalize on the DEFAULT stream) wall
73.65 s vs 92.49 s serial = −20.4 %, peak 76 GB. The earlier clip-2 OOM was
NOT side-stream pool stranding — it was the Gemma MISS above. Cross-stream
allocator pools still don't share (keep finalize on the default stream), but
the OOM blame belonged to the text encoder.
