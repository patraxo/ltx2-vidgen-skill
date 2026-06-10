# SageAttention on sm_120 (RTX PRO 6000 Blackwell) × LTX-2.3 — research + decision record

Date: 2026-06-10. Sources: thu-ml/SageAttention, woct0rdho fork,
ComfyUI core + KJNodes source, Lightricks repos, HF wheel repos, Reddit/community threads.

## TL;DR

SageAttention **2.2** (INT8-QK + FP16-PV, CUDA kernels, fp32 PV accumulation) = verified Blackwell sm_120 path for LTX-2.3 — community-measured **~16-35% faster diffusion**, no reported quality loss when invoked correctly. SageAttention **3** (FP4) also targets sm_120 but needs Python ≥3.13, quality-riskier (FP4 attention), matches previously observed black-frame path → skipped. Repo's old "SageAttn-2.2 = black output on LTX" dead-end **re-diagnosed as environment failure**, not kernel incompatibility.

## Why the old attempt black-framed / never ran

1. **Real blocker was image, not kernel.** Base `nvidia/cuda:12.4.0-devel`
   has nvcc 12.4; torch (via ltx-core) is 2.12.0+cu130. `torch.utils.cpp_extension`
   refuses mismatch → sageattention never pip-built in-image →
   `LTX_SAGE_ATTN=2` branch always hit ImportError → patch inert.
2. **Documented black-output causes elsewhere (none apply to our config):**
   - PyPI `pip install sageattention` ships **1.0.6** → black output on Blackwell
     (ComfyUI discussion #11583, mobcat40).
   - ComfyUI `--use-sage-attention` global flag → **Triton backend** on some paths →
     black output (Qwen/Wan reports; ComfyUI issue #9773: Triton PV kernel fp16
     accumulation overflow on Ampere).
   - `pv_accum_dtype="fp16"` pure-fp16 accumulation → overflow → NaN → black
     (thu-ml issue #93).
   - Global patching of ALL attention (cross-attn with masks, VAE, text encoder)
     instead of DiT self-attn only (SageAttention README warns explicitly).
3. Our deploy's patch already avoids all four: source-built 2.2, explicit
   `sageattn_qk_int8_pv_fp16_cuda` (default `pv_accum_dtype="fp32"`), self-attn-only
   routing (context is None and mask is None), per-call try/except → SDPA fallback.

## Version matrix (sm_120, linux, torch 2.12.0+cu130, py3.12)

| Version | sm_120 | Our env | Notes |
|---|---|---|---|
| PyPI 1.0.6 | ✗ | ✗ | Black output documented. Never use. |
| 2.1.1 | partial | ✗ | No dedicated sm_120 kernels. |
| **2.2.0 (source)** | **✓ dedicated kernels** | **✓ (this repo)** | `qk_int_sv_f8_cuda_sm120.cu` etc. ~30-35% on RTX 5090 (Qwen 40-step 14m30s→9m30s); LTX-2.3: 16-24% via KJ node (HF Lightricks/LTX-2.3-nvfp4 discussion #2, RTX 5090 laptop). |
| 3.0 (FP4) | ✓ (sm_120a is TARGET; sm_100 datacenter unsupported, issue #237) | ✗ py3.13 req + FP4 quality risk | Earlier "won't build sm_120" note = wrong env (nvcc 12.4) + wrong arch claim. |

**Wheels:** no linux torch-2.12+cu130 wheel anywhere (woct0rdho = Windows-only;
sntdismas SA3 = torch 2.11 ABI → undefined-symbol on 2.12; pseudobacon SA2.2 abi3 =
"torch2.11.0andhigher" unverified). → **source build** = deterministic path.

## Build recipe (what the image does)

```
base: nvidia/cuda:13.0.3-devel-ubuntu22.04   # nvcc 13.0 == torch cu130
git clone thu-ml/SageAttention (main, --depth 1; SHA echoed in build log)
EXT_PARALLEL=4 NVCC_APPEND_FLAGS='--threads 4' MAX_JOBS=6 \
TORCH_CUDA_ARCH_LIST='12.0' pip install --no-build-isolation -v .
```

- `TORCH_CUDA_ARCH_LIST='12.0'` ONLY. Add 9.0 → injects sm_90a `wgmma.*` PTX, illegal
  on sm_120 target → compile failure (thu-ml issue #291).
- AOT compile; no runtime JIT. Build-time import check asserts
  `sageattn_qk_int8_pv_fp16_cuda` importable — broken build fails image,
  not first request.

## Runtime contract (the `_gpu_init` patch)

- Patch `ltx_core...Attention.forward` (class-level mirror of pinned-commit
  forward). Sage path ONLY when `context is None and mask is None and not
  all_perturbed` — i.e. video/audio self-attn. Cross-attn (text, audio↔video) stays
  SDPA: Sage rejects arbitrary `attn_mask`.
- Layout: q,k,v reshaped (b, seq, h, d) → transpose(1,2) → (b, h, seq, d) = sage
  `tensor_layout="HND"` (same as torch SDPA). `is_causal=False`. bf16 in → bf16 out.
- head_dim: video stream 128, audio stream 64 — both natively supported (kernels do
  64/128; 65–127 padded; >128 unsupported).
- `pv_accum_dtype` stays default `"fp32"` (overflow-safe setting).
- Per-call try/except → original SDPA forward. One bad kernel call degrades, never
  crashes.
- **Runtime toggle:** `_SAGE22_RUNTIME[0]` (default from env `LTX_SAGE_ATTN=2`),
  flipped per request via `smoke_generate(sage_attn=0|1)`. Enables same-container
  A/B: identical resident pipeline + fused LoRA both arms (persist-pipeline LoRA
  fusion not reproducible across containers — cross-deploy A/B polluted).

## Bench protocol

`deploy/bench_sage_ab.py`: 768×1280, 121 frames, 8 steps, seed 42 →
build run + 2× SDPA warm + 2× SA2.2 warm (mp4 captured per arm) →
latency delta, ffmpeg PSNR/SSIM between arms, blackdetect, visual frame inspection.
Acceptance: no black/NaN frames, no visible artifacting vs SDPA, latency drop
in attention-bound regime, audio intact (audio self-attn also routes through sage).

## Expected ceiling (set expectations)

Sage accelerates attention kernel only. With FBCache skipping blocks + distilled 8-step schedule, end-to-end gain BELOW headline 30-35% (that figure = 40-step image-model, attention-dominated). Community LTX-2.3 number: 16-24% it/s gain. ≥10% end-to-end warm = win, stacks with existing levers. Bench decides.

## What to try next if SA2.2 validates

- ~~`sageattn_qk_int8_pv_fp8_cuda` FP8-PV kernel~~ — **excluded by the bf16-only quality rule
  2026-06-10: no fp8 anywhere, bf16 only.** fp16-PV + fp32-accum is the ceiling.
- SpargeAttn (thu-ml, Apache) — sparse attention on TOP of sage kernels,
  explicit FBCache-coexistence design (`skip_*blocks`), CUDA ≥12.8 Blackwell gate.
- Do NOT stack TeaCache/MagCache/TaylorSeer (compete with FBCache).

## MEASURED VERDICT (2026-06-10) — REJECTED as default

Bench (`deploy/bench_sage_ab.py`, same warm container, 768×1280×121f, 8 steps, seed 42):

| arm | warm latency | notes |
|---|---|---|
| SDPA | 22.81 / **22.72 s** | byte-identical mp4s run-to-run (deterministic) |
| SA2.2 first call | 93.97 s | torch.compile recompile (runtime-flag guard flip → 48 blocks) |
| SA2.2 steady | **23.88 s** | **-5.1% (SLOWER)**; mp4s differ run-to-run (nondeterministic) |

Quality gate: blackdetect 0 segments both arms; audio aac/48k intact (RMS -17.0 vs
-16.7 dB); frames artifact-free both arms; PSNR 33.7 dB / SSIM 0.947 between arms =
trajectory divergence (INT8-QK drift compounds over 8 steps × 2 stages), per-frame
quality comparable — but NOT "indistinguishable" in the strict same-output sense.

Why community +16-24% didn't reproduce: (1) baseline here = torch.compile'd
flash-SDPA, far stronger than ComfyUI eager; sage call = untraceable external op →
graph break per block per step eats the kernel gain; (2) LTX-2.3's 1:192 latent
compression → ~15k tokens at this res → attention share of step time too small;
(3) ltx-core's own sm_120 priority is cuDNN-first SDPA (also strong).

State: patch deployed, default OFF (env unset). Kernel proven healthy on sm_120 —
reusable substrate if a sparse-attention successor (SpargeAttn-class) lands sm_120
kernels. Earlier dead-end note "SA2.2 = black output on LTX" = CORRECTED (env failure,
not kernel): it builds + runs clean; rejection reason is throughput, not quality.

## Quality rule (user, 2026-06-10)

Weights/activations bf16 ONLY. No fp8, no fp4 — weights OR attention kernels.
SA2.2 fp16-PV is conditionally allowed ONLY because PV runs fp16 with fp32
accumulation and Q·K INT8 is attention-internal; acceptance gate = output
indistinguishable from bf16-SDPA (blackdetect + PSNR/SSIM + visual frames +
audio). ANY visible degradation → REJECT sage, stay SDPA.