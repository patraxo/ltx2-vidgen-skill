# Why SageAttention made our 22B video model 5% SLOWER (and when it will for you too)

*June 2026 · Measured on LTX-2.3 (22B) · RTX PRO 6000 Blackwell (sm_120) · torch 2.12 + CUDA 13*

SageAttention 2.2 advertises 1.3–2× attention speedups, and the ComfyUI community
reports +16–24% end-to-end on video diffusion. We built it from source for
Blackwell sm_120, verified clean output, benchmarked it A/B on the same warm
GPU — and our 22B video DiT got **+5.1% slower** (23.88s vs 22.72s per clip).

This is not a "sage is bad" post. It's a "know your serving stack before you
swap kernels" post.

## The three reasons, measured

**1. torch.compile graph breaks ate the kernel win.**
Our pipeline runs fully compiled (inductor). SageAttention's CUDA op is not
traceable, so every attention call becomes a graph break: ~770 breaks per
request (48 blocks × 8 steps × 2 stages). Each break = a host round-trip +
cache lookup. The community numbers come from **eager-mode** ComfyUI, where
there's no compiled graph to break.

**2. LTX-2's compression leaves little attention to accelerate.**
LTX-2 uses ~1:192 video compression, so a 5s 768×1280 clip is only ~15k
tokens. At that sequence length, attention is 20–30% of step time — even a
2× attention kernel caps the theoretical end-to-end win at ~10–15%, before
overhead. Models with longer sequences (lower compression) have more to gain.

**3. INT8 quantization overhead is per-call.**
Sage quantizes Q/K to INT8 on every call. Against a strong baseline (inductor-
fused flash-SDPA on Blackwell), the quantize step costs more than the kernel
saves at this sequence length.

## The rule we derived

> An attention swap must be **compile-traceable on your stack**, or graph-break
> overhead can erase the kernel gain entirely. Benchmark end-to-end on YOUR
> serving configuration — kernel microbenchmarks and eager-mode community
> numbers do not transfer to compiled pipelines.

## Bonus gotcha: you can't even benchmark backends the obvious way

While testing alternatives we found that **inductor captures the SDPA backend
at trace time**. Wrapping inference in `sdpa_kernel([SDPBackend.CUDNN_ATTENTION,...],
set_priority=True)` at runtime is a **no-op for compiled blocks** — we proved it
by byte-comparing output mp4s (identical). To test a backend you must set the
priority before the FIRST call on a fresh process, so it's captured into the
trace. (Result for the curious: cuDNN SDPA = exactly 0% vs flash here.)

## Reproduce it

Build notes for SageAttention 2.2 on sm_120 (CUDA 13 base image, wheel/
packaging fixes) plus the full A/B methodology are in
[`references/SAGEATTENTION_SM120.md`](../references/SAGEATTENTION_SM120.md) and
[`references/LATENCY_RESEARCH_2026_06.md`](../references/LATENCY_RESEARCH_2026_06.md).
The bench driver is [`deploy/bench_sage_ab.py`](../deploy/bench_sage_ab.py) —
same container, same seed, byte-compared outputs, medians over repeated runs.

The patch ships in this repo default-OFF (`smoke_generate(sage_attn=1)` to flip)
so you can reproduce both sides of the measurement.
