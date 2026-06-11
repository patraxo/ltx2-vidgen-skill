# The text-encoder OOM your benchmarks never catch

*June 2026 · LTX-2.3 (22B) + Gemma-3 12B text encoder · 96 GB RTX PRO 6000*

We ran dozens of warm-container benchmarks on our video-generation stack.
All green: 22.7s per clip, peak 74.3 GB of 96 GB, zero OOM over long runs.
Then a 3-clip batch died on clip 2 at **93.75 GB**.

The difference: clip 2 was the first request with a **new prompt**.

## The mechanism

LTX-2's `PromptEncoder` builds the full Gemma text encoder (~23 GB in bf16)
on GPU **inside every call**, uses it, and frees it on exit — a clean
build-use-free design that works fine when nothing else is resident.

But a persistent serving stack keeps the two 22B stage transformers GPU-
resident between requests (~70.8 GB) for warm latency. Add a text-embedding
cache, and every benchmark that reuses the same prompt hits the cache —
**Gemma never loads**. The first genuinely new prompt on a warm container is
the first time the full-GPU Gemma build runs against full residency:

```
70.8 GB resident transformers + ~23 GB Gemma build  →  93.75 GB  →  OOM
```

A dead production request, invisible to every same-prompt benchmark we ran.

## The fix (no quality cost)

Upstream's PromptEncoder already constructs a **streaming builder** for the
text encoder (`OffloadMode.CPU`: weights stream layer-by-layer to a small GPU
buffer, ~5 GB peak) — it's just not selected unless `offload_mode` is set, and
the CLI's `--offload` flag is **global**: it would stream the 22B transformers
too, defeating persistent serving.

So our embedding-cache wrapper flips `offload_mode` to `CPU` **only for the
cache-miss call, only when free VRAM is below a threshold** (default 28 GB,
env `LTX_EMB_STREAM_FREE_GB`), then restores it:

- Same weights, same math, same dtype → identical embeddings.
- Measured: the request that used to die now completes in 33.0s at 78.6 GB
  peak (vs 22.8s for a cached prompt — the +10s is once per prompt per
  container, then cached).
- Fresh containers (low residency) keep the fast full-GPU build.

Output quality verified the only way that counts for generative video: frame
inspection of the resulting clips — fully prompt-faithful through the streamed
encoder.

## The transferable lesson

> **Same-prompt benchmarks never exercise the text-encoder memory path.**
> If your serving stack caches embeddings (it should), every load test that
> recycles prompts is silently skipping a multi-GB allocation that production
> will hit. Always include a warm new-prompt arm in the bench matrix.

Implementation: the wrapper lives in [`deploy/ltx2_model.py`](../deploy/ltx2_model.py)
(search `_EMB_STREAM_FREE_GB`); measurements in
[`references/LATENCY_RESEARCH_2026_06.md`](../references/LATENCY_RESEARCH_2026_06.md).
