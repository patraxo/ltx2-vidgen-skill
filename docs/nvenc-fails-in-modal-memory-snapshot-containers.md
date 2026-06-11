# NVENC fails in Modal memory-snapshot containers (and how we proved it)

*June 2026 · Modal serverless · RTX PRO 6000 · PyAV 17.1 / driver 580.95*

If `avcodec_open2("h264_nvenc")` returns a generic
`UnknownError(1313558101, 'Unknown error occurred')` inside your Modal GPU
container — while the exact same open works in a scratch container — check one
thing first: **`enable_memory_snapshot=True`**.

## The evidence chain

We wanted hardware video encode (libx264 was 2.7s of a 23s request). Four
experiments, same image, same GPU class:

1. **Bare probe function** (plain `@app.function`, no snapshot): PyAV opens a
   real h264_nvenc encoder, encodes a frame, closes clean. Codec, driver, and
   `libnvidia-encode` all present and working.
2. **In-process in our serving class** (after torch initialized CUDA):
   `avcodec_open2` fails with the generic UnknownError — with ~21 GB VRAM free,
   so it's not memory.
3. **Torch-free child subprocess** inside the same container (fresh process,
   no torch import, clean dlopen space): **same failure.** This killed our
   "torch and PyAV's bundled ffmpeg collide over CUDA libs" theory — a clean
   process should not inherit a library conflict.
4. Only structural difference left between the passing and failing containers:
   the serving class runs with `enable_memory_snapshot=True`.

## Why snapshots break it

With memory snapshots, **every serving container runs a checkpoint-restored
process** — including the very first one (Modal checkpoints after your
snapshot-phase init, then restores for serving). The GPU is re-attached after
restore. That restore path covers the CUDA compute surface, but evidently not
the NVENC session/device-fd surface — and the broken state is container-wide,
which is why even a freshly spawned child process can't open an encoder
session.

## The trade

You can have NVENC by dropping memory snapshots. For us that trade is clearly
wrong: snapshots save 10–30s of cold-start on every cold container; NVENC
would save ~2s of encode per request. We kept snapshots, documented the
finding, and left the NVENC dispatch in the code default-off — with a
subprocess preflight that fails loudly back to libx264 and surfaces the real
error in the API response (`nvenc_last_error`), so nobody debugs a silent
fallback again.

## If you hit this

- Reproduce in a bare `@app.function` first ([`deploy/probe_nvenc.py`](../deploy/probe_nvenc.py)
  is our standalone probe). If bare passes and your app fails, suspect the
  container lifecycle, not ffmpeg.
- Don't trust an in-process preflight that silently falls back — your encode
  path will lie to you. Surface the open error in your response payload.
- Implementation details: [`references/LATENCY_RESEARCH_2026_06.md`](../references/LATENCY_RESEARCH_2026_06.md)
  (follow-up verdicts section).
