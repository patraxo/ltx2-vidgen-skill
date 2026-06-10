"""cuDNN-SDPA trace-time bench.

bench_opts_ab.py proved the runtime sdpa_cudnn flip is a NO-OP for compiled
blocks (cudnn arm mp4 byte-identical to base): inductor selects the SDPA
backend at TRACE time. So this driver assumes a FRESH container (run
`modal app stop` + redeploy first) and makes the FIRST call - the one that
torch.compiles the transformer - run under cuDNN-first priority. Everything
this container compiles is then cuDNN-traced.

Sequence (768x1280, 121f, 8 steps, seed 42 - same as all prior benches):
  1. build  sdpa_cudnn=1   -> traces/compiles under cuDNN priority
  2. cudnn warm x4         -> median latency; last returns cudnn_traced.mp4
  3. flash warm x2         -> flip to 0; expect byte-identical + same latency
                              (backend captured in graph; proves symmetry)

Compare median vs the flash-traced container numbers in opts_ab/results.json.

Usage:
  cd ltx2-fast && uv run modal app stop ltx2-fast-inference --yes \
    && uv run --with fastapi modal deploy deploy/ltx2_model.py \
    && PYTHONPATH=. uv run python deploy/bench_cudnn_trace.py
"""

import base64
import json
import os
import statistics
import time

import modal

APP = "ltx2-fast-inference"
OUT_DIR = os.path.join(os.path.dirname(__file__), "smoke_outputs", "opts_ab")

GEN = dict(
    height=1280,
    width=768,
    num_frames=121,
    num_inference_steps=8,
    prompt=(
        "cinematic dark fantasy: a cloaked figure walks through a torchlit "
        "stone corridor, embers drifting, camera slowly pushing in, "
        "volumetric light, photoreal detail"
    ),
    seed=42,
)


def run(m, label, sdpa, want_video=False):
    t0 = time.time()
    r = m.smoke_generate.remote(
        sage_attn=0, sdpa_cudnn=sdpa, return_video_b64=want_video, **GEN
    )
    wall = round(time.time() - t0, 2)
    keep = {
        k: r.get(k)
        for k in (
            "status", "error", "latency_s", "peak_vram_gb",
            "video_bytes", "sdpa_cudnn_active", "phases",
        )
    }
    keep["wall_s"] = wall
    keep["label"] = label
    print(json.dumps(keep), flush=True)
    if want_video and r.get("video_b64"):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"{label}.mp4")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["video_b64"]))
        print(f"saved {path}", flush=True)
    return keep


def main():
    Model = modal.Cls.from_name(APP, "Model")
    m = Model()

    results = [run(m, "build_cudnn", sdpa=1)]
    for i in range(1, 4):
        results.append(run(m, f"cudnn_traced_{i}", sdpa=1))
    results.append(run(m, "cudnn_traced_4", sdpa=1, want_video=True))
    results.append(run(m, "flash_flip_1", sdpa=0))
    results.append(run(m, "flash_flip_2", sdpa=0, want_video=True))

    warm = [
        r["latency_s"] for r in results
        if r["label"].startswith("cudnn_traced") and r.get("status") == "ok"
    ]
    if warm:
        print(
            f"\ncuDNN-traced warm: median {statistics.median(warm)}s  "
            f"min {min(warm)}s  all {warm}",
            flush=True,
        )
        print("flash-traced container (opts_ab): base warm 23.18/28.92s "
              "median ~26.05s - cross-container compare, noise +/-20%", flush=True)

    with open(os.path.join(OUT_DIR, "results_cudnn_trace.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
