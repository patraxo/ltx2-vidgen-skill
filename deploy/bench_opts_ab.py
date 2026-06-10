"""Latency fan-out A/B bench: cuDNN-SDPA priority x VAE no-tiling.

Same-container serial arms (identical resident pipeline + fused LoRA), so
latency AND quality deltas attribute to the toggled lever alone:

  1. build      (defaults)            -> pipeline build+persist; timing discarded
  2. base x2    sdpa_cudnn=0          -> baseline.mp4   (flash-SDPA, tiled VAE)
  3. cudnn x2   sdpa_cudnn=1          -> cudnn.mp4      (cuDNN-first SDPA)
  4. cudnn_notile x2  +tile_px=0      -> cudnn_notile.mp4 (cuDNN + reference decode)
  5. notile x2  sdpa_cudnn=0 tile_px=0 -> notile.mp4    (isolate tiling effect)

All arms: 768x1280 portrait, 121 frames, 8 steps, seed 42 (same as sage bench
for cross-comparability). sage_attn pinned 0 everywhere (rejected 2026-06-10).
cudnn.benchmark=True applies globally (deploy default) - folded into every arm.

Usage:  cd ltx2-fast && PYTHONPATH=. uv run python deploy/bench_opts_ab.py
"""

import base64
import json
import os
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

ARMS = [
    # (label, kwargs)
    ("base", dict(sage_attn=0, sdpa_cudnn=0)),
    ("cudnn", dict(sage_attn=0, sdpa_cudnn=1)),
    ("cudnn_notile", dict(sage_attn=0, sdpa_cudnn=1, tile_px=0)),
    ("notile", dict(sage_attn=0, sdpa_cudnn=0, tile_px=0)),
]


def run(m, label, arm_kwargs, want_video=False):
    t0 = time.time()
    r = m.smoke_generate.remote(return_video_b64=want_video, **arm_kwargs, **GEN)
    wall = round(time.time() - t0, 2)
    keep = {
        k: r.get(k)
        for k in (
            "status", "error", "is_oom", "latency_s", "peak_vram_gb",
            "free_vram_gb", "video_bytes", "sage_attn_active",
            "sdpa_cudnn_active", "phases",
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
        print(f"saved {path} ({keep['video_bytes']} bytes)", flush=True)
    return keep


def main():
    Model = modal.Cls.from_name(APP, "Model")
    m = Model()

    results = [run(m, "build", dict(sage_attn=0, sdpa_cudnn=0))]
    for label, kw in ARMS:
        results.append(run(m, f"{label}_1", kw))
        results.append(run(m, f"{label}_2", kw, want_video=True))

    print("\n=== summary (best warm per arm) ===", flush=True)
    base_best = None
    for label, _ in ARMS:
        warm = [
            r["latency_s"] for r in results
            if r["label"].startswith(label + "_") and r.get("latency_s")
            and r.get("status") == "ok"
        ]
        if not warm:
            print(f"{label:>14}: FAILED", flush=True)
            continue
        best = min(warm)
        if label == "base":
            base_best = best
        delta = (
            f"{round((base_best - best) / base_best * 100, 1):+}% vs base"
            if base_best and label != "base" else ""
        )
        print(f"{label:>14}: {best}s   {delta}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
