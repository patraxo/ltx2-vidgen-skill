"""SA2.2 vs SDPA same-container A/B bench for ltx2-fast-inference.

Runs against the DEPLOYED app (modal.Cls.from_name) — one warm container,
identical resident pipeline + fused LoRA for both arms, so latency AND quality
deltas are attributable to the attention kernel alone.

Sequence (all 768x1280 portrait, 121 frames @24fps, 8 steps, seed 42):
  1. sage=0 build run (pipeline build+persist; timing discarded)
  2. sage=0 warm x2  -> baseline latency; 2nd returns mp4 (baseline.mp4)
  3. sage=1 warm x2  -> sage latency;     2nd returns mp4 (sage.mp4)

Usage:  cd ltx2-fast && PYTHONPATH=. uv run python deploy/bench_sage_ab.py
"""

import base64
import json
import os
import time

import modal

APP = "ltx2-fast-inference"
OUT_DIR = os.path.join(os.path.dirname(__file__), "smoke_outputs", "sage_ab")

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


def run(m, label, sage, want_video=False):
    t0 = time.time()
    r = m.smoke_generate.remote(sage_attn=sage, return_video_b64=want_video, **GEN)
    wall = round(time.time() - t0, 2)
    keep = {
        k: r.get(k)
        for k in (
            "status", "error", "latency_s", "peak_vram_gb", "free_vram_gb",
            "video_bytes", "sage_attn_active", "sage_patch_installed",
            "persist_stage1_builds", "persist_stage2_builds",
            "persist_stage1_hits", "persist_stage2_hits",
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

    results = []
    results.append(run(m, "build_sdpa", sage=0))            # build+persist
    results.append(run(m, "warm_sdpa_1", sage=0))
    results.append(run(m, "warm_sdpa_2", sage=0, want_video=True))
    results.append(run(m, "warm_sage_1", sage=1))
    results.append(run(m, "warm_sage_2", sage=1, want_video=True))

    sdpa = [r["latency_s"] for r in results if r["label"].startswith("warm_sdpa")]
    sage = [r["latency_s"] for r in results if r["label"].startswith("warm_sage")]
    if all(sdpa) and all(sage):
        b, s = min(sdpa), min(sage)
        print(f"\nbaseline(SDPA) best warm: {b}s   sage(SA2.2) best warm: {s}s   "
              f"delta: {round((b - s) / b * 100, 1)}%", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
