"""Format/duration/encoder/overlap matrix bench (2026-06-10).

One warm container, ordered to minimize resolution switches (persist LRU=2
holds ONE res pair):

  PHASE 1a  768x1280 x 121f (5s portrait)   build + 2 warm + video
  PHASE 1b  768x1280 x 241f (10s portrait)  2 warm + video (same res = persist hit)
  PHASE 2   notile @ 10s portrait           1 call tile_px=0 (non-tiled 241f
                                            decode peak = the OOM question) + video
  PHASE 3   NVENC A/B @ anchor + @10s       frames deterministic per config ->
                                            PSNR vs phase-1 mp4 isolates encoder
  PHASE 4   batch overlap 3 clips           overlap=True vs False (bit-identical
                                            path, scheduling only)
  PHASE 5   1280x768 x 121f + 241f (landscape)  res-switch rebuild + 2 warm + video

Usage:  cd ltx2-fast && PYTHONPATH=. uv run python deploy/bench_matrix.py
"""

import base64
import json
import os
import time

import modal

APP = "ltx2-fast-inference"
OUT_DIR = os.path.join(os.path.dirname(__file__), "smoke_outputs", "matrix")

PROMPT = (
    "cinematic dark fantasy: a cloaked figure walks through a torchlit "
    "stone corridor, embers drifting, camera slowly pushing in, "
    "volumetric light, photoreal detail"
)
BATCH_PROMPTS = [
    PROMPT,
    "cinematic dark fantasy: rain-slicked castle battlements at night, a raven "
    "takes flight past torchlight, camera tracking sideways, photoreal detail",
    "cinematic dark fantasy: an ancient library lit by floating candles, dust "
    "motes in volumetric light, slow dolly forward, photoreal detail",
]

BASE = dict(num_inference_steps=8, prompt=PROMPT, seed=42)


def run(m, label, want_video=False, **kw):
    t0 = time.time()
    r = m.smoke_generate.remote(return_video_b64=want_video, **BASE, **kw)
    wall = round(time.time() - t0, 2)
    keep = {
        k: r.get(k)
        for k in (
            "status", "error", "is_oom", "latency_s", "peak_vram_gb",
            "video_bytes", "nvenc_active", "sdpa_cudnn_active", "phases",
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
    results = []
    P5 = dict(height=1280, width=768, num_frames=121)
    P10 = dict(height=1280, width=768, num_frames=241)
    L5 = dict(height=768, width=1280, num_frames=121)
    L10 = dict(height=768, width=1280, num_frames=241)

    # PHASE 1: portrait 5s + 10s
    results.append(run(m, "p5_build", **P5))
    results.append(run(m, "p5_warm1", **P5))
    results.append(run(m, "p5_warm2", want_video=True, **P5))
    results.append(run(m, "p10_warm1", **P10))
    results.append(run(m, "p10_warm2", want_video=True, **P10))

    # PHASE 2: non-tiled decode at 10s (the OOM question)
    results.append(run(m, "p10_notile", want_video=True, tile_px=0, **P10))

    # PHASE 3: NVENC (frames deterministic per config -> encoder-only delta)
    results.append(run(m, "p5_nvenc1", nvenc=1, **P5))
    results.append(run(m, "p5_nvenc2", want_video=True, nvenc=1, **P5))
    results.append(run(m, "p10_nvenc", want_video=True, nvenc=1, **P10))
    run(m, "nvenc_off_reset", nvenc=0, **P5)  # flip back; also a base re-measure

    # PHASE 4: batch overlap A/B (3 clips, portrait 5s)
    for ov in (False, True):
        t0 = time.time()
        r = m.smoke_generate_batch.remote(
            prompts=BATCH_PROMPTS, overlap=ov, return_video_b64=(ov is True),
            height=1280, width=768, num_frames=121, num_inference_steps=8, seed=42,
        )
        r["client_wall_s"] = round(time.time() - t0, 2)
        b64 = r.pop("video_b64", None)
        print(json.dumps({**r, "label": f"batch_overlap_{int(ov)}"}), flush=True)
        results.append({**r, "label": f"batch_overlap_{int(ov)}"})
        if b64:
            os.makedirs(OUT_DIR, exist_ok=True)
            path = os.path.join(OUT_DIR, "batch_overlap_clip3.mp4")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"saved {path}", flush=True)

    # PHASE 5: landscape (one res switch)
    results.append(run(m, "l5_build", **L5))
    results.append(run(m, "l5_warm", want_video=True, **L5))
    results.append(run(m, "l10_warm", want_video=True, **L10))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== matrix summary ===", flush=True)
    for r in results:
        ph = r.get("phases") or {}
        print(
            f"{r['label']:>18}: total {r.get('latency_s') or r.get('wall_s')}s"
            f"  inf {ph.get('inference_s', '-')}s  enc {ph.get('encode_s', '-')}s"
            f"  peak {r.get('peak_vram_gb', '-')}GB  status {r.get('status')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
