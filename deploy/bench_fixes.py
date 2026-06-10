"""Re-test after matrix-bench fixes: NVENC (preflight, no mid-stream fallback)
+ batch overlap (inter-clip allocator hygiene). One warm container, serial.

Usage:  cd ltx2-fast && PYTHONPATH=. uv run python deploy/bench_fixes.py
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
P5 = dict(height=1280, width=768, num_frames=121, num_inference_steps=8, seed=42)
# Warm single-call with a prompt NOT in the emb-cache: exercises the
# streaming-Gemma MISS path at full residency (the 93.75GB OOM repro).
NEW_PROMPT = (
    "cinematic dark fantasy: a lone knight kneels before a shattered obsidian "
    "throne, moonlight through broken stained glass, slow orbital camera, "
    "photoreal detail"
)


def gen(m, label, want_video=False, prompt=PROMPT, **kw):
    t0 = time.time()
    r = m.smoke_generate.remote(prompt=prompt, return_video_b64=want_video, **P5, **kw)
    keep = {k: r.get(k) for k in (
        "status", "error", "latency_s", "peak_vram_gb", "video_bytes",
        "nvenc_active", "nvenc_last_error", "phases")}
    keep["wall_s"] = round(time.time() - t0, 2)
    keep["label"] = label
    print(json.dumps(keep), flush=True)
    if want_video and r.get("video_b64"):
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, f"{label}.mp4"), "wb") as f:
            f.write(base64.b64decode(r["video_b64"]))
        print(f"saved {OUT_DIR}/{label}.mp4", flush=True)
    return keep


def batch(m, label, overlap):
    t0 = time.time()
    r = m.smoke_generate_batch.remote(
        prompts=BATCH_PROMPTS, overlap=overlap, return_video_b64=overlap, **P5
    )
    r["client_wall_s"] = round(time.time() - t0, 2)
    b64 = r.pop("video_b64", None)
    print(json.dumps({**r, "label": label}), flush=True)
    if b64:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "batch_overlap_clip3.mp4"), "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"saved {OUT_DIR}/batch_overlap_clip3.mp4", flush=True)
    return {**r, "label": label}


def main():
    Model = modal.Cls.from_name(APP, "Model")
    m = Model()
    results = []
    results.append(gen(m, "fx_build"))                       # build/persist
    results.append(gen(m, "fx_base", want_video=True))       # libx264 reference
    # warm MISS at full residency — the prod-OOM repro, now streaming-Gemma
    results.append(gen(m, "fx_newprompt", want_video=True, prompt=NEW_PROMPT))
    results.append(gen(m, "fx_nvenc1", nvenc=1))
    results.append(gen(m, "fx_nvenc2", want_video=True, nvenc=1))
    results.append(gen(m, "fx_nvenc_off", nvenc=0))
    results.append(batch(m, "fx_batch_serial", overlap=False))
    results.append(batch(m, "fx_batch_overlap", overlap=True))

    with open(os.path.join(OUT_DIR, "results_fixes.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== fixes summary ===", flush=True)
    for r in results:
        ph = r.get("phases") or {}
        extra = ""
        if "overlap_saved_s" in r:
            extra = (f"  serial_est {r.get('serial_estimate_s')}s"
                     f"  saved {r.get('overlap_saved_s')}s"
                     f"  denoise {r.get('denoise_s')}  fin {r.get('finalize_s')}")
        nverr = r.get("nvenc_last_error")
        if nverr:
            extra += f"  nv-err {str(nverr)[:90]}"
        print(f"{r['label']:>17}: {r.get('latency_s') or r.get('wall_s')}s"
              f"  enc {ph.get('encode_s', '-')}s  peak {r.get('peak_vram_gb')}GB"
              f"  {r.get('status')}{extra}", flush=True)


if __name__ == "__main__":
    main()
