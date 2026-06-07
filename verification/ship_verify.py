#!/usr/bin/env python3
"""Auth-free PROD verification driver for the 2026-06-07 optimization stack.

Runs a sequence of `smoke_generate.remote()` calls against the DEPLOYED prod
`ltx2` app (NO JWT, NO secret peek) on ONE warm container so the resident
persist registry + emb-cache survive across calls. Verifies:

  - 768x512  AND 768x1280 portrait: latency, peak VRAM, no-OOM, coherence.
  - resolution-switch: 512 then 1280 back-to-back in one warm container — the
    resolution-keyed persist must NOT crash / reuse a wrong-shape module.
  - bit-identical: baseline-config (persist off, emb off) vs all-on, SAME
    seed/prompt/res → decoded raw-RGB sha256 must MATCH.
  - audio-skip=false (audio kept) AND =true (silent) both work; the audio-skip
    video-RGB must equal the audio-kept video-RGB (audio track only dropped).

Writes raw mp4s + a JSON results blob into ship_clips/ for offline frame reads.

Run:  modal run deploy/ltx/ship_verify.py
"""

import base64
import hashlib
import json
import pathlib
import subprocess
import sys

# Repo layout: deploy/ltx2_model.py + utils/ live one level up from this file.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "deploy")]

from ltx2_model import app, Model

OUT = pathlib.Path(__file__).parent / "ship_clips"
OUT.mkdir(exist_ok=True)

# Small but real shapes. 17 frames / 6 steps keeps each gen cheap (~a few $ total
# across ~8 gens) while still exercising stage_1 + upsampler + stage_2 + (optional)
# audio decode. 768x512 = bench shape; 768x1280 = the REAL portrait reel shape.
NF = 17
STEPS = 6
PROMPT = "a calm mountain lake at dawn, gentle ripples, soft light"
SEED = 42


def _save_and_rgb_sha(b64, name):
    """Save mp4, then sha256 the decoded raw VIDEO-RGB (audio stripped) via ffmpeg."""
    if not b64:
        return None, None, 0
    raw = base64.b64decode(b64)
    mp4 = OUT / f"{name}.mp4"
    mp4.write_bytes(raw)
    # Decode VIDEO stream only to raw rgb24 → sha256 (audio-agnostic compare).
    try:
        rgb = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(mp4),
             "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            capture_output=True, check=True,
        ).stdout
        return mp4, hashlib.sha256(rgb).hexdigest(), len(raw)
    except Exception as e:
        return mp4, f"ffmpeg_err:{e}", len(raw)


def _streams(name):
    mp4 = OUT / f"{name}.mp4"
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type",
             "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.replace("\n", " | ")
    except Exception as e:
        return f"ffprobe_err:{e}"


def _extract_dense_frames(name, n_frames):
    """Extract every-other frame (dense ≥12/clip target) for coherence reading."""
    mp4 = OUT / f"{name}.mp4"
    fdir = OUT / f"{name}_frames"
    fdir.mkdir(exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(mp4),
             "-vf", "select=not(mod(n\\,2))", "-vsync", "0",
             str(fdir / "f%03d.png")],
            check=True,
        )
        return sorted(p.name for p in fdir.glob("*.png"))
    except Exception as e:
        return [f"extract_err:{e}"]


@app.local_entrypoint()
def main():
    results = {}
    m = Model()

    print("\n========== STEP 1: 512 BASELINE (persist off, emb off, audio KEPT) ==========")
    r = m.smoke_generate.remote(
        height=512, width=768, num_frames=NF, num_inference_steps=STEPS,
        prompt=PROMPT, seed=SEED,
        skip_audio=False, emb_cache=False, persist_mode="off",
        return_video_b64=True,
    )
    mp4, sha, nbytes = _save_and_rgb_sha(r.pop("video_b64", None), "base_512")
    r["rgb_sha256"], r["mp4_bytes"], r["streams"] = sha, nbytes, _streams("base_512")
    results["base_512"] = r
    print(json.dumps(r, indent=2))

    print("\n========== STEP 2: 512 ALL-ON build (persist both, emb on, audio KEPT) ==========")
    r = m.smoke_generate.remote(
        height=512, width=768, num_frames=NF, num_inference_steps=STEPS,
        prompt=PROMPT, seed=SEED,
        skip_audio=False, emb_cache=True, persist_mode="both",
        return_video_b64=False,
    )
    results["allon_512_build"] = r
    print(json.dumps(r, indent=2))

    print("\n========== STEP 3: 512 ALL-ON warm (steady-state; resident REUSE) ==========")
    r = m.smoke_generate.remote(
        height=512, width=768, num_frames=NF, num_inference_steps=STEPS,
        prompt=PROMPT, seed=SEED,
        skip_audio=False, emb_cache=True, persist_mode="both",
        return_video_b64=True,
    )
    mp4, sha, nbytes = _save_and_rgb_sha(r.pop("video_b64", None), "allon_512")
    r["rgb_sha256"], r["mp4_bytes"], r["streams"] = sha, nbytes, _streams("allon_512")
    results["allon_512_warm"] = r
    print(json.dumps(r, indent=2))
    _extract_dense_frames("allon_512", NF)

    print("\n========== STEP 4: 1280 PORTRAIT all-on (RESOLUTION SWITCH, build) ==========")
    # Back-to-back after 512 in the SAME warm container: the resolution-keyed
    # persist must build a NEW resident for 768x1280, NOT reuse the 512 module.
    r = m.smoke_generate.remote(
        height=1280, width=768, num_frames=NF, num_inference_steps=STEPS,
        prompt=PROMPT, seed=SEED,
        skip_audio=False, emb_cache=True, persist_mode="both",
        return_video_b64=False,
    )
    results["allon_1280_build"] = r
    print(json.dumps(r, indent=2))

    print("\n========== STEP 5: 1280 PORTRAIT all-on warm (resident REUSE) ==========")
    r = m.smoke_generate.remote(
        height=1280, width=768, num_frames=NF, num_inference_steps=STEPS,
        prompt=PROMPT, seed=SEED,
        skip_audio=False, emb_cache=True, persist_mode="both",
        return_video_b64=True,
    )
    mp4, sha, nbytes = _save_and_rgb_sha(r.pop("video_b64", None), "allon_1280")
    r["rgb_sha256"], r["mp4_bytes"], r["streams"] = sha, nbytes, _streams("allon_1280")
    results["allon_1280_warm"] = r
    print(json.dumps(r, indent=2))
    _extract_dense_frames("allon_1280", NF)

    print("\n========== STEP 6: 512 SWITCH BACK all-on (512 resident still valid?) ==========")
    # Switch resolution AGAIN back to 512 — must reuse the 512 resident (if not
    # LRU-evicted) or rebuild cleanly. Proves no wrong-shape reuse either way.
    r = m.smoke_generate.remote(
        height=512, width=768, num_frames=NF, num_inference_steps=STEPS,
        prompt=PROMPT, seed=SEED,
        skip_audio=False, emb_cache=True, persist_mode="both",
        return_video_b64=True,
    )
    mp4, sha, nbytes = _save_and_rgb_sha(r.pop("video_b64", None), "allon_512_back")
    r["rgb_sha256"], r["mp4_bytes"], r["streams"] = sha, nbytes, _streams("allon_512_back")
    results["allon_512_back"] = r
    print(json.dumps(r, indent=2))

    print("\n========== STEP 7: 512 audio-skip=TRUE (silent reel) ==========")
    r = m.smoke_generate.remote(
        height=512, width=768, num_frames=NF, num_inference_steps=STEPS,
        prompt=PROMPT, seed=SEED,
        skip_audio=True, emb_cache=True, persist_mode="both",
        return_video_b64=True,
    )
    mp4, sha, nbytes = _save_and_rgb_sha(r.pop("video_b64", None), "audioskip_512")
    r["rgb_sha256"], r["mp4_bytes"], r["streams"] = sha, nbytes, _streams("audioskip_512")
    results["audioskip_512"] = r
    print(json.dumps(r, indent=2))

    # ---- VERDICTS ----
    print("\n\n================= VERDICTS =================")
    b = results["base_512"]["rgb_sha256"]
    a = results["allon_512_warm"]["rgb_sha256"]
    ab = results["allon_512_back"]["rgb_sha256"]
    sk = results["audioskip_512"]["rgb_sha256"]
    print(f"base_512        rgb_sha = {b}")
    print(f"allon_512       rgb_sha = {a}")
    print(f"allon_512_back  rgb_sha = {ab}")
    print(f"audioskip_512   rgb_sha = {sk}")
    print(f"\nBIT-IDENTICAL (emb-cache+persist, audio kept): base==allon -> "
          f"{'PASS' if b and a and b == a else 'FAIL'}")
    print(f"BIT-IDENTICAL after res-switch-back: base==allon_back -> "
          f"{'PASS' if b and ab and b == ab else 'FAIL'}")
    print(f"AUDIO-SKIP video-RGB == audio-kept video-RGB (only track dropped): "
          f"{'PASS' if a and sk and a == sk else 'FAIL'}")
    print(f"audio-kept streams (allon_512):  {results['allon_512_warm']['streams']}")
    print(f"audio-skip streams (audioskip):  {results['audioskip_512']['streams']}")
    print(f"\nNO-OOM 512:  {not results['allon_512_warm']['persist_oom_fallback']} "
          f"(peak {results['allon_512_warm']['peak_vram_gb']} GB)")
    print(f"NO-OOM 1280: {not results['allon_1280_warm']['persist_oom_fallback']} "
          f"(peak {results['allon_1280_warm']['peak_vram_gb']} GB)")
    print(f"RES-SWITCH 1280 build OK (resident_count={results['allon_1280_build']['persist_resident_count']}, "
          f"oom_fallback={results['allon_1280_build']['persist_oom_fallback']})")

    (OUT / "ship_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT / 'ship_results.json'}")
