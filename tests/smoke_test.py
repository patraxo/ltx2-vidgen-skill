#!/usr/bin/env python3
"""Auth-free end-to-end smoke test for the LTX-2.3 Modal deploy (app `ltx2`).

Invokes the Model's `smoke_generate` method DIRECTLY via Modal (no JWT HTTP
endpoint, no api-key secret peek). Runs ONE tiny default-mode i2v job from an
in-container solid-color image (320x256, 17 frames, 4 steps) and prints the
returned status + wall-clock latency.

Run:
    modal run deploy/ltx/smoke.py
"""

import sys
import pathlib

# Repo layout: deploy/ltx2_model.py + utils/ live one level up from this file.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "deploy")]

import time

from ltx2_model import app, Model


@app.local_entrypoint()
def main():
    print("=== LTX-2.3 smoke test (direct method invoke, no auth) ===")
    t0 = time.time()
    result = Model().smoke_generate.remote(
        height=256,
        width=320,
        num_frames=17,
        num_inference_steps=4,
    )
    wall = time.time() - t0
    print(f"RESULT: {result}")
    print(f"CLIENT_WALL_CLOCK_S: {wall:.2f}")
    if result.get("status") == "ok" and result.get("video_bytes", 0) > 0:
        print("SMOKE_TEST: PASS")
    else:
        print("SMOKE_TEST: FAIL")
