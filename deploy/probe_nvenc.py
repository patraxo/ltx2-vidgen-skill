"""NVENC availability probe for the ltx2-fast image on RTX PRO 6000.

Answers: can PyAV (pip `av` wheel in the image) open an h264_nvenc encoder?
Is there a system ffmpeg with nvenc? What nvenc codecs exist at all?

Run:  cd ltx2-fast && PYTHONPATH=deploy uv run --with fastapi modal run deploy/probe_nvenc.py
"""

import modal

from ltx2_model import ltx2_image, volumes

app = modal.App("ltx2-nvenc-probe")


@app.function(image=ltx2_image, gpu="RTX-PRO-6000", volumes=volumes, timeout=600)
def probe() -> dict:
    import shutil
    import subprocess

    out: dict = {}

    # 1. PyAV codec inventory
    try:
        import av
        out["av_version"] = av.__version__
        out["av_nvenc_codecs"] = sorted(c for c in av.codecs_available if "nvenc" in c)
    except Exception as e:
        out["av_error"] = repr(e)

    # 2. Real encoder-open test (listing can lie - driver/session matters)
    try:
        import io
        import av
        import numpy as np
        buf = io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("h264_nvenc", rate=24, options={"preset": "p5", "cq": "19"})
        stream.width, stream.height, stream.pix_fmt = 256, 256, "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.zeros((256, 256, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        out["pyav_nvenc_open"] = f"OK ({buf.getbuffer().nbytes} bytes)"
    except Exception as e:
        out["pyav_nvenc_open"] = f"FAIL: {e!r}"

    # 3. System ffmpeg?
    ff = shutil.which("ffmpeg")
    out["system_ffmpeg"] = ff
    if ff:
        try:
            enc = subprocess.run(
                [ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30
            ).stdout
            out["ffmpeg_nvenc"] = [l.strip() for l in enc.splitlines() if "nvenc" in l]
        except Exception as e:
            out["ffmpeg_nvenc"] = repr(e)

    # 4. Driver / GPU sanity
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0)
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,encoder.stats.sessionCount",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        out["nvidia_smi"] = (smi.stdout or smi.stderr).strip()
    except Exception as e:
        out["gpu_error"] = repr(e)

    for k, v in out.items():
        print(f"{k}: {v}")
    return out


@app.local_entrypoint()
def main():
    probe.remote()
