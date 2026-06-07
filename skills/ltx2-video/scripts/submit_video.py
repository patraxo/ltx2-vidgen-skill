#!/usr/bin/env python3
"""Submit a video-generation job to the deployed `ltx2-fast-inference` Modal app
and save the resulting .mp4 locally (+ a preview frame).

Calls the deployed app's methods remotely via `modal.Cls.from_name` — no repo
source needed, just `pip install modal` + `modal token new` + the app deployed.

Modes:
  i2v       1 image  + prompt  -> animated clip
  keyframe  2 images + prompt  -> interpolation A->B
  v2v       1 video  + prompt  -> retake (regenerate a time window)
  t2v       prompt only        -> text-to-video (no image conditioning)

Examples:
  python3 submit_video.py --mode i2v --image photo.jpg --prompt "..." --frames 97
  python3 submit_video.py --mode keyframe --image a.jpg --image b.jpg --prompt "..."
  python3 submit_video.py --mode v2v --video clip.mp4 --prompt "..." --start 2 --end 5
"""
import argparse
import base64
import datetime
import pathlib
import subprocess
import sys

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _b64(path: str) -> str:
    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        sys.exit(f"ERROR: file not found: {p}")
    return base64.b64encode(p.read_bytes()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate video via the deployed ltx2-fast-inference app.")
    ap.add_argument("--mode", default="i2v", choices=["i2v", "keyframe", "v2v", "t2v"])
    ap.add_argument("--image", action="append", default=[], help="image path (repeat once more for keyframe)")
    ap.add_argument("--video", help="source video path (v2v)")
    ap.add_argument("--prompt", default="cinematic, natural motion, photorealistic")
    ap.add_argument("--frames", type=int, default=97, help="num frames (8k+1, e.g. 49/97/121/217)")
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--start", type=float, default=2.0, help="v2v: window start (s)")
    ap.add_argument("--end", type=float, default=5.0, help="v2v: window end (s)")
    ap.add_argument("--app", default="ltx2-fast-inference")
    ap.add_argument("--out-dir", default="./video_out")
    args = ap.parse_args()

    try:
        import modal
    except ImportError:
        sys.exit("ERROR: modal not installed. Run: pip install modal && modal token new")

    Model = modal.Cls.from_name(args.app, "Model")
    m = Model()

    out = pathlib.Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    print(f"[ltx2-video] mode={args.mode} app={args.app} frames={args.frames} {args.width}x{args.height}")
    print("[ltx2-video] first call is a cold start (~90s); warm calls are ~7-9s.")

    if args.mode == "v2v":
        if not args.video:
            sys.exit("ERROR: --video required for v2v")
        res = m.smoke_retake.remote(
            video_b64=_b64(args.video), prompt=args.prompt,
            start_time=args.start, end_time=args.end, num_inference_steps=args.steps,
        )
    else:
        if args.mode == "i2v":
            if not args.image:
                sys.exit("ERROR: --image required for i2v")
            image_b64 = _b64(args.image[0])
        elif args.mode == "keyframe":
            if len(args.image) < 2:
                sys.exit("ERROR: keyframe needs two --image args (A and B)")
            image_b64 = [_b64(args.image[0]), _b64(args.image[1])]
        else:  # t2v
            image_b64 = []  # empty -> no image conditioning
        res = m.smoke_generate.remote(
            height=args.height, width=args.width, num_frames=args.frames,
            num_inference_steps=args.steps, prompt=args.prompt,
            image_b64=image_b64, return_video_b64=True,
        )

    b64 = res.get("video_b64") if isinstance(res, dict) else None
    if not b64:
        sys.exit(f"ERROR: no video returned: {res}")
    mp4 = out / f"{ts}_{args.mode}.mp4"
    mp4.write_bytes(base64.b64decode(b64))
    print(f"SAVED {mp4}  ({res.get('video_bytes')} B, latency {res.get('latency_s')}s)")

    # Best-effort preview frame (needs ffmpeg).
    png = out / f"{ts}_{args.mode}.png"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
             "-vf", "select=eq(n\\,1)", "-vframes", "1", str(png)],
            check=False, timeout=30,
        )
        if png.is_file():
            print(f"PREVIEW {png}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
