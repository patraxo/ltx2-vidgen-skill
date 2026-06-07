---
name: ltx2-video
description: >-
  Generate video from a photo (or two) using self-hosted LTX-2.3 on Modal GPU.
  Use this whenever the user wants to turn an image into a video, animate a
  photo, make a reel/clip, do keyframe interpolation between two images, restyle
  a video (video-to-video / retake), or generate video from a text prompt — even
  if they don't say the word "video", e.g. "bring this photo to life", "make
  this move", "animate this", "turn these two shots into a transition". Calls the
  user's deployed `ltx2-fast-inference` Modal app and saves an .mp4 locally.
  Triggers: "make a video", "animate this photo", "image to video", "i2v",
  "keyframe", "interpolate", "video to video", "retake", "generate a clip/reel".
argument-hint: "[/abs/path/image.jpg] [\"prompt\"] [i2v|keyframe|v2v|t2v]"
allowed-tools: Bash(uv run *) Bash(python3 *) Bash(ffmpeg *) Bash(file *) Bash(realpath *) Bash(test *) Bash(modal token *) Bash(modal app *) Read
---

# ltx2-video — photo → video via self-hosted LTX-2.3

Turns a local image (or two, or a video) into an `.mp4` by calling the user's
**deployed** `ltx2-fast-inference` Modal app (LTX-2.3, 22B). Four modes:

| Mode | Input | What it does |
|---|---|---|
| `i2v` (default) | 1 image + prompt | animates the photo into a clip |
| `keyframe` | 2 images + prompt | interpolates A → B |
| `v2v` | 1 video + prompt | regenerates a time window (retake) |
| `t2v` | prompt only | text-to-video, no image |

The work is done by `scripts/submit_video.py`, which calls the deployed app's
methods remotely via `modal.Cls.from_name` (no repo path needed).

## Setup (one-time)

- `pip install modal && modal token new`
- The backend must be deployed: `modal app list | grep ltx2-fast-inference`.
  If absent, deploy it from the `ltx2-fast-inference` repo: `./deploy.sh`.

## Workflow

1. **Resolve + validate the image.** Get the absolute path and confirm it's an image:
   ```bash
   realpath "<user-path>"            # normalize ~, relative, drag-dropped paths
   file "<abs-path>"                 # must contain JPEG / PNG / image data
   ```
   If not found or not an image, report and stop.
2. **Confirm before running** (it costs GPU time). Use **AskUserQuestion**:
   - header: `LTX-2.3`
   - question: `Generate video from <name>? First run cold-starts ~90s; warm ~7–9s; a few cents.`
   - options:
     - `Quick smoke (cheap)` — low-res sanity check, confirms the container is warm
     - `Full quality` — 97 frames @ 768×1280 (vertical reel)
     - `Cancel`
3. **Run** the script (set `--timeout 300` on the Bash call — the first run cold-starts):
   ```bash
   # i2v (full)
   uv run --with modal python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py \
     --mode i2v --image "<abs>" --prompt "<prompt>" --frames 97 --height 1280 --width 768

   # quick smoke (cheap warm-check)
   uv run --with modal python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py \
     --mode i2v --image "<abs>" --prompt "<prompt>" --frames 17 --height 320 --width 512 --steps 8

   # keyframe (two images)
   uv run --with modal python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py \
     --mode keyframe --image "<absA>" --image "<absB>" --prompt "<prompt>"

   # video-to-video retake
   uv run --with modal python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py \
     --mode v2v --video "<abs.mp4>" --prompt "<prompt>" --start 2 --end 5

   # text-to-video
   python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py --mode t2v --prompt "<prompt>"
   ```
   Immediately tell the user "waiting for container cold start (~90s)…" so it doesn't look hung.
4. **Report.** The script prints `SAVED <path>` and `PREVIEW <png>`. **Read** the
   PREVIEW png so the user sees a still inline, then report the saved mp4 path +
   latency. Offer follow-ups (longer clip via `--frames`, keyframe, v2v restyle).

## Prompting

Subject + action first, then lighting/camera, photorealistic detail; keep it tight.
Generate native vertical 9:16 (768×1280) for reels — don't render wide and crop.
Frame counts must be `8k+1` (17, 49, 97, 121, 217). bf16, no quantization.

## Guardrails

- Always confirm via AskUserQuestion before a full run (GPU cost). Offer the cheap smoke first.
- First call after idle cold-starts (~90s); warm calls 7–9s. Use `--timeout 300`.
- Do NOT route through fal-mcp. For Hail Films / @patrawtf canon reels, use the `hail-films-reel` skill instead.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `modal not installed` | `pip install modal && modal token new` |
| `from_name` can't find app | deploy the backend: `./deploy.sh` in the ltx2-fast-inference repo |
| no mp4 / `no video returned` | check `modal app logs ltx2-fast-inference` |
| looks hung | normal cold start — wait up to ~120s |
