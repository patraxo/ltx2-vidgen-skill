# ltx2-video — Mode UX & Interaction Design

Implementation-ready interaction design for the `ltx2-video` Claude Code skill. Wraps the self-hosted **LTX-2.3 (22B)** backend on Modal (`ltx2-fast-inference`) via `scripts/submit_video.py`. Five modes: **i2v, t2v, keyframe (A→B), v2v (retake window), control (IC-LoRA union: canny/depth/pose)**.

Grounded in the actual code:
- Script: `/Users/sandeep.patra/.claude/skills/ltx2-video/scripts/submit_video.py`
- Backend: `/Users/sandeep.patra/Downloads/ltx2-fast/deploy/ltx2_model.py`
- Backend runs at **24 fps**; frame counts must be **8k+1** (17, 49, 97, 121, 217, 241). 97f ≈ 4s, 217f ≈ 9s, 241f ≈ 10s. Returns `latency_s` per job.

> Positioning note: Higgsfield is studied below **only as a UX reference** (the technique, not the platform). The skill never recommends Higgsfield or any SaaS; it drives the user's own self-hosted LTX-2.3. No paid per-clip APIs in this path.

---

## PART 1 — Higgsfield interaction patterns worth reusing (study, don't copy)

Higgsfield's product is a useful reference for *how to structure a camera-first, image-grounded video UX*. The reusable interaction primitives:

### 1.1 Named motion/camera preset library (the big one)
Higgsfield exposes **50–250+ named camera presets** as the primary control surface, not free-text. The named moves are an industry-standard cinematography vocabulary, grouped by camera move and VFX:

> **Dolly In / Dolly Out, Crane Up / Crane Down, Jib, Pan, Whip Pan, Tilt, 360 Orbit, Zoom, Crash Zoom In / Crash Zoom Out, Dolly Zoom In / Dolly Zoom Out (vertigo), Bullet Time, FPV Drone, Handheld, Snorricam, Dutch Angle, Focus Change, Hyperlapse** (Higgsfield Camera Controls; Kolbo.AI; Scribe; Clipia DoP).

**Reusable pattern:** a *preset name → expanded shot description* expansion. The user picks an intent word; the system expands it into a full, model-ready clause. This is exactly the pattern the user's memory endorses ("use the reference-image **usage pattern**, not the platform"). We replicate it as a **named motion-preset table** that expands into LTX-2.3 prompt text (PART 3.2), since LTX has no preset dropdown — the preset *is* prompt engineering.

### 1.2 Start Frame → End Frame keyframe framing
Higgsfield's keyframe UX (Cinema Studio "Keyframe Interpolation"; "Kling Start/End Frames") is framed as: **Start Frame = your primary image, End Frame = the frame you want the video to arrive at**, AI fills the in-between for a "smooth, morph-free transition." Two product truths to steal:
- **Same image for both ends → auto smooth loop / 360° camera loop.** (A clean default we can offer.)
- Explicit guidance: **"keyframes should match the theme for best results."** This is the coherence rule, stated in their own docs — unrelated A/B is where morph/dissolve happens. Higgsfield even sells a separate **"Morph" transition** product for the intentionally-unrelated case, which tells you they treat "coherent interpolation" and "morph between unrelated clips" as **two different features**. Our skill should make the same distinction explicit (PART 2 keyframe).

### 1.3 Reference-image / image-grounded control
- Reference attach pattern: users tag `@image1` in the prompt to bind an uploaded image; character art is dropped in directly and the *motion* is described on top ("turn this character art into a shonen battle scene"). The image carries identity; the prompt carries motion.
- Workflow is **image-first, then motion**: upload image → pick a camera preset → add an action line. The image is never re-described by the user; the motion is the only thing they author.

**Reusable pattern → our i2v contract:** *the skill describes the image first (image-grounded), then appends motion.* This is the single most important borrowed idea for i2v quality (PART 2 i2v).

### 1.4 Mode entry + clarifying questions (from GitHub Claude skills)
- **AKCodez/higgsfield-claude-skills** (19 skills): slash-command entry points map a request to a pre-configured "expertise layer"; clarifying is **context-specific, inline** ("TikTok ad for my wireless earbuds, premium unboxing feel") rather than a generic menu. Defaults stored as config: **Ratio 9:16, Duration 8s**, "camera-movement encyclopedia 15–20+ techniques." Confirms: **vertical-by-default + a finite named-move vocabulary** is the expected shape.
- **robonuggets/higgsfield-skill** (one MCP, 30+ models): notable for a **"built-in decision rule — only suggest Higgsfield when the user already subscribes."** Mirror this as a guardrail: *never* suggest a SaaS; we already have a self-hosted backend.
- **SamurAIGPT/Generative-Media-Skills**: explicit CLI mode split (`--mode i2v --file …` vs t2v `--subject …`), and motion lives **inside** the natural-language subject prompt ("camera slowly pulls back to reveal…") plus a named **`--intent`** preset (`epic`, `reveal`). No callable motion library — confirms most LLM video skills *don't* yet ship a preset table, which is where ours can be better.
- **digitalsamba / wilwaldon Claude-Code-Video-Toolkit, SamurAI**: structured "concept → image → render" pipelines; reinforce the *describe → confirm → render → preview* loop we already use.

**Honest gaps:** Higgsfield's exact per-preset prompt expansions and their internal coherence thresholds are not public (the camera-controls page couldn't be fetched directly; preset names are corroborated across Kolbo.AI, Scribe, Clipia). Treat the expansions in PART 3.2 as **our** authored equivalents tuned for LTX-2.3, not Higgsfield's literals.

**Sources:** [Higgsfield Camera Controls](https://higgsfield.ai/camera-controls) · [Higgsfield WAN Camera Control guide](https://higgsfield.ai/blog/WAN-AI-Camera-Control-Your-Guide-to-Cinematic-Motion) · [Kolbo.AI: 100+ camera presets](https://kolbo.ai/blog/higgsfield-suite-100-camera-presets) · [Scribe: 50+ camera presets](https://scribehow.com/page/I_Tried_Higgsfield_Cinema_Studios_50_Camera_Presets__Heres_What_Happened__Z2vkHHECTSKny70JtAzMNg) · [Clipia: Higgsfield DoP (Dolly/Pan/Orbit from photo)](https://clipia.ai/en/video-models/higgsfield-dop) · [Higgsfield Kling Start/End Frames](https://higgsfield.ai/blog/Kling-Start-End-Frames) · [Higgsfield Transitions (Morph)](https://higgsfield.ai/blog/The-Ultimate-Video-Transitions-Tool) · [AKCodez/higgsfield-claude-skills](https://github.com/AKCodez/higgsfield-claude-skills) · [robonuggets/higgsfield-skill](https://github.com/robonuggets/higgsfield-skill) · [SamurAIGPT/Generative-Media-Skills](https://github.com/SamurAIGPT/Generative-Media-Skills) · [digitalsamba/claude-code-video-toolkit](https://github.com/digitalsamba/claude-code-video-toolkit)

---

## PART 2 — Per-mode interaction contract

Each mode below gives: **when it fires**, **what the skill says/does**, **how it elicits the prompt**, and **the exact `submit_video.py` args** it maps to. Default render is **vertical 9:16, 768×1280**.

### 2.1 i2v — animate a single photo (DEFAULT when one image is present)

**Conversational contract.** User drops one image and says "animate this" / "make it move" / "bring this to life." Default to i2v.

**The image-grounded prompting rule (core technique borrowed from Higgsfield):** the user should NOT have to describe their own photo. The skill does the description, then adds motion. Concretely the skill should:
1. **Read the image first** (it already validates with `file`; also actually `Read` the pixels) and silently form a 1-sentence description of subject + setting + lighting.
2. **Author the motion**, not the subject. Final prompt = `<image description (subject/setting/light)> , <motion clause> , <camera clause>`. Keep the description faithful so identity/scene is preserved; only the motion + camera are new.
3. Offer **named motion presets** (the user picks an intent word; the skill expands it). Default to **"subtle idle"** if the user gives nothing — it's the safest, most coherent i2v motion.

**Named motion preset library (i2v)** — see full expansions in PART 3.2. Minimum set the SKILL.md should know:
`subtle idle`, `slow push-in`, `slow pull-out`, `hair / wind`, `turn to camera`, `orbit (arc)`, `parallax / drift`, `rack focus`, `ambient life` (crowd/water/fire moving), `handheld energy`.

**What to say (one image, ambiguous):** fire the **AskUserQuestion** in PART 3.3 (animate vs keyframe-A vs control). On "animate," optionally offer the preset chips.

**Args mapping:**
```bash
uv run --with modal python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py \
  --mode i2v --image "<abs>" --prompt "<image-desc + motion preset expansion>" \
  --frames 97 --height 1280 --width 768
```
- Cheap warm-check: `--frames 17 --height 320 --width 512 --steps 8`.
- Longer clip: bump `--frames` (97≈4s, 217≈9s). Keep 8k+1.

---

### 2.2 t2v — text only, no image

**Conversational contract.** User describes a scene with **no image** ("generate a video of a neon alley in the rain"). No conditioning frame exists, so identity is invented — use t2v.

**When t2v vs i2v:** If the user has *any* representative still, prefer **i2v** (far better subject/identity control and coherence — i2v is faster and more controllable). Use t2v only when there's genuinely no image, or the user explicitly wants the model to invent everything. State this tradeoff if they ask for t2v while holding an image.

**Prompt scaffolding (the skill builds this, in order):**
> **Subject → Action → Camera → Lighting → Style**

Template the skill fills:
```
<subject, concrete & specific>, <action/motion>, <camera move + framing>, <lighting>, <style/film-look>, 9:16 vertical.
```
Example: `a lone swordsman on a cliff edge, cloak whipping in the wind, slow push-in on his face, stormy overcast light, cinematic anamorphic film grain, 9:16 vertical.`

Keep it tight (one to two sentences). The backend already appends a strong negative prompt internally — the skill does **not** need to author negatives.

**Args mapping:**
```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py --mode t2v \
  --prompt "<scaffolded prompt>" --frames 97 --height 1280 --width 768
```

---

### 2.3 keyframe — interpolate A → B (the coherence-critical mode)

**Conversational contract.** User has two images, or says "transition from this to that," "morph A to B," "first frame / last frame." Maps to `--mode keyframe --image A --image B`.

**THE key UX problem: coherence.** LTX interpolates between A and B. If A and B are the **same subject/scene** (same character, same room, slightly different pose/expression/camera), the result is a **clean, coherent motion**. If A and B are **unrelated** (different person, different place), there is no coherent in-between — the model produces a **morph/cross-dissolve** (smearing, identity melt). This is identical to Higgsfield's own rule ("keyframes should match the theme") and why they ship a separate "Morph" product for the unrelated case.

**How the skill should guide A/B selection (in priority order):**
1. **Best — B derived from A.** If the user has only A, offer to **produce B by editing A** (same subject, changed pose/expression/lighting/angle) using an image-edit skill (`qwen-image-edit`). This guarantees a coherent pair. Say: *"For a clean transition, B should be the same subject as A with one thing changed. Want me to generate B by editing A (e.g. 'turn head left', 'open eyes', 'sunset light')?"*
2. **Good — A and B are first/last frames of an intended shot** (same scene, two moments). Coherent by construction.
3. **Same image for A and B** → smooth **loop / 360° pulse** (borrowed from Higgsfield's loop trick). Offer this when the user wants a seamless looping clip.
4. **Risky — unrelated A and B.** Allow it, but **warn first** (PART 3.3 keyframe warning). Frame it honestly: *"These look like different subjects/scenes — expect a morph/dissolve, not a clean motion. Proceed as a morph, or should I make B a variant of A instead?"*

**Prompt for keyframe:** describe the *shared* subject + the *change* across the transition, not two separate scenes. e.g. `the same woman, head turning from profile to facing camera, soft window light, subtle smile forming`.

**Args mapping:**
```bash
uv run --with modal python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py \
  --mode keyframe --image "<absA>" --image "<absB>" \
  --prompt "<shared subject + the change>" --frames 97 --height 1280 --width 768
```
(Script enforces two `--image` args; errors if fewer.)

---

### 2.4 v2v — retake / restyle a time window (slowest mode)

**Conversational contract.** User has a **video** and wants to "change the look but keep the motion," "restyle this clip," "redo seconds 2–5," "retake." Maps to `--mode v2v --video … --start … --end …`.

**Two intents to disambiguate (restyle vs retake-window):**
- **Restyle (whole clip, keep motion):** keep the original motion/composition, change the aesthetic (e.g. "make it anime," "make it night"). Express as a **style-only prompt** over the **full duration** (`--start 0 --end <clip length>`). The prompt should name the *new look* and assert *keep the motion/subject*.
- **Retake-window (fix a segment):** regenerate only a slice of the timeline (e.g. a botched 2–5s stretch) while leaving the rest. Use a **tight `--start`/`--end`** around the bad window. The prompt describes what should happen *in that window*.

**How to choose the window.** Ask the user the segment in seconds, or default to the script's `--start 2 --end 5`. For full-clip restyle, set the window to the entire clip. Keep windows as short as the fix allows — it directly drives cost (below).

**Strength / adherence.** The v2v path (`smoke_retake`) is the denoise-window retake; there is **no exposed strength slider** on v2v in the current script (unlike `control`, which has `--control-strength`). So adherence is governed by **window size + prompt specificity**, not a numeric knob. Set expectations accordingly: a wider window changes more; a narrower window changes less. (If a strength lever is wanted later, it would be added to `smoke_retake`; today, don't promise it.)

**Realistic expectations — SET THESE OUT LOUD.** v2v is the **slowest mode by far: ~470s for a 10s window** (plus cold start on first call). Always warn before running: *"Heads up — v2v is the slow one, roughly 8 minutes for a 10s window. Want a shorter window or a quick low-res test first?"* Offer a short-window or low-res smoke before a full v2v.

**Args mapping:**
```bash
# restyle whole clip (keep motion)
uv run --with modal python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py \
  --mode v2v --video "<abs.mp4>" --start 0 --end <clip_len> \
  --prompt "<NEW look>, keep the original motion and composition"

# retake a bad window
... --mode v2v --video "<abs.mp4>" --start 2 --end 5 --prompt "<what should happen in 2–5s>"
```
(`--steps` is the only other v2v lever exposed.)

---

### 2.5 control — IC-LoRA structural control (union: canny / depth / pose)

**Conversational contract.** User wants the **structure/motion of a source** to drive a **new render** — "use the motion of this video," "match this pose/depth/edges," "same camera move, different content." Maps to `--mode control` (union IC-LoRA: Canny+Depth+Pose; the backend also has `motion_track` for trajectory control).

**control vs i2v — when to use which:**
- **i2v** = "animate *this image*." One still, model invents the motion. Use for "bring a photo to life."
- **control** = "follow *this structure/motion*." A control video dictates edges/depth/pose per frame; the prompt + optional init image dictate appearance. Use when the user **already has the motion they want** (a reference clip) and wants to re-skin or re-content it while preserving structure. More setup, more control, tighter motion fidelity.

**Control-render acquisition — explain the three cases clearly:**
1. **Canny (auto, built-in).** Edges are derivable from any source video **with no extra model** — the script renders them via `ffmpeg edgedetect`. User just supplies `--video <src>`; the skill derives the canny control automatically. *Easiest path; offer this first.*
2. **Depth (needs a pre-rendered control video).** Depth maps require a depth estimator (e.g. Depth-Anything / MiDaS) run beforehand. The skill **cannot** auto-derive depth — ask the user for a pre-rendered **depth-map video** and pass it via `--control-video`.
3. **Pose (needs a pre-rendered control video).** OpenPose/DWPose skeleton video must be rendered beforehand; pass via `--control-video`.

**What to ask the user (control):**
- "Do you want to follow **edges (canny)**, **depth**, or **pose**?"
- If **canny** → "Give me the source video; I'll derive the edge map automatically."
- If **depth/pose** → "I can't generate the depth/pose map here — please provide a pre-rendered <depth-map / OpenPose> video (same length/res). Want pointers on how to render it?" (Depth-Anything for depth; OpenPose/DWPose for pose.)
- Optional **init image** (`--image`) seeds the first-frame appearance (IC-LoRA supports an optional init frame).
- **Adherence:** `--control-strength` (default 1.0). Lower (~0.6–0.8) loosens adherence for more prompt freedom; 1.0 sticks tightly to the control structure.

**Args mapping:**
```bash
# canny auto-derived from a source clip
uv run --with modal python3 ${CLAUDE_SKILL_DIR}/scripts/submit_video.py \
  --mode control --video "<src.mp4>" --control-type canny \
  --prompt "<new appearance>" --frames 97 --height 1280 --width 768

# user-supplied depth or pose control video (+ optional init image)
... --mode control --control-video "<depth_or_pose.mp4>" --image "<init.jpg>" \
    --control-strength 0.9 --prompt "<new appearance>"
```
(Script: `--control` selects `union` vs `motion_track`; `--control-type` only auto-derives `canny`; depth/pose must arrive via `--control-video`.)

---

## PART 3 — Skill UX deliverables

### 3.1 Decision tree — "user has X, wants Y → mode Z"

```
What did the user provide?
│
├─ Nothing but a text description ─────────────────────────────► t2v
│       (subject→action→camera→lighting→style; 9:16)
│
├─ ONE image
│     └─ What do they want?
│          ├─ "animate / bring to life / make it move" ────────► i2v   (DEFAULT)
│          ├─ "transition to another look/pose" ──────────────► keyframe
│          │        → offer to generate B by editing A (qwen-image-edit)
│          ├─ "loop it / seamless" ───────────────────────────► keyframe (A==B loop)
│          └─ "follow this image's structure for a render" ────► control (init image)
│          └─ AMBIGUOUS → AskUserQuestion 3.3-A (animate vs keyframe-A vs control)
│
├─ TWO images ────────────────────────────────────────────────► keyframe (A→B)
│       → coherence check: same subject? yes=clean / no=warn morph (3.3-B)
│
├─ ONE video
│     └─ What do they want?
│          ├─ "change the look, keep the motion" ─────────────► v2v restyle (window = full clip)
│          ├─ "redo seconds X–Y" ─────────────────────────────► v2v retake-window
│          ├─ "use this motion/edges/pose for a NEW render" ──► control
│          │        ├─ edges → canny (auto via ffmpeg)
│          │        └─ depth/pose → ask for pre-rendered control video
│          └─ AMBIGUOUS → AskUserQuestion 3.3-C (restyle vs retake vs control)
│
└─ A control-render video already (depth/pose/canny map) ──────► control (--control-video)
```

Speed tiebreaker to mention when the user is undecided: **control (4s) ≈ fastest structural path; i2v fast; keyframe ~ i2v; v2v slowest by 5–10×.**

### 3.2 Named-preset table (preset → expanded prompt + recommended frames/res/mode)

Default res for all: **768×1280 (9:16)**. `{img}` = the skill's auto-description of the source image (i2v); for t2v, replace `{img}` with the user's subject.

| Preset | Mode | Frames (≈dur) | Expanded prompt fragment (appended after `{img}`) |
|---|---|---|---|
| **subtle idle** *(i2v default)* | i2v | 97 (4s) | `subtle natural motion — slight breathing, micro head movement, blinking, gentle ambient sway; locked-off camera; photorealistic` |
| **slow push-in** | i2v / t2v | 97 (4s) | `slow steady dolly push-in toward the subject, gradually tightening framing; cinematic, shallow depth of field` |
| **slow pull-out** | i2v / t2v | 97 (4s) | `slow dolly pull-out revealing more of the environment; cinematic, stable` |
| **hair / wind** | i2v | 97 (4s) | `a steady breeze moves the hair and fabric, natural flowing motion; subject otherwise still; soft directional light` |
| **turn to camera** | i2v | 97 (4s) | `the subject slowly turns their head to face the camera, eyes meeting lens, subtle expression shift` |
| **orbit (arc)** | i2v / t2v | 121 (5s) | `smooth arcing orbit around the subject, parallax revealing form and depth; steady cinematic motion` |
| **parallax / drift** | i2v | 97 (4s) | `slow lateral camera drift creating parallax between foreground and background; subtle, dreamlike` |
| **rack focus** | i2v | 97 (4s) | `focus racks from foreground to subject, cinematic shallow depth of field, otherwise still` |
| **ambient life** | i2v | 121 (5s) | `the scene comes alive — {water/fire/crowd/foliage} moves naturally, atmospheric particles drift; subject anchored` |
| **handheld energy** | i2v / t2v | 97 (4s) | `subtle handheld camera movement, organic micro-shake, documentary energy; natural motion` |
| **crash zoom in** | t2v / i2v | 49 (2s) | `rapid crash zoom in toward the subject, sudden punch-in, energetic` |
| **360 loop** | keyframe (A==B) | 97 (4s) | `seamless looping motion, smooth continuous camera arc returning to start; perfect loop` |
| **establishing** | t2v | 217 (9s) | `wide establishing shot, slow majestic camera move across the landscape, epic scale, cinematic lighting` |

Notes:
- Frame counts are all **8k+1** (valid for the backend). Bump to 217 (≈9s) / 241 (≈10s) only when the user wants a long clip — cost scales with frames.
- For t2v, prepend the **subject→action** scaffold; for i2v, the preset is appended after the auto image-description so identity is preserved.

### 3.3 Exact AskUserQuestion prompts the SKILL.md should use

**3.3-A — One image dropped, intent unclear** (the most common fork):
- `header`: `What to do with this image?`
- `question`: `You gave me one image. How should I use it?`
- options:
  - **Animate it (i2v)** — *"Bring the photo to life with motion. Fastest, keeps the subject. (~45s for 4s, warm.)"*
  - **Use as start frame (keyframe A)** — *"Transition from this image to a second one. I can generate frame B by editing this image so the result stays coherent."*
  - **Use as structural control** — *"Drive a new render from this image's structure (pose/edges). More setup; tightest control."*
  - **Cancel**

**3.3-B — Two images that look unrelated** (coherence warning):
- `header`: `A and B look different`
- `question`: `These two images look like different subjects/scenes. Interpolating unrelated images gives a morph/dissolve, not a clean motion. How do you want to proceed?`
- options:
  - **Make B a variant of A (recommended)** — *"I'll regenerate B as the same subject as A with one change, for a clean transition."*
  - **Proceed as a morph** — *"Render the dissolve between them as-is."*
  - **Switch to i2v on A** — *"Just animate the first image instead."*
  - **Cancel**

**3.3-C — One video dropped, intent unclear:**
- `header`: `What to do with this video?`
- `question`: `You gave me a video. What's the goal?`
- options:
  - **Restyle, keep the motion (v2v)** — *"Change the look across the whole clip, preserve motion. Slowest mode (~8 min for 10s)."*
  - **Retake a time window (v2v)** — *"Regenerate just seconds X–Y. Shorter window = faster/cheaper."*
  - **Use its motion/structure for a NEW render (control)** — *"Follow edges/depth/pose. Edges (canny) are automatic; depth/pose need a pre-rendered control video."*
  - **Cancel**

**3.3-D — Control sub-type** (after user picks "control"):
- `header`: `Which control signal?`
- `question`: `Structural control follows one signal per frame. Which?`
- options:
  - **Edges (canny) — automatic** — *"I'll derive the edge map from your source video. No extra files."*
  - **Depth — needs a control video** — *"Give me a pre-rendered depth-map video (Depth-Anything/MiDaS)."*
  - **Pose — needs a control video** — *"Give me a pre-rendered OpenPose/DWPose skeleton video."*
  - **Cancel**

**3.3-E — Cost/quality gate before any full run** (keep the existing one, per-mode latency injected):
- `header`: `LTX-2.3`
- `question`: `Generate <mode> from <name>? <per-mode latency line from 3.4>. First run cold-starts ~90–200s. A few cents of GPU.`
- options: **Quick smoke (cheap)** · **Full quality (9:16)** · **Cancel**

### 3.4 Cost / latency expectations to surface, per mode

Backend is **24 fps**; warm = container already running; cold start (first call after idle) adds **~90–200s** on top. State the warm number, then mention cold start once.

| Mode | Warm latency | Notes to say out loud |
|---|---|---|
| **i2v** | **4s clip (97f) ≈ 45s** · **10s clip (241f) ≈ 100s** | Fastest image path. Default to 4s; offer 10s if they want length. |
| **t2v** | ≈ i2v at same frame count (~45s @ 97f) | No image to load; comparable to i2v. |
| **keyframe** | ≈ i2v at same frame count (~45s @ 97f) | Two-image interpolation; similar cost to i2v. |
| **v2v** | **10s window ≈ 470s (~8 min)** | **Slowest by far (5–10×).** Always warn; push short window / low-res smoke first. |
| **control** | **4s clip (97f) ≈ 28s** | Fastest structural path (distilled base, few steps). Canny adds a quick local ffmpeg pass. |
| **cold start** | **+90–200s on first call** | One-time per idle container. Say "waiting for container cold start (~90s)…" so it doesn't look hung. Use `--timeout 300` on the Bash call. |

**Cheap smoke (any image mode):** `--frames 17 --height 320 --width 512 --steps 8` → confirms the container is warm and the prompt direction is right before spending on a full render.

---

## Summary for SKILL.md authoring

- **Default mode = i2v, default render = 9:16 768×1280, default motion = "subtle idle."**
- **i2v technique: describe the image first (skill-authored), then append a named motion preset.** Never make the user describe their own photo.
- **keyframe is coherence-gated:** same-subject A/B = clean; unrelated = morph → warn (3.3-B) and offer to derive B by editing A.
- **v2v is the slow one (~470s/10s):** always set expectations; prefer a tight window.
- **control: canny is auto (ffmpeg), depth/pose need a pre-rendered control video** — ask for it explicitly (3.3-D).
- **Always gate a full run behind AskUserQuestion (3.3-E)** with the per-mode latency line; offer the cheap smoke.
- **Never recommend a SaaS / Higgsfield / paid per-clip API; never route through fal-mcp.** This is the user's own self-hosted LTX-2.3.
