# Learnings & gotchas — ltx2-clip-skill

Hard-won, measured findings from running LTX-2.3 (22B) self-hosted on one RTX PRO 6000 (Blackwell sm_120, 96 GB, bf16) on Modal. Read this before touching the persist/eviction/tiling paths.

## Memory & OOM

- **2 resident stage transformers FIT at 768×1280×97 on a CLEAN container** — measured peak **~75 GB** (23 GB free), zero OOM. The forward activation is only **~5 GB**, not the ~24 GB once assumed. Tile-size sweep: 768→74.94 GB, ≤512→73.5 GB (a ~1.4 GB lever).
- **The real OOM cause was memory ACCUMULATION over long-lived warm containers + cross-resolution stale residents**, not the inherent 2-resident footprint. A container that had built residents for one resolution (e.g. a 512 batch) then served a different resolution (1280) would stack weights and hit 94 GB.
  - Fix shipped: `_purge_stale_residents()` drops residents whose `(h,w)` ≠ the current request before any allocation (transformer weights are resolution-independent, ~35 GB each). Verified: 512→1280→youtube→1280 switches all pass.
  - `_max_residents_for(h,w,frames)` caps residents by a VRAM estimate. NOTE: it is currently **conservative** (forces 1 resident at high-res); the clean-container sweep shows 2 fit and are *faster* warm (~35 s vs ~43 s). Raising it to 2 is gated on a longevity/creep test (does peak drift back to OOM over many gens?).
- **`torch.cuda.empty_cache()` alone doesn't reclaim** — pipeline ref-cycles need `gc.collect()` *first*, then `empty_cache()` then `synchronize()`. `expandable_segments:True` is set and helps fragmentation but can't free genuinely-live blocks.
- A remote `torch.OutOfMemoryError` **fails to deserialize** in a local env without torch → surfaces as a generic client exception with no "out of memory" string. The smoke methods therefore catch and **return a diagnostic dict** (status/ is_oom/ free_vram_gb/ resident_count) instead of raising.
- **Stale warm containers serve OLD code** after a `modal deploy` until they drain. A new method kwarg → `TypeError: unexpected keyword argument` from the lingering container. Force a clean state with `modal app stop <app> --yes` then redeploy when testing signature changes.

## Precision / attention (all rejected — kept bf16)

- **fp8** (cast / scaled-mm): rejected — quality bar is bf16. (Studied LTX repos lean on fp8 to fit; we don't.)
- **SageAttention / SageAttention-3 (FP4)**: black frames / noise on this path.
- **flash-attn**: no sm_120 kernel → PyTorch SDPA (exact) is used.
- **max-autotune** (torch.compile): tested, slower here than default compile.

## VAE tiling

- `TilingConfig.default()` = spatial **768 px / 64 overlap**, temporal **80 frames / 24 overlap** (same as Lightricks + community repos). Env-tunable via `LTX_VAE_TILE_PX/OVERLAP/TEMPORAL_FRAMES/TEMPORAL_OVERLAP` + per-request `tile_px`/`temporal_frames`. Smaller tiles = smaller decode peak but only a ~1 GB lever here, and may add overlap-blend seams — fine-tune, not the OOM fix.

## Versions (verified 2026-06-08)

- HF weights `Lightricks/LTX-2.3`: `MODEL_REVISION=76730e6…` = **HF `main` HEAD (latest)**.
- LTX-2 git: pinned `1799988…` (vs latest `d605370…`). Pinned for a deliberate issue-#216 (multigpu import) workaround; `tiling.py`/TilingConfig is byte-identical to latest. Bumping = fresh image rebuild + full retest.
- Gemma `68f7ee4`, IC-LoRA union `b4d1c4d`, motion-track `572bb9c`, fp8 `1d756cd`.

## Behaviour notes

- **Resolution is W×H** in this repo: `768×1280` = vertical 9:16 reel (768 wide, 1280 tall). (Some earlier notes wrote it backwards.)
- **Canny control**: derive a DENSE edge map (`edgedetect=low=0.05:high=0.2`); sparse thresholds on a soft-lit face yield only a hair outline → the model drifts.
- **Keyframe coherence**: A and B must be the same subject/scene or you get a morph/identity-swap, not motion. Derive B by editing A, or use a clip's first/last frames.
- **v2v (retake)** is the slowest mode (~470 s for a 10 s window — single-stage full-CFG).
