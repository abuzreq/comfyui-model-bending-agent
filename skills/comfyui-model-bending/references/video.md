# Bending video models (WAN 2.1 / 2.2), experimental

ComfyUI-Model-Bending 0.3 and newer can bend video diffusion transformers of the WAN family: text-to-video and
image-to-video. Upstream marks all of it **experimental**: it was tested on tiny random-weight WAN models, not yet on
real weights. Say so once, in plain words, the first time a session bends a video ("bending video models is new, so
some results may surprise us both"), and treat every heuristic below as a starting point to verify with a sweep.

Contents:
- What is there to bend
- Getting started: presets, models, cost
- Attention bends in the bends JSON
- Ops for video, and bends that change over the clip
- Where and when to bend (heuristics)
- Measuring and showing videos
- Starting a video from a picture
- Limits

## What is there to bend

| route | node / JSON | what it changes |
|---|---|---|
| cross-attention maps (video ↔ prompt) | `attention_bends` item `"attention": "cross_text"`, or **Attention Map Bending** | where each word lands in the frame, frame by frame: layout, which object goes where |
| cross-attention maps (video ↔ starting picture), WAN 2.1 I2V | `"attention": "cross_image"` | how the video reads the starting picture |
| self-attention, where results land | `"attention": "self_query"` | how parts of the frame relate: moves or mirrors content |
| self-attention, where positions read from | `"attention": "self_key"` | drags content around while the layout is kept |
| block outputs on the video grid | `bends` with `path: "blocks.N"` / `"blocks.N.ffn"`, or **DiT Block Bending** `double:13-18` | like image DiTs; outputs are laid out on frames × height × width, so spatial ops act on the picture of each frame |
| over time | `temporal_shift`, `temporal_blur`, `frame_reverse`, `frame_ramp` | motion and timing: lag, smear, reverse, a bend that grows over the clip |
| inspection (L3) | **Attention Map Capture** → **Read Attention Maps** | where chosen words are attended to, for every step, rendered as a video (correlational) |

WAN 2.1 1.3B has **30 blocks** (12 heads), WAN 2.1 / 2.2 14B has **40** (40 heads). One latent frame is 4 video
frames (after the first), so `frames: "0-5"` targets roughly the first 21 video frames.

## Getting started: presets, models, cost

| preset | model files (ComfyUI `models/` folders) | notes |
|---|---|---|
| `video_sweep_wan21_t2v` | `diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors`, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `vae/wan_2.1_vae.safetensors` | baseline + 3 attention bends (rotate early, sharpen everywhere, a rotation that grows over the clip). The light one: ~8 GB VRAM comfortable; 6 GB works with offloading, slowly |
| `video_sweep_wan21_i2v` | `diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors`, the same text encoder and VAE, `clip_vision/clip_vision_h.safetensors` | starts from a picture (`start_image` node); bends how the video reads the picture vs the prompt. 14B: 16 GB+ VRAM, or a GGUF quant |

Both render 640×368, 25 frames (≈1.5 s at 16 fps), 10 steps, and save with `SaveAnimatedWEBP`, which every script
reads without ffmpeg. Each candidate's `ApplyBendsFromJSON` has `strict: true` and its `report` wired to a
`PreviewAny`, so `run_workflow` / `queue --wait` returns what was bent.

- Check the files first (`list_models("diffusion_models")`, `"text_encoders"`, `"vae"`, `"clip_vision"`). If they are
  missing, say what they are for and where they go, and ask before suggesting a download. Never download them
  yourself.
- A video render costs many image renders: on a mid-range GPU the 1.3B preset takes minutes per candidate. State the
  cost before a sweep, keep sweeps to baseline + 2–3, and keep the clip short until a direction is chosen. Then
  re-render the pick longer (`length` 49 or 81) or larger.
- Free VRAM before and after switching between image and video models (`free_memory`).
- WAN 2.2 14B is two models (a high-noise expert for layout and motion, a low-noise one for detail) chained by two
  `KSamplerAdvanced`. Bend each model separately; the high-noise one for layout. Model-Bending's
  `workflows/video/wan22_t2v_14b_attention_bending.json` shows the wiring. WAN 2.2 5B (TI2V) is untested.

## Attention bends in the bends JSON

`Apply Bends from JSON` takes an `attention_bends` list next to `bends` (either may be empty), so the skill's usual
path (one JSON per candidate, `strict`, the report) works for video too:

```json
{"version": 1.1, "bends": [],
 "attention_bends": [
  {"label": "A", "module_type": "rotate", "module_args": {"angle_degrees": 12},
   "attention": "cross_text", "blocks": "13-18", "tokens": "all", "renormalize": "keys", "steps": "0-2"}]}
```

| key | default | meaning |
|---|---|---|
| `module_type`, `module_args`, `inner` | | any op (below), `frame_ramp` / `subset` wrap an `inner` op |
| `attention` | `cross_text` | `cross_text`, `cross_image` (WAN 2.1 I2V), `self_query`, `self_key` |
| `blocks` | `*` | block indices, e.g. `13-18` |
| `tokens` | `all` | cross attention only: `all`, `prompt` (without padding) or indices `0-3, 7`. Prompt **words** (`"horse"`) need the text encoder, so they work only in the **Attention Map Bending** node (its `clip` and `prompt` inputs), not in the JSON |
| `renormalize` | `keys` | `keys` (each position's attention sums to 1 again), `per_token_mass`, `none`. **`multiply` (amplify) needs `none`**: with `keys` it is undone, and the report warns |
| `apply_to` | `both` | CFG passes: `both`, `cond` (stronger, glitchier: amplified by the guidance scale), `uncond` |
| `heads`, `frames`, `steps` | `*` | index lists; `frames` are latent frames |
| `t`, `blend`, `label` | | as for layer bends |

The report lists each attention bend under `attention_bends` with its blocks, steps and arguments made explicit, and
marks it `experimental`. Read it like any report: an empty `blocks` list is a spec bug.

## Ops for video, and bends that change over the clip

Every op acts on each frame separately, except the temporal ones:

| op | arguments | on video |
|---|---|---|
| `translate` | `dx`, `dy` (fractions, ±1), `padding` border / zeros / reflection | slides content; with `self_key` it drags things across the frame |
| `flip` | `direction` horizontal / vertical / both | mirrors (strong; try `self_query` on few steps) |
| `blur` | `sigma` (cells / tokens) | on attention: softer, less texture, composition kept |
| `sharpen` | `amount`, `sigma` | on attention: more defined content, richer surface texture |
| `rotate`, `scale` | as for images, plus `padding` | keep them small on video: angles ~10–15°, scales ~1.04 |
| `temporal_shift` | `frames` (latent frames, ±), `padding` border / wrap / zeros | lag or lead: parts of the clip run ahead of others |
| `temporal_blur` | `sigma` (latent frames) | smears motion over time |
| `frame_reverse` | — | runs that layer's activations backwards in time |
| `frame_ramp` | `w_start`, `w_end` (±4), `curve` linear / ease_in / ease_out / smooth / triangle, `inner` | **the inner op's strength changes over the frames**: a bend that grows, fades, or peaks mid-clip |

`frame_ramp` is the video counterpart of a bend animation: one render instead of one render per frame
(`animate.py` refuses video workflows and says so). `triangle` peaks in the middle and returns, which loops.

## Where and when to bend (heuristics)

Upstream's tips, to verify with sweeps:
- **Blocks**: the middle blocks change the most (13–18 of 30 in 1.3B, ~17–24 of 40 in 14B). Every block (`*`) is the
  strongest; early blocks act on layout, late ones on surface.
- **Steps**: the early steps change layout and coherence; late steps change little. WAN samples with shift 8, so
  **most steps sit in the structure window** (t ≥ 0.7): at 10 steps, steps 0–7; at 20, 0–15, and no step reaches the
  detail window. Use `steps` (e.g. `"0-2"` = the very start) or a narrow `t` like `[1.0, 0.9]` to mean "early";
  `timestep_windows(arch="wan", steps=…)` shows the mapping.
- **Tokens**: `all` is much stronger than a single word.
- **Strength**: keep it small. A 12° rotation or ×1.04 scale of the attention maps is already visible; amplify
  (multiply above 1) with `renormalize: none`.
- **I2V**: `cross_image` bends how the video reads the starting picture (identity, layout of the subject);
  `cross_text` bends the prompt's influence (the motion and what happens). Bend both, separately, to see which one
  carries the effect the artist wants.
- **Cost**: cross-attention bending builds the full attention map, so at high resolution bend a few blocks and steps.

A first round for an open "make this video strange" goal: baseline, rotate `cross_text` 12° on the middle blocks for
steps 0–2, sharpen `cross_text` on every block, and a `frame_ramp` rotation 0 → 30° (the `video_sweep_wan21_t2v`
preset). Then bisect on the axis the artist picks.

## Measuring and showing videos

- `compare_images` / `metrics.py compare` match frames by their position in the clip (8 evenly spaced) and add
  `motion` (mean change between consecutive frames), `motion_ratio` (candidate ÷ baseline) and three video flags:
  `frozen` (the bend stopped the motion), `flicker` (motion more than 2.5× the baseline: temporal blow-out) and
  `static` (the baseline barely moves: judge motion bends on another prompt). These thresholds are provisional.
  `mae_by_frame` shows whether a change grows over the clip (as a `frame_ramp` should).
- `view_images` / `metrics.py sheet` put each video on its own row as a filmstrip (`frames`, default 6). Look at it
  before judging; a still frame hides flicker, so read `motion_ratio` too.
- `diff_image` / `metrics.py diff` averages the change over matched frames.
- The in-chat board plays each version as a short silent loop (a small animated preview), with the starting picture
  (if any) as the original. Captions say what happens over time ("the horse's path bends left halfway through").
- `.mp4` / `.webm` outputs need ffmpeg to be read; `SaveAnimatedWEBP` outputs do not. Prefer `SaveAnimatedWEBP` in
  agent-built workflows, and add a `SaveVideo` (mp4) only for the final deliverable.
- `log_round` stores the first frame as the record's picture, and records each attention bend with the path `attention.<route>.blocks[…]` and the routes `txt2video` / `img2video`.

## Starting a video from a picture

Image-to-video needs an image-to-video model and workflow (`video_sweep_wan21_i2v`); a text-to-video workflow cannot
take a picture, and `build_workflow(start_image=…)` refuses one with that message. Get the picture into ComfyUI with
`upload_image` (a path on the user's computer, or a version from an earlier round), then
`build_workflow(spec, name, start_image=…)`: in the I2V preset this sets the `start_image` node. A strong pairing:
bend a still with an image model first, then hand the chosen version to the video model as its start.

## Limits

- Experimental upstream: no measured safe ranges exist for video, and `clamp: safe` has no table. Start small.
- Prompt words in `tokens` need the **Attention Map Bending** node (with `clip` and `prompt` wired); the JSON takes
  `all`, `prompt` or indices.
- Attention bends are supported for the WAN family only. Other video DiTs (LTX, HunyuanVideo) take block bends through
  **DiT Block Bending**, untested.
