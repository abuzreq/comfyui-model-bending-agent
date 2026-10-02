# ComfyUI-Model-Bending and ComfyUI-Agent-Bridge: node and API reference for agents

This covers the agent-ready ComfyUI-Model-Bending (`ApplyBendsFromJSON` has a `report` output), ComfyUI 0.18+ and
frontend 1.40+. Check a node's exact inputs with `comfy_canvas.py node CLASS` (tool: `node_info`), and the bridge
with `comfy_canvas.py bridge`. For older installs, see "Older Model-Bending installs" at the end of SKILL.md.

## Contents
- `ApplyBendsFromJSON`: the bends JSON, per-bend keys, placeholders, clamping, the report
- Discrete nodes: Model Bending, DiT Block Bending, Timestep Gated Bending, feature maps, probes, steering
- Video nodes (Model-Bending 0.3+, experimental): attention-map bending and capture, temporal ops
- Behaviour to rely on
- Interactive Bending WebUI
- ComfyUI-Agent-Bridge: nodes, routes, events
- comfy-mcp (optional)

## `ApplyBendsFromJSON` ("Apply Bends from JSON")

Inputs:
- `model`
- `bends_json`
- *(optional)* `strict` (BOOLEAN)
- *(optional)* `a`, `b`, `c`, `d`: FLOAT sliders, substituted for `{{a}}`…`{{d}}`
- *(optional)* `clamp`: none | hard | safe
- *(optional)* `safe_ranges`: JSON

Outputs: `MODEL`, `report` (STRING, JSON), `resolved_json` (STRING).

```json
{"bends": [
  {"label": "recompose", "path": "middle_block.1", "module_type": "rotate",
   "module_args": {"angle_degrees": {{a}}}, "t": [1.0, 0.7]},
  {"path": "output_blocks.*.1", "module_type": "multiply", "module_args": {"scalar": 1.4}, "steps": "3-",
   "blend": 0.5, "guard": {"nan": "zero", "max_std_ratio": 8}},
  {"path": "input_blocks.[4-8].0", "module_type": "subset", "module_args": {"percentage": 0.3, "dimension": "channel"},
   "inner": {"module_type": "add_noise", "module_args": {"noise_std": 0.8, "seed": 7}}}]}
```

**Per-bend keys** (JSON version 1.1; v1 JSON still works):

| key | meaning |
|---|---|
| `t: [hi, lo]` | Diffusion-time window: 1 = noise, 0 = image. It follows the noise level, not the step count, so it is **correct for shifted Flux/SD3 schedules and at any denoise**. |
| `steps: "0-3,7"` | A window in executed step indices. The old node-level `steps_min`/`steps_max` still work, and a single bound is honoured. |
| `blend: 0..1` | Mixes the unbent and bent activation. This is what makes smooth sliders possible. |
| `guard` | `nan`: zero, clamp or none (NaN→0 is on by default everywhere). `max_std_ratio`: caps how much the bend may raise the activation's std. `preserve_norm`: keeps per-channel energy. |
| `label` | Echoed in `report`. Use it for provenance, e.g. a candidate letter or a hypothesis name. |
| wildcards | Per path segment: `output_blocks.*.1`, `input_blocks.[4-8].0`. `report.patterns` lists what each one matched. |

**Ops and their arguments:**

| op | arguments |
|---|---|
| `multiply` | `scalar` |
| `add_scalar` | `scalar` |
| `add_noise` | `noise_std`, `seed` |
| `threshold` | `threshold` |
| `rotate` | `angle_degrees` |
| `scale` | `scale_factor` |
| `erosion`, `dilation`, `gradient` | `kernel_size` |
| `sobel` | `normalized` |
| `fourier` | `cutoff_freq` 0–10, `amp_factor` −10–10. Per sample and per channel. |
| `subset` | `percentage`, `dim` (batch, channel or spatial), `seed`, plus an `inner` op. Needs 4-D input. |
| `translate` (0.3+) | `dx`, `dy` (fractions of width / height, ±1), `padding` border / zeros / reflection |
| `flip` (0.3+) | `direction`: horizontal, vertical or both |
| `blur` (0.3+) | `sigma` (0–20, in latent cells / tokens) |
| `sharpen` (0.3+) | `amount` (±10), `sigma`. Unsharp mask: `x + amount·(x − blur(x))` |
| `temporal_shift` (0.3+, video) | `frames` (±64 latent frames), `padding` border / wrap / zeros |
| `temporal_blur` (0.3+, video) | `sigma` (0–16 latent frames) |
| `frame_reverse` (0.3+, video) | none |
| `frame_ramp` (0.3+, video) | `w_start`, `w_end` (±4), `curve` linear / ease_in / ease_out / smooth / triangle, plus an `inner` op whose strength changes over the frames |

`rotate` and `scale` also take `padding` (0.3+). Ops act on each frame of a video activation, except the temporal
ones, which act across frames.

**`attention_bends`** (0.3+, video, experimental): a second top-level list next to `bends`, for the attention maps of
WAN video models. Each item: an op (`module_type`, `module_args`, `inner`) plus `attention` (cross_text | cross_image
| self_query | self_key), `blocks`, `tokens` (all | prompt | indices), `renormalize` (keys | per_token_mass | none),
`apply_to` (both | cond | uncond), `heads`, `frames`, `steps`, `t`, `blend`, `label`. `bends` may be empty. The report
and `resolved_json` list each one under `attention_bends`. Details and heuristics: `references/video.md`.

**`clamp: safe`** takes `safe_ranges` in the form `{op: {arg: [lo, hi]}, "paths": {"<glob>": {op: {arg: [lo, hi]}}}}`.
Use the bundled tables `data/safe_ranges_sd15.json` / `data/safe_ranges_sdxl.json`, or the user's own
measurements in the same format.
Use it for unattended runs (mode A).

**Read the `report` before judging an image.** It has these keys:
- `resolved`: every layer, its op, and its `steps`, `t` and `blend`
- `expanded`: container → last child
- `skipped`: with a reason
- `clamped`: requested → applied
- `patterns`
- `warnings`: misspelled keys or arguments

Two ways to get it back:
- Wire `report` into a `PreviewAny` node; the text lands in `/history` and shows in `last-run`.
- Wire it into an `AgentReportSink` (bridge); it arrives as an `events` entry.

This replaces the old pixel no-op check (`metrics.py` `noop`), though that check is still a useful cross-check.

## Discrete nodes (category `model_bending`)

**Module factories** output `BENDING_MODULE`:
- `Rotate Module (Bending)`, `Scale Module (Bending)`, `Multiply Scalar Module (Bending)`,
  `Add Scalar Module (Bending)`, `Add Noise Module (Bending)`, `Threshold Module (Bending)`
- `Fourier Amplify Module (Bending)`
- `Erosion`, `Dilation`, `Gradient`, `Sobel`
- `Apply To Subset (Bending)`
- `Latent Operation To Module`
- **`Timestep Gated Bending`** (`bending_module`, `t_start`, `t_end`, `ramp`: hard | linear | cosine, `ramp_width`).
  It gates any module by t, and works with every applier. A ramp fades across the edge of the window.

**Appliers** output MODEL:
- **`Model Bending`**: model, bending_module, path (comma list), `steps_to_bend_str`, `max_denoising_steps`,
  and *optional* `t_start`, `t_end`, `strict`.
- `Model Bending (SD Blocks)`: block_type, block_index. It bends the block's last child.
- `Model Bending (SD Layers)`.
- **`DiT Block Bending`**: model, bending_module, `blocks` (`double:0-6, single:25-37`, `joint:0-11`, and for other
  DiTs their list name), `stream` img | txt | both, `spatial` (maps image tokens onto the real latent grid, so
  rotate and scale are spatial), `t_start`, `t_end`, `strict`. Outputs MODEL and `report`. It works with Flux, SD3,
  WAN, Qwen-Image, LTX and HunyuanVideo.
  - A block patch set earlier in the graph is kept: the node calls it rather than overwriting it.
  - Skip Layer Guidance replaces these block patches during its extra guidance pass, so the bend is absent from
    that pass only.
- `Model VAE Bending`, `LoRA Bending`, `LoRA Bending (list)`.

**Inspection and steering (L2/L3):**
- **`Bendable Layer Catalogue`** (`model`, `filter`, `max_depth`) → JSON:
  `{model, block_lists: {name: {count, dit_block_bending}}, layers: [{path, class, hookable, note}]}`.
  Run it once per new architecture instead of guessing paths.
- **`Visualize Feature Map`**: `timestep` is the **1-based step** (0 = last).
  - It sees upstream bends.
  - Maps are min-max normalised, 3-channel, and nearest-upscaled ×8.
  - Token layers are laid out on the image grid.
  - Output 1 (`Feature Map`) is the map. Output 0 is the raw latent.
- **`ActivationProbe`** (`model`, `sites` = auto | globs, `store`) → `MODEL`, `PROBE`.
  - Observe-only: images are bit-identical with and without it.
  - It records **after** all bends, per site and per step, on the cond half of the batch.
  - Use one probe per sampler.
- **`ReadActivationProbe`** (`probe`, `latent`, which forces ordering; `stat` std | rms | mean | absmax | dead_frac;
  `alert_ratio`; *reference* ACTIVATIONS) → `activations`, `heatmap` (IMAGE, sites × steps; log2 ratio when a
  reference is given), `report` (JSON: `alerts` of kind nonfinite | dead_channels | blowup | collapse, and
  `per_site` first/last/max plus ratio min/max).
  - Probe an unbent run first and feed its `activations` in as `reference`. That gives the causal footprint of a
    bend in activation space.
- **`SteeringVectorFromActivations`** (`activations_a`, `activations_b`, `site`, `mode` channel | spatial) and
  **`ApplySteeringVector`** (`strength`, `t_start`, `t_end`, `apply_to` both | cond, `normalize`).
  - This is a difference-of-means steering vector between two prompt runs, added at a site.
  - It is the real "activation steering" of Level 2.
  - `site` must be an exact probed path; the error lists the valid ones.
- `Compute PCA` → `HSpace Bending` (UNet h-space directions, step-safe).
- `LatentApplyOperationCFGToStep`, `Latent Operation (…)`, `ConditioningApplyOperation`, `NoiseVariations`.
- `Model Inspector`, `Model VAE Inspector`.

## Video nodes (category `model_bending/video (experimental)`, Model-Bending 0.3+)

Validated upstream on tiny random-weight WAN models only. `video_ready` in the status says they are installed.

- **`Attention Map Bending`** (`model`, `bending_module`, `attention`, `blocks`, `tokens`, `renormalize`,
  `apply_to`; optional `steps`, `t_start`, `t_end`, `heads`, `frames`, `strength` −4–4, `clip` + `prompt` to find
  prompt **words** in `tokens`, `strict`) → `MODEL`, `report`. The node form of an `attention_bends` item; use it when
  the bend should follow particular words.
- **`Attention Map Capture`** (`model`, `attention` cross_text | cross_image, `blocks` default 13-18, `tokens` words,
  indices or `prompt` (at most 32); optional `steps`, `heads`, `clip`, `prompt`) → `MODEL`, `ATTENTION_MAPS`, `report`.
  Records after any upstream attention bends.
- **`Read Attention Maps`** (`attention_maps`, `latent` = that sampler's output, `token` sum | index, `block` mean |
  index, `step` all (side by side) | mean | index, `normalize` per_video | per_frame, `colormap`,
  `match_video_frames`) → `frames` (IMAGE batch: save it with `SaveAnimatedWEBP`), `report`.
- Module factories: `Translate Module`, `Flip Module`, `Gaussian Blur Module`, `Sharpen Module`,
  `Temporal Shift Module`, `Temporal Blur Module`, `Frame Reverse Module` (all `… (Bending)`), and
  `Frame Ramp (Bending)` (`bending_module`, `w_start`, `w_end`, `curve`). They work with every applier.
- `DiT Block Bending` and `Model Bending` work on WAN too: `blocks: "double:13-18"` or paths `blocks.15`,
  `blocks.15.ffn`; outputs are laid out on the frames × height × width grid, so spatial ops act per frame.

## Behaviour to rely on
1. **Paths.**
   - Containers (`middle_block`, `output_blocks.4`) are bent at their last child and logged as
     `[model-bending] expanded container …`.
   - `ModuleList` paths are skipped with a hint.
   - Typos are skipped with a close-match hint.
   - `strict` turns all of these into errors. Use `strict` in mode A.
2. **Tuple outputs** (Flux `double_blocks.N`, SD3 `joint_blocks.N` via hooks) warn once and bend `output[0]`. Prefer
   `DiT Block Bending`, which picks the stream explicitly.
3. **3-D token outputs** through the generic hook are still treated as `(1, B, L, C)` by spatial ops. Use
   `DiT Block Bending spatial=true` for real spatial ops on DiT image tokens.
4. **Windows fail closed.** If t cannot be resolved, a t-windowed bend is skipped for that pass and a warning is
   logged. It never falls back to every step.
5. **No leak.** Unbent branches stay unbent, and each applier has its own step state, so a shared module node is
   fine.
6. **Logs.** Every message carries `[model-bending]`. Run `comfy_canvas.py logs --grep model-bending` after a
   surprising result.

## Interactive Bending WebUI (Model-Bending's own UI)
Node `InteractiveBendingWebUI` serves `/web_bend_demo/`. Its routes:
- `GET /web_bend_demo/layers?session_id=S`: the module tree and model_type.
- `GET` / `POST /web_bend_demo/selection`: the live selection. It is agent-writable.

`selected_part` is honoured, so bends on Flux `single_blocks` copied from the UI target what the UI shows.

## ComfyUI-Agent-Bridge (optional pack; check with `comfy_canvas.py bridge`)

| node | inputs → outputs | purpose |
|---|---|---|
| `AgentVariantSwitch` | `candidates` (autogrow, any type, lazy), `selected` (1-based), `labels` (one per line) → `selected`, `index`, `label` | Radio-button N-way pick. Only the chosen branch runs. The pick lands in `/history` and in the `run` events. |
| `AgentFeedback` | `images`, `labels`, `session_id`, `round`, `rating` 0–5, `push` −1..1, `direction`, plus votes and chips (managed by its UI) → `feedback_json`, `chosen_index` | Thumbnails with 👍/👎, keep/change chips (composition, subject, palette, lighting, texture, style), and a **Send to agent** button that needs no run. |
| `AgentReportSink` | `reports` (autogrow, any type), `channel`, `session_id` | Forwards `report` / `resolved_json` / probe reports / the catalogue to the agent. JSON strings are parsed. |

**Routes** live under `/api/agent_bridge/`. Agent routes need the `X-Agent-Bridge-Token` header;
`comfy_canvas.py` reads the token file for you.

| route | what it does |
|---|---|
| `propose` | A confirm dialog in every open tab. The graph opens as `workflows/agent_bridge/<name>.json`, so Ctrl+S writes back to it. |
| `events?since=&session=&kinds=&wait=` | A long-poll over proposal, proposal_status, feedback, report and run events. `run` events are summarised from `/history` and include `switch_picks` and `feedback`. |
| `canvas` | The user's active graph, available only while they turn on Settings → Agent Bridge → Share canvas. |
| `notify` | A toast in the open tabs. |

## comfy-mcp (optional; Comfy-Org, `uv tool install comfy-mcp`)

The skill does not need it. If the user has it registered, call its tools by the server name they registered it
under. Useful extras:

| need | tool |
|---|---|
| server and install info | `server_info`, `which` |
| pre-flight a workflow file | `validate_workflow(workflow_path)`. Read `.valid`. |
| stage an input image for img2img | `upload_file(paths)` (the skill's own `upload_image` / `comfy_canvas.py upload` does this too) |

It is stdio-only and file-based, so it cannot touch the user's open canvas. Its `get_logs` only covers a ComfyUI
that comfy-cli launched; otherwise use `comfy_canvas.py logs`.
