# On-demand canvas UI: building, proposing and reading back

## Contents
- Transports (with and without ComfyUI-Agent-Bridge) and the tested presets
- The loop (with the bridge)
- Spec format
- Preset 1: Comparison Switchboard
- Preset 2: Bending Slider Control Board
- Preset 3: Feature Map Inspector
- Feedback channels
- Adapting the base block to other architectures

All of this goes through `scripts/comfy_canvas.py`. It uses only the standard library, and `COMFYUI_URL` sets the
server.

There are two transports. Check which one is available with `comfy_canvas.py bridge`:
- **With ComfyUI-Agent-Bridge** (recommended):
  - `propose` shows a consent dialog in the user's open tabs.
  - `events` long-polls for the user's decisions, feedback, reports and finished runs.
  - `canvas` reads the live graph when the user shares it.
  - `notify` sends a toast.
- **Without it** (plain ComfyUI):
  - `push` saves the graph into the Workflows sidebar.
  - `wait-run` / `last-run` read the user's runs from `/history`.
  - `pull --baseline` diffs what the user saved with Ctrl+S.

Every preset in `presets/` was built and run against ComfyUI 0.18.2, frontend 1.44, the agent-ready Model-Bending and the
SD1.5 LCM checkpoint. To use one: copy it, then change the loaders, prompt, paths and values.

| preset | needs |
|---|---|
| `switchboard_bridge_sd15.spec.json` | ComfyUI-Agent-Bridge |
| `switchboard_sd15.spec.json` | core only (switch chain + Note/Primitive feedback) |
| `slider_board_sd15.spec.json` | agent-ready Model-Bending (`{{a}}`…`{{d}}`, `clamp: safe`) |
| `feature_inspector_sd15.spec.json` | agent-ready Model-Bending (bent feature maps, activation probe) |
| `video_sweep_wan21_t2v.spec.json` | Model-Bending 0.3+, WAN 2.1 1.3B text-to-video files (experimental; see `references/video.md`) |
| `video_sweep_wan21_i2v.spec.json` | Model-Bending 0.3+, WAN 2.1 I2V 14B files; its `start_image` node takes the starting picture |

The two video sweeps were built against ComfyUI 0.18.2's node definitions and their bends checked on Model-Bending's
tiny test WAN; they have not been rendered with real WAN weights. Any preset starts from a picture with
`build --start-image NAME` (tool: `build_workflow(start_image=…)`), which swaps its empty latents for the picture.

## The loop (with the bridge)

```
write spec ─► build (UI + API json; refuses unknown classes and inputs)
     ─► queue --api --ui --wait      pre-run, so the user's Run is instant (cache) and you have the metrics
     ─► propose ui.json --name <session>_r<N>_<board> --session S --round N --message "<what to do>"
     ─► events --session S --since <last> --kinds proposal_status,feedback,run --wait 60   (repeat)
     ─► interpret: switch_picks, feedback (votes, chips, rating, push, direction), reports, errors
     ─► next round (or notify "working on round N+1…")
```

- **Proposals persist.** The graph is saved as `workflows/agent_bridge/<name>.json`, so the user's Ctrl+S writes
  back to it. A tab opened after the broadcast still gets the dialog, because pending proposals are fetched on load.
  If `delivered_to_clients` is 0, say "open ComfyUI" in chat.
- **A proposal can be dismissed.** Treat a `proposal_status: dismissed` event as feedback: ask in chat what was
  wrong.
- **Metadata travels with the graph.** Put round context in the spec's `meta`. It is written to
  `workflow.extra.agent` and comes back in every `run` event, whatever the user did to the graph.
- **Runs you queue don't show on the user's canvas**, because previews go only to the client that queued them. The
  user presses Run, and identical nodes hit the cache.

## Spec format (an API-format superset)
```json
{"meta": {...},
 "groups": [{"id": "candA", "title": "A · rotate mid 90° (new composition?)", "color": "#3f5159"}],
 "nodes": {"key": {"class_type": "...", "inputs": {"widget": 1.5, "socket": ["other_key", 0],
                                                   "candidates": [["bendA", 0], ["bendB", 0]]},
                   "title": "...", "group": "candA", "mode": "active|mute|bypass", "pos": [x, y], "size": [w, h]}}}
```
- A value `[node_key, output_index]` is a link.
- A **list of links under an autogrow input** (`candidates`, `reports`, `images` on `BatchImagesNode`) fills
  `name.prefix0…N`, and the builder adds the spare slot the frontend expects.
- You can link into a widget input, for example a `PrimitiveFloat` into `a`.
- `Note` and `MarkdownNote` are canvas-only.
- Layout is automatic: groups stack as bands in spec order, and columns follow graph depth.
- Put the **hypothesis** in the group titles, and the safe range and neutral value in the slider titles.

## Preset 1 — Comparison Switchboard (`switchboard_bridge_sd15.spec.json`)

**Lanes.** It shares one base (loaders, prompt, seed), then has one lane per candidate:
`ApplyBendsFromJSON → KSampler → VAEDecode`. It also has a baseline lane.

**FEEDBACK band.** `BatchImagesNode(candidates)` → `AgentFeedback`:
- thumbnails with 👍/👎 per labelled candidate
- keep/change chips
- rating 0–5
- push −1..1
- free text
- a **Send to agent** button that needs no run

**PICK band.** `AgentVariantSwitch` shows radio buttons labelled like the candidates. Only the chosen bend runs,
through a 4-seed confirmation sampler. It tests whether the bend **generalises** beyond one seed.

**Reports band.** Each lane's `report` feeds `AgentReportSink` (channel `bends`), so you see what each bend actually
resolved to.

**Reading it back:**
- `run` events carry `switch_picks: [{index, label, labels}]` and `feedback: [...]`.
- `feedback` events with `source: "button"` arrive without a run, and include the image refs the user saw.

The core-only fallback (`switchboard_sd15.spec.json`) uses a chain of `ComfySwitchNode` questions
("Pick: B instead of A?") plus a Note and primitives. Read it with `wait-run` or `pull`.

## Preset 2 — Bending Slider Control Board (`slider_board_sd15.spec.json`)

**Controls.** A 🎚 CONTROL BOARD holds `PrimitiveFloat` sliders linked into `ApplyBendsFromJSON`'s `a`…`d`.

**Bends.** A single JSON template uses them: `"angle_degrees": {{a}}`, `"t": [1.0, {{d}}]`, and so on.
- `clamp: safe` with `safe_ranges` from `data/safe_ranges_sd15.json` keeps any slider position inside the measured safe
  range. Clamps are listed in the report.
- The `report` goes to `PreviewAny`. It shows `placeholders`, `resolved`, `clamped` and `patterns`; it appears in
  `last-run` and in the user's canvas.
- Put neutral values in the slider titles ("1 = off"), so the user can disable a bend without rewiring.
- `PrimitiveInt` gets `control_after_generate = "fixed"` from the builder. Keep it that way, or the seed drifts.

**Reading it back:**
- `run` events and `last-run` give the values used.
- `pull --baseline` or `canvas` gives values moved but not yet run.

**Discrete-node variant.** Use this when a slider must drive an op's argument that JSON can't reach:
`PrimitiveFloat` → `Rotate Module (Bending).angle_degrees` → `Model Bending(path, t_start, t_end)`.

## Preset 3 — Feature Map Inspector (`feature_inspector_sd15.spec.json`)

**Feature maps.** A 🎚 step slider feeds `Visualize Feature Map.timestep` (1-based step, 0 = last) for the
**unbent** and the **bent** model, side by side, at two layers.
- Maps are normalised, and they respect upstream bends.
- Token layers work too; they are laid out on the image grid.

**Activation probe band.**
- `ActivationProbe` on an unbent run → `ReadActivationProbe` gives the reference.
- `ActivationProbe` on the bent run → `ReadActivationProbe(reference)` gives a **sites × steps heat map of
  log2(std bent / std unbent)** and a JSON report with `alerts` (blowup, collapse, nonfinite, dead_channels) and
  `per_site` ratio min/max.
- Together they show where in the network, and when, the bend's effect travels. That is the L3 causal
  footprint, measured before decoding.

Pair it with `metrics.py diff baseline.png bent.png` for where the change landed in the pixels.

## Feedback channels, from least to most effort for the user

1. **Button** (bridge): vote, chip or rate, then press Send to agent. Nothing runs.
2. **Queue-as-vote**: pick a radio, move sliders, press Run. `run` event, or `wait-run` without the bridge.
3. **Save-as-message**: edit anything, press Ctrl+S, say "done". `pull --baseline` gives the exact diff: nodes added,
   deleted or rewired, and muted lanes.
4. **Live canvas** (bridge, opt-in): read the current graph at any moment with `canvas`.
5. **Model-Bending WebUI**: `GET`/`POST /web_bend_demo/selection`.
6. **Chat**: a `metrics.py sheet` contact sheet, plus a letter or a sentence.

## Adapting the base block to other architectures
- **SDXL**: `CheckpointLoaderSimple` gives MODEL, CLIP and VAE (slots 0, 1, 2). Use 1024², `dpmpp_2m` / `karras`,
  20 steps, cfg 5–7.
- **Flux**: `UNETLoader` (or a GGUF loader), `DualCLIPLoader(type=flux)`, `VAELoader`,
  `ModelSamplingFlux(width, height)`, `FluxGuidance(3.5)` on the positive, KSampler at cfg 1.0 with `euler` /
  `simple`, and `EmptySD3LatentImage`. Bend with `DiT Block Bending`.
- **SD3.5**: `CheckpointLoaderSimple` (or `TripleCLIPLoader`), `ModelSamplingSD3(shift 3)`, `EmptySD3LatentImage`,
  cfg 4–5. Use `DiT Block Bending` with `joint:…`.
- Check exact input names with `comfy_canvas.py node CLASS` (tool: `node_info`). The builder refuses unknown classes and inputs, and re-fetches
  `object_info` once if a class seems to be missing, for example after a restart that added a pack.
