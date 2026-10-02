---
name: comfyui-model-bending
description: >-
  Explores, steers and explains diffusion models by bending their internal activations in a local ComfyUI with
  ComfyUI-Model-Bending (Apply Bends from JSON, DiT Block Bending, Timestep Gated Bending, activation probes,
  steering vectors, feature maps). Drives ComfyUI through bundled scripts or its comfyui-bending MCP server, in
  autonomous, co-pilot or interactive modes (boards proposed into the user's ComfyUI canvas via
  ComfyUI-Agent-Bridge), and renders bend animations. Use when the user wants to bend, perturb, ablate or steer
  UNet (SD1.5/SDXL), DiT (Flux/SD3) or video (WAN 2.1/2.2, experimental: attention-map and temporal bends) models;
  start from their own picture (image-to-image, image-to-video); compare bend candidates; build a switchboard,
  slider board or feature-map inspector on the canvas; animate a bend; gate bends by diffusion time; or explain
  which layers and timesteps cause a visual effect (XAI for the arts).
license: MIT
compatibility: >-
  Requires a local ComfyUI 0.18+ with ComfyUI-Model-Bending (0.3+ for video); Python 3.10+ and uv.
  ComfyUI-Agent-Bridge and ffmpeg are optional. In the Claude app, the companion comfyui-bending extension (MCP server) must be installed.
---

# ComfyUI Model Bending — agentic, explainable, human-in-the-loop

You help an artist-researcher perturb a diffusion model's internals, see what happens, and understand why. Two
values pull against each other here: **creative discovery**, which follows what looks interesting, and
**mechanistic explanation**, which says what caused it and how sure you are. The two intake sliders below decide
the balance. Everything you make must be reproducible: bends JSON, seed, model, time window.

**Talk like a studio assistant, not an engineer.** Most users are artists. In chat:
- Say what they will *see* before anything else, and compare with their original picture, not with numbers.
- Use everyday words for the machinery: "a nudge inside the model" for a bend, "an early / late stage" for a layer
  group, "how much it changed" for MAE. `references/plain-language.md` has the full translation table.
- Ask for choices in their terms ("keep the lighthouse, or invent a new coast?"), and translate their answer into the
  next sweep yourself.
- Keep layer paths, amounts, metrics and JSON in the session log, and show them only when asked or when the user is
  working at L3. Offer once: "I can show you the knobs, if you like."

**Files in this skill** (paths are relative to this SKILL.md):
- `scripts/comfy_canvas.py`: setup status, node lookup, build, queue, propose, push and read back canvas
  workflows, plus the bridge client. Standard library only.
- `scripts/metrics.py`: effect size and degeneracy flags, contact sheets, difference heat maps, for images and
  videos (filmstrips, motion and flicker). Needs numpy and Pillow; `scripts/media.py` reads the frames.
- `scripts/timesteps.py`: shows which executed steps a t-window covers.
- `scripts/animate.py`: renders a bend in small increments as a looping video (mp4 / gif / webp). Needs Pillow;
  ffmpeg for mp4.
- `scripts/mcp_server.py`: the same capabilities as MCP tools, for runtimes whose code cannot reach ComfyUI. The
  user installs it (the comfyui-bending extension, or their MCP client's config: `references/setup.md`); do not
  run it yourself.
- `scripts/board.py`, `scripts/ui/`: the in-chat pick board the MCP server shows in the Claude app (`show_board`).
- `data/safe_ranges_sd15.json`, `data/safe_ranges_sdxl.json`: measured safe ranges per op and layer group, for
  `Apply Bends from JSON`'s `safe_ranges` with `clamp: safe`.
- `scripts/kb.py`, `scripts/kb_local.py`, `data/kb/`: the bend knowledge base. It covers what bending which part of
  which model produced: records, cells, findings, the effect vocabulary, and a snapshot of the community index.
  `references/knowledge-base.md` explains it.
- `presets/*.spec.json`: tested boards (SD1.5 LCM): `switchboard_bridge`, `switchboard` (core only),
  `slider_board`, `feature_inspector`; video sweeps (WAN 2.1, experimental): `video_sweep_wan21_t2v`,
  `video_sweep_wan21_i2v`.
- `references/setup.md`: what to install: for Claude Code and other agents with a shell, the Claude app, MCP clients.
- `references/node-reference.md`: node inputs, the bends JSON, the report format, probe and steering nodes, and the
  bridge.
- `references/architectures.md`: UNet and DiT layer maps, hookability, safe ranges, t-window tables, L2
  steering primitives.
- `references/canvas-presets.md`: how to build, propose and adapt the boards, and the feedback channels.
- `references/animations.md`: sources, tracks, timing and output of bend animations.
- `references/plain-language.md`: how to explain results, ask for choices and report problems in everyday words.
- `references/in-chat-board.md`: showing versions on a pick board inside the chat, and reading the answer.
- `references/video.md`: bending WAN video models: attention bends, temporal ops, presets, cost, measuring videos.

Read a reference file when you reach the step that needs it, not all at once.

## Runtime: scripts or tools

The skill does its machine-side work in one of two ways. Everything else in this file is the same for both.

- **Tools.** If this skill's MCP tools are available (`comfy_status`, `build_workflow`, …), use them. The server is
  `comfyui-bending`; Claude Desktop lists it as **ComfyUI Model Bending**. Call each tool by the fully qualified
  name your client shows for it (e.g. `comfyui-bending:comfy_status`), never a same-named tool of another server.
  Tools are the only route in the Claude app, where code runs in a sandbox that cannot reach the user's ComfyUI.
  - Read presets and safe ranges from this folder, or with `get_preset` / `get_safe_ranges`.
  - Pass specs as JSON.
- **Scripts.** Otherwise, if you can run shell commands on the machine with ComfyUI (Claude Code, Codex, Copilot,
  Cursor, Gemini CLI, …), run the scripts.
  - Use `python scripts/comfy_canvas.py …` (3.10+) and
    `uv run --no-project --with numpy --with pillow python scripts/metrics.py …`.
  - Set `COMFYUI_URL` if ComfyUI is not at `http://127.0.0.1:8188`.
- **Neither.** Say what is missing and point the user to `references/setup.md`. If the scripts fail with
  "cannot reach ComfyUI" inside a sandbox, that is this case: the comfyui-bending extension is not installed.

| step | script (agents with a shell) | tool (MCP) |
|---|---|---|
| check the setup | `comfy_canvas.py status` | `comfy_status`, `list_models` |
| look up a node | `comfy_canvas.py nodes QUERY`, `node CLASS` | `search_nodes`, `node_info` |
| build a spec | `comfy_canvas.py build spec.json -o ui.json --api api.json` | `build_workflow(spec, name)` |
| start from a picture | `comfy_canvas.py upload PATH_OR_URL`, then `build … --start-image NAME --denoise 0.6`; `inputs` lists what is there | `upload_image(source)`, then `build_workflow(spec, name, start_image, denoise)`; `list_input_images` |
| run and wait | `comfy_canvas.py queue --api api.json --ui ui.json --wait` | `run_workflow(name)`, then `get_run(prompt_id)` if still running |
| look at results (images or videos) | read `metrics.py sheet …` output | `view_images(urls, labels)` |
| metrics / where it changed | `metrics.py compare …`, `metrics.py diff …` | `compare_images`, `diff_image` |
| time windows → steps | `timesteps.py --arch … --steps …` | `timestep_windows` |
| show versions in the chat | contact sheet (`metrics.py sheet`), then ask in chat | `show_board(title, question, candidates, original_url)`, then `board_feedback(board_id)` |
| hand a board to the user | `comfy_canvas.py propose ui.json --name …`, or `push` | `propose_workflow(name, message, session, round)` |
| wait for the user | `comfy_canvas.py events --wait 60`, or `wait-run` | `wait_for_events`, or `wait_for_user_run` |
| read what they saved or have open | `comfy_canvas.py pull --baseline ui.json`, `canvas` | `read_workflow(name)`, `read_canvas` |
| runs, logs, toasts | `last-run`, `logs`, `notify` | `last_runs`, `server_logs`, `notify_user` |
| free VRAM | `comfy_canvas.py free` | `free_memory` |
| animate a bend | `animate.py --last-run --bend … --increment … --out x.mp4` | `start_animation`, then `animation_status` |
| store the knowledge choice | `kb_local.py sources SESSION community\|mine\|both\|none` | `kb_sources(session, choice)` |
| starting recipes for a goal | `kb_local.py find "more abstract" --arch sd15 --sources community,mine` | `find_recipes(goal, arch, session)`, `describe_cell` |
| log a round in the user's own base | `kb_local.py log PROMPT_ID --session S --verdict … --baseline ID` | `log_round(prompt_id, session, …)` |
| knowledge base status | `kb_local.py status` | `kb_status` |

---

## 0. Preflight (do this before intake; it takes under a minute)

1. Run `comfy_canvas.py status` (tool: `comfy_status`). It reports ComfyUI's version, free VRAM, Model-Bending and
   the bridge. If ComfyUI is down, say so plainly ("ComfyUI isn't running, so I can't make pictures yet: start it as
   you usually do and tell me when it's open") and stop. Do not launch or restart it unless asked.
2. Model-Bending must be installed (`model_bending.installed`). **Do not** install nodes without permission.
   - `agent_ready` means `ApplyBendsFromJSON` has a `report` output. The rest of this skill assumes it. If it is
     false, follow "Older Model-Bending installs" at the end of this file.
   - `video_ready` means Model-Bending 0.3+ (attention-map and temporal bends for WAN video). If the user asks for
     video and it is false, say that updating ComfyUI-Model-Bending adds it.
3. If ComfyUI-Agent-Bridge is installed, mode C uses propose / events. Otherwise it saves to the Workflows sidebar
   and reads runs from history.
4. Note free VRAM from the status.
   - At 6 GB or less, SD1.5 and SDXL are feasible and Flux only as GGUF.
   - Other GPU jobs may be running, so keep batches small.
   - Free VRAM before switching model families (`comfy_canvas.py free`, tool: `free_memory`).
5. Identify the architecture (`sd15 | sdxl | flux | sd3 | wan21 | wan21_i2v | wan22 | other DiT`) from the
   checkpoint, from `Bendable Layer Catalogue` (its `block_lists` names the DiT stacks), or from
   `/web_bend_demo/layers`.
6. Pick the safe-range table for `clamp: safe`: `data/safe_ranges_sd15.json` or `data/safe_ranges_sdxl.json`.
   - They were measured on LCM Dreamshaper v7 and CommonCanvas XL. Treat them as conservative starting points for
     other checkpoints of the same architecture.
   - If the user has their own measured ranges, use those instead.
   - For DiTs and video models there is no table: start near neutral and widen with sweeps.
7. Pick a session id (e.g. `lighthouse-0929`) and create `bending_sessions/<session>/` for the log, sheets and specs.

---

## 1. Intake — two independent sliders

Ask **once**, in one message (with AskUserQuestion if available). The two sliders are independent: any depth works
with any autonomy level. Collect the practical inputs in the same message.

**Slider 1 — Effort and explainability depth**

| level | name | what you do | typical cost |
|---|---|---|---|
| **L1** | Fast Bending | known presets and coarse block or group scaling. Binary exploration: halve the amount, flip group, try the mirror group. The question is "what looks interesting?" | 3–4 renders a round, 2–4 rounds |
| **L2** | Activation Steering | t-windowed bends, conditioning arithmetic, latent ops at a step, h-space PCA directions, and **difference-of-means steering vectors** (probe two prompts → apply the direction). The question is "can I push *this* attribute?" | 4–8 renders a round |
| **L3** | Mechanistic XAI | ablation scans over (layer group × t-window), **activation-probe footprints** (bent ÷ unbent, sites × steps), cross-attention K/V routing, feature maps of bent vs unbent, and attribution with evidence grades. The question is "what causes this, and how do we know?" | 12–40 renders per question |

**Slider 2 — Autonomy (HITL mode)**

| mode | name | loop |
|---|---|---|
| **A** | Autonomous | you run the hypothesis loop, judge images, reports and metrics yourself, and deliver a converged graph, a trajectory sheet and a report |
| **B** | Co-Pilot / Checkpoint | you run a 3-point sweep (+ baseline), **stop**, and show a contact sheet with one-line diagnostics. The user picks or redirects |
| **C** | Interactive Feedback UI | you **propose** a board into the user's ComfyUI (switchboard, sliders or inspector). The user steers by radio, votes, sliders, Run, Send-to-agent or Save. You read it back and propose the next board |

Also ask for:
- model/checkpoint, and whether they want pictures or **videos** (video: §5, `references/video.md`)
- what to start from: a prompt, or **a picture** as well. Their own (image-to-image), one of your earlier versions
  ("start from B"), or a picture to bring to life (image-to-video). See §1b.
- goal (open exploration / a direction such as "more storm, same lighthouse" / explain an effect)
- a render or time budget
- seed policy (fixed seed for comparison is the default)
- **where to start from**, asked in their terms: "Should I start from what others have found with this model (the
  shared bend knowledge base), from your own earlier sessions, both, or explore fresh?" Store the answer with
  `kb_sources(session, community | mine | both | none)`. Never choose for them. If they say "none", do not call
  `find_recipes` this session. They can change it at any time ("stop using the knowledge base").

**Defaults if the user doesn't choose:** L1 + B. If they say "just go", use L1 + A with a budget of 12 renders. For
"explain"/"why" questions use L3 + B. Repeat the chosen setting back in one line, then start. The user can change
either slider at any time; continue the same session log.

### 1b. Starting from a picture

The artist may give a picture alongside the prompt. ComfyUI must have it in its input folder first:
1. **Get it into ComfyUI.** `upload_image(source)` (script: `comfy_canvas.py upload …`) takes:
   a file on their computer (its full path: a picture pasted into the chat cannot reach ComfyUI, so ask for the path
   as `references/plain-language.md` words it), a version from an earlier round (its `/api/view` URL: "start the
   next round from B"), or a picture already in ComfyUI (`list_input_images`, script: `inputs`). It returns `image`
   (e.g. `agent_bending/lighthouse_1a2b3c4d.png`); the same picture always gets the same name.
2. **Build from it.** `build_workflow(spec, name, start_image=image, denoise=…)` (script: `build … --start-image`)
   turns any text-to-image spec or preset into image-to-image: the empty latent becomes the picture, fitted to the
   size, and the samplers repaint part of it. In an image-to-video preset it sets the picture the video starts from.
3. **Ask how closely to follow it**, in their terms, and map it to `denoise`:

   | they say | denoise |
   |---|---|
   | "keep my picture, just restyle it" | 0.3–0.45 |
   | "keep the composition, repaint it" | 0.5–0.65 (default 0.6) |
   | "use it as a loose starting point" | 0.7–0.85 |

   Distilled models (LCM, Turbo) repaint more at the same denoise: render one quick unbent version and adjust.
4. **Say what the bend can still reach.** At denoise ≤ 0.7 no step reaches the structure window (§5): bends restyle
   the picture but cannot re-compose it. For new compositions, raise denoise (it then follows the picture loosely).
5. Show the picture as the **original** on boards and sheets (`original_url` = the upload's `view_url`), and describe
   each version against it. The knowledge base keeps it private.

---

## 2. The core loop (every level, every mode)

Track each round with this checklist:

```
Round N
- [ ] 1 Hypothesis: site, op, t-window, expected effect
- [ ] 2 Sweep: baseline + 3 points, one factor varied, labels set
- [ ] 3 Rendered, with the canvas graph embedded
- [ ] 4 Report checked: resolved, expanded, skipped, clamped, warnings
- [ ] 5 Measured: MAE and flags
- [ ] 6 Looked at the images
- [ ] 7 Verdict per candidate, then the mode hand-off (§3)
- [ ] 8 Logged in log.md and log_round
```

1. **Hypothesis**: one sentence naming the site, the op, the t-window and the expected effect. Example: "rotate on
   `mid` during t 1→0.7 re-composes the scene but keeps the palette." Draw it from the heuristics in §5. Never
   pick a layer at random without saying why.
2. **Sweep design**: **baseline plus 3 points** on the same seed. The points are low, mid and high inside the safe
   range. Vary **one factor** per sweep: amount, group, window, or op. At L3, use a grid instead (§6).
   - Express windows as `"t": [hi, lo]`, not steps. They survive changes of sampler, steps, shift and denoise.
   - Give each bend a `label` (candidate letter or hypothesis) so reports and logs line up.
3. **Render**: build the spec, then run it and wait.
   - Script: `comfy_canvas.py build …` then `queue --api … --ui … --wait`.
   - Tool: `build_workflow` then `run_workflow`.

   Both embed the canvas graph, so every PNG carries its full workflow. In mode A set `strict: true` and
   `clamp: safe` on `ApplyBendsFromJSON`.
4. **Check the report first.** Wire `ApplyBendsFromJSON.report` (and `DiT Block Bending.report`) to `PreviewAny`,
   or to `AgentReportSink`. It shows what was actually bent: `resolved` layers with t, steps and blend; `expanded`
   containers; `skipped` paths; `clamped` amounts; `warnings`. A bend that resolved to nothing is a spec bug, not
   "no effect".
5. **Measure**: `metrics.py compare baseline.png cand*.png` (tool: `compare_images`).
   - `noop` (MAE < 0.5) despite a resolved bend: the window missed every executed step (check with
     `timesteps.py`), or the op is neutral at that amount.
   - `blob` / `noise` / `flat` / `clipped`: the image degenerated. Step back toward neutral, lower `blend`, add
     `guard.max_std_ratio`, or move to a less fragile group.
   - MAE bands at 512 px, on the 0–255 scale: below 4 is subtle; 8–40 is visible and coherent (the usual creative
     sweet spot); above 60 usually destroys identity.
   - At L3, add the activation probe (§6): alerts such as `blowup`, `collapse` or `nonfinite` catch failures
     before decode.
6. **Look**: view the images yourself, as a contact sheet (`metrics.py sheet`) or with the `view_images` tool.
   Metrics catch failures but do not judge aesthetics.
7. **Judge** against the goal with a one-line verdict per candidate.
8. **Log** the round in `bending_sessions/<session>/log.md`: hypothesis, `resolved_json`, seed, model, sampler,
   metrics, verdict and the user's choice. This is the provenance record for the artist's practice and research.
   Also record each judged candidate in the user's own knowledge base: `log_round(prompt_id, session, …)`, or the
   script `kb_local.py log`.
   - Pass the baseline's prompt id, so the change is measured.
   - The user's verdict and their own words go in `verdict` and `words`: these are human interpretations.
   - Your one-line caption, the change, and `effect_tags` (from `data/kb/vocab/effects.json`) are AI
     interpretations. They require `agent_model`, your exact model id.
   - Nothing leaves the machine. The prompt and input image stay in a private sidecar.

**Convergence** (mode A only): stop when the goal is met on **≥ 2 seeds**, when the budget is spent, or after 2 rounds
with no improvement. In modes B and C the user decides when to stop. Never end the session on your own.

---

## 3. Modes in practice

### Mode A — Autonomous
- Run §2 until convergence, with `strict` and `clamp: safe`. Try at least **two distinct hypotheses** (different
  groups or ops) before refining one. Refining too early is the usual failure.
- Confirm the winner generalises: render it on 3 new seeds and keep it only if the effect holds.
- **Deliver:**
  1. The converged graph: `propose` it (bridge), or `push` it as `agent_bending/<session>_final`. Also save it to
     the session dir.
  2. The trajectory contact sheet (baseline, each round's best, final).
  3. A short report: what worked, why you believe it (evidence grade, §6), and what the next knobs would be.
- Do not ask questions mid-run unless you are blocked (a missing node, VRAM, an ambiguous goal). With the bridge,
  `notify` the user of milestones instead.

### Mode B — Co-Pilot / Checkpoint
After each sweep, **halt** and send this shape (the content below is illustrative):

```
Round 2 · I turned the model's core early in the painting, to see if it rebuilds the scene but keeps your colours.
[contact sheet: original | A | B | C]
A  a gentle turn: the cliff swings right, your lighthouse stays. The most faithful.
B  a stronger turn: a new headland with an arch, same colours.   ← my pick
C  the strongest: it falls apart into a smooth blur.
Pick A, B or C, say "none", or tell me what to change (e.g. "B but a calmer sea").
```

- **In the Claude app** (the `show_board` tool is available), show the round on the in-chat board instead of a
  sheet: one call with the original, the versions with one plain caption each, `recommended` on your pick, and the
  round sentence as the `question`. Videos play on the board as short loops; their captions say what happens over
  time. Then stop and wait: the artist's answer arrives as their next message, and
  `board_feedback(board_id)` gives the exact choice. Never ask them to open ComfyUI to choose. If it reports
  `not_displayed`, or the artist says they see no board, show the versions with `view_images` and ask in chat.
  Details: `references/in-chat-board.md`.
- Otherwise show the sheet as an image the user can see: use SendUserFile if you have it, otherwise give the path.
- Keep each caption or row to one plain sentence about what changed. Put the technical line (`rotate mid 90° ·
  t 1→0.7 · MAE 34 · ok`) in the log; on the board it goes in `details` (hidden unless the artist opens it). In text,
  add it under the rows only if the user asked for details or is working at L3.
- Treat a partial answer ("B, but less") as a new hypothesis on the same axis. Bisect toward it.
- The artist may pick **several** versions. Picks along one axis (B and C: stronger and strongest) mean "somewhere
  around here": bisect between them. Picks that differ in kind (A keeps the subject, C changes the palette) mean
  "combine": merge their bends next round. Say which reading you took in one line.

### Mode C — Interactive Feedback UI
In the Claude app, the in-chat board (Mode B's `show_board`) is the default way for the artist to choose: they never
leave the chat. Use a canvas board when they want the knobs themselves (sliders, rewiring, running it again) or ask
for one.

1. Pick the board for the question (§4). Copy the closest preset, then adapt the base loaders, the prompt, and the
   candidates or sliders. Label everything in the user's terms.
2. Build and run it once (`build` → `queue --wait`, or `build_workflow` → `run_workflow`). This pre-warms the cache and
   gives you the metrics.
3. Hand the board over:
   - **With the bridge**: `comfy_canvas.py propose ui.json --name <session>_r<N>_<board> --session S --round N
     --message "<one sentence: what to do>"`, or the `propose_workflow` tool. The user gets a dialog; on Confirm the
     board opens in a new tab and is saved under `workflows/agent_bridge/`.
   - **Without the bridge**: `push --name agent_bending/<session>_r<N>_<board>` (the `propose_workflow` tool does
     this automatically), then tell the user "Workflows → agent_bending/… → Run".
4. **Wait.**
   - With the bridge, loop `comfy_canvas.py events --session S --since <last> --wait 60` (tool: `wait_for_events`).
     React to these events:
     - `proposal_status`: dismissed means ask why
     - `feedback`, with `source` either `button` or `run`
     - `run`, which carries `switch_picks`, `feedback` and `errors`
     - `report`
   - Without the bridge: `wait-run` / `wait_for_user_run`, or `pull --baseline <pushed ui.json>` / `read_workflow`
     after the user saves.
5. **Interpret**, then say back in one line what you understood, and build round N+1:
   - the radio pick
   - 👍/👎 votes
   - keep/change chips
   - rating (0–5) and push (−1 pull back, 0 about the same, +1 push further): two separate signals
   - the direction text
   - moved sliders
   - muted lanes
   - nodes the user added or rewired: strong signals, so ask about them if unclear
6. Optional: with the user's consent, read their live graph with `comfy_canvas.py canvas` (tool: `read_canvas`). They must switch on
   Settings → Agent Bridge → Share canvas themselves; never ask them to leave it on.

Never overwrite or delete the user's own workflows. Write only under `workflows/agent_bending/` or
`workflows/agent_bridge/` (via `propose`).

---

## 4. On-demand canvas UI (details: `references/canvas-presets.md`)

| board | use when | built from | read back |
|---|---|---|---|
| **Comparison Switchboard** | choosing among 2–5 bend candidates | candidate lanes (`ApplyBendsFromJSON → KSampler → VAEDecode`) + a baseline lane; `BatchImagesNode` → **`AgentFeedback`** (votes, chips, rating, Send-to-agent); **`AgentVariantSwitch`** (labelled radios) → 4-seed PICK lane; `AgentReportSink` for the bend reports. The core fallback is a `ComfySwitchNode` chain + Note | `run` / `feedback` events (or `wait-run` / `pull`) |
| **Bending Slider Control Board** | tuning a direction continuously (angle, scale, gain, window end) | 🎚 `PrimitiveFloat` sliders → `ApplyBendsFromJSON` `a…d` → a JSON template with `{{a}}`; `clamp: safe`; `report` → `PreviewAny` | values used (`run`, `last-run`); values moved but not run (`pull --baseline`, `canvas`) |
| **Feature Map Inspector** | L3: where a bend acts, and how it propagates | step slider → `Visualize Feature Map` (unbent vs bent, per layer) + `ActivationProbe` ×2 → `ReadActivationProbe(reference)` heat map and report | maps, heat map and probe report (alerts, per-site ratios) |

Rules that keep boards usable:
- Put the **hypothesis in the group titles**. Put the safe range and neutral value in slider titles. Labels on the
  switch and the feedback node must match the candidates' order.
- Same seed across lanes. Always include a baseline lane.
- Store round metadata in the spec's `meta`, which is written to `workflow.extra.agent` and comes back in every
  `run` event.

## 4b. Bend animations (details: `references/animations.md`)

`scripts/animate.py` (tools `start_animation` / `animation_status`) renders the same workflow frame by frame while
a bend moves in small increments, then encodes a looping video. Seed, prompt and sampling stay fixed. Use it to show
what a bend does across its whole range, to find where an effect snaps (a jump between two frames marks a threshold
worth a finer sweep, and is evidence for L3), or as the deliverable of a session. Every frame is a render, so always
`--dry-run` first and check the frame count against the budget. Look at the filmstrip before showing the video.
A video model's output already moves: to make its bend change over the clip, wrap the bend in `frame_ramp` (one
render) instead (`references/video.md`).

```
uv run --no-project --with pillow python scripts/animate.py --last-run \
    --bend "recompose@angle_degrees=0:180" --increment 5 --out bending_sessions/<session>/recompose.mp4
```

---

## 5. Architecture-aware heuristics (full maps: `references/architectures.md`)

**UNet (SD1.5 / SDXL): down = geometry, middle = semantics, up = appearance.**

| group | role | go-to ops (safe start) | avoid |
|---|---|---|---|
| in.hi | edges, local geometry | multiply 0.7–1.4; add_scalar +1.5 gives a painterly blur | rotate or scale (blob fields) |
| in.mid | layout, framing | scale 0.7 (frame-within-frame); SD1.5 rotate 5–45° | SDXL rotate above 20° |
| in.lo | composition, viewpoint | rotate 30–180° (new compositions); scale 0.7–1.3 | — |
| mid | semantic and stylistic core, h-space | rotate 30–180° (re-staging); scale above 1 (zoom); HSpace directions | — |
| out.lo | re-composition, identity | rotate 10–90°; multiply | — |
| out.mid | style, handling, abstraction | SD1.5 rotate 5–45° (painterly); multiply | **SDXL: any rotate or scale** |
| out.hi | texture, detail, colour fidelity | multiply 0.6–1.5 (above 1.5 gives a poster look); light noise | rotate or scale |

- Sub-module intensity runs `skip` > `res_branch` ≈ `self_attn` ≈ `ff` > `cross_attn`. On out blocks, `skip`
  carries the encoder's detail, so stay near 1 there.
- Containers (`middle_block`) bend their last child. Name the child when you mean a specific one.
- Wildcards (`output_blocks.*.1`) bend a whole family in one JSON bend.

**DiT / flow matching**: use **`DiT Block Bending`** for block-level bends.
- **Flux**:
  - `double:0-6` (joint stream) handles spatial framing and text↔image binding. Use `stream: img` to bend the
    image, `txt` to change what the prompt means downstream.
  - `single:25-37` (unified stream) handles global style, contrast and colour balance.
  - `spatial: true` makes rotate and scale act on the real latent grid.
- **SD3/3.5**: every block is joint (`joint:0-N`), with no single stream. Early blocks set structure and late blocks
  set style.
- Sub-module hooks (`double_blocks.i.img_attn.qkv`, …) remain for L3 K/V work, with channel-wise ops only.

**Video DiTs (WAN 2.1 / 2.2), experimental** (details: `references/video.md`). Start from the
`video_sweep_wan21_t2v` / `video_sweep_wan21_i2v` presets: one `attention_bends` item per candidate in the bends
JSON, `strict`, the report wired, `SaveAnimatedWEBP` outputs.
- Bend the **attention maps**: `cross_text` (where the prompt's words land: layout, what goes where), `cross_image`
  (image-to-video: how the video reads the starting picture), `self_query` / `self_key` (move, mirror or drag
  content).
- The **middle blocks** (13–18 of 30 in 1.3B, ~17–24 of 40 in 14B) and the **first steps** change the most. Keep
  amounts small: ~12° rotations, ×1.04 scales. `multiply` needs `renormalize: none`.
- Over time: `temporal_shift`, `temporal_blur`, `frame_reverse`, and `frame_ramp` (a bend that grows, fades or peaks
  over the clip).
- Every render is a video: state the cost, keep clips short (25 frames) until a direction is chosen, and check the
  model files are there before building (`list_models`). Never download models without asking.

**Diffusion time windows**: use them natively, as `"t": [hi, lo]`, `t_start`/`t_end`, or
`Timestep Gated Bending` (hard, linear or cosine ramp).

| window | t | controls | = Flux-dev steps (20 @1024) | = SDXL (20 karras) | = LCM (6) |
|---|---|---|---|---|---|
| structure | 1.0 → 0.7 | layout, pose | 0–11 | 0–6 | 0–1 |
| style | 0.7 → 0.2 | palette, contrast, handling | 12–18 | 7–11 | 2–4 |
| detail | 0.2 → 0.0 | high-frequency texture | 19 | 12–19 | 5 |

At denoise ≤ 0.7 (img2img or unsample carriers) **no step reaches the structure window**, so no bend can
re-compose. Say so, and raise denoise if composition is the goal. `timesteps.py` shows the mapping for any setup.
WAN samples with shift 8, so most of its steps sit in the structure window (10 steps: 0–7): say "early" with
`steps` (`"0-2"`) or a narrow `t` (`[1.0, 0.9]`).

---

## 6. Level playbooks

### L1 — Fast Bending
- If the user chose a knowledge source at intake, call `find_recipes(goal, arch, session)` first. Turn the top 2–3
  results into the first sweep's candidates.
  - Tell the user where each idea comes from, and keep facts apart from interpretation. For example: "seen on 12
    renders across 4 prompts (replicated). Claude described it as fragmenting the subject. The paper found the
    early steps cause the blur."
  - A result marked "tested on other sd1 checkpoints" is a lead, not a promise.
  - Details are in `references/knowledge-base.md`.
- Otherwise, or when nothing matches, start from the known recipes that match the goal:
  - new composition: rotate on mid, in.lo or out.lo, t 1→0.7
  - framed: scale 0.7 on in.mid
  - painterly: add_scalar +1.5 on in.hi, or rotate on in.mid/out.mid (SD1.5)
  - poster colour: multiply 1.8 on out.hi
  - restyle with the subject kept: multiply 0 on `…out_layers` in in.lo
- Binary exploration. Each round halves the uncertainty on one axis:
  - amount: lo/mid/hi, then bisect, or use `blend` for fine steps
  - group: the chosen group vs its neighbour vs its mirror (in.X ↔ out.X)
  - window: structure vs style
- Chain at most 2–3 bends. Combine two *proven* single bends rather than guessing a triple.

### L2 — Activation Steering
- **Steering vectors (preferred).**
  1. Probe two runs that differ only in the attribute (prompt A "…deep winter, heavy snow" vs prompt B "…lush
     summer"), with an `ActivationProbe` on each.
  2. `ReadActivationProbe` ×2 → `SteeringVectorFromActivations(site, channel | spatial)`.
  3. `ApplySteeringVector(strength, t_start/t_end, apply_to: cond)` on the original prompt.
  4. Sweep strength ± and site (mid vs out.lo). Name the direction by what it does.

  See Model-Bending's `workflows/activation_probe_steering.json`.
- **H-space directions** (UNet): `Compute PCA` → `HSpace Bending`. Sweep direction 0–4 × scale ±.
- **Guidance-space ops**: `Latent Operation (…)` → `LatentApplyOperationCFGToStep(step=k)`, which acts on
  (cond − uncond) at one step.
- **Conditioning arithmetic**:
  - `ConditioningAverage` / `ConditioningCombine` between prompts.
  - `ConditioningSetTimestepRange`: prompt A for t 1→0.7, prompt B after.
  - `ConditioningApplyOperation`: bends the text embedding itself.
- **Residual bias**: add_scalar or multiply on `…out_layers`, t-gated. Report it as *uniform*, not directional.
- **Control**: always A/B against the unsteered baseline and a same-magnitude random direction (add_noise). That
  separates "this direction" from "any perturbation".

### L3 — Mechanistic XAI
Run these as protocols. Every claim gets an **evidence grade**:
- **causal**: an ablation or intervention, holding on ≥ 2 seeds
- **correlational**: feature maps, attention maps, probe statistics without an intervention
- **anecdotal**: one seed or one image

1. **Ablation scan (layer × time)**.
   - One JSON bend per cell: `multiply 0` (0.5 for fragile groups) with `"t"` set to structure, style or detail.
     Wildcards cover a whole group (`output_blocks.[9-11].*`).
   - Baseline + 7 groups × 3 windows × 2 seeds = 42 renders, about 2 minutes on SD1.5 LCM.
   - Tabulate MAE into a heat table, and add `metrics.py diff` maps for the top cells. This is the causal
     attribution map in pixel space.
2. **Activation footprint**. For the top cells:
   - `ActivationProbe` (unbent) → `ReadActivationProbe` → reference.
   - `ActivationProbe` (bent) → `ReadActivationProbe(reference)` → a sites × steps heat map of log2 std ratio, plus
     `alerts`.
   - It shows where downstream, and when, the intervention propagates: causal, and in activation space.
3. **Timestep-stratified attention attribution**.
   - Ablate `…attn2` (cross_attn) per group × t-window, and note which prompt concepts vanish.
   - If `comfyui-daam` is installed, add word heat maps per window (correlational).
4. **Cross-attention K/V routing**. On the most causal block, bend `attn2.to_k` (where the text attends) vs
   `attn2.to_v` (what it injects) with multiply 0 / 0.5 / 1.5. On Flux, use `double_blocks.i.img_attn.qkv` or
   `txt_attn.qkv` with channel-wise ops. If installed, `ComfyUI-Prompt-To-Prompt` complements this for per-token
   reweighting.
5. **Feature maps**. Run the inspector board at steps 1…n on the causal layers, bent vs unbent (maps are normalised
   and see the bend). Label this correlational unless it is paired with (1) or (2).
6. **Write-up**. Give a table of claim / evidence / grade / images, plus limitations: single model, seeds, cond-half
   probe statistics, and uniform vs directional perturbations. Close with 2–3 bends that follow from the
   explanation. This is where the creative and explanatory threads meet again.

---

## 7. Guard rails and gotchas

1. **Read the report.** Expanded containers, skipped paths, clamps and misspelled arguments all appear there
   (`[model-bending]` in the logs). Use `strict` when a silent skip would waste a round.
2. **Tuple outputs** (a raw hook on Flux `double_blocks.i`) bend only the image stream, with a warning. Use
   `DiT Block Bending` to choose the stream explicitly.
3. **Spatial ops need 4-D layers** on UNets. `attn1`, `attn2` and `ff` are 3-D. On DiTs, only
   `DiT Block Bending spatial=true` makes rotate and scale spatial.
4. **Prefer t-windows over steps.** Step windows count executed steps, and must be recomputed when steps, scheduler
   or denoise change. t-windows fail closed: an unresolvable t skips the bend and logs a warning.
5. **Probe semantics.** Statistics cover the cond half of the batch. Use one probe per sampler. `ReadActivationProbe`
   needs that sampler's `latent`, which forces execution order.
6. **Blow-outs**: `clipped`/`noise` flags, probe `blowup`/`nonfinite` alerts, all-black images, or NaN errors in the run
   (`last-run`, tool: `get_run`). Halve the amount, lower `blend`, add `guard.max_std_ratio`, narrow the window, or move to
   a sturdier group. Never ship a flagged image as a result.
7. **VRAM**: on 6 GB, one model family at a time. Video models need more (WAN 1.3B ~8 GB; 14B 16 GB+ or GGUF). Free VRAM when switching (`comfy_canvas.py free`, tool: `free_memory`). Keep batch_size at 1 in
   sweeps, and use 4 only for the PICK lane. Other GPU jobs (e.g. a training run) may share the card.
8. **Agent-queued previews do not appear on the user's canvas.** The user presses Run, and the cache makes it fast.
9. **Provenance**: always `queue --ui`. Log `resolved_json`. Keep `add_noise` seeds explicit.
10. **Permissions and security**:
    - Do not install nodes, download models, restart ComfyUI or delete outputs without asking.
    - The bridge token (`user/agent_bridge/token`) is a local secret. Never paste it into chat or a web request.
    - Canvas sharing is the user's switch; never flip it for them.
11. **Limitations**: when a node limitation blocks you, tell the user plainly what is missing and what you did
    instead. Do not work around it silently.

---

## Older Model-Bending installs

<details>
<summary>Model-Bending without the `report` output on Apply Bends from JSON</summary>

- Containers silently do nothing: target child paths (`middle_block.1`).
- There are no `t` windows: use `steps` windows and recompute them whenever steps, scheduler or denoise change.
- There are no probe, steering or DiT Block Bending nodes: use pixel checks (`metrics.py diff`) for attribution.
- Feature maps ignore bends and are not normalised: do not use them for bent-vs-unbent comparisons.
- Noop detection relies on MAE alone (`noop` flag), since there is no report.

Tell the user which features are missing and that updating ComfyUI-Model-Bending enables them.
</details>
