# Bending a workflow the user already has

Someone in the middle of a project has their own workflow: their model, LoRAs, prompt and settings. Do not rebuild
it inside a preset. Bend a **copy of their last run**, so everything of theirs is kept and their picture is the
unbent original.

Steps name the MCP tools; with scripts, `inspect_run` is `comfy_canvas.py run-setup` and `bend_run` is
`comfy_canvas.py bend-run` followed by `queue --api … --ui …` (SKILL.md, "Scripts and tools, step by step", has the
rest).

Contents:
- Steps
- What the copy is
- Limits, and what to say
- A custom fragment (Flux, SD3)

## Steps

1. **Ask them to run it once**, as it is: "Run your workflow once in ComfyUI the way you have it, and tell me when
   the picture is there." If they just did, go on. ComfyUI's history holds the run: no bridge and no canvas
   sharing are needed.
2. **Read it**: `inspect_run()` (script: `comfy_canvas.py run-setup`). It takes their newest own run (runs this
   skill queued are skipped). For another one, list runs with `last_runs(n)` and pass its `prompt_id`.
3. **Say back what you found in one line**, so they can tell it is the right run: "That's Dreamshaper XL with your
   two LoRAs, 30 steps, the prompt about the harbour at night." If it plainly is not what they described (another
   model, no LoRAs), say what you found and ask them to run their project once, or pick from `last_runs(5)`
   together. Do not bend the wrong run.
4. **Read what else it reports**:
   - `can_bend`: false means no sampler was found or Model-Bending is missing (the `notes` say which).
   - `copy`: `full` (a bent copy of their workflow can be opened too) or `pictures_only` (renders only).
   - `model.arch`: a guess from the file name. If it is missing, ask which kind of model it is, or read it with
     `Bendable Layer Catalogue`, and pass `arch` to `bend_run`.
   - `already_bent` and `bending_nodes`: bends already in their run. New ones are added after them, and their
     picture is then not an unbent original: say so.
   - `setup.batch_size`: each version renders that many pictures. Say what a round costs.
   - `images`: their picture. It is the original on boards and sheets, and the baseline for metrics.
5. **Make three versions**: `bend_run(name, bends, prompt_id)` (script: `comfy_canvas.py bend-run`) once per
   candidate, each under its own name (`<session>_r1_A`), then `run_workflow(name)`.
   - A goal → candidates from §5 and §6, or from `find_recipes` once they chose a knowledge source.
   - No goal → one bend per part of the model: `surprise_bends(arch, wild=false)` draws them. If they asked to
     be surprised, keep the wild card. Pass each candidate's `clamp`. A model type without a safe-range table has
     no draw: pick one gentle bend per part by hand (§5).
   - Always pass the same `prompt_id` (their run), so every version starts from their original.
   - Several bends in one version go in one `bends` document.
6. **Check and show** as in §2: the report (a "What the bend did" box is added with the bend), `compare_images`
   against their picture, then the board with their picture as `original_url`.
7. **On a pick, hand them the copy** when `copy` is `full`: `propose_workflow(name, message)`. Say what it is:
   "a copy of your workflow with one new green group, *Bend added by the assistant*. Your own workflow is
   untouched."
8. **Later rounds**: `bend_run` again from their run with the refined bends. `start_animation(name=…)` animates a
   bent copy, and `log_round(prompt_id=<bent run>, baseline_prompt_id=<their run>)` records it.

Their settings still decide what a bend can reach:
- A starting picture at denoise ≤ 0.7 never reaches the structure window (§5): bends restyle but cannot re-compose.
- A batch of 4 renders 4 pictures per version: say what a round costs.
- Their models stay loaded: do not free graphics memory or switch model families without asking.
- A pictures-only version has no canvas graph to embed, so it is queued without one (`queue --api` alone).
- Time windows as `"t": [hi, lo]` hold for their sampler and step count; step numbers do not.

## What the copy is

- The bend sits on the model link **just before their samplers**, after their LoRAs and model-sampling nodes, the
  same place the presets use.
- Their nodes keep their positions, groups and notes. The copy adds one group with `Apply Bends from JSON` and a
  `Preview Any` box showing its report.
- The seed is fixed to the one their run used, so the copy stays comparable with their picture. Tell them, in case
  they expect a new picture on every run.
- Pictures from the copy are saved in ComfyUI's output folder inside `agent_bending`, apart from their own.
- It is saved only under `agent_bending/` or `agent_bridge/`. Their own workflow file is never written.

## Limits, and what to say

| what `inspect_run` / `bend_run` reports | meaning | what to do |
|---|---|---|
| `copy: pictures_only`, "no canvas graph" | the run was queued by a script or another tool, not from the canvas | show the versions; to give them the bend, describe where to add the box (below) |
| `copy: pictures_only`, "inside a subgraph or a group node" | the sampler is nested where the copy cannot reach | the same |
| several samplers (high-res fix, base + refiner) | all samplers that take a model are bent | say so; `only` (node ids or titles) bends some of them |
| no `arch` | the file name does not tell the model type | ask, or use `Bendable Layer Catalogue`; pass `arch` |
| "no safe-range table" (Flux, SD3, WAN, others) | amounts are limited only by each op's own range | start near "no change" and widen; no surprise draw |
| `already_bent` or `bending_nodes` is not empty | their run already bends the model (a run from the bending web UI always does) | by default new bends stack after theirs: say so, and compare against their (already bent) picture. `replace_bends=true` puts the new bends in place of theirs instead, and with no bends it renders their run unbent |
| the run failed (`status: error`) | there is no picture to compare with | ask them to fix and run it first |

**Adding the box by hand** (when there is no openable copy), in click-path words:

> In your workflow, double-click the empty canvas, type "Apply Bends from JSON" and add it. Unplug the purple
> *model* wire going into your sampler, plug it into the new box, and connect the new box's *MODEL* output to the
> sampler. Then paste this into its text field: …

Give the `bends_json` of the version they picked. `bend_run`'s notes name the two nodes the box goes between.

Never overwrite, rename or delete the user's own workflows, and never change their run's settings to make a bend
look better: if a setting stands in the way (denoise too low to re-compose), say so and let them decide.

## A custom fragment (Flux, SD3)

Block-level bends on DiT models use `DiT Block Bending`, which takes a module node instead of JSON. Pass it to
`bend_run` as `fragment` (in place of `bends`): a small spec in the format of `references/canvas-presets.md`, where
an input linked to `["@model", 0]` receives the user's model and `out` names the model their samplers get.

```json
{"nodes": {
   "turn":  {"class_type": "Rotate Module (Bending)", "inputs": {"angle_degrees": 12}},
   "bend":  {"class_type": "DiT Block Bending", "title": "Bend (early joint blocks)",
             "inputs": {"model": ["@model", 0], "bending_module": ["turn", 0], "blocks": "double:0-6",
                        "stream": "img", "spatial": true, "t_start": 1.0, "t_end": 0.7, "strict": true}},
   "report": {"class_type": "PreviewAny", "inputs": {"source": ["bend", 1]}}},
 "out": ["bend", 0]}
```

Video models (WAN) take the default fragment: put `attention_bends` in the `bends` document
(`references/video.md`).
