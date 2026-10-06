# The first minutes of a session

People arrive in five ways: curious about what bending is, wanting to play, in the middle of a project of their own,
with a goal, or with a bend in hand. Meet each where they are, and get a result in front of them before asking how
they want to work.

Steps name the MCP tools. With scripts, use the equivalents in SKILL.md, "Scripts and tools, step by step"
(`inspect_run` is `comfy_canvas.py run-setup`, `bend_run` is `bend-run`, `view_images` is `metrics.py sheet`,
`last_runs` is `last-run`, `list_models` is `comfy_canvas.py node CheckpointLoaderSimple`, which lists the
installed checkpoints).

Contents:
- Which way in
- What is asked, and when
- Intro: what bending is, with examples
- Surprise round
- Their own project
- A bend they bring
- After the first round

## Which way in

Run the setup check (SKILL.md §0), then read the first message and pick one way in. Guess when the message makes it
clear. Ask only when it does not.

| the first message sounds like | way in | the first thing they get |
|---|---|---|
| "what is this?", "what can I do with it?", "how does bending work?" | **Intro** | a short map of what can be bent and 3–4 example pictures, then a choice of what to try |
| "surprise me", "just play", "anything", "show me something weird" with no direction | **Surprise** | two random bends within safe limits and one wild card, on their prompt or a default one |
| "my workflow", "my project", "what I have open in ComfyUI", "this LoRA of mine" | **Own project** | their own last picture, with three bent versions of it |
| a goal or a prompt: "stormier sea, same lighthouse", "bend this prompt" | **Goal** | the core loop (§2), starting at once |
| a bend as JSON, pasted or in a file: "make this bend", "I copied this from the web UI" | **A given bend** | what the bend does in one plain line, then that bend next to the unbent picture |
| none of these ("hi", "let's start") | ask the one question below | — |

The one question, when it is unclear (wording: `references/plain-language.md`, "The opening question"): offer
four ways in (a pasted bend is never unclear, so it is not among them), in one short message, with AskUserQuestion
if available. That message holds nothing else: no other question, no setup report, no working defaults.

A message can fit two ways in ("I have a workflow, surprise me"): combine them (a surprise round on a copy of their
run).

## What is asked, and when

A result comes before the questions about how to work. Ask only for what the first render needs, and state the
rest.

| | Goal | Intro | Surprise | Own project | A given bend |
|---|---|---|---|---|---|
| prompt or picture | from their message | when they choose what to try | theirs if given, otherwise a default prompt: say which, and that they can give their own | their run has it | their last run has it; with no run, theirs if given, otherwise the default prompt: say which |
| model | named, or pick and say which | pick and say which | pick and say which | their run has it | their last run has it; with no run, the model the bend was made for |
| knowledge source (§1d) | before the first sweep only if they asked for recipes or what others found; otherwise with the first result | with the first result | with the first result | with the first result | with the first result |
| depth and autonomy (§1d) | with the first result | with the first result | with the first result | with the first result | with the first result |
| render budget, seed policy | stated, not asked | — | stated | stated | stated |

- **Picking a model** when they did not name one, in this order: the `base_sd15` preset's own model if it is
  installed (`list_models`), because the preset then runs as it is; otherwise the checkpoint of their newest run
  (`last_runs`); otherwise the first installed SD1.5-type checkpoint. Say which one in a clause ("on Dreamshaper,
  a small fast model you have installed"). A checkpoint other than the preset's needs the preset's loaders and
  sampler settings adapted (`references/canvas-presets.md`, the base blocks).
- **The default prompt** when they gave none: the `base_sd15` preset's, "a lighthouse on a sea cliff at dusk, oil
  painting". Say that you used it and that they can give their own.
- **Until they choose a knowledge source**, use only the built-in recipes (§5, §6). Using none is not a choice
  made for them; calling `find_recipes` would be.
- **Defaults you state in one clause**, when the first round starts: three versions a round on one starting point,
  stopping for their pick (L1 + B, fixed seed). Never ask about L1–L3 or A–C before a first result unless they
  brought it up.
- If they answer the opening question with detail ("surprise me, on SDXL, and run on your own"), take it all.

## Intro: what bending is, with examples

For someone who does not know what bending is or what they could bend. It needs no render, so it also works while
ComfyUI is not running: say that pictures of their own have to wait until it is, and give the intro anyway.

1. **Say what bending is in three lines** (wording: `references/plain-language.md`, "What bending is").
2. **Give the map of what can be bent**: where, when, how, and what (same file, "The map"). Keep it to the short
   form unless they ask for more.
3. **Show examples.** `intro_examples(arch)` (script: `kb_local.py intro --arch sd15`) returns a few examples from
   the community knowledge base; the first three cover the early stages, the core and the late stages. Use `sd15`
   unless you already know their model. Show them as one sheet (`view_images`; script: `metrics.py sheet` with two
   columns), the unbent picture next to the bent one where `unbent_url` is given, labelled in plain words ("early
   stages, weakened"). The pictures are fetched from the knowledge base on Hugging Face.
   - Say where they come from: examples other people made, on the models listed in `tested_on`.
   - The first ones are hand-picked (`data/kb/intro.json`) and carry a `caption`; the rest come from the index
     and carry `looked_like` tags and `keywords`. All three are interpretations: `caption_by` / `by` says whose
     (`ai:<model>`: "described by an AI model"; `human:<name>`: that person). The recipe and the pictures are
     facts. `evidence` says how often it was seen; `when` says at what point in the painting.
   - A `note` means there are no examples for this model type: say that these are from SD1.5-type models and theirs
     may respond differently.
   - If the pictures cannot be loaded, give the map alone.
4. **Hand over the choice**: try one of these on a prompt of theirs, be surprised, or bend something of their own.
   - "That one" → its `recipe` is the first candidate, next to two neighbours (weaker and stronger), on their
     prompt. They chose that one recipe; still ask the knowledge-source question before any `find_recipes`.

The examples are illustrations. Showing them is not a choice of knowledge source, and `intro_examples` stores none.

## Surprise round

For playing without a goal. The draw is random on purpose: this is the one case where a bend is picked without a
hypothesis, and it is always labelled as a surprise.

1. `surprise_bends(arch)` (script: `python scripts/surprise.py --arch sd15`). It returns three candidates: two
   inside the measured safe ranges and one **wild card** past them, each on a different part of the model (early
   stages, core, late stages) with a different kind of nudge.
2. Render them as bent copies of one unbent run, so model, prompt and starting point are the same:
   - **the unbent picture**: build and run the `base_sd15` preset with their prompt or the default one (model: see
     "Picking a model"). For someone mid-project it is their own last run instead (`inspect_run`).
   - **each candidate**: `bend_run(name, bends=<its bends_json>, prompt_id=<that run>, clamp=<its clamp>)`, then
     `run_workflow(name)`. `bend_run` adds the safe-range table, the report box and the same seed by itself.
3. Check reports and metrics as always (§2 steps 4–5).
4. Show all of them, the wild card labelled "wild card". **A wild card that fell apart is still shown**: say that
   it broke, in plain words, and let them judge. A safe candidate that fell apart is shown too, with the same
   honesty. Never present either as a clean result.
5. Captions say what they will see, not what was bent (on the board, `what_was_bent` goes in `details`). In the
   message after their pick, or when they ask, say what each one was: "B was the late stages (style), shaken up."
   That is the explanation the round owes them.
6. Next:
   - a pick → the normal loop from that bend (§2): weaker and stronger, a neighbouring part, another time window.
   - "again" → a new draw with `avoid` set to the earlier candidates' `key`s. The returned `seed` repeats a draw.
   - log what they kept (`log_round`), with `words` in their own terms.

Models without a safe-range table (Flux, SD3, WAN): the tool says there is nothing to draw from. Pick two or three
bends near "no change" by hand (§5), say that these are your picks and not a random draw, and widen from what you
see.

Offer the surprise round at the start. Later in a session, run one whenever they ask ("surprise me again"), but do
not add it to every choice.

## Their own project

For someone who already has a ComfyUI workflow and wants to see what bending does to it. Follow
`references/own-workflow.md`: read their last run, say back what it is made of, and show three bent versions of
their own picture. Their workflow is never changed.

## A bend they bring

Someone pastes a bend, or points to a file with one: copied from the Model-Bending web UI ("Copy Bends"), taken
from an `Apply Bends from JSON` node, from the knowledge base, a paper or a friend. They want that bend made, not a
search around it.

1. **Read it**: `check_bends(bends, arch)` (script: `python scripts/bendjson.py BENDS --arch sd15`). Pass what
   they gave as it is: the document, a list of bends, one bend, or the pasted text with its ``` fence. Give `arch`
   when you know the model, so each bend is placed in the model and its amount is checked.
2. **Say back what it does**, one plain line per bend, from `summary` ("the model's core, turned by 90°, early in
   the painting"). This is the moment they learn what their JSON means.
3. **Settle what is wrong before rendering.**
   - `errors` are bends the node would refuse (a misspelled op, no path). Say which and what you think was meant
     ("`multipy`: did you mean multiply?"), and wait for their answer. Do not drop or fix a bend silently.
   - `problems` are things the node would ignore (an unknown key or argument). Mention them in one line.
   - Notes on a bend ("outside the range that usually keeps a picture together", "this layer is not
     image-shaped") are worth one plain sentence each: they are what an artist cannot see in the JSON.
4. **Render exactly that bend** next to the unbent picture: `bend_run(name, bends=<bends_json>, prompt_id, clamp)`,
   then `run_workflow(name)`.
   - **Which run to copy**: their own last run when they have one (`inspect_run`); otherwise one run of the
     `base_sd15` preset with their prompt, on the model the bend was made for.
   - **`clamp`**: they asked for this bend, so keep its amounts: `clamp="none"` when an amount is outside the safe
     range and they want it as given (say that it may fall apart), `safe` otherwise.
   - **A run that is already bent** (`inspect_run` lists `bending_nodes`: a run made in the web UI always does):
     pass `replace_bends=true`, so the given bend stands in for the run's own. The same call with no bends gives
     their run unbent: render that once as the original to compare with.
5. **Show both** with one line on what changed. Then offer the next step in their terms: weaker or stronger, the
   same bend on another picture, keep it as a workflow (`propose_workflow(name)` when `copy` is `full`; otherwise
   say where to add the box by hand, as in `references/own-workflow.md`), or an animation of it growing
   (`start_animation(name=…)`).

A bend from the knowledge base or the intro examples (`recipe`) is the same case: it is already in this format.
Its `kb` key says where in the knowledge base it came from: pass it along unchanged. Model-Bending 0.3.1+ keeps it
in its `resolved_json`. For older plugins the tools drop it from what they send to ComfyUI on their own
(`comfy_status` shows `keeps_kb`), so you never need to remove it yourself.

### A tray link from the navigator

The Model Bending Navigator lets people collect bends in a tray and share it as one link (`…/#tray=…`).
1. **Read it** with `open_tray(link, arch)` (script: `python scripts/bendjson.py --tray LINK --arch sd15`). Each
   bend is checked on its own, as in step 1 above.
2. **Say back** one plain line per bend from its `summary`. When a bend has a `link`, offer it: that page shows the
   bend's before/after examples and how often it broke.
3. **Render each bend separately** on their run: one `bend_run` per bend, each with its own name (its `bends_json`),
   then show them together on a board with the unbent picture. Every bend in a tray was tested alone; do not
   combine them unless they ask, and then say the combination has not been seen before.
4. To give bends back to keep or share (the round they liked, say), `tray_link(bends)` makes a link of the same
   kind (script: `bendjson.py BENDS --to-tray`).

## After the first round

At the end of the message that shows the first result, and not before, offer the dials in one or two lines:

> I'll keep showing you three versions and stopping for your pick. I can also dig into *why* something happens, or
> run a few rounds on my own and report back: just say so.

If the knowledge-source question has not been asked, add it here, once (§1d). If they do not answer it, carry on
with the built-in recipes and do not call `find_recipes`. Then continue with the core loop (§2) in the mode they
chose.
