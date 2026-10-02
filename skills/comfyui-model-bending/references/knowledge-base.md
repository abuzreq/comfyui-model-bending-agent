# The bend knowledge base

The knowledge base records what bending which part of which model produced. Artists consult it, and you use it to
pick starting recipes. There are two:

- **community**: a public dataset, `abuzreq/model-bending-knowledge-base` on Hugging Face.
  - The skill bundles a snapshot of its index (`data/kb/community/`).
  - `find_recipes(..., refresh=true)` fetches the latest version without a token.
- **mine**: the user's own rounds, recorded with `log_round`, kept in the work folder (`kb/`).

Contents:
- Asking first
- What a record holds: facts apart from interpretation
- Cells and evidence grades
- Reading `find_recipes` results to the user
- Logging rounds, and privacy

## Asking first

At intake, ask whether to start from the community knowledge base, the user's own earlier sessions, both, or neither.
Store the answer with `kb_sources`. Do not call `find_recipes` before they have chosen, and never after they said
neither.

## What a record holds: facts apart from interpretation

Each record folder keeps three kinds of content apart:

| file | holds | author |
|---|---|---|
| `record.json` | **facts**: model and checkpoint, sampler, steps, cfg, seed, size, route, the bends as actually sent (path, op, args, step window), the output | the producer |
| `measurements.json` | **numbers**: change vs the unbent baseline (MAE; latent cosine; LPIPS, DINOv2 and CLIP distances), degeneracy | each value names its method, and its model when a learned model computed it |
| `interpretations.jsonl` | **interpretation**: caption, what changed, keywords, effect tags, concept tags, notes, verdicts | every entry names its author: `{type: human, name}` or `{type: ai, model, prompt_version}` |

`findings/` holds claims about many records, such as a paper's results.
- A `cell` finding marks the cells its scope matches as study-backed.
- A `general` finding is context for the whole study (e.g. "layer type matters more than location").

Prompts and input images are kept only when their owner agreed. Otherwise a key stands in for them.

## Cells and evidence grades

A **cell** groups single-bend records that share all of these:
- model family
- layer group (in.hi … out.hi, mid, embed)
- sub-module kind (skip, res_branch, self_attn, cross_attn, ff, norm, proj, res_block, attn_block …)
- module type (Conv2d, Linear, LayerNorm …, or `*` when unknown)
- op
- amount bucket (e.g. multiply: zero, damp, boost, strong; rotate: slight, quarter, half)
- step window (all, early, early_to_mid, middle, mid_to_end, late; fractions of the steps, where early ≈ structure
  and late ≈ detail)
- route

Evidence grades:
- **anecdotal**: one seed and one prompt
- **multi-seed** / **multi-prompt**: more than one of either
- **replicated**: at least two seeds and at least two prompts
- **study-backed**: a cited finding covers the cell. It is shown alongside the grade.

## Reading `find_recipes` results to the user

Each result carries:
- the bend (`recipe`, in Apply Bends from JSON form)
- where it acts, the amount bucket and tested argument range, the step window
- `evidence`: grade, counts, the checkpoints it was tested on, sources
- `effects`: effect tags with their share of the cell's records and **who assigned them**, e.g.
  `["ai:claude-opus-5-5"]` or `["human:…"]`
- example record ids and image URLs (view them with `view_images`)
- cited findings, with their authors

When you pass these on:
- **Separate fact from interpretation.**
  - Facts: "this bend changed the picture a lot on 8 of 8 renders"
  - AI interpretation: "Claude described the results as fragmented"
  - Human interpretation: "the paper's authors found …"
- **Say how sure it is, in words.** "Seen across several prompts and seeds" (replicated) is different from "seen
  once" (anecdotal).
- **Name the model gap.** A note such as "no records on sd14 itself; tested on other sd1 checkpoints" means the
  results are leads: say so.
- **Treat results as a first sweep, not an answer.** Render the top 2–3 on the user's own prompt and seed, and judge
  as usual (§2).
- If nothing matched, say the knowledge base has nothing for this goal on this model yet. Then explore with §5 and
  log what works.

## Logging rounds, and privacy

`log_round(prompt_id, session, verdict, words, caption, change, effect_tags, agent_model, baseline_prompt_id)`:
- **Facts** are read from ComfyUI's history: the checkpoint loader, sampler, latent size, the bends (preferring
  `resolved_json`), and the output image.
- **Measurements:** pass `baseline_prompt_id` (the unbent run of the same setup) so the change is measured.
- **The user's interpretations:** `verdict` and `words` are recorded as theirs (human).
- **Your interpretations:** `caption`, `change` and `effect_tags` are recorded as AI, with `agent_model`, which is
  required. Use tags from `data/kb/vocab/effects.json`; others are dropped.
- **Privacy:**
  - The record is written to the user's own knowledge base only. The prompt and input image go to a private
    `private.json`.
  - `sharing.json` starts as `share_ok: false`. Nothing is uploaded.
  - Sharing to the community dataset is a separate step that the user must agree to, field by field.
