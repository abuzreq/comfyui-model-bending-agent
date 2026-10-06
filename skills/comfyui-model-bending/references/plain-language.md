# Talking with artists in plain language

Most people using this skill are artists, not engineers. Use this file to turn technical findings into words they can
act on. The technical record (layer paths, amounts, metrics, JSON) still goes in the session log in full.

Contents:
- Translation table
- The opening question
- What bending is
- The map: what can be bent
- Describing a result
- Asking for a choice
- Asking for their picture
- When something needs fixing on their machine

## Translation table

Use the plain phrase in conversation. Add the technical term in brackets only when it helps the user find something
on the canvas, or when they ask.

| technical | say instead |
|---|---|
| bend, activation bend | a nudge inside the model while it paints |
| layer, block, site | a stage of the model |
| in.hi / in.mid / in.lo | the early stages: fine detail / layout / overall composition |
| mid | the model's core, where it decides what the scene is |
| out.lo / out.mid / out.hi | the late stages: rebuilding the composition / style and handling / texture and fine detail |
| attn, self-attention | how parts of the picture relate to each other |
| cross-attention | how closely the picture follows the prompt |
| skip connection | the shortcut that carries early detail to the end |
| rotate / scale / multiply / add_noise | turn / zoom / strengthen or weaken / shake up |
| amount, gain | strength |
| t-window, structure / style / detail window | when in the painting process: early (layout), middle (style), late (fine detail) |
| step | one pass of the painting process |
| seed | the random starting point (same seed = fair comparison) |
| MAE | how much it changed: barely / clearly / completely |
| degenerate, blob, noise flags | it fell apart into mush or static |
| baseline | the picture with no nudge |
| sweep | a few versions side by side |
| denoise | how much of the original image is repainted ("how closely to follow your picture") |
| start image, img2img | starting from your picture |
| image-to-video, I2V | bringing your picture to life |
| attention map (video) | where the model looks for each word (or for your picture) in every frame |
| temporal shift / blur / frame ramp | parts of the clip lagging behind / motion smeared / a change that grows over the clip |
| frames, fps | the stills that make up the clip / how many per second |
| flicker / frozen flags | it jitters from frame to frame / the motion stopped |
| VRAM | graphics-card memory |
| checkpoint | the model |
| node, workflow, canvas | a box in ComfyUI / the whole setup / the ComfyUI screen |
| report, resolved, clamped | what the model actually did / it was kept within a safe strength |
| evidence grade causal / correlational / anecdotal | tested and repeatable / a strong hint / seen once |
| surprise round, random draw | a few nudges picked at random, within safe limits |
| wild card, past the safe range | one pushed too far on purpose: it may fall apart |
| safe range, clamp | the strengths that usually keep a picture together |
| their run, history | the last picture you made in ComfyUI |
| bent copy, splice | a copy of your workflow with one new box in it |

## The opening question

Ask it only when the first message does not show which way in they want (`references/first-session.md`). One short
message, nothing else in it:

> How would you like to start?
> - **Show me what bending is**: a short tour with example pictures.
> - **Surprise me**: I pick a few nudges at random and we see what comes out.
> - **I'm working on something**: bend a copy of a workflow you already have in ComfyUI.
> - **I know what I'm after**: tell me the picture or the change you want.

## What bending is

Three lines, in their terms. For example:

> An image model paints in stages: it settles the layout first, then the style, then the fine detail. Bending means
> reaching in while it paints and nudging one of those stages: turning it, zooming it, strengthening or weakening
> it, or shaking it up. The prompt stays the same; what changes is how the model handles it, which gives results
> a prompt cannot ask for.

Do not explain networks, layers or activations unless they ask. If they know the field (they say "UNet", "layer",
"activation"), drop the metaphor and use their words.

## The map: what can be bent

Four short lists. Give the one-line form of each; expand a list only when they ask about it.

- **Where** in the model:
  - the early stages: fine detail, then layout, then overall composition
  - the core: what the scene is
  - the late stages: rebuilding the composition, then style and handling, then texture and colour
- **When** in the painting: early (layout), middle (style), late (fine detail). The same nudge does different things
  at different moments.
- **How**: turn, zoom, strengthen or weaken, shift, shake up. Rougher ones too: keep only the edges, swell or eat
  away the strong areas.
- **What**: an image model, a video model, a picture of your own as the starting point, or a workflow you already
  have, LoRAs and all.

One sentence ties it together: "A nudge is a *where*, a *when* and a *how*: for example, turning the core early in
the painting rebuilds the scene but keeps the colours."

## Describing a result

Lead with what they will see, then why it matters, then (optionally) what you would try next. One or two sentences
per image. Compare with the original picture, not with numbers.

> **B** turns the core of the model by 90°, early in the painting. The cliff becomes a new headland with an arch, and
> the colours stay the same. This is the one I'd keep.

Not: "B: rotate mid 90°, t 1→0.7, MAE 34, ok."

For explanations ("why does this happen?"), give the finding first and how sure you are in words ("tested on two
starting points, it holds"), then the evidence. Keep the full evidence table for the write-up, and offer it.

## Asking for a choice

- Offer the choice in their terms: "A keeps your lighthouse, B invents a new coastline, C goes abstract. Which way?"
- Accept answers in their terms ("B, but calmer") and translate them into the next sweep yourself.
- Do not ask them to pick layers, ops or numbers unless they want that level of control. Offer it once: "I can also
  show you the knobs, if you like."

## Asking for their picture

A picture attached to the chat reaches you but not ComfyUI. In the Claude app, open the picture box
(`ask_for_picture`) and say one line:

> I can see your picture, but ComfyUI needs its own copy: drop it into the box below (or click the box to choose the
> file).

When the box is not available, offer both other ways at once, with the click path:

> ComfyUI needs its own copy of your picture. Either find the file, hold **Shift** and right-click it, choose **Copy
> as path**, and paste that here (on a Mac: right-click it, hold **Option**, and choose **Copy "…" as Pathname**).
> Or drag the picture into a **Load Image** box in ComfyUI and tell me its name.

If it is already in ComfyUI (they used it in a Load Image box), list what is there and let them pick by name. Do not
suggest the Picture folder setting as a way to load a picture: it only limits where pictures may come from.
Then ask how closely to follow it: "keep it and just restyle", "keep the composition, repaint it", or "use it as a
loose starting point".

## When something needs fixing on their machine

Say what is wrong, what it means for them, and the one thing to do, in that order:

> ComfyUI isn't running, so I can't make pictures yet. Start it the way you usually do (for example
> `run_nvidia_gpu.bat` in the ComfyUI folder) and tell me when the page has opened.

Give click paths ("Workflows → agent_bending → open it → press Run"), not file paths, and never paste error traces
unless asked. If the fix needs installing something, say what it is for in one line and ask first.
