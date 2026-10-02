# The in-chat pick board

In the Claude app, the `comfyui-bending` extension can show the versions you made as a pick board right in the chat.
The artist sees big pictures with a plain caption each, chooses one or several, and sends their answer without
opening ComfyUI. Use it for every Mode B checkpoint, and as the default Mode C board in the app.

Contents:
- When to use it
- Showing a board
- Reading the answer
- Apps that cannot show boards
- Limits

Versions may be videos: the board plays each as a short silent loop (a small animated preview made by the
extension), next to the original, which may be the artist's starting picture (`original_url` = the `view_url` from
`upload_image`). Write captions about what happens over time.

## When to use it

| situation | use |
|---|---|
| choosing between versions you rendered (any level) | `show_board` |
| the artist wants to move sliders, rewire or re-run things themselves | a canvas board (`propose_workflow`, §4) |
| Claude Code and other agents with a shell | the contact sheet (`metrics.py sheet`), then ask in chat |

## Showing a board

Render the round first (baseline plus the versions, same seed), then call `show_board` once:

```json
{
  "title": "Round 2",
  "question": "I turned the model's core early in the painting. Which way should we go?",
  "original_url": "http://127.0.0.1:8188/api/view?filename=base_00001_.png&subfolder=&type=output",
  "candidates": [
    {"label": "A", "image_url": "…/api/view?filename=a_00001_.png…", "caption": "A gentle turn: the cliff swings right, your lighthouse stays."},
    {"label": "B", "image_url": "…", "caption": "A stronger turn: a new headland with an arch, same colours.", "recommended": true,
     "details": "rotate mid 90° · t 1→0.7 · MAE 34 · ok"},
    {"label": "C", "image_url": "…", "caption": "The strongest: it falls apart into a smooth blur."}
  ],
  "round": 2,
  "ask": ["keep_change", "strength", "direction"]
}
```

- `image_url`: the `/api/view` URLs a run returns (`run_workflow` / `get_run` → `images[].url`). Other addresses are
  refused.
- `caption`: one plain sentence about what changed, compared with the original (`references/plain-language.md`).
- `recommended`: on your pick only. `details`: the technical line; the artist sees it only if they open the details.
- `ask`: which extras to show. `votes` (👍/👎 per version), `keep_change` (composition, subject, palette, lighting,
  texture, style), `strength` (pull back / about the same / push further), `direction` (a free note). Default: all.
  Fewer is calmer: for a quick A/B, `["direction"]` is enough.
- 1–6 versions. The original is shown first and cannot be picked.

After the call, **stop and wait**. Do not describe the pictures again; the board shows them.

## Reading the answer

The artist's answer arrives as their next chat message, e.g. *"I like B and C. Keep the palette. "calmer sea" (my
answer on board board_3f2a…)"*. Then call `board_feedback(board_id)` for the exact fields:

| field | meaning |
|---|---|
| `picks` | the chosen labels (one or several) |
| `none` | "none of these" |
| `votes` | `{label: "up" | "down"}` |
| `keep`, `change` | aspects to keep / to change |
| `push` | −1 pull back, 0 about the same, +1 push further |
| `direction` | their note, in their words |
| `revision` | goes up each time they send again; pass it as `since_revision` to wait for a change |

Say back in one plain line what you understood, then build the next round. Several picks: along one axis, bisect
between them; different in kind, combine their bends (SKILL.md, Mode B). If the artist answers in the chat instead of
on the board, that answer counts; `board_feedback` then simply reports `waiting`.

## Apps that cannot show boards

- Where the host says it cannot show interactive views (some MCP clients), `show_board` returns the same versions
  as one labelled contact sheet plus a note. Show it with a plain caption per version and ask in chat.
- Some hosts accept the board but never display it, for example when they reach the extension through a relay. The
  board fetches its first picture as soon as it is displayed. If it never does, `board_feedback` reports
  `not_displayed` once the board is more than 20 seconds old. Then show the versions with `view_images`, one plain
  caption each, and ask in chat. Do the same whenever the artist says they cannot see a board. Do not show the board
  again in that conversation.

## Limits

- Each picture travels to the board as a JPEG of at most 640 px. Full resolution stays in ComfyUI's output folder.
- The board needs ComfyUI to be running while the artist looks at it (the pictures are fetched from it).
- One board per round. A new round means a new board; old boards stay in the chat as a record.
