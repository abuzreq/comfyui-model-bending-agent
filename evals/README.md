# Scenario suite: how does the agent behave?

Unit tests (`tests/`) check the scripts. This suite checks the *behaviour*: it plays realistic conversations with
the skill and gives you a report to read.

## What a run does

Each scenario in `scenarios/` is a small play with three roles:

| role | who plays it | what it sees |
|---|---|---|
| **the agent** | a fresh, headless Claude Code session in an empty folder | only the **built** skill (`dist/comfyui-model-bending.zip`) and the **built** MCP server (`dist/_mcpb`). No shell and no web, and it is told it runs in the Claude app's chat (otherwise it knows it is in a terminal and avoids the pick board). Your real ComfyUI renders real pictures. |
| **the artist** | a second model with a persona and a goal | what a person would see: the agent's messages, the in-chat pick board, the pictures. It looks at the pictures before it picks. |
| **the judge** | a third model, after the conversation | the whole record, including the agent's tool calls. It marks each expected behaviour *met / partly / not met / not reached*, with evidence. |

Claude Code cannot draw the Claude app's in-chat views (pick board, picture box). The runner plays the artist's
side of them: it opens the board, shows its pictures to the artist and stores their pick, as the view would
(`app_server.py` starts the built server in that mode).

## Running it

Needs: ComfyUI running with ComfyUI-Model-Bending and the SD1.5 LCM models the presets use, the `claude` CLI signed
in, `uv`, and Python 3.10+.

```bash
python tools/build_dist.py
```

```bash
python evals/run.py --list
```

```bash
python evals/run.py --tag smoke
```

```bash
python evals/run.py --only goal-stormier,own-workflow
```

```bash
python evals/run.py
```

In a Claude Code session, ask for it in words ("run the smoke scenarios of evals/run.py in the background and show
me the report when it is done"). A scenario takes two to fifteen minutes; a full run takes one to two hours and is
best started in the background. Every turn is a full model call with tools: with a large model, count a few
dollars of usage per scenario (the two trial scenarios cost about $2 each).

Useful options:

| option | what it does |
|---|---|
| `--build` | build the skill and extension first, so the current files are what is tested |
| `--agent-model NAME`, `--effort LEVEL` | the model and effort of the agent under test (default: your CLI's default) |
| `--artist-model`, `--judge-model` | default `sonnet` for both |
| `--max-turns N` | cap every scenario's length |
| `--no-judge` | record the conversations only |
| `--rejudge RUN`, `--report RUN` | judge an earlier run again, or only rebuild its report |

## What you get

`evals/runs/<date-time>/`:

- **`report.html`**: open it in a browser. A table of all scenarios, then for each one: the checklist with the
  judge's verdicts and evidence, what the judge also noticed, a short "for you to look at" list, and the whole
  conversation with the pick boards, the pictures and (folded) every step the agent took behind the scenes.
- `summary.md`: the table, and every "not met" and "partly" in one list.
- `<scenario>/record.json`: the conversation as data. `<scenario>/pictures/`: every picture. `<scenario>/raw/`:
  the raw event streams of the agent, the artist and the judge.

The checklist is a model's reading, not a measurement. Read the conversation where a verdict surprises you.
Two runs of the same scenario will differ: the agent, the artist and the draw of a surprise round are not fixed.

## What a run leaves behind

- Pictures and workflows in ComfyUI: the agent's renders in `output/agent_bending/`, the pictures made for the
  simulated artist in `output/scenario_artist/`, entries in ComfyUI's history, and proposed workflows under
  `workflows/agent_bending/` or `workflows/agent_bridge/`. Nothing is written next to your own pictures.
- One Claude Code session folder per scenario under `~/.claude/projects/` (the agent's conversation).
- Nothing in your own bend knowledge base: each scenario keeps its state in its own run folder.

## Limits

- **Real ComfyUI only.** Cases this machine cannot show (ComfyUI not running, an old Model-Bending install, real
  WAN video renders) are not covered. `no-video-models` assumes there are no WAN models installed.
- **MCP tools only.** The script route (Claude Code with a shell) is not exercised here.
- **The canvas is not played.** Nobody clicks in ComfyUI, so `sliders-on-canvas` tests the hand-over only.
- The simulated artist is patient and literate in a way real people are not. Treat a pass as "no obvious
  problem", not as proof.

## Adding a scenario

Copy a file in `scenarios/` and change it. The file name must be the `id`.

| field | meaning |
|---|---|
| `title`, `tags`, `why` | for the report; `why` says what the scenario is there to show |
| `first_message` | the artist's opening message, word for word |
| `artist` | `who`, `wants`, `style`, `rules` (what to answer when), `ends_when` |
| `max_turns` | how many messages the artist may send |
| `setup` | optional. `{"do": "user_run", "kind": "plain" \| "bent", "prompt": …}` puts a run of "their own" into ComfyUI's history; `{"do": "picture", "prompt": …, "name": …}` gives the artist a picture file |
| `expected` | behaviours the judge checks, one sentence each. Write what can be seen in the record. |
| `watch_for` | questions for you, the human reader; not judged |

`evals.json` in this folder is older and simpler: single messages with expected behaviours, without a runner.
