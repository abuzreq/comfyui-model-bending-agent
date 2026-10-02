# Contributing

Thanks for helping. Bug reports, presets, measurements and wording fixes are all welcome.

## Ways to help

- **Report what broke.** Open an issue with your agent (Claude app, Claude Code, Codex, …), ComfyUI and
  ComfyUI-Model-Bending versions, the model, and what the agent said or did. The output of *"check my ComfyUI bending
  setup"* helps.
- **Presets for more models.** `presets/*.spec.json` are tested boards and sweeps. A preset for SDXL, Flux, SD3 or
  another video model, built and run on your machine, is very useful.
- **Safe ranges.** `data/safe_ranges_*.json` hold measured ranges per operation and layer group. Measurements for other
  checkpoints or architectures, with how you measured them, are welcome.
- **Video.** Video bending is new upstream and was tested only on tiny random models. What you see with real WAN
  weights (which blocks, steps and amounts give what) can go into `references/video.md`.
- **Wording.** Most users are artists. If the agent says something confusing, suggest plainer words for
  `references/plain-language.md`.
- **The knowledge base.** Bends and their effects go to the
  [Model Bending Knowledge Base](https://huggingface.co/datasets/abuzreq/model-bending-knowledge-base) as a pull
  request there (see the README's *Contribute your findings*).

## Layout

```
skills/comfyui-model-bending/   the skill: SKILL.md, scripts/, references/, presets/, data/, assets/
  scripts/comfy_canvas.py       ComfyUI client, spec builder, starting pictures, Agent-Bridge client (stdlib only)
  scripts/mcp_server.py         the same capabilities as MCP tools, and the in-chat pick board (scripts/ui/)
  scripts/metrics.py, media.py  image and video metrics, contact sheets, filmstrips
  scripts/kb.py, kb_local.py    the knowledge base format, and the user's own records
  scripts/animate.py            bend animations
.claude-plugin/                 Claude Code plugin and marketplace
gemini-extension.json           Gemini CLI extension
pyproject.toml                  the MCP server as a package (uvx)
packaging/mcpb/                 Claude Desktop extension: manifest, pyproject, lockfile, icon
tools/build_dist.py             builds dist/comfyui-model-bending.zip and dist/comfyui-bending.mcpb
tests/                          tests, against a stub server standing in for ComfyUI
evals/evals.json                scenarios describing how the agent should behave
```

## Rules the skill keeps

- **Self-contained.** The skill folder must work on its own once copied anywhere: no absolute paths, no references to
  files outside it, nothing machine-specific. Settings come from environment variables (`COMFYUI_URL`,
  `COMFY_BENDING_WORKDIR`).
- **Ask before acting on the user's machine.** The agent never installs nodes, downloads models, restarts ComfyUI,
  deletes outputs or overwrites the user's workflows without asking. Keep it that way in new instructions and tools.
- **Facts apart from interpretation.** In the knowledge base, measured facts and interpretations stay in separate
  files, and every interpretation names its author (a person, or the exact AI model).
- **Plain language in chat.** Instructions that make the agent talk to the user follow
  `references/plain-language.md`; technical detail goes in the session log.
- **Scripts and tools stay in step.** A capability added to the scripts gets its MCP tool (and the other way round),
  and a row in SKILL.md's runtime table.

## Development

Run the tests (no ComfyUI or GPU needed):

```
uv run --no-project --with "mcp>=2.2" --with pytest --with pillow --with numpy pytest tests
```

Try the scripts against your own ComfyUI:

```
python skills/comfyui-model-bending/scripts/comfy_canvas.py status
```

Build the release files (uses `npx @anthropic-ai/mcpb` when Node is installed):

```
python tools/build_dist.py
```

## Releasing

1. Bump the version in all five manifests: `pyproject.toml`, `packaging/mcpb/pyproject.toml`,
   `packaging/mcpb/manifest.json`, `.claude-plugin/plugin.json`, `gemini-extension.json`. The build stops if they
   differ. If the extension's dependencies changed, re-lock `packaging/mcpb/uv.lock`.
2. Add a `CHANGELOG.md` entry.
3. Push a tag `v<version>`. CI runs the tests, builds the skill ZIP and the `.mcpb`, and attaches them to a GitHub
   release.
