# Setup: what the skill needs on the machine that runs ComfyUI

## Everywhere
1. **ComfyUI** (0.18+ recommended, frontend 1.40+) running locally. The default address is `http://127.0.0.1:8188`;
   set `COMFYUI_URL` otherwise.
2. **ComfyUI-Model-Bending** in `ComfyUI/custom_nodes/`.
   - Check it is the agent-ready release: `Apply Bends from JSON` has `report` and `resolved_json` outputs.
3. **Optional: ComfyUI-Agent-Bridge** in `custom_nodes/`. It adds propose-to-canvas, the feedback and switch nodes,
   and the event stream.
   - Without it, the skill falls back to saving workflows into the Workflows sidebar and reading runs from history.
4. **For video (optional):** ComfyUI-Model-Bending 0.3 or newer, and WAN model files in ComfyUI's `models/`
   folders. The light text-to-video set (about 8 GB of graphics memory): `diffusion_models/wan2.1_t2v_1.3B_fp16`,
   `text_encoders/umt5_xxl_fp8_e4m3fn_scaled`, `vae/wan_2.1_vae`. Image-to-video (WAN 2.1 I2V 14B, 16 GB+ or a GGUF
   quant) also needs `clip_vision/clip_vision_h`. Model-Bending's `workflows/video/*.json` link to the downloads.
   The skill never downloads models itself.
5. **Optional: comfy-mcp** (`uv tool install comfy-mcp`). It covers installation, models and lifecycle tasks. The
   skill does not require it.

## Claude Code (terminal or the desktop app's Code tab) on that machine
Place the skill folder in `~/.claude/skills/` (every project) or in `<project>/.claude/skills/`. The scripts run
directly:
- `python scripts/comfy_canvas.py …` uses only the standard library.
- `uv run --no-project --with "numpy<3" --with "pillow<13" python scripts/metrics.py …` for image metrics.
- `uv run --no-project --with "pillow<13" python scripts/animate.py …` for bend animations. `ffmpeg` on the PATH gives
  `.mp4`; without it, write `.gif` or `.webp`.

No MCP server is needed.

## Claude desktop / claude.ai (chat)
Skill code runs in a sandbox that cannot reach your ComfyUI, so the app needs two installs: the skill, and its
**comfyui-bending** extension, which runs on your computer and talks to ComfyUI.

1. **Skill.** Turn on code execution (Settings → Capabilities). Then go to Customize → Skills → **+** → Create skill →
   Upload a skill, and choose `comfyui-model-bending.zip`.
2. **Extension.** Double-click `comfyui-bending.mcpb`, or install it from Settings → Extensions. Claude Desktop
   installs Python and the dependencies itself; there are no paths to edit.
   - Settings: **ComfyUI address** (default `http://127.0.0.1:8188`), **Picture folder** (optional: the only folder
     starting pictures are read from), and **Agent-Bridge token** (optional: only for a ComfyUI on another machine).
     There is no folder to choose: animations are saved in ComfyUI's output folder (`output/agent_bending/`), and
     the extension's own files in the system's app-data folder.
3. Start ComfyUI, open a new chat and ask Claude to check the ComfyUI bending setup. It should call
   `comfyui-bending:comfy_status`.
   - If Claude says the ComfyUI tools are not connected, or that its code "cannot reach localhost", the extension
     is not installed or is disabled. The skill alone cannot reach ComfyUI from the app's sandbox.
   - Animations are `.mp4` when `ffmpeg` is on the PATH, otherwise `.gif`. They are saved in ComfyUI's output
     folder, inside `agent_bending`.

## Other agents (Codex, GitHub Copilot, Cursor, Gemini CLI, …)
Agents that run shell commands on your machine use the scripts, like Claude Code: install the skill folder in the
agent's skills directory (for example `~/.agents/skills/`) and nothing else. Per-agent steps and one-command
installs: https://github.com/abuzreq/comfyui-model-bending-agent

Optional, for MCP tools instead of scripts: any MCP client can start the server with [uv](https://docs.astral.sh/uv/)
and no paths, as `"command": "uvx"` with
`"args": ["--from", "git+https://github.com/abuzreq/comfyui-model-bending-agent@v0.4.0", "comfyui-bending-mcp"]`. Set
`COMFYUI_URL` in the client's `env` to change the default.

## Settings

| variable | default | used for |
|---|---|---|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | where ComfyUI runs |
| `COMFY_BENDING_WORKDIR` | the system's app-data folder (`~/.comfyui-model-bending` if an earlier version made it) | the skill's own files: built workflows, boards, caches, sessions and the user's own knowledge base (animations go to ComfyUI's output folder) |
| `AGENT_BRIDGE_TOKEN` | read from `ComfyUI/user/agent_bridge/token` when ComfyUI runs on this computer | Agent-Bridge authentication (set it for a ComfyUI on another machine) |
| `BEND_KB_DATASET` | `abuzreq/model-bending-knowledge-base` | the Hugging Face dataset behind the community knowledge base |
| `COMFY_BENDING_PICTURE_DIRS` | not set (any folder) | if set, the only folders starting pictures are read from (separated like `PATH`) |
| `BEND_KB_REVISION` | `main` | the dataset version a knowledge-base refresh fetches: a branch, tag, or a commit to pin it |

## When something goes wrong
- "answered with a redirect": ComfyUI sits behind a proxy or tunnel that redirects; set the ComfyUI address to the
  final address (for example the `https://` one). A login page in front of ComfyUI is not supported.
- "not on this computer … AGENT_BRIDGE_TOKEN": the ComfyUI is on another machine; the user pastes that machine's
  `ComfyUI/user/agent_bridge/token` into the extension's Agent-Bridge token setting (or `AGENT_BRIDGE_TOKEN`).
- "pictures from this computer are not sent there": the ComfyUI is on another machine; the user uploads the picture
  in ComfyUI and you pick it with `list_input_images`.
- The picture box does not appear (`ask_for_picture` says the app cannot show it, or `picture_received` reports
  `not_displayed`): ask for the file's path, or for the picture in a Load Image box (`references/plain-language.md`).

## Security notes
- ComfyUI has no authentication. With `--listen` it is reachable from your network. With
  `--enable-cors-header="*"` any website you visit can call it.
- The Agent-Bridge token (`ComfyUI/user/agent_bridge/token`) is read by the scripts and the MCP server locally.
  Never paste it anywhere.
