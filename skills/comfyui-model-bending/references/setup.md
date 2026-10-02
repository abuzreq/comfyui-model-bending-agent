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
- `uv run --no-project --with numpy --with pillow python scripts/metrics.py …` for image metrics.
- `uv run --no-project --with pillow python scripts/animate.py …` for bend animations. `ffmpeg` on the PATH gives
  `.mp4`; without it, write `.gif` or `.webp`.

No MCP server is needed.

## Claude desktop / claude.ai (chat)
Skill code runs in a sandbox that cannot reach your ComfyUI, so the app needs two installs: the skill, and its
**comfyui-bending** extension, which runs on your computer and talks to ComfyUI.

1. **Skill.** Turn on code execution (Settings → Capabilities). Then go to Customize → Skills → **+** → Create skill →
   Upload a skill, and choose `comfyui-model-bending.zip`.
2. **Extension.** Double-click `comfyui-bending.mcpb`, or install it from Settings → Extensions. Claude Desktop
   installs Python and the dependencies itself; there are no paths to edit.
   - Settings: **ComfyUI address** (default `http://127.0.0.1:8188`) and **Working folder**, where built
     workflows and animations are saved (default `~/.comfyui-model-bending`).
3. Start ComfyUI, open a new chat and ask Claude to check the ComfyUI bending setup. It should call
   `comfyui-bending:comfy_status`.
   - If Claude says the ComfyUI tools are not connected, or that its code "cannot reach localhost", the extension
     is not installed or is disabled. The skill alone cannot reach ComfyUI from the app's sandbox.
   - Animations are `.mp4` when `ffmpeg` is on the PATH, otherwise `.gif`.

## Other agents (Codex, GitHub Copilot, Cursor, Gemini CLI, …)
Agents that run shell commands on your machine use the scripts, like Claude Code: install the skill folder in the
agent's skills directory (for example `~/.agents/skills/`) and nothing else. Per-agent steps and one-command
installs: https://github.com/abuzreq/comfyui-model-bending-agent

Optional, for MCP tools instead of scripts: any MCP client can start the server with [uv](https://docs.astral.sh/uv/)
and no paths, as `"command": "uvx"` with
`"args": ["--from", "git+https://github.com/abuzreq/comfyui-model-bending-agent", "comfyui-bending-mcp"]`. Set
`COMFYUI_URL` and `COMFY_BENDING_WORKDIR` in the client's `env` to change the defaults.

## Security notes
- ComfyUI has no authentication. With `--listen` it is reachable from your network. With
  `--enable-cors-header="*"` any website you visit can call it.
- The Agent-Bridge token (`ComfyUI/user/agent_bridge/token`) is read by the scripts and the MCP server locally.
  Never paste it anywhere.
