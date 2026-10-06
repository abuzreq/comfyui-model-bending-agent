#!/usr/bin/env python3
"""The built comfyui-bending MCP server, started the way the Claude app sees it: as a client that can show the
in-chat pick board and picture box. Claude Code cannot display those views, so `run.py` plays the artist's side of
them (it opens the board, and stores the pick the simulated artist makes).

Only for the scenario suite. Started by run.py as:

    uv run --directory dist/_mcpb python evals/app_server.py

It reads the server from the folder it is run in (the unpacked extension), so what is tested is what was built.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))

import mcp_server  # noqa: E402

mcp_server.client_supports_apps = lambda ctx: True  # the Claude app shows boards; run.py stands in for its view

if __name__ == "__main__":
    mcp_server.main()
