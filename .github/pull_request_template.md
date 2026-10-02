**What this changes**

**How it was checked**
- [ ] `uv run --no-project --with "mcp>=2.2" --with pytest --with pillow --with numpy pytest tests` passes
- [ ] Tried against a real ComfyUI (say which model), if it touches rendering, presets or ranges
- [ ] The skill folder stays self-contained (no absolute paths, no files outside it)
- [ ] New script capabilities have their MCP tool and a row in SKILL.md's runtime table (or the other way round)
