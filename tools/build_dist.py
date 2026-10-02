#!/usr/bin/env python3
"""Build the distributables of the comfyui-model-bending skill into dist/:

  comfyui-model-bending.zip   the skill, for Claude app uploads (Customize > Skills > Upload a skill)
  comfyui-bending.mcpb        its MCP server as a Claude Desktop extension (double-click to install)

The extension bundles the skill's scripts/, presets/ and data/ with packaging/mcpb/ (manifest.json,
pyproject.toml, uv.lock, icon.png). Its tool list is read from scripts/mcp_server.py, so it cannot drift. The .mcpb is
validated and packed with the official CLI (`npx @anthropic-ai/mcpb`) when Node is available; otherwise it is
zipped directly (the format is a zip with manifest.json at the root).

  python tools/build_dist.py [--no-cli]

It first checks that every manifest carries the same version (see VERSIONED).
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "comfyui-model-bending"
PACKAGING = ROOT / "packaging" / "mcpb"
DIST = ROOT / "dist"
STAGE = DIST / "_mcpb"
BUNDLED = ("scripts", "presets", "data")  # skill folders the MCP server reads at runtime
# files that state the release version; a release bumps them all
VERSIONED = ("packaging/mcpb/manifest.json", "packaging/mcpb/pyproject.toml", "pyproject.toml",
             ".claude-plugin/plugin.json", "gemini-extension.json")


def _files(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            yield p


def build_skill_zip() -> Path:
    out = DIST / "comfyui-model-bending.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in _files(SKILL):
            z.write(p, f"{SKILL.name}/{p.relative_to(SKILL).as_posix()}")
    return out


def _model_visible(d: ast.expr) -> bool:
    """@tool, or @ui_tool(...) unless it is board-only (visibility=["app"], hidden from the model)."""
    if isinstance(d, ast.Name):
        return d.id == "tool"
    if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "ui_tool":
        vis = next((k.value for k in d.keywords if k.arg == "visibility"), d.args[0] if d.args else None)
        return vis is None or "model" in ast.literal_eval(vis)
    return False


def tool_list(server: Path) -> list[dict]:
    """Every tool in mcp_server.py the model can call, with the first sentence of its docstring."""
    tree = ast.parse(server.read_text(encoding="utf-8"))
    tools = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(_model_visible(d) for d in node.decorator_list):
            doc = " ".join((ast.get_docstring(node) or "").split())
            first = doc.split(". ")[0].rstrip(".")
            tools.append({"name": node.name, "description": first + "." if first else ""})
    if not tools:
        sys.exit(f"no @tool functions found in {server}")
    return tools


def stage_extension() -> dict:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    for name in BUNDLED:
        for p in _files(SKILL / name):
            dst = STAGE / name / p.relative_to(SKILL / name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    shutil.copy2(PACKAGING / "pyproject.toml", STAGE)
    shutil.copy2(PACKAGING / "uv.lock", STAGE)  # pins the tested dependency versions
    shutil.copy2(PACKAGING / "icon.png", STAGE)
    manifest = json.loads((PACKAGING / "manifest.json").read_text(encoding="utf-8"))
    manifest["tools"] = tool_list(SKILL / "scripts" / "mcp_server.py")
    (STAGE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def check_versions() -> str:
    """The one version every manifest states; exits listing them when they differ."""
    found = {}
    for rel in VERSIONED:
        text = (ROOT / rel).read_text(encoding="utf-8")
        if rel.endswith(".json"):
            found[rel] = json.loads(text).get("version")
        else:
            line = next((ln for ln in text.splitlines() if ln.startswith("version = ")), "")
            found[rel] = line.split("=", 1)[1].strip().strip('"') if line else None
    if len(set(found.values())) != 1:
        sys.exit("versions differ; bump them together:\n" + "\n".join(f"  {v}  {k}" for k, v in found.items()))
    return next(iter(found.values()))


def _npx(*args: str) -> subprocess.CompletedProcess | None:
    npx = shutil.which("npx")
    if not npx:
        return None
    return subprocess.run([npx, "-y", "@anthropic-ai/mcpb", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def pack_extension(use_cli: bool) -> Path:
    out = DIST / "comfyui-bending.mcpb"
    out.unlink(missing_ok=True)
    if use_cli:
        r = _npx("validate", str(STAGE / "manifest.json"))
        if r is not None:
            print(r.stdout.strip() or r.stderr.strip())
            if r.returncode:
                sys.exit("manifest validation failed")
            r = _npx("pack", str(STAGE), str(out))
            print(r.stdout.strip()[-1500:] or r.stderr.strip()[-1500:])
            if r.returncode or not out.exists():
                sys.exit("mcpb pack failed")
            return out
        print("npx not found; zipping the extension without the mcpb CLI")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in _files(STAGE):
            z.write(p, p.relative_to(STAGE).as_posix())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-cli", action="store_true", help="zip the .mcpb directly instead of using npx mcpb")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    version = check_versions()
    print(f"version {version}")
    DIST.mkdir(exist_ok=True)
    skill_zip = build_skill_zip()
    manifest = stage_extension()
    mcpb = pack_extension(not a.no_cli)
    print(f"{skill_zip.relative_to(ROOT)}  ({skill_zip.stat().st_size // 1024} KB)")
    print(f"{mcpb.relative_to(ROOT)}  ({mcpb.stat().st_size // 1024} KB, v{manifest['version']}, "
          f"{len(manifest['tools'])} tools)")


if __name__ == "__main__":
    main()
