#!/usr/bin/env python3
"""MCP server (stdio) exposing this skill's scripts as tools. Use it where the scripts cannot reach ComfyUI
themselves, e.g. the Claude desktop app, whose skill code runs in a sandbox. Claude Code can call the scripts directly
and does not need it.

It runs on the machine that can reach ComfyUI (default http://127.0.0.1:8188; set COMFYUI_URL otherwise). The
ComfyUI-Agent-Bridge node pack is optional: without it, proposals fall back to saving into the Workflows sidebar and
feedback is read from runs.

In the Claude desktop app, install it as the `comfyui-bending.mcpb` extension (double-click it), which needs no
paths or Python setup. For other MCP clients, start it with uv, pinned to a release:
  uvx --from git+https://github.com/abuzreq/comfyui-model-bending-agent@v0.4.0 comfyui-bending-mcp

Trust boundary: tools open only ComfyUI /api/view URLs on COMFYUI_URL (and the knowledge base's example images), never
local files or other addresses; upload_image reads only picture files the user points to, and the picture box takes
only pictures the artist drops into it. Pictures from this computer go only to a ComfyUI on this computer.
Animations are saved into ComfyUI's output folder (agent_bending/). The server's own files (built workflows,
boards, caches, sessions) live in the system's app-data folder, or COMFY_BENDING_WORKDIR.
"""

from __future__ import annotations

import functools
import io
import json
import re
import os
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import board as boards  # noqa: E402
import comfy_canvas as cc  # noqa: E402
import media  # noqa: E402
import timesteps as ts  # noqa: E402
from mcp.server.apps import Apps, client_supports_apps  # noqa: E402  (mcp >= 2.2)
from mcp.server.mcpserver import Image, MCPServer  # noqa: E402
from mcp.server.mcpserver.context import Context  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402

SKILL = HERE.parent
WORK = cc.work_dir()
media.ALLOW_LOCAL_PATHS = False  # tool arguments come from the model: only ComfyUI and knowledge-base URLs
BUILT = WORK / "workflows"
BOARDS = WORK / "boards"
BOARD_URI = "ui://comfyui-bending/board.html"
PICTURE_URI = "ui://comfyui-bending/picture.html"
# MCP clients commonly time a tool call out at 60 s, so blocking tools return before that and say
# how to keep waiting.
MAX_WAIT_S = 55

# Tools are collected here and registered when the server is built at the end of this file: MCP Apps tools (the
# in-chat board) are an extension, and extensions are fixed when the server is constructed.
_TOOLS: list = []
apps = Apps()


def _readable_errors(fn):
    """Expected failures reach the model (or the board) as readable messages, not masked as internal errors."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (RuntimeError, ValueError, OSError) as e:  # CanvasError, BoardError, AnimationError, bad input, I/O
            raise ToolError(str(e)) from e
    return wrapped


def tool(fn):
    """Register a tool."""
    _TOOLS.append(_readable_errors(fn))
    return fn


def ui_tool(visibility: list[str] | None = None, view: str = BOARD_URI):
    """Register a tool bound to an in-chat view (the board, or the picture box). visibility ["app"] = callable by the
    view only, hidden from the model."""
    def decorator(fn):
        # the flat "ui/resourceUri" key is the older spelling of _meta.ui.resourceUri; some hosts still read only it
        apps.tool(resource_uri=view, visibility=visibility, meta={"ui/resourceUri": view})(_readable_errors(fn))
        return fn
    return decorator


def _scratch(suffix: str) -> Path:
    """A fresh private file for a sheet or heat map, in the work folder (not the shared temp folder)."""
    d = WORK / "tmp"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(suffix=suffix, dir=d)
    os.close(fd)
    return Path(name)


def _shown(path: Path, max_side: int = 1024):
    """The picture for the model, and the scratch file removed."""
    try:
        return _image(path, max_side)
    finally:
        path.unlink(missing_ok=True)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.removesuffix(".json")).strip("._") or "workflow"


def _built(name: str) -> tuple[dict, dict]:
    ui_p, api_p = BUILT / f"{_safe(name)}.ui.json", BUILT / f"{_safe(name)}.api.json"
    if not ui_p.exists():
        raise cc.CanvasError(f"no built workflow named {name!r}; call build_workflow first "
                             f"(built so far: {sorted(p.name[:-8] for p in BUILT.glob('*.ui.json'))})")
    return json.loads(ui_p.read_text(encoding="utf-8")), json.loads(api_p.read_text(encoding="utf-8"))


def _remember(name: str, saved: str) -> None:
    """Where the user's copy of a built workflow lives (relative to workflows/), for read_workflow."""
    (BUILT / f"{_safe(name)}.saved").write_text(saved, encoding="utf-8")


def _image(path, max_side: int = 1024):
    """A JPEG no larger than max_side, so images stay light in the model's context. 1024 px keeps a 512-1024 px
    render at full detail; filmstrips use 1568 px, the largest edge Claude sees without downscaling."""
    from PIL import Image as PILImage
    im = PILImage.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


def _poll(pid: str, max_wait: float) -> dict:
    deadline = time.time() + max(0.0, min(max_wait, MAX_WAIT_S))
    while True:
        h = cc._req("GET", f"/api/history/{pid}")
        if pid in h and (h[pid].get("status") or {}).get("completed") is not None:
            return cc.describe_run(pid, h[pid])
        if time.time() >= deadline:
            return {"prompt_id": pid, "status": "running",
                    "hint": "call get_run with this prompt_id to keep waiting"}
        time.sleep(1)


# ------------------------------------------------------------------------------------------------ discovery
@tool
def comfy_status() -> dict:
    """Is ComfyUI reachable; free VRAM; whether Model-Bending is installed and agent-ready (ApplyBendsFromJSON
    has a `report` output); whether the ComfyUI-Agent-Bridge pack answers. Call this first."""
    return cc.status()


@tool
def free_memory() -> dict:
    """Unload all models and free VRAM (do this before switching model families). Returns free VRAM per device."""
    return cc.free_memory()


@tool
def list_models(folder: str = "checkpoints") -> list[str]:
    """Model files ComfyUI can load from a folder: checkpoints, diffusion_models, loras, vae, text_encoders, …"""
    return cc._req("GET", f"/api/models/{urllib.parse.quote(folder)}")


@tool
def search_nodes(query: str, limit: int = 30) -> list[dict]:
    """Find node classes whose name, display name or category contains every word of the query."""
    return cc.search_nodes(query, limit)


@tool
def node_info(class_type: str) -> dict:
    """Inputs (with types, defaults, ranges, choices) and outputs of one node class, as ComfyUI reports them."""
    return cc.node_info(class_type)


@tool
def get_preset(name: str = "") -> dict:
    """A tested board spec from the skill's presets/ (switchboard_bridge_sd15, switchboard_sd15, slider_board_sd15,
    feature_inspector_sd15; base_sd15: one unbent picture to bend copies of; video, experimental:
    video_sweep_wan21_t2v, video_sweep_wan21_i2v), or the list of presets when name is empty. Adapt it, then
    build_workflow."""
    presets = {p.name.removesuffix(".spec.json"): p for p in (SKILL / "presets").glob("*.spec.json")}
    if not name:
        return {"presets": sorted(presets)}
    if name not in presets:
        raise cc.CanvasError(f"unknown preset {name!r}; available: {sorted(presets)}")
    return json.loads(presets[name].read_text(encoding="utf-8"))


@tool
def get_safe_ranges(arch: str = "sd15") -> dict:
    """Measured safe ranges (per op, and per layer-group path glob) for Apply Bends from JSON's `safe_ranges`
    with clamp = safe. arch: sd15 or sdxl."""
    p = SKILL / "data" / f"safe_ranges_{arch}.json"
    if arch not in ("sd15", "sdxl") or not p.exists():
        raise cc.CanvasError("no table for this architecture (available: sd15, sdxl); start near neutral (for WAN "
                             "video: small angles, scales near 1.04, the middle blocks, early steps)")
    return json.loads(p.read_text(encoding="utf-8"))


@tool
def timestep_windows(arch: str, steps: int, scheduler: str = "normal", resolution: int = 1024,
                     denoise: float = 1.0, window_hi: float | None = None, window_lo: float | None = None) -> dict:
    """Which executed steps the structure (t 1→0.7), style (0.7→0.2) and detail (0.2→0) windows — and an optional
    custom [window_hi, window_lo] — cover. arch: sd15 | sdxl | flux | sd3 | wan (shift 8); scheduler for eps models:
    normal | simple | sgm_uniform | karras."""
    window = (window_hi, window_lo) if window_hi is not None and window_lo is not None else None
    return ts.table(arch, steps, scheduler, resolution, None, None, denoise, window)


# ------------------------------------------------------------------------------------------------ build & run
@tool
def build_workflow(spec: dict, name: str, start_image: str = "", denoise: float | None = None) -> dict:
    """Build a ComfyUI workflow from a spec (see the skill's references/canvas-presets.md: nodes keyed by name,
    [node_key, output_index] = link, lists of links for autogrow inputs, optional groups/titles/meta). Stores the
    canvas (UI) and API versions under `name` for run_workflow / propose_workflow.
    start_image (a name from upload_image or list_input_images) starts every sampler from that picture instead of an
    empty latent (image-to-image), with denoise (default 0.6; lower keeps more of the picture). In an image-to-video
    preset it sets the picture the video starts from."""
    notes = []
    if start_image:
        spec, notes = cc.use_start_image(spec, start_image, denoise)
    BUILT.mkdir(parents=True, exist_ok=True)
    ui, _ = cc.build_files(spec, BUILT / f"{_safe(name)}.ui.json", BUILT / f"{_safe(name)}.api.json")
    return {"name": name, "nodes": len(ui["nodes"]), "links": len(ui["links"]),
            "groups": [g["title"] for g in ui["groups"]], "stored_in": str(BUILT), **({"notes": notes} if notes else {})}


@tool
def upload_image(source: str, name: str = "") -> dict:
    """Put a starting picture into ComfyUI's input folder, from a file on the user's computer (its full path) or a
    ComfyUI /api/view URL from a run (e.g. "use version B as the new start"). Returns `image`, the value for
    build_workflow(start_image=...) or a LoadImage node. The same picture always gets the same name. Only use a path
    the user gave you in the chat. Files go only to a ComfyUI on this computer, and only from the user's picture
    folders when COMFY_BENDING_PICTURE_DIRS is set. For a picture attached to the chat, use ask_for_picture."""
    return cc.upload_image(source, name)


@tool
def list_input_images() -> list[str]:
    """Pictures already in ComfyUI's input folder (what its Load Image node lists), usable as start_image."""
    return cc.input_images()


@tool
def run_workflow(name: str, max_wait: float = 50) -> dict:
    """Queue a built workflow (the canvas graph is embedded in every PNG for provenance) and wait up to max_wait
    seconds. Returns status, errors, the steering inputs used, texts (e.g. bend reports from PreviewAny) and image
    URLs; if still running, returns status 'running' — then call get_run."""
    ui, api = _built(name)
    return _poll(cc.queue_prompt(api, ui if ui.get("nodes") else None)["prompt_id"], max_wait)


# ------------------------------------------------------------------------------------------------ the user's own run
@tool
def inspect_run(prompt_id: str = "") -> dict:
    """What a run the user made in ComfyUI is made of, to bend a copy of it: the model and a guess of its type,
    LoRAs, sampler settings, prompt, the samplers found, bends already in it, its pictures (the unbent original for
    a board), and whether a bend can be added (`copy`: full = an openable copy of their workflow too; pictures_only
    = renders only, the notes say why). prompt_id empty = their newest own run (runs this skill queued are
    skipped): ask them to run their workflow once, as it is, first."""
    import splice
    pid, entry = splice.load_run(prompt_id)
    api, ui = splice.run_graphs(entry)
    run = cc.describe_run(pid, entry)
    return {"prompt_id": pid, "status": run["status"], **splice.inspect(api, ui), "images": run["images"],
            "errors": run["errors"]}


@tool
def check_bends(bends: dict | list | str, arch: str = "") -> dict:
    """Read a bend someone hands over as JSON, in the format of Apply Bends from JSON (what the web UI's "Copy
    Bends" puts on the clipboard): a document {"bends": [...]}, a list of bends, or one bend, as an object or as
    pasted text. Checks it the way the node would, without rendering. Returns `summary` (one plain sentence per
    bend: where in the model, what kind of nudge, when), `bends` (the same in parts, with `inside_safe_range` when
    arch has a safe-range table, and notes such as "this layer is not image-shaped"), `errors` (bends the node would
    refuse: fix or drop them with the user), `problems` (things the node would ignore), and `bends_json`, the tidied
    document to pass to bend_run. Say the summary back to the user before rendering."""
    import bendjson
    return bendjson.check(bends, arch)


@tool
def open_tray(link: str, arch: str = "") -> dict:
    """Read a tray link the user brings from the Model Bending Navigator (…/#tray=…): bends they picked, each tested
    on its own in the knowledge base. Checks each one like check_bends. Returns `bends`, one entry per bend with
    `summary`, `bends_json` (a one-bend document for bend_run), `kb` and `link` (where it came from, to show the user
    its before/after examples), `inside_safe_range`, `notes`, `errors` and `problems`. Say the summaries back, then
    try them one at a time on the user's own run (one bend_run per bend, each with its own name) and show them
    together on a board. Combine bends only if the user asks: no combination of them has been tested."""
    import bendjson
    return bendjson.open_tray(link, arch)


@tool
def tray_link(bends: list | dict | str) -> dict:
    """A navigator link that carries bends to the user as a tray, to look at and keep: pass the bends of the round
    they liked (an Apply Bends from JSON document, a list of bends, or one bend; each stays a separate bend). Bends
    from find_recipes keep their `kb` source, so the navigator can show their before/after examples next to them."""
    import bendjson
    import kb
    r = bendjson.check(bends)
    if not r["ok"]:
        raise ValueError("; ".join(r["errors"]) or "there are no bends in it")
    return {"link": kb.tray_link(r["document"]["bends"]), "count": len(r["document"]["bends"]),
            "problems": r["problems"]}


@tool
def bend_run(name: str, bends: dict | list | str | None = None, prompt_id: str = "", arch: str = "",
             clamp: str = "safe", only: list[str] | None = None, fragment: dict | None = None,
             session: str = "", replace_bends: bool = False) -> dict:
    """Bend a COPY of a run the user made (prompt_id; empty = their newest own run): their model, LoRAs, prompt,
    sampler settings and seed are kept, and the bends are inserted on the model link into their samplers. Their own
    workflow is never changed. Stores the result under `name`, like build_workflow: then run_workflow(name), and
    propose_workflow(name) to hand them the bent copy (when `copy` is full). Call it once per candidate, with
    different names; the picture of their own run is the unbent original.
    bends: an Apply Bends from JSON document, a list of bends, or one bend, as an object or as the text the user
    pasted (see check_bends); bends the node would refuse are an error here. clamp: safe (the model type's
    safe-range table, which pulls amounts back into it; arch overrides the guess from the file name) | hard | none
    (a wild card, or a bend the user wants exactly as given). only: sampler node ids or
    titles, when not all samplers should be bent. fragment: instead of bends, a small spec inserted on the model
    link (inputs linked to ["@model", 0] take the user's model; "out": [node_key, output] feeds the samplers), e.g.
    DiT Block Bending for Flux or SD3. replace_bends: when their run already bends its model (inspect_run lists
    bending_nodes, e.g. a run made in the bending web UI), take the model from before those nodes, so the given
    bends stand in for theirs; with no bends it renders their run unbent, as the original to compare with."""
    import splice
    if clamp not in ("none", "hard", "safe"):
        raise ValueError("clamp must be none, hard or safe")
    pid, entry = splice.load_run(prompt_id)
    api, ui = splice.run_graphs(entry)
    meta = {"bent_copy_of": pid, **({"session": session} if session else {})}
    bent, bent_ui, info = splice.bend_copy(api, ui, bends, arch=arch, clamp=clamp, only=only, fragment=fragment,
                                           name=_safe(name), meta=meta, replace_bends=replace_bends)
    if bent_ui is None:  # no canvas graph can take the bend: a marker, so the bent run is not labelled as unbent
        bent_ui = {"nodes": [], "links": [], "extra": {"agent": {**meta, "pictures_only": True}}}
    BUILT.mkdir(parents=True, exist_ok=True)
    (BUILT / f"{_safe(name)}.ui.json").write_text(json.dumps(bent_ui, indent=1), encoding="utf-8")
    (BUILT / f"{_safe(name)}.api.json").write_text(json.dumps(bent, indent=1), encoding="utf-8")
    return {"name": name, "from_run": pid, **info, "next": "run_workflow(name)"}


@tool
def get_run(prompt_id: str, max_wait: float = 50) -> dict:
    """Result of a queued run (waits up to max_wait seconds)."""
    return _poll(prompt_id, max_wait)


@tool
def last_runs(n: int = 1) -> list[dict]:
    """The newest n finished runs, including runs the user started from the canvas: their steering inputs (switch
    picks, sliders, bends), notes and primitives on the canvas, texts, images, errors."""
    return cc.last_runs(n)


@tool
def wait_for_user_run(after_prompt_id: str = "", max_wait: float = 50) -> dict:
    """Wait (up to max_wait s) for the user to run something on the canvas after `after_prompt_id` (default: the
    newest run now). Use without the bridge; with it prefer wait_for_events. Call again if it times out."""
    try:
        return cc.wait_run(after_prompt_id or None, max_wait)
    except cc.CanvasError as e:
        return {"status": "no new run yet", "detail": str(e)}


# ------------------------------------------------------------------------------------------------ looking
@tool
def view_images(urls: list[str], labels: list[str] | None = None, title: str = "", columns: int = 4,
                frames: int = 6):
    """Look at images or videos (ComfyUI /api/view URLs from a run) as one labelled contact sheet. Videos (and any
    sheet that has one) show one row per source: a filmstrip of `frames` evenly spaced frames."""
    import metrics
    out = _scratch(".jpg")
    metrics.sheet(str(out), urls, labels or [], max(1, min(columns, len(urls))), 320 if len(urls) > 1 else 640,
                  title or None, max(1, min(frames, 8)))
    return _shown(out, 1568 if any(len(metrics.frames(u)) > 1 for u in urls) else 1024)


@tool
def compare_images(baseline_url: str, candidate_urls: list[str]) -> list[dict]:
    """Effect size (MAE vs the baseline) and degeneracy flags (noop, subtle, blob, noise, flat, clipped) per
    candidate. Videos: frames matched by position, plus motion, motion_ratio and the flags frozen / flicker (and
    static on a baseline that barely moves); a still baseline against a video measures drift from the starting
    picture. Metrics catch failures; judge aesthetics by looking (view_images)."""
    import metrics
    return metrics.compare(baseline_url, candidate_urls)


@tool
def diff_image(baseline_url: str, candidate_url: str):
    """Where did the bend act? A heat map of |candidate − baseline| over the baseline (for videos, averaged over
    matched frames, over the middle frame), plus summary numbers."""
    import metrics
    out = _scratch(".png")
    stats = metrics.diff(baseline_url, candidate_url, str(out))
    return [_shown(out, 768), json.dumps(stats)]


# ------------------------------------------------------------------------------------------------ the user's canvas
@tool
def propose_workflow(name: str, message: str, session: str = "", round: int | None = None) -> dict:
    """Put a built workflow in front of the user. With ComfyUI-Agent-Bridge: a confirm dialog in their open ComfyUI
    tabs; on Confirm it opens in a new tab (saved as workflows/agent_bridge/<name>.json). Without it: saved into the
    Workflows sidebar as agent_bending/<name> — tell the user to open it and press Run."""
    ui, _ = _built(name)
    if not ui.get("nodes"):
        raise cc.CanvasError(f"{name!r} is a pictures-only copy of the user's run (bend_run could not add the bend "
                             f"to their canvas graph), so there is no workflow to open; tell them where to add the "
                             f"box by hand instead")
    if cc.bridge_info():
        prop = cc.propose(ui, name, message, session or None, round)
        _remember(name, (prop.get("saved_path") or "").removesuffix(".json"))
        return {"via": "agent_bridge", "proposal": prop,
                "next": "wait_for_events(kinds='proposal_status,feedback,run')"}
    where = cc.push_workflow(ui, f"agent_bending/{_safe(name)}")
    _remember(name, f"agent_bending/{_safe(name)}")
    return {"via": "workflows_sidebar", "saved_as": where,
            "next": f"tell the user: open Workflows → {where}, press Run; then wait_for_user_run or read_workflow"}


@tool
def wait_for_events(session: str = "", since: int = 0, kinds: str = "", max_wait: float = 50) -> dict:
    """(ComfyUI-Agent-Bridge) Long-poll the user's side: proposal decisions (proposal_status), feedback (votes,
    keep/change chips, rating, push, direction; source 'button' or 'run'), reports, and finished runs (with
    switch_picks and feedback). Pass the returned `last` as `since` next time. kinds: comma-separated filter."""
    return cc.events(since, session or None, kinds, min(max_wait, MAX_WAIT_S))


@tool
def read_workflow(name: str) -> dict:
    """What the user did to a proposed/pushed workflow after saving it (Ctrl+S): notes, primitives, switches,
    muted groups, and every change against what was built (widgets, modes, added/removed nodes, rewiring)."""
    ui, _ = _built(name)
    marker = BUILT / f"{_safe(name)}.saved"
    if not marker.exists():
        raise cc.CanvasError(f"{name!r} was never proposed or pushed; call propose_workflow first")
    saved = marker.read_text(encoding="utf-8").strip()
    return cc.pull_workflow(saved, ui)[0] | {"saved_as": f"workflows/{saved}.json"}


@tool
def read_canvas() -> dict:
    """(ComfyUI-Agent-Bridge) The user's live canvas — only while they share it (Settings → Agent Bridge → Share
    canvas). Never ask them to leave sharing on."""
    return cc.read_canvas()[0]


@tool
def notify_user(message: str, severity: str = "info") -> dict:
    """(ComfyUI-Agent-Bridge) A toast in the user's open ComfyUI tabs, e.g. 'round 3 is ready'."""
    return {"tabs": cc.notify(message, severity)}


@tool
def server_logs(grep: str = "model-bending", tail: int = 80) -> str:
    """The tail of ComfyUI's log, filtered (default: Model-Bending's messages about expanded/skipped paths)."""
    return cc.server_logs(grep or None, tail)


# ------------------------------------------------------------------------------------------------ knowledge base
_KB_CHOICES = {"community": ["community"], "mine": ["mine"], "both": ["community", "mine"], "none": []}


@tool
def kb_sources(session: str, choice: str = "") -> dict:
    """Which stored knowledge this session may use, as the user chose: community (the shared bend
    knowledge base), mine (their own logged rounds), both, or none (explore fresh). With choice empty, returns the
    stored choice. Ask the user before the first find_recipes of a session; never pick for them."""
    import kb_local
    if choice:
        if choice not in _KB_CHOICES:
            raise ValueError(f"choice must be one of {sorted(_KB_CHOICES)}")
        return {"session": session, "sources": kb_local.session_sources(session, _KB_CHOICES[choice])}
    return {"session": session, "sources": kb_local.session_sources(session)}


@tool
def find_recipes(goal: str, arch: str = "", session: str = "", sources: list[str] | None = None, limit: int = 6,
                 refresh: bool = False) -> dict:
    """Starting recipes for a goal in plain words ("more abstract", "keep the face, change the palette") from the
    bend knowledge base. arch: sd14 | sd15 | sdxl … (results from the same family count, labelled with the
    checkpoints they were tested on). sources: community and/or mine; default = the session's choice (kb_sources).
    Each result: the bend (Apply Bends from JSON form), where it acts, amount and step window, evidence (grade, seeds,
    prompts, sources), effect tags with WHO assigned them (human, or ai:<model>), cited findings, and example images.
    The recipe carries `kb` (the cell it came from; keep it when rendering) and community results a `link` to the
    navigator page with the cell's before/after examples and risks.
    Facts (recipe, setup, measurements) and interpretations (captions, tags) are kept apart: say which is which.
    refresh fetches the latest community index (no token needed)."""
    import kb_local
    if sources is None:
        if not session:
            raise ValueError("pass sources, or a session whose choice was stored with kb_sources")
        sources = kb_local.session_sources(session)
        if sources is None:
            raise ValueError("the user has not chosen yet: ask whether to use the community knowledge base, their "
                             "own previous runs, both, or neither; store it with kb_sources")
    return kb_local.find(goal, arch, sources, limit, refresh)


@tool
def describe_cell(cell: str, sources: list[str] | None = None) -> dict:
    """Everything the knowledge base holds for one cell (family|group|kind|module_type|op|amount|window|route, as
    find_recipes returns it): counts, tested argument ranges, measurement summaries, effect tags by author, findings,
    example record ids."""
    import kb_local
    return kb_local.describe_cell(cell, sources)


@tool
def intro_examples(arch: str = "sd15", n: int = 4) -> dict:
    """For someone new to bending: a few example bends from the community knowledge base, one per part of the model
    (early stages, core, late stages), each with what was bent in plain words, what it looked like and who said so,
    how well it is evidenced, and `image_url` / `unbent_url` to show with view_images. Illustrations only: it needs
    no knowledge-source choice and makes none (ask before find_recipes as usual). Examples exist for SD1.5-type
    models; for others the note says so."""
    import kb_local
    return kb_local.intro_examples(arch, n)


@tool
def surprise_bends(arch: str, n: int = 3, wild: bool = True, seed: int | None = None,
                   avoid: list[str] | None = None) -> dict:
    """A surprise round: n random single-bend candidates, one per part of the model (early stages, core, late
    stages), drawn inside the measured safe ranges; with wild, the last one is a wild card that goes past them on
    purpose (run it with its clamp, none, and show it labelled even if it fell apart). arch: sd15 | sd14 | sdxl
    (other model types have no table: the error says what to do). Each candidate: `bends_json` for Apply Bends from
    JSON, `clamp`, and `what_was_bent` in plain words (tell the artist after they have looked). The same seed
    repeats a draw; pass earlier candidates' `key`s as avoid for a fresh round."""
    import surprise
    return surprise.draw(arch, n, wild, seed, avoid)


@tool
def kb_status(session: str = "") -> dict:
    """The community knowledge base in use (snapshot or latest; date and counts), the user's own records and how many
    they marked for sharing, and the session's stored source choice."""
    import kb_local
    return kb_local.status(session)


@tool
def log_round(prompt_id: str, session: str, verdict: str = "", words: str = "", caption: str = "", change: str = "",
              effect_tags: list[str] | None = None, agent_model: str = "", baseline_prompt_id: str = "",
              arch: str = "", keywords: list[str] | None = None, prompt_version: str = "") -> dict:
    """Record a finished bent run in the user's OWN knowledge base (stays on their machine; nothing is shared). Facts
    are read from ComfyUI's history (model, sampler, seed, size, the bends as resolved). Interpretations are labelled:
    verdict (kept / rejected / pinned …) and words are the user's; caption, change, keywords and effect_tags (from
    the vocabulary) are yours, and then agent_model (your exact model id) is required. When you wrote them with
    description_prompt, pass its prompt_version, so they compare with the community's captions. baseline_prompt_id =
    the unbent run of the same setup, for the change measurement. The prompt and input image are kept privately."""
    import kb_local
    return kb_local.log_round(prompt_id, session, verdict, words, caption, change, effect_tags, agent_model,
                              baseline_prompt_id, arch, "", keywords, prompt_version)


@tool
def description_prompt() -> dict:
    """The knowledge base's prompt for describing a bent result (caption, change, keywords, effect tags), with its
    version and steps. Use it whenever the user wants you to describe their results, e.g. when logging or sharing
    rounds: it is the prompt the community captions were written with, so descriptions stay consistent. Look at the
    unbent and bent pictures, describe only what is visible, show the user, then log_round(..., prompt_version)."""
    import kb_local
    return kb_local.description_prompt()


# ------------------------------------------------------------------------------------------------ animation
_JOBS: dict[str, dict] = {}


def _anim_source(name: str, prompt_id: str, last_run: bool):
    import animate
    if name:
        ui, api = _built(name)
        return api, ui if ui.get("nodes") else None
    return animate.load_source(prompt_id=prompt_id or None, last_run=last_run)


@tool
def start_animation(name: str = "", prompt_id: str = "", last_run: bool = False, bends: list[str] | None = None,
                    inputs: list[str] | None = None, frames: int = 24, increment: float | None = None,
                    easing: str = "ease", loop: str = "boomerang", hold: int = 0, repeats: int = 2, fps: int = 12,
                    output_node: str = "", format: str = "mp4", dry_run: bool = False) -> dict:
    """Animate bends in small increments: render the workflow frame by frame while values move, then encode a looping
    video (same seed and prompt in every frame). Source: a built workflow `name`, a `prompt_id`, or `last_run`
    (e.g. the run the user just made). Tracks are "SELECTOR@name=FROM:TO" strings (FROM defaults to the neutral
    value, TO to the current one):
      bends:  "recompose@angle_degrees=0:180" (a JSON bend by label or path), "A bends#0@blend=0:1" (NODE#index;
              `blend` works for any op)
      inputs: "Bends@a=0:180", "12@angle_degrees=0:90" (any numeric input, node id or title; for a linked slider
              animate the slider node's `value`)
    With no tracks, every JSON bend moves from neutral to its current amount. `increment` = step of the first
    moving track in its own units (sets the frame count); otherwise `frames`. loop: boomerang | restart | none.
    format: mp4 (needs ffmpeg; falls back to gif) | gif | webp. dry_run returns the plan without rendering.
    Returns a job id; poll animation_status."""
    import shutil
    import threading

    import animate
    api, ui = _anim_source(name, prompt_id, last_run)
    tracks = animate.build_tracks(api, bends or [], inputs or [])
    kw = dict(frames=frames, increment=increment, easing=easing, loop=loop, hold=hold, repeats=repeats, fps=fps,
              output_node=output_node or None)
    p = animate.plan(api, tracks, **kw)
    summary = {"rendered_frames": len(p["times"]), "video_frames": len(p["order"]), "duration_s": p["duration_s"],
               "output_node": p["output_node"],
               "tracks": [{k: t[k] for k in ("label", "name", "from", "to")} for t in tracks],
               "first_values": p["values"][:3], "last_value": p["values"][-1]}
    if dry_run:
        return {"plan": summary}
    if format == "mp4" and shutil.which("ffmpeg") is None:
        format = "gif"
    job = f"anim_{int(time.time() * 1000):x}"
    out = WORK / "animations" / animate.video_name(name or "", format)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = _JOBS[job] = {"status": "rendering", "done": 0, "total": summary["rendered_frames"], "plan": summary}

    def work():
        try:
            m = animate.animate(api, ui, tracks, out, progress=lambda d, n: state.update(done=d), **kw)
            state["manifest"] = {**m, **animate.deliver(Path(m["video"]))}
            state["status"] = "done"
        except Exception as e:  # noqa: BLE001 - reported through animation_status
            state.update(status="failed", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True).start()
    return {"job": job, "plan": summary, "next": "animation_status(job)"}


@tool
def animation_status(job: str, max_wait: float = 50):
    """Progress of an animation job (waits up to max_wait s). When done: `where` (a plain sentence telling the
    artist where the video is saved: ComfyUI's output folder, inside agent_bending), `view_url` (plays it in a
    browser), and a filmstrip image of evenly spaced frames so you can see the motion."""
    state = _JOBS.get(job)
    if state is None:
        raise ValueError(f"unknown job {job!r}; jobs: {sorted(_JOBS)}")
    deadline = time.time() + max(0.0, min(max_wait, MAX_WAIT_S))
    while state["status"] == "rendering" and time.time() < deadline:
        time.sleep(0.5)
    if state["status"] == "rendering":
        return {"status": "rendering", "frames_done": state["done"], "frames_total": state["total"]}
    if state["status"] == "failed":
        raise ValueError(state["error"])
    m = state["manifest"]
    info = {k: m[k] for k in ("where", "file", "view_url", "video", "rendered_frames", "video_frames", "fps",
                              "duration_s") if m.get(k)}
    return [_image(m["filmstrip"], 1568), json.dumps({"status": "done", **info})]


# ------------------------------------------------------------------------------------------------ in-chat board
@ui_tool()
def show_board(ctx: Context, title: str, question: str, candidates: list[dict], original_url: str = "",
               session: str = "", round: int | None = None, ask: list[str] | None = None):
    """Show the artist a pick board inside the chat (the Claude app): the original and 1-6 versions as big pictures
    with a plain caption each (videos play as short loops); they choose one or several, can mark what to keep or
    change, how strong the next try should be, and type a note. candidates: [{label, image_url (ComfyUI /api/view URL
    of an image or video from a run; original_url may be the starting picture's view_url), caption (one plain
    sentence about what changed), recommended (true on your pick), details (the technical line, hidden by default)}].
    ask: which extras to show, from votes, keep_change, strength, direction (default all). Their answer arrives as
    their next chat message; read the exact choice with board_feedback. Apps that cannot show boards get a labelled
    contact sheet instead: then ask in chat."""
    b = boards.create(BOARDS, title=title, question=question, candidates=candidates, original_url=original_url,
                      session=session, round=round, ask=ask)
    labels = ", ".join(c["label"] for c in b["candidates"])
    if client_supports_apps(ctx):
        text = (f"The board was sent to the chat ({b['board_id']}, versions {labels}). Wait: the artist's answer "
                f"arrives as their next message. Then read the exact choice with "
                f"board_feedback(board_id='{b['board_id']}'). If it reports not_displayed (some apps cannot show "
                f"boards), show the versions with view_images and ask in chat. Do not ask them to open ComfyUI.")
        # some hosts hand the model the structured part instead of the text, so the note goes in both
        return CallToolResult(content=[TextContent(type="text", text=text)],
                              structured_content={"note_for_claude": text, "board": boards.view(b)})
    import metrics
    srcs = ([b["original_url"]] if b["original_url"] else []) + [c["image_url"] for c in b["candidates"]]
    names = (["original"] if b["original_url"] else []) + [
        f"{c['label']}{'  (my pick)' if c['recommended'] else ''}" for c in b["candidates"]]
    out = _scratch(".jpg")
    metrics.sheet(str(out), srcs, names, max(1, min(4, len(srcs))), 320, title or None)
    text = (f"This app cannot show the in-chat board, so here are the versions as one contact sheet. Show it to the "
            f"artist with a one-line plain caption per version, and ask in chat: pick one or more of {labels}, "
            f'"none", or what to change.')
    return [_shown(out), text]


# Visible to the model as well as to the board: hosts that relay tools to a local extension (the Claude desktop Chat
# tab) pass on only model-visible tools, and the board cannot work without these two.
@ui_tool()
def board_image(board_id: str, index: int, max_side: int = 640):
    """(Used by the board; you do not need to call it.) One picture of a board, in view order (the original first),
    as a downscaled JPEG, or a short looping WEBP for a video."""
    b = boards.load(BOARDS, board_id)
    url = boards.image_url(b, index)
    data, fmt = boards.picture(cc.fetch_bytes(url), media.name_of(url), max(64, min(max_side, 1024)))
    boards.mark_opened(BOARDS, board_id)
    return Image(data=data, format=fmt)


@ui_tool()
def submit_board(board_id: str, feedback: dict) -> dict:
    """(Used by the board; you do not need to call it.) Store the artist's answer: picks (labels), none, votes, keep,
    change, rating, push, direction."""
    fb = boards.submit(BOARDS, board_id, feedback)
    return {"saved": True, "revision": fb["revision"]}


@tool
def board_feedback(board_id: str, since_revision: int = 0, max_wait: float = 50) -> dict:
    """The artist's answer on an in-chat board (waits up to max_wait s): picks (one or several labels), none, votes
    (up/down per label), keep and change (composition, subject, palette, lighting, texture, style), push (-1 pull back,
    0 about the same, +1 push further), direction (their note), revision (it goes up if they send a new answer; pass
    the last one as since_revision to wait for a change)."""
    return boards.wait_feedback(BOARDS, board_id, min(max_wait, MAX_WAIT_S), since_revision)


# ------------------------------------------------------------------------------------------------ picture box
def _picture_ways(local: bool) -> str:
    """What to offer when the picture box cannot be used."""
    load = ("drag the picture into a Load Image box in ComfyUI and tell you its name (it then appears in "
            "list_input_images; use that name as start_image)")
    if not local:
        return (f"ComfyUI runs on another machine, so pictures from this computer are not sent there. Ask the artist, "
                f"in plain words, to {load}.")
    return ("Offer the artist two ways, in plain words: (1) paste the picture's full path into the chat (Windows: "
            "Shift + right-click the file in Explorer, then Copy as path; Mac: right-click it, hold Option, then Copy "
            "as Pathname), and call upload_image with it; or (2) " + load + ".")


@ui_tool(view=PICTURE_URI)
def ask_for_picture(ctx: Context, purpose: str = "", session: str = ""):
    """Show the artist a box in the chat where they drop (or choose, or paste) a picture to start from; it goes
    straight into ComfyUI. Call it as soon as the artist attaches a picture to the chat or says they will bring one: a
    picture attached to the chat is not a file ComfyUI can open. purpose: one plain line shown in the box (e.g. "The
    painting the next round starts from"). The result arrives as their next chat message; read it with
    picture_received. When the box cannot be shown, this returns the other ways to ask for instead."""
    if not cc.is_local():
        return _picture_ways(False)
    if not client_supports_apps(ctx):
        return "This app cannot show the picture box. " + _picture_ways(True)
    req = boards.create_request(BOARDS, purpose=purpose, session=session)
    rid = req["board_id"]
    text = (f"The picture box was sent to the chat ({rid}). Tell the artist in one line to drop their picture into it "
            f"(they can also paste it or choose the file). Their answer arrives as their next message; then read it "
            f"with picture_received(request_id='{rid}'), which gives the start_image name. If it reports "
            f"not_displayed, offer the other ways: " + _picture_ways(True))
    return CallToolResult(content=[TextContent(type="text", text=text)],
                          structured_content={"note_for_claude": text, "request": boards.request_view(req)})


@ui_tool(view=PICTURE_URI)
def add_picture(request_id: str, name: str = "", data: str = "", part: int = 0, parts: int = 1) -> dict:
    """(Used by the picture box; you do not need to call it.) Receive the dropped picture in base64 parts and upload
    it to ComfyUI; with no data it only says the box is open."""
    if not data:
        boards.mark_opened(BOARDS, request_id)
        return {"ready": True, "max_bytes": boards.MAX_PICTURE_BYTES, "part_bytes": boards.PART_BYTES}
    if not cc.is_local():
        raise cc.CanvasError(f"ComfyUI at {cc.BASE} is on another machine: upload the picture in a Load Image box there")
    whole = boards.add_part(BOARDS, request_id, data, part, parts)
    if whole is None:
        return {"received_part": part}
    uploaded = cc.upload_bytes(whole, name or "picture.png", Path(name).stem if name else "")
    width, height = boards.picture_size(whole)
    pic = boards.record_picture(BOARDS, request_id, {**uploaded, "width": width, "height": height,
                                                     "name": (name or "")[:120]})
    return {"saved": True, "image": pic["image"], "width": width, "height": height, "revision": pic["revision"]}


@tool
def picture_received(request_id: str, since_revision: int = 0, max_wait: float = 50) -> dict:
    """The picture the artist dropped into a picture box (waits up to max_wait s): image (the start_image value for
    build_workflow or a LoadImage node), view_url (use it as a board's original_url), width, height, name (their file
    name), revision (it goes up if they drop another picture; pass the last one as since_revision to wait for a
    change)."""
    return boards.wait_picture(BOARDS, request_id, min(max_wait, MAX_WAIT_S), since_revision)


# ------------------------------------------------------------------------------------------------ server
apps.add_html_resource(BOARD_URI, boards.html(), name="board", title="Pick a version",
                       description="Pick board for comparing bend versions", prefers_border=False)
apps.add_html_resource(PICTURE_URI, boards.html("picture.html"), name="picture", title="Add your picture",
                       description="Drop a picture to start from", prefers_border=False)
mcp = MCPServer(
    "comfyui-bending",
    instructions=(
        "Tools for the comfyui-model-bending skill: build ComfyUI workflows from compact specs, run them, look at "
        "the results, take the artist's starting picture through an in-chat picture box, show them an in-chat pick "
        "board or propose boards to their open ComfyUI, and read their feedback back. For a first session: example "
        "bends to show a newcomer (intro_examples), a random surprise round (surprise_bends), and bending a copy of "
        "a workflow the user already ran (inspect_run, bend_run), and reading a bend they bring as JSON (check_bends). "
        "Follow the skill's SKILL.md for the method; this server only does the machine-side work."),
    extensions=[apps],
)
for _fn in _TOOLS:
    mcp.add_tool(_fn)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
