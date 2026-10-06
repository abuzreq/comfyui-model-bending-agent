#!/usr/bin/env python3
"""Build, push, queue and read back agent-authored ComfyUI canvas workflows. Standard library only.

Talks to ComfyUI's HTTP API directly, so no other MCP server or package is needed. Commands:

  status                                       ComfyUI reachable? free VRAM; Model-Bending installed and
                                               agent-ready?; Agent-Bridge installed? Run this first.
  nodes QUERY [--limit 30]                     node classes whose name, display name or category match every word
  node CLASS                                   one node's inputs (types, defaults, ranges, choices) and outputs
  free                                         unload models and free VRAM (before switching model families)
  build SPEC.json -o UI.json [--api API.json]  spec (an API-format superset, see below) -> frontend workflow with
        [--start-image NAME [--denoise 0.6]]   widgets, links, groups and an automatic layout; --start-image starts
                                               every sampler from that picture instead of an empty latent
  upload PATH_OR_URL [--name NAME]             put a picture into ComfyUI's input folder (agent_bending/...): a
                                               file on this computer or a ComfyUI /api/view URL from a run. Prints
                                               the value for LoadImage / --start-image
  inputs                                       pictures already in ComfyUI's input folder (LoadImage's list)
  push UI.json --name foo                      save into the user's Workflows sidebar, always under agent_bending/
                                               (or agent_bridge/), never over the user's own workflows
  pull --name agent/foo [--baseline UI.json]   read it back: notes, primitives, switches, muted/bypassed groups,
                                               and (with --baseline) every widget, mode or note the user changed
  queue --api API.json [--ui UI.json] [--wait] queue a prompt; --ui embeds the canvas graph in the PNG metadata
  last-run [--n 1]                             newest executed prompts: bend/steering inputs, errors, images
  wait-run [--after PROMPT_ID] [--timeout 900] block until the user queues and finishes a new prompt
  logs [--grep TEXT] [--tail 200]              ComfyUI server log tail (bend path warnings land here)
  run-setup [--prompt-id ID]                   what the user's own last run is made of (model, LoRAs, sampler,
                                               prompt, samplers) and whether a bend can be added to it
  bend-run --bends BENDS --name N --api API.json [-o UI.json] [--prompt-id ID] [--only NODE] [--clamp safe]
           [--replace-bends]                   a copy of that run with the bends inserted before its samplers
                                               (their model, LoRAs, settings and seed kept); BENDS is a file or
                                               the JSON text (check it first with bendjson.py). --replace-bends
                                               puts them in place of the run's own bends; see splice.py

With the ComfyUI-Agent-Bridge node pack installed (checked via /api/agent_bridge/info):
  bridge                                       is the bridge there? version, capabilities, sharing state
  propose UI.json --name r2_board [--message TEXT] [--session S] [--round N]
                                               ask the user (dialog in every open ComfyUI tab) to open the graph
                                               in a new tab; it is also saved under workflows/agent_bridge/
  events [--since ID] [--session S] [--kinds feedback,run] [--wait 30]
                                               long-poll: proposal decisions, feedback, reports, finished runs
  canvas [--save FILE]                         the user's active graph (only while they share it in Settings)
  notify TEXT [--severity info|success|warn|error]   toast in the user's open ComfyUI tabs

Add --json to pull / last-run / wait-run / events for machine-readable output.
Environment: COMFYUI_URL (default http://127.0.0.1:8188); AGENT_BRIDGE_TOKEN (default: read the bridge's token
file, ComfyUI/user/agent_bridge/token, which works only when ComfyUI runs on this computer); COMFY_BENDING_WORKDIR
(the skill's own files, see work_dir); COMFY_BENDING_PICTURE_DIRS (optional: the only folders `upload` reads
pictures from, separated like PATH).

SPEC format
-----------
{
  "meta":   {...},                                   # copied to workflow.extra.agent (survives user saves)
  "groups": [{"id": "cand_a", "title": "A · rotate mid 90°", "color": "#3f5159"}],
  "nodes": {
    "ckpt":  {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
    "bendA": {"class_type": "ApplyBendsFromJSON", "group": "cand_a", "title": "A bends",
              "inputs": {"model": ["ckpt", 0], "bends_json": "{...}"}},
    "fb":    {"class_type": "MarkdownNote", "group": "feedback", "inputs": {"text": "Write here..."}},
    "off":   {"class_type": "PreviewImage", "mode": "mute", ...}         # mode: active | mute | bypass
  }
}
A list value [node_key, output_index] is a link; anything else is a widget value. Linking into a widget input
(e.g. a PrimitiveFloat into angle_degrees) is allowed. Optional per node: title, group, pos [x, y], size [w, h].
Note / MarkdownNote are canvas-only and are left out of the API export.

Starting picture: `build --start-image NAME` (tool: build_workflow(start_image=NAME)) rewrites a text-to-image spec
into image-to-image. Each empty image latent becomes LoadImage -> ImageScaleToTotalPixels (the latent's pixel count,
the picture's own shape: nothing is cropped) -> VAEEncode (with the VAE the spec decodes with), repeated for
batch_size > 1, and every KSampler that starts from it gets denoise (default 0.6 where it was 1.0). A spec that already has a LoadImage keyed "start_image" (image-to-video
presets) just gets its picture set. Empty video latents are refused: image-to-video needs an image-to-video model.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

def _env(name: str) -> str | None:
    """An environment value, ignoring empty ones and unfilled ${placeholders} from an extension manifest."""
    v = (os.environ.get(name) or "").strip()
    return v if v and not v.startswith("${") else None


def work_dir() -> Path:
    """Where the skill keeps its own files: caches, boards, built workflows, sessions and the user's own knowledge
    base. COMFY_BENDING_WORKDIR if set; else ~/.comfyui-model-bending if an earlier version made it; else the system's
    app-data folder. Nothing the artist opens lives here: animations go to ComfyUI's output folder (save_output)."""
    if (v := _env("COMFY_BENDING_WORKDIR")):
        return Path(v).expanduser()
    legacy = Path.home() / ".comfyui-model-bending"
    if legacy.is_dir():
        return legacy
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "comfyui-model-bending"


BASE = (_env("COMFYUI_URL") or "http://127.0.0.1:8188").rstrip("/")
CACHE = work_dir() / "cache"
AGENT_FOLDERS = ("agent_bending/", "agent_bridge/")  # the only places the agent writes workflows
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32,128}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """ComfyUI never redirects its API; following one could send requests to other addresses."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CanvasError(f"ComfyUI at {BASE} answered with a redirect (HTTP {code}) to {newurl}. Set the ComfyUI "
                          f"address to the final address (for example the https:// one), or to the server itself "
                          f"rather than a login page in front of it")


_OPENER = urllib.request.build_opener(_NoRedirect)




class CanvasError(RuntimeError):
    """A message for the caller: ComfyUI unreachable, a bad spec, a missing node, a timeout."""


WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
VIRTUAL = {"Note", "MarkdownNote"}
IMAGE_VIEWERS = {"PreviewImage", "SaveImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveVideo", "SaveWEBM"}
VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp")  # .webp / .gif count only when ComfyUI says animated
EMPTY_IMAGE_LATENTS = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyFlux2LatentImage", "EmptyHunyuanImageLatent"}
EMPTY_VIDEO_LATENTS = {"EmptyHunyuanLatentVideo", "EmptyMochiLatentVideo", "EmptyLTXVLatentVideo",
                       "EmptyCosmosLatentVideo", "Wan22ImageToVideoLatent"}
INPUT_SUBFOLDER = "agent_bending"
UPLOAD_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MODES = {"active": 0, "mute": 2, "never": 2, "bypass": 4}
MODE_NAMES = {0: "active", 2: "muted", 4: "bypassed"}
STEERING_HINTS = ("Bend", "Bending", "Switch", "Primitive", "KSampler", "Feature Map", "Latent Operation",
                  "ConditioningApplyOperation", "Any Switch")
# canvas layout in px: column width fits a default node plus its links; gaps keep group titles readable
COL_W, ROW_GAP, BAND_GAP, X0, Y0 = 420, 30, 90, 40, 60


# --------------------------------------------------------------------------- HTTP
def _req(method: str, path: str, body: bytes | None = None, ctype: str = "application/json",
         headers: dict | None = None, timeout: float = 60):
    req = urllib.request.Request(BASE + path, data=body, method=method,
                                 headers={"Content-Type": ctype, **(headers or {})})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise CanvasError(f"{method} {path} -> HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:800]}")
    except urllib.error.URLError as e:
        raise CanvasError(f"cannot reach ComfyUI at {BASE}: {e.reason}")
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")


def fetch_bytes(url: str, timeout: float = 60) -> bytes:
    """Raw bytes of a ComfyUI image (an /api/view URL on BASE). Other hosts and paths are refused."""
    u, base = urllib.parse.urlsplit(url), urllib.parse.urlsplit(BASE)
    if (u.scheme, u.netloc) != (base.scheme, base.netloc) or u.path not in ("/api/view", "/view"):
        raise CanvasError(f"not a ComfyUI image URL on {BASE}: {url}")
    try:
        with _OPENER.open(url, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise CanvasError(f"GET {u.path} -> HTTP {e.code}")
    except urllib.error.URLError as e:
        raise CanvasError(f"cannot reach ComfyUI at {BASE}: {e.reason}")


def object_info(fresh: bool = False) -> dict:
    """GET /object_info, cached for 10 minutes (it is several MB) in the user's own work folder."""
    cache = CACHE / f"object_info_{hashlib.sha256(BASE.encode()).hexdigest()[:16]}.json"
    if not fresh and cache.exists() and time.time() - cache.stat().st_mtime < 600:
        return json.loads(cache.read_text(encoding="utf-8"))
    info = _req("GET", "/api/object_info")
    CACHE.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = cache.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(info), encoding="utf-8")
    os.replace(tmp, cache)
    return info


# --------------------------------------------------------------------------- the "kb" key on bends
BEND_NODE = "ApplyBendsFromJSON"
KB_PROBE_SESSION = "agent-model-bending-kb-probe"
_KB_OK: dict[str, bool] = {}


def bends_accept_kb(fresh: bool = False) -> bool:
    """Does this ComfyUI's Model-Bending know a bend's "kb" key (where it came from in the bend knowledge base;
    plugin 0.3.1+, bends JSON 1.2)? Older plugins call it an unknown key, which fails a run with strict on. Asked once
    per ComfyUI address: a one-bend document with a "kb" key goes to the plugin's own /web_bend_demo/selection route
    under a session of its own, and a warning naming 'kb' (or no such route) means no."""
    if not fresh and BASE in _KB_OK:
        return _KB_OK[BASE]
    probe = {"session_id": KB_PROBE_SESSION, "bends": [{"path": "probe", "module_type": "multiply",
                                                        "module_args": {"scalar": 1.0}, "kb": {"dataset": "probe"}}]}
    try:
        r = _req("POST", "/web_bend_demo/selection", json.dumps(probe).encode(), timeout=15)
    except CanvasError as e:
        if "HTTP" not in str(e):
            raise  # ComfyUI unreachable: no answer to remember
        r = {}  # no such route: a plugin older than the web UI's JSON format
    ok = isinstance(r, dict) and r.get("ok") is True and not any("'kb'" in str(w) for w in r.get("warnings") or [])
    _KB_OK[BASE] = ok
    return ok


def _without_kb(text):
    """A bends JSON text without its per-bend "kb" keys, or None when it has none (or is not a bends document)."""
    if not isinstance(text, str) or '"kb"' not in text:
        return None
    try:
        doc = json.loads(text)
    except ValueError:
        return None  # e.g. {{a}} placeholders the node fills in: left as it is
    if not isinstance(doc, dict):
        return None
    hit = False
    for key in ("bends", "bend", "attention_bends"):
        items = doc.get(key)
        for b in (items.values() if isinstance(items, dict) else items if isinstance(items, list) else []):
            if isinstance(b, dict) and "kb" in b:
                del b["kb"]
                hit = True
    return (json.dumps(doc, indent=1) if "\n" in text else json.dumps(doc)) if hit else None


def fit_bends(api: dict | None = None, ui: dict | None = None) -> tuple[dict | None, dict | None, int]:
    """(api, ui, how many bend texts changed): the graphs as this ComfyUI can run them. When its Model-Bending does
    not know the "kb" key (bends_accept_kb), every Apply Bends from JSON text loses its "kb" keys, in copies, so a
    strict run does not fail; the skill keeps "kb" in its own results and links. Graphs whose bends carry no "kb"
    are returned as they are, without asking ComfyUI anything."""
    api_hits = {nid: t for nid, n in (api or {}).items() if isinstance(n, dict) and n.get("class_type") == BEND_NODE
                for t in [_without_kb((n.get("inputs") or {}).get("bends_json"))] if t is not None}
    ui_hits = {}
    for i, n in enumerate((ui or {}).get("nodes") or []):
        if isinstance(n, dict) and n.get("type") == BEND_NODE and isinstance(n.get("widgets_values"), list):
            fixed = [_without_kb(v) for v in n["widgets_values"]]
            if any(t is not None for t in fixed):
                ui_hits[i] = [v if t is None else t for v, t in zip(n["widgets_values"], fixed)]
    if not (api_hits or ui_hits) or bends_accept_kb():
        return api, ui, 0
    if api_hits:
        api = {**api, **{nid: {**api[nid], "inputs": {**api[nid]["inputs"], "bends_json": t}}
                         for nid, t in api_hits.items()}}
    if ui_hits:
        ui = {**ui, "nodes": [{**n, "widgets_values": ui_hits[i]} if i in ui_hits else n
                              for i, n in enumerate(ui["nodes"])]}
    return api, ui, len(api_hits) + len(ui_hits)


def agent_workflow_name(name: str) -> str:
    """A sidebar name inside the agent's own folders: agent_bending/<name> unless it is already under agent_bending/
    or agent_bridge/. Refuses '..', absolute paths and unusual characters."""
    n = name.replace("\\", "/").strip().removesuffix(".json").strip("/")
    parts = n.split("/")
    if not n or any(p in ("", ".", "..") or not re.fullmatch(r"[\w .()+-]{1,120}", p) for p in parts):
        raise CanvasError(f"unusable workflow name {name!r}: use letters, digits, spaces and . _ - ( ) +")
    return n if n.startswith(AGENT_FOLDERS) else f"agent_bending/{n}"


def _userdata_path(name: str) -> str:
    name = name.removesuffix(".json")
    return "/api/userdata/" + urllib.parse.quote(f"workflows/{name}.json", safe="")


# --------------------------------------------------------------------------- ComfyUI-Agent-Bridge client
def bridge_info() -> dict | None:
    req = urllib.request.Request(BASE + "/api/agent_bridge/info")
    try:
        with _OPENER.open(req, timeout=10) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ValueError):
        return None


def is_local(base: str = "") -> bool:
    """Does the ComfyUI address point at this computer?"""
    host = urllib.parse.urlsplit(base or BASE).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _bridge_token() -> str:
    """The bridge token: AGENT_BRIDGE_TOKEN, or the bridge's token file when ComfyUI runs on this computer. The file
    path comes from the server, so it is accepted only if it is the bridge's own token file (…/agent_bridge/token,
    after resolving links), small, and holds a token; nothing else is ever read and sent."""
    tok = _env("AGENT_BRIDGE_TOKEN")
    if tok:
        return tok.strip()
    info = bridge_info()
    if not info:
        raise CanvasError("ComfyUI-Agent-Bridge is not installed on this ComfyUI (no /api/agent_bridge/info); "
                 "use push / pull / wait-run instead")
    if not is_local():
        raise CanvasError(f"ComfyUI at {BASE} is not on this computer, so its bridge token cannot be read here: set "
                          f"AGENT_BRIDGE_TOKEN to the contents of ComfyUI/user/agent_bridge/token on that machine")
    raw = info.get("token_file") or ""
    try:
        path = Path(raw).resolve(strict=True)
    except (OSError, RuntimeError):
        raise CanvasError("cannot find the bridge token file; set AGENT_BRIDGE_TOKEN") from None
    if (path.parent.name, path.name) != ("agent_bridge", "token") or not path.is_file() or path.stat().st_size > 256:
        raise CanvasError(f"refusing to read {raw!r}: it is not the bridge's token file (…/agent_bridge/token); set "
                          f"AGENT_BRIDGE_TOKEN")
    tok = path.read_text(encoding="utf-8").strip()
    if not TOKEN_RE.fullmatch(tok):
        raise CanvasError("the bridge token file does not hold a token; set AGENT_BRIDGE_TOKEN")
    return tok


def bridge_call(method: str, path: str, payload: dict | None = None, timeout: float = 60):
    body = json.dumps(payload).encode() if payload is not None else None
    return _req(method, "/api/agent_bridge/" + path.lstrip("/"), body,
                headers={"X-Agent-Bridge-Token": _bridge_token()}, timeout=timeout)


# --------------------------------------------------------------------------- node schema
def _spec_inputs(info: dict) -> list[tuple[str, object, dict]]:
    """(name, type, options) for required then optional inputs, in the node's declared order."""
    order = info.get("input_order") or {}
    out = []
    for sect in ("required", "optional"):
        d = (info.get("input") or {}).get(sect) or {}
        for name in order.get(sect) or list(d):
            if name not in d:
                continue
            spec = d[name]
            typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            out.append((name, typ, opts))
    return out


def _is_widget(typ, opts: dict) -> bool:
    if opts.get("forceInput"):
        return False
    return isinstance(typ, list) or typ in WIDGET_TYPES


def _default(typ, opts: dict):
    if "default" in opts:
        return opts["default"]
    if isinstance(typ, list):
        return typ[0] if typ else ""
    if typ == "COMBO":
        return (opts.get("options") or [""])[0]
    return {"INT": 0, "FLOAT": 0.0, "BOOLEAN": False}.get(typ, "")


def widget_layout(class_type: str, oi: dict) -> list[tuple[str, str]]:
    """The node's widgets_values layout as (widget name, role); role is 'value', 'control' or 'upload'."""
    if class_type in VIRTUAL:
        return [("text", "value")]
    info = oi.get(class_type)
    if info is None:
        return []
    out = []
    for name, typ, opts in _spec_inputs(info):
        if not _is_widget(typ, opts):
            continue
        out.append((name, "value"))
        if opts.get("control_after_generate"):
            out.append((f"{name}.control_after_generate", "control"))
        if opts.get("image_upload"):
            out.append((f"{name}.upload", "upload"))
    return out


# --------------------------------------------------------------------------- build
def build(spec: dict) -> tuple[dict, dict]:
    oi = object_info()
    nodes_spec: dict = spec["nodes"]
    missing = sorted({n["class_type"] for n in nodes_spec.values()
                      if n["class_type"] not in oi and n["class_type"] not in VIRTUAL})
    if missing:  # the cache may predate a restart that added nodes
        oi = object_info(fresh=True)
        missing = [c for c in missing if c not in oi]
    if missing:
        raise CanvasError(f"node classes not installed in this ComfyUI: {missing} (find the right class with `comfy_canvas.py nodes QUERY` or the search_nodes tool)")

    nodes_spec = {k: {**n, "inputs": _expand_autogrow(n, oi)} for k, n in nodes_spec.items()}
    keys = list(nodes_spec)
    ids = {k: i + 1 for i, k in enumerate(keys)}
    deps = {k: [v[0] for v in (nodes_spec[k].get("inputs") or {}).values() if _is_link(v)] for k in keys}
    for k, ds in deps.items():
        for d in ds:
            if d not in ids:
                raise CanvasError(f"node {k!r} links to unknown node {d!r}")
    depth, order = _depths(keys, deps)

    ui_nodes, links, api, defaults = {}, [], {}, {}
    for k in keys:
        n = nodes_spec[k]
        ct, inputs = n["class_type"], dict(n.get("inputs") or {})
        node = {"id": ids[k], "type": ct, "pos": [0, 0], "size": [0, 0], "flags": {}, "order": order[k],
                "mode": MODES.get(n.get("mode", "active"), 0), "inputs": [], "outputs": [],
                "properties": {"Node name for S&R": ct}, "widgets_values": []}
        if n.get("title"):
            node["title"] = n["title"]
        n_multiline = 0
        if ct in VIRTUAL:
            node["widgets_values"] = [inputs.get("text", "")]
            n_multiline = 3
        else:
            info = oi[ct]
            socket_inputs, widget_inputs = [], []
            required = set(((info.get("input") or {}).get("required") or {}))
            for name, typ, opts in _spec_inputs(info):
                if typ == AUTOGROW:
                    slots = _autogrow_slots(name, opts)
                    used = [s for s in slots if f"{name}.{s}" in inputs]
                    shown = slots[: min(len(slots), (slots.index(used[-1]) + 2) if used else 1)]
                    for s in shown:  # the frontend always shows one spare slot after the last connected one
                        val = inputs.pop(f"{name}.{s}", None)
                        socket_inputs.append({"name": f"{name}.{s}", "label": s, "type": _autogrow_type(opts),
                                              "link": ("PENDING", val) if _is_link(val) else None})
                    continue
                val = inputs.pop(name, None)
                if _is_widget(typ, opts):
                    # frontend >= 1.16 lists every widget as an input slot (after the sockets), linked or not
                    widget_inputs.append({"name": name, "type": _tname(typ), "widget": {"name": name},
                                          "link": ("PENDING", val) if _is_link(val) else None})
                    if _is_link(val):
                        val = None
                    elif val is None and name in required and "default" in opts:
                        # a required widget the spec leaves out (e.g. one a newer ComfyUI added) runs at its
                        # default, as it would from the canvas: the API export needs it written out
                        defaults.setdefault(k, {})[name] = opts["default"]
                    node["widgets_values"].append(_default(typ, opts) if val is None else val)
                    if opts.get("control_after_generate"):
                        node["widgets_values"].append("fixed")
                    if opts.get("image_upload"):
                        node["widgets_values"].append("image")
                    if opts.get("multiline"):
                        n_multiline += 1
                else:
                    socket_inputs.append({"name": name, "type": _tname(typ),
                                          "link": ("PENDING", val) if _is_link(val) else None})
                    if val is not None and not _is_link(val):
                        raise CanvasError(f"{k}.{name} is a {typ} socket; give a link [node_key, output_index]")
            if inputs:
                raise CanvasError(f"{k} ({ct}) has no inputs named {sorted(inputs)}")
            node["inputs"] = socket_inputs + widget_inputs
            names = info.get("output_name") or info.get("output") or []
            node["outputs"] = [{"name": names[i] if i < len(names) else str(t), "type": _tname(t), "links": None,
                                "slot_index": i} for i, t in enumerate(info.get("output") or [])]
            api[str(ids[k])] = {"class_type": ct, "inputs": {}, "_meta": {"title": n.get("title") or ct}}
        n_widgets = len(node["widgets_values"])
        n_sockets = sum(1 for i in node["inputs"] if "widget" not in i)
        if ct in IMAGE_VIEWERS:
            auto = [380, 440]
        else:
            auto = [420 if ct in VIRTUAL else 340,
                    40 + 24 * max(n_sockets, len(node["outputs"])) + 26 * n_widgets + 90 * n_multiline]
        node["size"] = n.get("size") or auto
        ui_nodes[k] = node

    # links, now that every node's slots exist
    for k in keys:
        node = ui_nodes[k]
        for slot, inp in enumerate(node["inputs"]):
            if not (isinstance(inp["link"], tuple) and inp["link"][0] == "PENDING"):
                continue
            src_key, src_slot = inp["link"][1]
            src = ui_nodes[src_key]
            if src_slot >= len(src["outputs"]):
                raise CanvasError(f"{k}.{inp['name']} links to output {src_slot} of {src_key}, which has "
                         f"{len(src['outputs'])} outputs")
            lid = len(links) + 1
            out = src["outputs"][src_slot]
            out["links"] = (out["links"] or []) + [lid]
            inp["link"] = lid
            links.append([lid, src["id"], src_slot, node["id"], slot, out["type"]])
        if str(ids[k]) in api:
            for name, val in (nodes_spec[k].get("inputs") or {}).items():
                api[str(ids[k])]["inputs"][name] = [str(ids[val[0]]), val[1]] if _is_link(val) else val
            for name, val in defaults.get(k, {}).items():
                api[str(ids[k])]["inputs"].setdefault(name, val)

    groups = _layout(spec, keys, ui_nodes, depth)
    ui = {"last_node_id": len(keys), "last_link_id": len(links), "nodes": list(ui_nodes.values()),
          "links": links, "groups": groups, "config": {},
          "extra": {"ds": {"scale": 0.75, "offset": [0, 0]}, "agent": spec.get("meta") or {}}, "version": 0.4}
    return ui, api


AUTOGROW = "COMFY_AUTOGROW_V3"


def _autogrow_slots(name: str, opts: dict) -> list[str]:
    t = opts.get("template") or {}
    if t.get("names"):
        return list(t["names"])
    return [f"{t.get('prefix', name)}{i}" for i in range(int(t.get("max", 10)))]


def _autogrow_type(opts: dict) -> str:
    inner = ((opts.get("template") or {}).get("input") or {})
    for sect in ("required", "optional"):
        for spec in (inner.get(sect) or {}).values():
            return _tname(spec[0] if isinstance(spec, (list, tuple)) else spec)
    return "*"


def _expand_autogrow(node: dict, oi: dict) -> dict:
    """Autogrow inputs (V3 variable-length sockets) may be given as a list of links under the input's name,
    e.g. "candidates": [["bendA", 0], ["bendB", 0]]; they become the dotted keys ComfyUI uses
    ("candidates.candidate0", …). Dotted keys given directly are kept."""
    inputs = dict(node.get("inputs") or {})
    info = oi.get(node["class_type"])
    if not info:
        return inputs
    for name, typ, opts in _spec_inputs(info):
        if typ != AUTOGROW or not isinstance(inputs.get(name), list) or _is_link(inputs[name]):
            continue
        slots = _autogrow_slots(name, opts)
        values = inputs.pop(name)
        if len(values) > len(slots):
            raise CanvasError(f"{node['class_type']}.{name} takes at most {len(slots)} entries")
        for s, v in zip(slots, values):
            if v is not None:
                inputs[f"{name}.{s}"] = v
    return inputs


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def _tname(t) -> str:
    if isinstance(t, list):
        return "COMBO"
    return "*" if str(t).startswith("COMFY_MATCHTYPE") else str(t)


def _depths(keys, deps):
    depth, order, seen, stack = {}, {}, set(), set()

    def visit(k):
        if k in depth:
            return depth[k]
        if k in stack:
            raise CanvasError(f"cycle through node {k!r}")
        stack.add(k)
        depth[k] = 1 + max((visit(d) for d in deps[k]), default=-1)
        stack.discard(k)
        order[k] = len(order)
        return depth[k]

    for k in keys:
        visit(k)
    return depth, order


def _layout(spec, keys, ui_nodes, depth) -> list[dict]:
    """Group bands stacked top to bottom (spec order), columns by graph depth so lanes line up across bands."""
    gspecs = spec.get("groups") or []
    band_of = {k: spec["nodes"][k].get("group") or "" for k in keys}
    unknown = {g for g in band_of.values() if g} - {g["id"] for g in gspecs}
    if unknown:
        raise CanvasError(f"nodes reference undeclared groups: {sorted(unknown)}")
    bands = [""] + [g["id"] for g in gspecs]
    y = Y0
    boxes = {}
    for band in bands:
        members = [k for k in keys if band_of[k] == band]
        if not members:
            continue
        col_y: dict[int, float] = {}
        top = y + (40 if band else 0)
        for k in members:
            node, n = ui_nodes[k], spec["nodes"][k]
            if n.get("pos"):
                node["pos"] = list(n["pos"])
                continue
            d = depth[k]
            node["pos"] = [X0 + d * COL_W, col_y.get(d, top)]
            col_y[d] = col_y.get(d, top) + node["size"][1] + ROW_GAP + 30
        xs = [ui_nodes[k]["pos"][0] for k in members]
        ys = [ui_nodes[k]["pos"][1] for k in members]
        x2 = [ui_nodes[k]["pos"][0] + ui_nodes[k]["size"][0] for k in members]
        y2 = [ui_nodes[k]["pos"][1] + ui_nodes[k]["size"][1] for k in members]
        boxes[band] = [min(xs) - 20, min(ys) - 70, max(x2) - min(xs) + 40, max(y2) - min(ys) + 100]
        y = max(y2) + BAND_GAP
    out = []
    for i, g in enumerate(gspecs):
        if g["id"] in boxes:
            out.append({"id": i + 1, "title": g.get("title", g["id"]), "bounding": boxes[g["id"]],
                        "color": g.get("color", "#3f789e"), "font_size": 24, "flags": {}})
    return out


# --------------------------------------------------------------------------- read back
def _group_of(node: dict, groups: list[dict]) -> str | None:
    x, y = node["pos"][:2] if isinstance(node["pos"], list) else (node["pos"]["0"], node["pos"]["1"])
    for g in groups:
        gx, gy, gw, gh = g["bounding"]
        if gx <= x <= gx + gw and gy <= y <= gy + gh:
            return g["title"]
    return None


def _named_widgets(node: dict, oi: dict) -> dict:
    layout = widget_layout(node["type"], oi)
    vals = node.get("widgets_values")
    if isinstance(vals, dict):  # some nodes save a dict
        return dict(vals)
    vals = vals or []
    if not layout:
        return {f"w{i}": v for i, v in enumerate(vals)}
    return {name: vals[i] for i, (name, role) in enumerate(layout) if role == "value" and i < len(vals)}


def summarize(wf: dict, oi: dict) -> dict:
    groups = wf.get("groups") or []
    notes, primitives, switches, group_modes = [], [], [], {}
    for n in wf.get("nodes") or []:
        t, title = n["type"], n.get("title") or n["type"]
        g = _group_of(n, groups)
        if g:
            group_modes.setdefault(g, []).append(MODE_NAMES.get(n.get("mode", 0), str(n.get("mode"))))
        w = _named_widgets(n, oi)
        if t in VIRTUAL:
            notes.append({"id": n["id"], "title": title, "group": g, "text": (n.get("widgets_values") or [""])[0]})
        elif t.startswith("Primitive"):
            primitives.append({"id": n["id"], "title": title, "group": g, "value": w.get("value")})
        elif "Switch" in t:
            switches.append({"id": n["id"], "title": title, "group": g, "type": t, "widgets": w,
                             "mode": MODE_NAMES.get(n.get("mode", 0))})
    groups_state = {}
    for g, modes in group_modes.items():
        groups_state[g] = "active" if all(m == "active" for m in modes) else \
            ("off" if all(m != "active" for m in modes) else "partial")
    return {"agent_meta": (wf.get("extra") or {}).get("agent"), "notes": notes, "primitives": primitives,
            "switches": switches, "groups": groups_state}


def diff(base: dict, cur: dict, oi: dict) -> list[str]:
    bn = {n["id"]: n for n in base.get("nodes") or []}
    cn = {n["id"]: n for n in cur.get("nodes") or []}
    out = []
    for i, n in cn.items():
        label = f"#{i} {n.get('title') or n['type']}"
        if i not in bn:
            out.append(f"{label}: ADDED by user ({n['type']})")
            continue
        b = bn[i]
        if b.get("mode", 0) != n.get("mode", 0):
            out.append(f"{label}: mode {MODE_NAMES.get(b.get('mode', 0))} -> {MODE_NAMES.get(n.get('mode', 0))}")
        if (b.get("title") or "") != (n.get("title") or ""):
            out.append(f"{label}: title {b.get('title')!r} -> {n.get('title')!r}")
        bw, cw = _named_widgets(b, oi), _named_widgets(n, oi)
        for name in sorted(set(bw) | set(cw)):
            if bw.get(name) != cw.get(name):
                out.append(f"{label}.{name}: {_short(bw.get(name))} -> {_short(cw.get(name))}")
    for i, b in bn.items():
        if i not in cn:
            out.append(f"#{i} {b.get('title') or b['type']}: DELETED by user")
    bl = {tuple(link[1:5]) for link in base.get("links") or [] if isinstance(link, list)}
    cl = {tuple(link[1:5]) for link in cur.get("links") or [] if isinstance(link, list)}
    if bl != cl:
        out.append(f"links: {len(cl - bl)} added, {len(bl - cl)} removed (user rewired the graph)")
    return out


def _short(v, n=160):
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------- history
def _history(n: int) -> list[tuple[str, dict]]:
    h = _req("GET", f"/api/history?max_items={max(n, 1)}")
    items = sorted(h.items(), key=lambda kv: kv[1]["prompt"][0], reverse=True)
    return items[:n]


def describe_run(pid: str, e: dict) -> dict:
    prompt, extra = e["prompt"][2], e["prompt"][3] if len(e["prompt"]) > 3 else {}
    wf = ((extra or {}).get("extra_pnginfo") or {}).get("workflow") or {}
    titles = {str(n["id"]): n.get("title") for n in wf.get("nodes") or []}
    steering = {}
    for nid, node in prompt.items():
        ct = node["class_type"]
        if any(h in ct for h in STEERING_HINTS):
            vals = {k: v for k, v in node["inputs"].items() if not _is_link(v)}
            steering[f"#{nid} {titles.get(nid) or node.get('_meta', {}).get('title') or ct}"] = vals
    images, texts = [], []
    for nid, out in (e.get("outputs") or {}).items():
        ttl = titles.get(nid) or (prompt.get(nid) or {}).get("_meta", {}).get("title")
        for t in out.get("text") or []:  # PreviewAny / AgentReportSink, e.g. a bend report
            texts.append({"node": nid, "title": ttl, "text": t if len(t) <= 6000 else t[:6000] + "\n..."})
        animated = any(out.get("animated") or [])  # SaveAnimatedWEBP / SaveVideo mark their outputs
        for im in (out.get("images") or []) + (out.get("gifs") or []) + (out.get("videos") or []):
            q = urllib.parse.urlencode({k: im.get(k, "") for k in ("filename", "subfolder", "type")})
            title = titles.get(nid) or (prompt.get(nid) or {}).get("_meta", {}).get("title")
            name = str(im.get("filename", "")).lower()
            video = (animated or "gifs" in out or "videos" in out) and name.endswith(VIDEO_EXT) \
                or name.endswith((".mp4", ".webm", ".mov", ".mkv"))
            images.append({"node": nid, "title": title, "url": f"{BASE}/api/view?{q}", **im,
                           **({"video": True} if video else {})})
    status = e.get("status") or {}
    errors = [m[1] for m in status.get("messages") or [] if m[0] == "execution_error"]
    canvas = summarize(wf, object_info()) if wf.get("nodes") else None
    if canvas:
        canvas.pop("agent_meta", None)
    return {"prompt_id": pid, "number": e["prompt"][0], "status": status.get("status_str"),
            "agent_meta": (wf.get("extra") or {}).get("agent"), "steering_inputs": steering,
            "canvas": canvas, "texts": texts, "images": images,
            "errors": [{k: err.get(k) for k in ("node_id", "node_type", "exception_type", "exception_message")}
                       for err in errors]}


def _output_key(d: dict) -> tuple[str, str, str]:
    return str(d.get("filename") or ""), str(d.get("subfolder") or ""), str(d.get("type") or "output")


def runs_for_urls(urls: list[str], max_items: int = 300) -> dict[str, tuple[str, dict]]:
    """The finished run behind each /api/view URL, from ComfyUI's history: {url: (prompt_id, entry)}. URLs no run in
    the last max_items made (an uploaded picture, an old run) are left out."""
    want = {}
    for u in urls:
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(u or "").query)
        if q.get("filename"):
            want[_output_key({k: (q.get(k) or [""])[0] for k in ("filename", "subfolder", "type")})] = u
    if not want:
        return {}
    found: dict[str, tuple[str, dict]] = {}
    for pid, e in _req("GET", f"/api/history?max_items={max_items}").items():
        for out in (e.get("outputs") or {}).values():
            for im in (out.get("images") or []) + (out.get("gifs") or []) + (out.get("videos") or []):
                u = want.get(_output_key(im))
                if u and u not in found:
                    found[u] = (pid, e)
    return found


def bend_reports(entry: dict) -> list[dict]:
    """The bend reports a run showed (Apply Bends from JSON's `report` output in a PreviewAny or report sink)."""
    reports = []
    for out in (entry.get("outputs") or {}).values():
        for t in out.get("text") or []:
            try:
                d = json.loads(t)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(d, dict) and "bends" not in d and ({"clamped", "warnings", "clamp", "resolved"} & set(d)):
                reports.append(d)
    return reports


def run_bends(entry: dict) -> list[dict]:
    """The bends a run asked for, as written in its Apply Bends from JSON boxes (with their labels and kb sources)."""
    bends = []
    for node in entry["prompt"][2].values():
        if node.get("class_type") == "ApplyBendsFromJSON" and isinstance(node["inputs"].get("bends_json"), str):
            try:
                bends += json.loads(node["inputs"]["bends_json"]).get("bends") or []
            except json.JSONDecodeError:
                continue
    return bends


def is_bent(entry: dict) -> bool:
    """Whether a run bends its model at all (an unbent run is a baseline)."""
    for n in entry["prompt"][2].values():
        ct = n.get("class_type", "")
        if ct == "ApplyBendsFromJSON":
            raw = n["inputs"].get("bends_json")
            try:
                doc = json.loads(raw) if isinstance(raw, str) else None
            except json.JSONDecodeError:
                doc = None
            if doc is None or doc.get("bends") or doc.get("attention_bends"):  # a linked document counts as bent
                return True
        elif "Bend" in ct and "Catalogue" not in ct:
            return True
    return False


# --------------------------------------------------------------------------- API (used by the CLI and mcp_server.py)
def use_start_image(spec: dict, image: str, denoise: float | None = None) -> tuple[dict, list[str]]:
    """The spec rewritten to start from a picture in ComfyUI's input folder (see "Starting picture" above);
    returns (new spec, notes for the caller)."""
    spec = json.loads(json.dumps(spec))
    nodes: dict = spec["nodes"]
    notes: list[str] = []
    if not image:
        raise CanvasError("no starting picture given")
    if denoise is not None and not 0 < denoise <= 1:
        raise CanvasError("denoise must be in (0, 1]: lower keeps more of the picture")
    meta = spec.setdefault("meta", {})
    meta["start_image"] = image
    if nodes.get("start_image", {}).get("class_type") == "LoadImage":
        nodes["start_image"]["inputs"]["image"] = image
        notes.append("set the starting picture of the existing start_image node")
        return spec, notes
    video = sorted({n["class_type"] for n in nodes.values() if n["class_type"] in EMPTY_VIDEO_LATENTS})
    if video:
        raise CanvasError(f"this is a text-to-video workflow ({', '.join(video)}); starting a video from a picture "
                          f"needs an image-to-video model and workflow (the video_sweep_wan21_i2v preset)")
    latents = [k for k, n in nodes.items() if n["class_type"] in EMPTY_IMAGE_LATENTS]
    if not latents:
        raise CanvasError(f"no empty image latent ({', '.join(sorted(EMPTY_IMAGE_LATENTS))}) to replace in this spec")
    vae = next((n["inputs"].get("vae") for n in nodes.values()
                if n["class_type"] in ("VAEDecode", "VAEDecodeTiled") and _is_link(n["inputs"].get("vae"))), None)
    if vae is None:
        raise CanvasError("cannot find the VAE this spec decodes with (no VAEDecode with a linked vae)")
    first_group = nodes[latents[0]].get("group")
    nodes["start_image"] = {"class_type": "LoadImage", "title": "Starting picture", "inputs": {"image": image},
                            **({"group": first_group} if first_group else {})}
    for k in latents:
        n = nodes[k]
        w, h = n["inputs"].get("width", 512), n["inputs"].get("height", 512)
        batch = int(n["inputs"].get("batch_size", 1) or 1)
        extra = {"group": n["group"]} if n.get("group") else {}
        # the picture keeps its shape: scaled to the latent's pixel count, never cropped to the latent's shape
        # (VAEEncode trims the last few pixels to a multiple of 8)
        nodes[f"{k}_fit"] = {"class_type": "ImageScaleToTotalPixels", "title": f"Fit to about {w}×{h} pixels",
                             **extra, "inputs": {"image": ["start_image", 0], "upscale_method": "lanczos",
                                                 "megapixels": max(0.01, round(w * h / 2**20, 2))}}
        enc = {"class_type": "VAEEncode", "inputs": {"pixels": [f"{k}_fit", 0], "vae": vae}}
        if batch > 1:
            nodes[f"{k}_enc"] = {**enc, **extra}
            nodes[k] = {"class_type": "RepeatLatentBatch", "title": n.get("title") or f"Starting picture × {batch}",
                        **extra, "inputs": {"samples": [f"{k}_enc", 0], "amount": batch}}
        else:
            nodes[k] = {**enc, "title": n.get("title") or "Starting picture (encoded)", **extra}
    d = 0.6 if denoise is None else denoise
    for k, n in nodes.items():
        src = (n.get("inputs") or {}).get("latent_image")
        if not (_is_link(src) and src[0] in latents):
            continue
        if n["class_type"] == "KSampler":
            if denoise is not None or float(n["inputs"].get("denoise", 1.0)) >= 0.999:
                n["inputs"]["denoise"] = d
        else:
            notes.append(f"{k} ({n['class_type']}) has no denoise input: set how much it repaints yourself "
                         f"(e.g. start_at_step)")
    meta["denoise"] = d
    notes.append(f"replaced {', '.join(latents)} with the starting picture; KSampler denoise {d}; the picture keeps "
                 f"its shape (same pixel count as the latent, not cropped)")
    if d <= 0.7:
        notes.append("at denoise <= 0.7 no executed step reaches the structure window (t 1-0.7): bends can restyle "
                     "the picture but not re-compose it")
    return spec, notes


def upload_image(src: str, name: str = "") -> dict:
    """Put a picture into ComfyUI's input folder (subfolder agent_bending). src: a path on this computer or a
    ComfyUI /api/view URL. Returns {"image": value for LoadImage, ...}. Same picture, same name: no duplicates."""
    if src.startswith(("http://", "https://")):
        data = fetch_bytes(src)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(src).query)
        orig = (q.get("filename") or ["picture.png"])[0]
    else:
        if not is_local():
            raise CanvasError(f"ComfyUI at {BASE} is on another machine, and pictures from this computer are not sent "
                              f"there: upload the picture in that ComfyUI (a Load Image box), then pick it with "
                              f"list_input_images (script: inputs)")
        path = Path(src.strip().strip('"')).expanduser()
        if not path.is_file():
            raise CanvasError(f"no file at {path} (give the full path to the picture on this computer)")
        allowed = [Path(d).expanduser().resolve() for d in (_env("COMFY_BENDING_PICTURE_DIRS") or "").split(os.pathsep)
                   if d.strip()]
        if allowed and not any(path.resolve().is_relative_to(d) for d in allowed):
            raise CanvasError(f"{path} is outside the picture folders set in COMFY_BENDING_PICTURE_DIRS")
        data, orig = path.read_bytes(), path.name
    ext = Path(orig).suffix.lower()
    if ext not in UPLOAD_EXT or not _looks_like_image(data):
        raise CanvasError(f"{orig}: not a picture ComfyUI can load ({', '.join(sorted(UPLOAD_EXT))})")
    return upload_bytes(data, orig, name)


def upload_bytes(data: bytes, orig: str, name: str = "") -> dict:
    """Upload picture bytes into agent_bending/, checked by their first bytes whatever the name says; the file
    extension follows the picture's real format when the name's does not match it."""
    real = _image_ext(data)
    if not real:
        raise CanvasError(f"{orig or 'that file'}: not a picture ComfyUI can load (PNG, JPEG, WEBP, BMP or TIFF)")
    ext = Path(orig).suffix.lower()
    if {".jpeg": ".jpg", ".tif": ".tiff"}.get(ext, ext) != real:
        ext = real
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (name or Path(orig).stem))[:60] or "picture"
    filename = f"{stem}_{hashlib.sha256(data).hexdigest()[:8]}{ext}"
    r = _post_file(data, filename, "input")
    value = f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"]
    return {"image": value, "bytes": len(data), "view_url": r["view_url"],
            "use": f'LoadImage "image": "{value}", or build with start_image="{value}"'}


def save_output(data: bytes, filename: str) -> dict:
    """Put a file the skill made (an animation) into ComfyUI's output folder, under agent_bending/, next to the
    pictures ComfyUI saves. Returns {"file": "agent_bending/<name>", "view_url"}; the view URL plays it in a browser."""
    r = _post_file(data, filename, "output")
    return {"file": f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"], "view_url": r["view_url"]}


def _post_file(data: bytes, filename: str, folder: str) -> dict:
    """POST one file to ComfyUI's /api/upload/image into <folder>/agent_bending/ (folder: input or output)."""
    boundary = f"----agentbending{hashlib.sha1(data[:64] + filename.encode()).hexdigest()}"
    parts = []
    for field, value in (("subfolder", INPUT_SUBFOLDER), ("type", folder), ("overwrite", "true")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    r = _req("POST", "/api/upload/image", body, ctype=f"multipart/form-data; boundary={boundary}")
    if not isinstance(r, dict) or "name" not in r:
        raise CanvasError(f"upload failed: {str(r)[:300]}")
    q = urllib.parse.urlencode({"filename": r["name"], "subfolder": r.get("subfolder", ""), "type": folder})
    return {**r, "view_url": f"{BASE}/api/view?{q}"}


def _image_ext(data: bytes) -> str | None:
    """PNG, JPEG, WEBP, BMP or TIFF by their first bytes (so only pictures are ever uploaded, whatever the name)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith(b"BM"):
        return ".bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    return None


def _looks_like_image(data: bytes) -> bool:
    return _image_ext(data) is not None


def input_images() -> list[str]:
    """Pictures in ComfyUI's input folder, as LoadImage lists them (its top level; uploads by this skill go to
    agent_bending/ and are not listed here)."""
    info = object_info(fresh=True).get("LoadImage") or {}
    spec = ((info.get("input") or {}).get("required") or {}).get("image") or [[]]
    return list(spec[0]) if isinstance(spec[0], list) else []


def build_files(spec: dict, ui_path: Path | None = None, api_path: Path | None = None) -> tuple[dict, dict]:
    ui, api = build(spec)
    if ui_path:
        Path(ui_path).write_text(json.dumps(ui, indent=1), encoding="utf-8")
    if api_path:
        Path(api_path).write_text(json.dumps(api, indent=1), encoding="utf-8")
    return ui, api


def push_workflow(ui: dict, name: str) -> str:
    """Save a frontend-format graph into the user's Workflows sidebar, inside the agent's own folder (see
    agent_workflow_name); returns its sidebar path. Only the agent's own workflows are ever overwritten."""
    name = agent_workflow_name(name)
    ui = fit_bends(ui=ui)[1]
    _req("POST", _userdata_path(name) + "?overwrite=true", json.dumps(ui).encode())
    return f"workflows/{name}.json"


def pull_workflow(name: str, baseline: dict | None = None) -> tuple[dict, dict]:
    """(summary, raw graph) of a saved workflow; with a baseline, summary['changes_since_push'] lists edits."""
    wf = _req("GET", _userdata_path(name))
    if not isinstance(wf, dict):
        raise CanvasError(f"workflow {name!r} not found or not JSON")
    oi = object_info()
    out = summarize(wf, oi)
    if baseline is not None:
        out["changes_since_push"] = diff(baseline, wf, oi)
    return out, wf


def queue_prompt(api: dict, ui: dict | None = None, wait: bool = False, timeout: float = 900) -> dict:
    """Queue an API prompt (embedding the canvas graph for provenance); with wait, return the run's description."""
    api, ui, _ = fit_bends(api, ui)
    payload = {"prompt": api, "client_id": "agent-model-bending"}
    if ui is not None:
        payload["extra_data"] = {"extra_pnginfo": {"workflow": ui}}
    r = _req("POST", "/api/prompt", json.dumps(payload).encode())
    if r.get("node_errors"):
        raise CanvasError(f"node errors: {json.dumps(r['node_errors'])[:1500]}")
    pid = r["prompt_id"]
    return _wait_for(pid, timeout) if wait else {"prompt_id": pid, "status": "queued"}


def last_runs(n: int = 1) -> list[dict]:
    return [describe_run(pid, e) for pid, e in _history(n)]


def wait_run(after: str | None = None, timeout: float = 900) -> dict:
    """Block until a prompt newer than `after` (default: the newest now) finishes; return its description."""
    after = after or next((pid for pid, _ in _history(1)), None)
    deadline = time.time() + timeout
    while time.time() < deadline:
        items = _history(1)
        if items and items[0][0] != after and (items[0][1].get("status") or {}).get("completed") is not None:
            return describe_run(*items[0])
        time.sleep(2)
    raise CanvasError(f"no new run within {timeout:.0f} s (last seen {after})")


def propose(ui: dict, name: str, message: str = "", session: str | None = None, round_: int | None = None,
            save: bool = True) -> dict:
    ui = fit_bends(ui=ui)[1]
    r = bridge_call("POST", "propose", {"name": name, "workflow": ui, "message": message, "session": session,
                                        "round": round_, "save": save})
    return r["proposal"]


def events(since: int = 0, session: str | None = None, kinds: str = "", wait: float = 0) -> dict:
    q = urllib.parse.urlencode({k: v for k, v in {"since": since, "session": session, "kinds": kinds,
                                                    "wait": wait}.items() if v not in (None, "")})
    return bridge_call("GET", f"events?{q}", timeout=wait + 30)


def read_canvas() -> tuple[dict, dict]:
    """(summary, raw graph) of the user's active canvas; only while they share it."""
    r = bridge_call("GET", "canvas")
    out = summarize(r["workflow"], object_info())
    out["name"], out["shared_at"] = r.get("name"), r.get("ts")
    return out, r["workflow"]


def notify(message: str, severity: str = "info", session: str | None = None) -> int:
    r = bridge_call("POST", "notify", {"message": message, "severity": severity, "session": session})
    return int(r.get("clients", 0))


def status() -> dict:
    """Is ComfyUI reachable; free VRAM; whether Model-Bending is installed and agent-ready (ApplyBendsFromJSON has a
    `report` output); whether the ComfyUI-Agent-Bridge pack answers."""
    stats = _req("GET", "/api/system_stats")
    oi = object_info(fresh=True)
    ab = oi.get("ApplyBendsFromJSON") or {}
    bridge = bridge_info()
    if bridge:
        bridge = {k: v for k, v in bridge.items() if k != "token_file"}
    return {
        "comfyui": BASE,
        "comfyui_version": (stats.get("system") or {}).get("comfyui_version"),
        "devices": [{k: d.get(k) for k in ("name", "vram_total", "vram_free")} for d in stats.get("devices") or []],
        "model_bending": {"installed": bool(ab), "agent_ready": "report" in (ab.get("output_name") or []),
                          # Model-Bending 0.3+: WAN video bending (Attention Map Bending, temporal ops, and the
                          # bends JSON "attention_bends" list); experimental upstream
                          "video_ready": "Attention Map Bending" in oi and "Frame Ramp (Bending)" in oi,
                          # Model-Bending 0.3.1+ keeps a bend's "kb" key; before that the skill drops it from
                          # what it sends (fit_bends) and keeps it in its own results
                          "keeps_kb": bool(ab) and bends_accept_kb(fresh=True),
                          "nodes": sorted(k for k, v in oi.items() if "model_bending" in (v.get("category") or ""))},
        "video_models": {"image_to_video": "WanImageToVideo" in oi, "save_animated_webp": "SaveAnimatedWEBP" in oi},
        "agent_bridge": bridge or {"installed": False},
    }


def search_nodes(query: str, limit: int = 30) -> list[dict]:
    """Node classes whose name, display name or category contains every word of the query."""
    words = query.lower().split()
    out = []
    for k, v in object_info().items():
        hay = f"{k} {v.get('display_name', '')} {v.get('category', '')}".lower()
        if all(w in hay for w in words):
            out.append({"class_type": k, "display_name": v.get("display_name"), "category": v.get("category")})
    return out[:limit]


def node_info(class_type: str) -> dict:
    """Inputs (types, defaults, ranges, choices) and outputs of one node class, as ComfyUI reports them."""
    info = object_info().get(class_type) or object_info(fresh=True).get(class_type)
    if not info:
        near = [n["class_type"] for n in search_nodes(class_type, 8)]
        raise CanvasError(f"unknown node class {class_type!r}" + (f"; similar: {near}" if near else
                                                                   "; search for it (`comfy_canvas.py nodes QUERY`, or the search_nodes tool)"))
    return {k: info.get(k) for k in ("display_name", "category", "description", "input", "input_order", "output",
                                      "output_name", "output_node")}


def free_memory() -> dict:
    """Unload all models and free VRAM, then report free VRAM. ComfyUI reloads models on the next run."""
    _req("POST", "/api/free", json.dumps({"unload_models": True, "free_memory": True}).encode())
    stats = _req("GET", "/api/system_stats")
    return {"devices": [{k: d.get(k) for k in ("name", "vram_total", "vram_free")} for d in stats.get("devices") or []]}


def server_logs(grep: str | None = None, tail: int = 200) -> str:
    r = _req("GET", "/internal/logs/raw")
    lines = "".join(e["m"] for e in (r.get("entries") or [])).replace("\r", "\n").splitlines()
    lines = [x for x in lines if x.strip() and (not grep or grep.lower() in x.lower())]
    return "\n".join(lines[-tail:])


def _wait_for(pid: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = _req("GET", f"/api/history/{pid}")
        if pid in h and (h[pid].get("status") or {}).get("completed") is not None:
            return describe_run(pid, h[pid])
        time.sleep(1)
    raise CanvasError(f"prompt {pid} did not finish within {timeout:.0f} s")


# --------------------------------------------------------------------------- CLI
def _print(obj, as_json: bool):
    if as_json:
        print(json.dumps(obj, indent=1, ensure_ascii=False))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{k}: {json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}")
    else:
        for x in obj:
            print(json.dumps(x, ensure_ascii=False))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("spec"); p.add_argument("-o", "--out", required=True)
    p.add_argument("--api"); p.add_argument("--start-image"); p.add_argument("--denoise", type=float)
    p = sub.add_parser("upload"); p.add_argument("src"); p.add_argument("--name", default="")
    sub.add_parser("inputs")
    p = sub.add_parser("push"); p.add_argument("ui"); p.add_argument("--name", required=True)
    p = sub.add_parser("pull"); p.add_argument("--name", required=True); p.add_argument("--baseline")
    p.add_argument("--save"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("queue"); p.add_argument("--api", required=True); p.add_argument("--ui")
    p.add_argument("--wait", action="store_true"); p.add_argument("--timeout", type=float, default=900)
    p = sub.add_parser("last-run"); p.add_argument("--n", type=int, default=1); p.add_argument("--json", action="store_true")
    p = sub.add_parser("wait-run"); p.add_argument("--after"); p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("logs"); p.add_argument("--grep"); p.add_argument("--tail", type=int, default=200)
    p = sub.add_parser("run-setup"); p.add_argument("--prompt-id", default="")
    p = sub.add_parser("bend-run"); p.add_argument("--bends"); p.add_argument("--fragment")
    p.add_argument("--name", required=True); p.add_argument("-o", "--out"); p.add_argument("--api", required=True)
    p.add_argument("--prompt-id", default=""); p.add_argument("--arch", default="")
    p.add_argument("--clamp", default="safe", choices=["none", "hard", "safe"]); p.add_argument("--only", default="")
    p.add_argument("--replace-bends", action="store_true")
    sub.add_parser("status")
    p = sub.add_parser("nodes"); p.add_argument("query"); p.add_argument("--limit", type=int, default=30)
    p = sub.add_parser("node"); p.add_argument("class_type")
    sub.add_parser("free")
    sub.add_parser("bridge")
    p = sub.add_parser("propose"); p.add_argument("ui"); p.add_argument("--name", required=True)
    p.add_argument("--message", default=""); p.add_argument("--session"); p.add_argument("--round", type=int)
    p.add_argument("--no-save", action="store_true")
    p = sub.add_parser("events"); p.add_argument("--since", type=int, default=0); p.add_argument("--session")
    p.add_argument("--kinds", default=""); p.add_argument("--wait", type=float, default=0)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("canvas"); p.add_argument("--save")
    p = sub.add_parser("notify"); p.add_argument("message")
    p.add_argument("--severity", default="info", choices=["info", "success", "warn", "error"])
    p.add_argument("--session")
    a = ap.parse_args(argv)
    try:
        _run(a)
    except CanvasError as e:
        sys.exit(str(e))
    except (OSError, ValueError) as e:  # missing file, bad JSON
        sys.exit(f"{type(e).__name__}: {e}")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run(a):
    if a.cmd == "build":
        spec = _load(a.spec)
        if a.start_image:
            spec, notes = use_start_image(spec, a.start_image, a.denoise)
            for n in notes:
                print(f"start image: {n}")
        ui, _ = build_files(spec, Path(a.out), Path(a.api) if a.api else None)
        print(f"wrote {a.out} ({len(ui['nodes'])} nodes, {len(ui['links'])} links, {len(ui['groups'])} groups)"
              + (f" and {a.api}" if a.api else ""))
    elif a.cmd == "upload":
        _print(upload_image(a.src, a.name), False)
    elif a.cmd == "inputs":
        _print(input_images(), False)
    elif a.cmd == "push":
        where = push_workflow(_load(a.ui), a.name)  # always under agent_bending/
        print(f"saved to the Workflows sidebar as '{where}' "
              f"(ComfyUI: Workflows panel -> open it; press Ctrl+S after editing so the agent can read it)")
    elif a.cmd == "pull":
        out, wf = pull_workflow(a.name, _load(a.baseline) if a.baseline else None)
        if a.save:
            Path(a.save).write_text(json.dumps(wf, indent=1), encoding="utf-8")
        _print(out, a.json)
    elif a.cmd == "queue":
        r = queue_prompt(_load(a.api), _load(a.ui) if a.ui else None, a.wait, a.timeout)
        print(f"queued prompt_id={r['prompt_id']}")
        if a.wait:
            _print(r, False)
    elif a.cmd == "last-run":
        _print(last_runs(a.n), a.json)
    elif a.cmd == "wait-run":
        _print(wait_run(a.after, a.timeout), a.json)
    elif a.cmd == "status":
        print(json.dumps(status(), indent=1, ensure_ascii=False))
    elif a.cmd == "nodes":
        _print(search_nodes(a.query, a.limit), False)
    elif a.cmd == "node":
        print(json.dumps(node_info(a.class_type), indent=1, ensure_ascii=False))
    elif a.cmd == "free":
        _print(free_memory(), False)
    elif a.cmd == "bridge":
        _print(bridge_info() or {"installed": False}, False)
    elif a.cmd == "propose":
        prop = propose(_load(a.ui), a.name, a.message, a.session, a.round, not a.no_save)
        print(f"proposed {prop['id']} to {prop.get('delivered_to_clients', 0)} open ComfyUI tab(s); "
              f"saved as workflows/{prop.get('saved_path')}. Poll: events --kinds proposal_status")
    elif a.cmd == "events":
        r = events(a.since, a.session, a.kinds, a.wait)
        if a.json:
            print(json.dumps(r, indent=1, ensure_ascii=False))
        else:
            for e in r["events"]:
                print(json.dumps(e, ensure_ascii=False))
            print(f"last: {r['last']}")
    elif a.cmd == "canvas":
        out, wf = read_canvas()
        if a.save:
            Path(a.save).write_text(json.dumps(wf, indent=1), encoding="utf-8")
        _print(out, False)
    elif a.cmd == "notify":
        print(f"notified {notify(a.message, a.severity, a.session)} tab(s)")
    elif a.cmd == "logs":
        print(server_logs(a.grep, a.tail))
    elif a.cmd in ("run-setup", "bend-run"):
        import splice  # imports this module back, so not at the top
        pid, entry = splice.load_run(a.prompt_id)
        api, ui = splice.run_graphs(entry)
        if a.cmd == "run-setup":
            run = describe_run(pid, entry)
            print(json.dumps({"prompt_id": pid, **splice.inspect(api, ui), "images": run["images"]}, indent=1,
                             ensure_ascii=False))
            return
        if a.bends and a.fragment or not (a.bends or a.fragment or a.replace_bends):
            raise CanvasError("give --bends (a file or JSON text) or --fragment (a spec file); --replace-bends alone "
                              "gives the run without its own bends")
        # a file with the JSON, or the JSON text itself (the web UI's "Copy Bends" format; see bendjson.py)
        bends = a.bends if not a.bends or a.bends.lstrip().startswith(("{", "[", "`")) \
            else Path(a.bends).read_text(encoding="utf-8")
        bent, bent_ui, info = splice.bend_copy(
            api, ui, bends, arch=a.arch, clamp=a.clamp, only=[s for s in a.only.split(",") if s], name=a.name,
            fragment=_load(a.fragment) if a.fragment else None, meta={"bent_copy_of": pid},
            replace_bends=a.replace_bends)
        Path(a.api).write_text(json.dumps(bent, indent=1), encoding="utf-8")
        if bent_ui is not None and a.out:
            Path(a.out).write_text(json.dumps(bent_ui, indent=1), encoding="utf-8")
        print(json.dumps({"from_run": pid, **info, "api": a.api,
                          "ui": a.out if bent_ui is not None and a.out else None}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
