#!/usr/bin/env python3
"""Bend a copy of a workflow the user already ran: read their run from ComfyUI's history, say what it is made of,
and insert a bend on the model link going into its samplers. Their model, LoRAs, prompt, sampler settings and seed are
kept, so the picture their run made is the unbent original. Their own workflow is never changed: the result is a new
API prompt to queue and, when the graph allows it, a copy of their canvas graph with the bend added (their layout,
groups and notes kept). Standard library only.

  python comfy_canvas.py run-setup [--prompt-id ID]
  python comfy_canvas.py bend-run --bends BENDS.json --name NAME -o UI.json --api API.json [--prompt-id ID]

A fragment is a small spec (comfy_canvas.py's format) inserted on the model link: one or more inputs linked to
["@model", 0] receive the user's model, and "out": [node_key, output_index] feeds their samplers. The default
fragment is Apply Bends from JSON with its report shown in a Preview Any box.
"""

from __future__ import annotations

import copy
import json

import comfy_canvas as cc
import kb

BEND_NODE = "ApplyBendsFromJSON"
MODEL_IN = "@model"
AGENT_CLIENT = "agent-model-bending"  # the client id comfy_canvas.queue_prompt queues with
# used when ComfyUI does not describe a node class (it always should: the class just ran)
KNOWN_SAMPLERS = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "BasicGuider", "CFGGuider", "DualCFGGuider"}
FRAGMENT_TITLE = "Bend added by the assistant"


# --------------------------------------------------------------------------- the user's run
def load_run(prompt_id: str = "") -> tuple[str, dict]:
    """(prompt id, history entry) of a finished run: the given one, or the newest one the user made themselves
    (runs queued by this skill are skipped)."""
    if prompt_id:
        h = cc._req("GET", f"/api/history/{prompt_id}")
        if prompt_id not in h:
            raise cc.CanvasError(f"no run {prompt_id!r} in ComfyUI's history")
        return prompt_id, h[prompt_id]
    for pid, e in cc._history(30):
        extra = e["prompt"][3] if len(e["prompt"]) > 3 and isinstance(e["prompt"][3], dict) else {}
        if extra.get("client_id") != AGENT_CLIENT:
            return pid, e
    raise cc.CanvasError("ComfyUI's history has no run made by the user yet: ask them to run their workflow once, "
                         "as it is")


def run_graphs(entry: dict) -> tuple[dict, dict | None]:
    """(API prompt, canvas graph or None) of a history entry, as copies."""
    prompt = entry["prompt"]
    extra = prompt[3] if len(prompt) > 3 and isinstance(prompt[3], dict) else {}
    ui = (extra.get("extra_pnginfo") or {}).get("workflow")
    return copy.deepcopy(prompt[2]), copy.deepcopy(ui) if isinstance(ui, dict) and ui.get("nodes") else None


# --------------------------------------------------------------------------- reading the graph
def _title(api: dict, nid: str) -> str:
    n = api.get(nid) or {}
    return f"#{nid} {(n.get('_meta') or {}).get('title') or n.get('class_type', '?')}"


def sinks(api: dict, oi: dict) -> list[dict]:
    """Where a model is used to make a picture: every linked MODEL input of a node that outputs a latent or a
    guider (samplers and guiders; not LoRA loaders, model patches or schedulers)."""
    out = []
    for nid, node in api.items():
        ct, ins = node.get("class_type"), node.get("inputs") or {}
        info = oi.get(ct)
        if info is None:
            if ct in KNOWN_SAMPLERS and cc._is_link(ins.get("model")):
                out.append({"node": nid, "input": "model", "source": tuple(ins["model"])})
            continue
        if not {"LATENT", "GUIDER"} & {str(t) for t in info.get("output") or []}:
            continue
        for name, typ, _ in cc._spec_inputs(info):
            if typ == "MODEL" and cc._is_link(ins.get(name)):
                out.append({"node": nid, "input": name, "source": tuple(ins[name])})
    return out


def _bender_input(api: dict, oi: dict, nid: str) -> str | None:
    """The model input of a node that bends a model (Apply Bends from JSON, the web UI node, Model Bending, …)."""
    node = api.get(nid) or {}
    ct = str(node.get("class_type", ""))
    info = oi.get(ct)
    if "bend" not in ct.lower() or info is None or "MODEL" not in [str(t) for t in info.get("output") or []]:
        return None
    return next((name for name, typ, _ in cc._spec_inputs(info)
                 if typ == "MODEL" and cc._is_link((node.get("inputs") or {}).get(name))), None)


def _before_bends(api: dict, oi: dict, source: tuple) -> tuple[tuple, tuple | None]:
    """(where the model comes from before any bending nodes, the first bending node's (id, input) or None)."""
    tap = None
    for _ in range(50):
        name = _bender_input(api, oi, source[0])
        if not name:
            break
        tap = (source[0], name)
        source = tuple(api[source[0]]["inputs"][name])
    return source, tap


def _loras(api: dict) -> list[dict]:
    out = []
    for node in api.values():
        for k, v in (node.get("inputs") or {}).items():
            if k.startswith("lora_name") and isinstance(v, str) and v.lower() != "none":
                out.append({"name": v, "strength": node["inputs"].get("strength_model",
                                                                      node["inputs"].get("strength"))})
    return out


def _select(found: list[dict], api: dict, only: list[str] | None) -> list[dict]:
    if not only:
        return found
    want = {str(o).lower() for o in only}
    hit = [s for s in found if s["node"].lower() in want
           or ((api[s["node"]].get("_meta") or {}).get("title") or "").lower() in want]
    if not hit:
        raise cc.CanvasError(f"none of {only} is a sampler in this run; samplers: "
                             f"{[_title(api, s['node']) for s in found]}")
    return hit


def _ui_node(ui: dict, nid: str) -> dict | None:
    return next((n for n in ui.get("nodes") or [] if str(n.get("id")) == nid), None)


def _ui_link(ui: dict, node: dict, input_name: str) -> list | None:
    """The link row feeding a node's input in the canvas graph: [id, from node, from slot, to node, to slot, type]."""
    lid = next((i.get("link") for i in node.get("inputs") or [] if i.get("name") == input_name), None)
    return next((row for row in ui.get("links") or [] if isinstance(row, list) and row and row[0] == lid), None) \
        if lid is not None else None


def _why_no_copy(ui: dict | None, group: list[dict]) -> str | None:
    """Why the canvas graph cannot take the bend (None when it can)."""
    if ui is None:
        return "the run carries no canvas graph (it was queued through the API)"
    for s in group:
        node = _ui_node(ui, s["node"])
        if node is None:
            return (f"sampler #{s['node']} is not on the top level of the canvas graph (it sits inside a subgraph "
                    f"or a group node)")
        row = _ui_link(ui, node, s["input"])
        if row is None or _ui_node(ui, str(row[1])) is None:
            return f"the model link into sampler #{s['node']} could not be found in the canvas graph"
        if s.get("tap"):  # replacing the run's own bends: the link into the first of them is needed too
            first = _ui_node(ui, s["tap"][0])
            row = _ui_link(ui, first, s["tap"][1]) if first else None
            if row is None or _ui_node(ui, str(row[1])) is None:
                return f"the run's own bending node #{s['tap'][0]} could not be found in the canvas graph"
    return None


def inspect(api: dict, ui: dict | None = None) -> dict:
    """What a run is made of, and whether a bend can be added: model and a guess of its type, LoRAs, sampler
    settings, prompt, the samplers found, any bends already in it."""
    oi = cc.object_info()
    model, setup, bends = kb.setup_from_api_prompt(api)
    found = sinks(api, oi)
    arch, family = model.get("arch"), kb.family_of(model.get("arch"))
    notes = []
    if not found:
        notes.append("no sampler with a model input was found: nothing to bend in this run")
    if BEND_NODE not in oi:
        notes.append("ComfyUI-Model-Bending is not installed: bends cannot be added")
    if not model.get("checkpoint"):
        notes.append("no model loader was recognised: ask which model this is")
    elif not arch:
        notes.append("the model's type could not be told from its file name: ask the user, or read it with the "
                     "Bendable Layer Catalogue node")
    elif family not in ("sd1", "sdxl"):
        notes.append(f"{arch}: no safe-range table for this model type, so start near 'no change'; block-level bends "
                     f"on Flux / SD3 use DiT Block Bending (pass a custom fragment), video models use "
                     f"attention_bends")
    if len({s["source"] for s in found}) > 1:
        notes.append("the samplers use different models (for example a base and a refiner): all are bent unless "
                     "`only` names some")
    why = _why_no_copy(ui, found) if found else None
    if why:
        notes.append(f"pictures only, no openable copy: {why}")
    # any bending node counts, also ones whose bends cannot be read back (the web UI's, module nodes)
    bending = [_title(api, i) for i, n in api.items() if "bend" in str(n.get("class_type", "")).lower()
               and "MODEL" in ((oi.get(n["class_type"]) or {}).get("output") or ["MODEL"])]
    if bending:
        notes.append("this run already bends its model: new bends are added after those, and their picture is "
                     "not an unbent original. With replace_bends, a copy takes the model from before them: to try "
                     "a bend in place of theirs, or (with no bends given) to render the unbent picture")
    batch = max([v for n in api.values() for k, v in (n.get("inputs") or {}).items()
                 if (k == "batch_size" or (k == "amount" and n.get("class_type") == "RepeatLatentBatch"))
                 and isinstance(v, int) and not isinstance(v, bool)], default=1)
    if batch > 1:
        setup = {**setup, "batch_size": batch}
        notes.append(f"each version renders {batch} pictures (their batch size): say what a round costs")
    return {"model": model, "loras": _loras(api), "setup": setup, "bending_nodes": bending,
            "samplers": [{"node": s["node"], "title": _title(api, s["node"]), "class": api[s["node"]]["class_type"],
                          "model_from": _title(api, s["source"][0])} for s in found],
            "already_bent": bends, "can_bend": bool(found) and BEND_NODE in oi,
            "copy": "none" if not found else "pictures_only" if why else "full", "notes": notes}


# --------------------------------------------------------------------------- the fragment
def bends_document(bends, arch: str = "") -> tuple[str, dict]:
    """(Apply Bends from JSON text, what bendjson.check found) for bends given as a document, a list of bends, one
    bend, or any of them as text (the web UI's "Copy Bends" format). Bends the node would refuse are an error
    here, before anything is rendered."""
    import bendjson
    try:
        found = bendjson.check(bends, arch)
    except bendjson.BendJSONError as e:
        raise cc.CanvasError(f"no usable bends given: {e}") from None
    if found["errors"]:
        raise cc.CanvasError("these bends cannot be used as they are: " + "; ".join(found["errors"]))
    return found["bends_json"], found


def default_fragment(bends, clamp: str = "safe", safe_ranges: dict | None = None, strict: bool = True) -> dict:
    """The bend as a fragment: Apply Bends from JSON, with its report shown. bends: the JSON text (bends_document)."""
    oi = cc.object_info()
    info = oi.get(BEND_NODE)
    if info is None:
        raise cc.CanvasError("ComfyUI-Model-Bending is not installed in this ComfyUI (no Apply Bends from JSON node)")
    optional = set((info.get("input") or {}).get("optional") or {})
    ins = {"model": [MODEL_IN, 0], "bends_json": bends if isinstance(bends, str) else bends_document(bends)[0]}
    if "strict" in optional:
        ins["strict"] = strict
    if "clamp" in optional:
        if clamp == "safe" and not safe_ranges:
            clamp = "hard"  # no table for this model type: each op's own limits
        ins["clamp"] = clamp
        if clamp == "safe" and "safe_ranges" in optional:
            ins["safe_ranges"] = json.dumps(safe_ranges)
    nodes = {"bend": {"class_type": BEND_NODE, "title": "Bends", "inputs": ins}}
    if "report" in (info.get("output_name") or []) and "PreviewAny" in oi:
        nodes["report"] = {"class_type": "PreviewAny", "title": "What the bend did (report)",
                           "inputs": {"source": ["bend", 1]}}
    return {"nodes": nodes, "out": ["bend", 0]}


def _build_fragment(fragment: dict) -> tuple[dict, dict, dict, list[tuple[str, str]], tuple[str, int]]:
    """(canvas nodes and links, API nodes, node key -> local id, inputs that take the model, the output)."""
    nodes = copy.deepcopy(fragment.get("nodes") or {})
    entries = []
    for k, n in nodes.items():
        for name, v in list((n.get("inputs") or {}).items()):
            if cc._is_link(v) and v[0] == MODEL_IN:
                entries.append((k, name))
                del n["inputs"][name]
    out = fragment.get("out")
    if not entries:
        raise cc.CanvasError(f'the fragment has no input linked to ["{MODEL_IN}", 0]')
    if not (cc._is_link(out) and out[0] in nodes):
        raise cc.CanvasError('the fragment needs "out": [node_key, output_index], the model its samplers receive')
    ui, api = cc.build({"nodes": nodes})
    return ui, api, {k: i + 1 for i, k in enumerate(nodes)}, entries, (out[0], out[1])


# --------------------------------------------------------------------------- canvas geometry
def _xy(v) -> tuple[float, float]:
    return (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v["0"]), float(v["1"]))


def _free_spot(ui: dict, x: float, y: float, w: float, h: float) -> tuple[float, float]:
    """Move a w×h box up from (x, y) until it covers no node."""
    rects = [(*_xy(n["pos"]), *_xy(n.get("size") or [300, 100])) for n in ui.get("nodes") or [] if n.get("pos")]
    for _ in range(60):
        if not any(x < rx + rw and rx < x + w and y < ry + rh + 40 and ry - 40 < y + h for rx, ry, rw, rh in rects):
            break
        y -= 120
    return x, y


def _set_widget(node: dict, oi: dict, name: str, value) -> bool:
    vals = node.get("widgets_values")
    if not isinstance(vals, list):
        return False
    for i, (wname, _) in enumerate(cc.widget_layout(node.get("type", ""), oi)):
        if wname == name and i < len(vals):
            vals[i] = value
            return True
    return False


# --------------------------------------------------------------------------- the splice
def bend_copy(api: dict, ui: dict | None, bends=None, *, arch: str = "", clamp: str = "safe", strict: bool = True,
              only: list[str] | None = None, fragment: dict | None = None, name: str = "",
              meta: dict | None = None, replace_bends: bool = False) -> tuple[dict, dict | None, dict]:
    """(bent API prompt, bent canvas graph or None, info). The fragment (default: Apply Bends from JSON with these
    bends) goes on the model link into every sampler, or into the samplers `only` names (node ids or titles).
    replace_bends takes the model from before the bending nodes the run already has, so the new bends stand in
    for them; with no bends and no fragment it gives the run without its bends (the unbent picture, API only).
    Nothing given is changed."""
    oi = cc.object_info()
    api = copy.deepcopy(api)
    found = _select(sinks(api, oi), api, only)
    if not found:
        raise cc.CanvasError("no sampler with a model input was found in this run: there is nothing to bend")
    notes: list[str] = []
    if replace_bends:
        for s in found:
            s["source"], tap = _before_bends(api, oi, s["source"])
            if tap:
                s["tap"] = tap
        if not any(s.get("tap") for s in found):
            notes.append("replace_bends: this run has no bending nodes of its own to replace")
    unbent = replace_bends and bends is None and fragment is None
    if unbent:
        for s in found:
            api[s["node"]]["inputs"][s["input"]] = list(s["source"])
        described, ui, fragment = [], None, {}
        notes.append("the run without its own bends: the unbent picture to compare with (pictures only)")
    elif fragment is None:
        from surprise import TABLES, DATA
        if not arch:
            arch = kb.setup_from_api_prompt(api)[0].get("arch") or ""
        table = TABLES.get(kb.family_of(arch))
        ranges = json.loads((DATA / f"safe_ranges_{table}.json").read_text(encoding="utf-8")) if table else None
        if clamp == "safe" and not ranges:
            notes.append(f"no safe-range table for {arch or 'this model type'}: amounts are limited only by each "
                         f"op's own range, so keep them gentle")
        text, read = bends_document(bends, arch)
        fragment = default_fragment(text, clamp, ranges, strict)
        notes += [f"the bends as given: {p}" for p in read["problems"]]
        notes += [f"bend #{b['index']} ({b['path']}): {n}" for b in read["bends"] for n in b["notes"]
                  if not (clamp == "safe" and "clamp = safe would pull it back" in n)]
        described = read["summary"]
    elif bends is not None:
        raise cc.CanvasError("give bends or a fragment, not both")
    else:
        described = []

    groups: dict[tuple, list[dict]] = {}
    for s in [] if unbent else found:
        groups.setdefault(s["source"], []).append(s)
    why = "an unbent copy is for comparison pictures" if unbent else _why_no_copy(ui, found)
    ui = copy.deepcopy(ui) if ui is not None and not why else None
    ids = [int(i) for i in api if i.isdigit()] + [n["id"] for n in (ui or {}).get("nodes") or []
                                                  if isinstance(n.get("id"), int)]
    next_id = max(ids, default=0)
    next_link = max([row[0] for row in (ui or {}).get("links") or [] if isinstance(row, list) and row] +
                    [(ui or {}).get("last_link_id") or 0], default=0)
    spliced = []
    for (src, src_slot), group in groups.items():
        f_ui, f_api, local, entries, (out_key, out_slot) = _build_fragment(fragment)
        nid = {lid: str(next_id + lid) for lid in local.values()}
        for lid, node in f_api.items():
            node = copy.deepcopy(node)
            node["inputs"] = {k: [nid[int(v[0])], v[1]] if cc._is_link(v) else v for k, v in node["inputs"].items()}
            api[nid[int(lid)]] = node
        for key, input_name in entries:
            api[nid[local[key]]]["inputs"][input_name] = [src, src_slot]
        for s in group:
            api[s["node"]]["inputs"][s["input"]] = [nid[local[out_key]], out_slot]
        if ui is not None:
            next_link = _graft(ui, f_ui, local, entries, (out_key, out_slot), group, next_id, next_link, oi)
        next_id += len(local)
        spliced.append({"after": _title(api, src), "into": [_title(api, s["node"]) for s in group],
                        "added": [_title(api, i) for i in nid.values() if i in api]})

    prefix = f"agent_bending/{name}" if name else ""
    for nid_, node in api.items():
        if prefix and isinstance((node.get("inputs") or {}).get("filename_prefix"), str):
            node["inputs"]["filename_prefix"] = prefix
            u = _ui_node(ui, nid_) if ui is not None else None
            if u is not None:
                _set_widget(u, oi, "filename_prefix", prefix)
    if prefix:
        notes.append(f"pictures are saved as {prefix}_… in ComfyUI's output folder, apart from the user's own")
    if ui is not None:
        for s in found:  # the copy repeats the run's seed, so it stays comparable with their original
            u, ins = _ui_node(ui, s["node"]), api[s["node"]]["inputs"]
            for seed in ("seed", "noise_seed"):
                if seed in ins and not cc._is_link(ins[seed]) and _set_widget(u, oi, seed, ins[seed]):
                    _set_widget(u, oi, f"{seed}.control_after_generate", "fixed")
        ui["last_node_id"], ui["last_link_id"] = next_id, next_link
        ui.setdefault("extra", {})["agent"] = {**(meta or {}), "bent_copy": True}
    elif not unbent:
        notes.append(f"pictures only, no openable copy: {why}. To give the user the bend in their own graph, tell "
                     f"them where to add an Apply Bends from JSON box: {spliced[0]['after']} → the new box → "
                     f"{', '.join(spliced[0]['into'])}")
    return api, ui, {"spliced": spliced, "copy": "full" if ui is not None else "pictures_only",
                     "bends": described, "notes": notes}


def _graft(ui: dict, f_ui: dict, local: dict, entries: list, out: tuple[str, int], group: list[dict],
           id_base: int, link_base: int, oi: dict) -> int:
    """Add a built fragment to the canvas graph on the model link of a group of samplers; returns the last link id."""
    by_local = {n["id"]: n for n in f_ui["nodes"]}
    first = _ui_node(ui, group[0]["node"])
    tap = group[0].get("tap")  # with replace_bends: the model as it enters the run's own first bending node
    origin_row = _ui_link(ui, _ui_node(ui, tap[0]), tap[1]) if tap else _ui_link(ui, first, group[0]["input"])
    origin, origin_slot, link_type = _ui_node(ui, str(origin_row[1])), origin_row[2], origin_row[5]

    # place the fragment above and to the left of the first sampler, where nothing else is
    xs = [n["pos"][0] for n in f_ui["nodes"]]
    ys = [n["pos"][1] for n in f_ui["nodes"]]
    w = max(n["pos"][0] + n["size"][0] for n in f_ui["nodes"]) - min(xs)
    h = max(n["pos"][1] + n["size"][1] for n in f_ui["nodes"]) - min(ys)
    sx, sy = _xy(first["pos"])
    x, y = _free_spot(ui, sx - w - 80, sy - h - 120, w, h)
    for n in f_ui["nodes"]:
        n["pos"] = [n["pos"][0] - min(xs) + x, n["pos"][1] - min(ys) + y]
        n["id"] += id_base
        for o in n["outputs"]:
            o["links"] = [lid + link_base for lid in o["links"]] if o["links"] else None
        for i in n["inputs"]:
            if i["link"] is not None:
                i["link"] += link_base
    for row in f_ui["links"]:
        ui.setdefault("links", []).append([row[0] + link_base, row[1] + id_base, row[2], row[3] + id_base, row[4],
                                           row[5]])
    ui["nodes"] += f_ui["nodes"]
    last = link_base + len(f_ui["links"])

    def connect(src_node: dict, src_slot: int, dst_node: dict, input_name: str) -> None:
        nonlocal last
        slot = next(i for i, inp in enumerate(dst_node["inputs"]) if inp["name"] == input_name)
        last += 1
        ui["links"].append([last, src_node["id"], src_slot, dst_node["id"], slot, link_type])
        dst_node["inputs"][slot]["link"] = last
        o = src_node["outputs"][src_slot]
        o["links"] = (o.get("links") or []) + [last]

    for key, input_name in entries:  # the user's model goes into the fragment
        connect(origin, origin_slot, by_local[local[key]], input_name)
    for s in group:  # and the fragment's model replaces it at each sampler
        node = _ui_node(ui, s["node"])
        old = _ui_link(ui, node, s["input"])
        ui["links"].remove(old)
        src = _ui_node(ui, str(old[1]))
        if src is not None and old[2] < len(src.get("outputs") or []):
            o = src["outputs"][old[2]]
            o["links"] = [lid for lid in o.get("links") or [] if lid != old[0]] or None
        connect(by_local[local[out[0]]], out[1], node, s["input"])
    gids = [g.get("id") for g in ui.get("groups") or [] if isinstance(g.get("id"), int)]
    ui.setdefault("groups", []).append({"id": max(gids, default=len(ui.get("groups") or [])) + 1,
                                        "title": FRAGMENT_TITLE, "bounding": [x - 20, y - 70, w + 40, h + 100],
                                        "color": "#8A8", "font_size": 24, "flags": {}})
    return last
