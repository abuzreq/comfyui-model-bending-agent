#!/usr/bin/env python3
"""Read a bend someone hands over as JSON, in the format of ComfyUI-Model-Bending's Apply Bends from JSON node (the
web UI's "Copy Bends" clipboard format): check it the way the node would, tidy it, and say in plain words what it
does. Standard library only.

  python bendjson.py BENDS.json [--arch sd15]
  python bendjson.py '{"bends": [{"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 90}}]}'
  python bendjson.py --tray 'https://…static.hf.space/#tray=…' [--arch sd15]   # a navigator tray: one check per bend

Accepted: the document ({"bends": [...]}, with the optional steps_min / steps_max / max_denoising_steps /
selected_part / version / attention_bends), a list of bends, or one bend; as an object or as text, with or without a
``` fence around it. The older forms are understood too: "bend" for "bends", and {"path", "angle"} for a rotation.
A bend may carry "kb": {"dataset", "cell", "record"}, where in the bend knowledge base it came from. This check keeps
it and adds the navigator link. Model-Bending 0.3.1+ (bends JSON 1.2) keeps it too, in its resolved_json; for older
plugins comfy_canvas.fit_bends drops it from what is sent to ComfyUI, so a strict run does not fail.
To render it, pass the document to bend_run (comfy_canvas.py bend-run), or put it in an Apply Bends from JSON node.
"""

from __future__ import annotations

import difflib
import json
import re
import sys

import kb

# op -> {argument: (type, default, hard limits or allowed values)}: Model-Bending 0.3's table (nodes.py BEND_OPS)
OPS: dict[str, dict[str, tuple]] = {
    "add_scalar": {"scalar": (float, 0.0, (-100.0, 100.0))},
    "add_noise": {"noise_std": (float, 0.0, (-100.0, 100.0)), "seed": (int, 42, None)},
    "multiply": {"scalar": (float, 1.0, (-100.0, 100.0))},
    "rotate": {"angle_degrees": (float, 0.0, (-360.0, 360.0))},
    "threshold": {"threshold": (float, 0.0, None)},
    "scale": {"scale_factor": (float, 1.0, (-100.0, 100.0))},
    "erosion": {"kernel_size": (int, 3, (1, 10))},
    "dilation": {"kernel_size": (int, 3, (1, 10))},
    "gradient": {"kernel_size": (int, 3, (1, 10))},
    "sobel": {"normalized": (bool, True, None)},
    "fourier": {"cutoff_freq": (float, 5.0, (0.0, 10.0)), "amp_factor": (float, 2.0, (-10.0, 10.0))},
    "subset": {"percentage": (float, 0.5, (0.0, 1.0)), "dim": (str, "batch", ("batch", "channel", "spatial")),
               "seed": (int, 0, None)},
    "translate": {"dx": (float, 0.0, (-1.0, 1.0)), "dy": (float, 0.0, (-1.0, 1.0)),
                  "padding": (str, "border", ("zeros", "border", "reflection"))},
    "flip": {"direction": (str, "horizontal", ("horizontal", "vertical", "both"))},
    "blur": {"sigma": (float, 1.0, (0.0, 20.0))},
    "sharpen": {"amount": (float, 1.0, (-10.0, 10.0)), "sigma": (float, 1.0, (0.05, 20.0))},
    "temporal_shift": {"frames": (int, 1, (-64, 64)), "padding": (str, "border", ("border", "wrap", "zeros"))},
    "temporal_blur": {"sigma": (float, 1.0, (0.0, 16.0))},
    "frame_reverse": {},
    "frame_ramp": {"w_start": (float, 0.0, (-4.0, 4.0)), "w_end": (float, 1.0, (-4.0, 4.0)),
                   "curve": (str, "linear", ("linear", "ease_in", "ease_out", "smooth", "triangle"))},
}
INNER_OPS = ("subset", "frame_ramp")  # ops that wrap an "inner" op
NEEDS_03 = {"translate", "flip", "blur", "sharpen", "temporal_shift", "temporal_blur", "frame_reverse", "frame_ramp"}
VIDEO_OPS = {"temporal_shift", "temporal_blur", "frame_reverse", "frame_ramp"}
# ops that move or filter the picture plane: they need an image-shaped (4-D) layer on a UNet
SPATIAL_OPS = {"rotate", "scale", "translate", "flip", "blur", "sharpen", "erosion", "dilation", "gradient", "sobel",
               "fourier"}
TOP_KEYS = {"bends", "bend", "steps_min", "steps_max", "max_denoising_steps", "selected_part", "version",
            "attention_bends"}
BEND_KEYS = {"path", "module_type", "module_args", "angle", "steps", "t", "blend", "label", "guard", "inner", "kb"}
OPTIONAL_BEND_KEYS = ("steps", "t", "blend", "label", "guard", "inner", "kb")
ATTENTIONS = ("cross_text", "cross_image", "self_query", "self_key")
GUARD_NAN = ("zero", "clamp", "none")
_STEPS = re.compile(r"\s*(\*|(\d+\s*-\s*\d*|-\s*\d+|\d+)(\s*,\s*(\d+\s*-\s*\d*|-\s*\d+|\d+))*)?\s*")
_FENCE = re.compile(r"^\s*```[\w-]*\s*\n(.*?)\n?\s*```\s*$", re.S)
PLAIN_OP = {"threshold": "cut off below a level", "erosion": "its strong areas eaten away",
            "dilation": "its strong areas swollen", "gradient": "reduced to its outlines",
            "sobel": "reduced to its edges", "fourier": "its fine patterns amplified",
            "subset": "changed in a random part only", "translate": "shifted sideways", "flip": "mirrored",
            "blur": "blurred", "sharpen": "sharpened", "temporal_shift": "shifted in time",
            "temporal_blur": "smeared over time", "frame_reverse": "played backwards",
            "frame_ramp": "changed more and more over the clip"}


class BendJSONError(ValueError):
    """The JSON cannot be used at all: not JSON, or nothing in it that is a bend."""


# --------------------------------------------------------------------------- reading
def parse(given) -> dict:
    """Whatever was handed over, as one document: {"bends": [...], ...other top-level keys as given}."""
    if isinstance(given, (bytes, bytearray)):
        given = given.decode("utf-8")
    if isinstance(given, str):
        text = given.strip().lstrip("﻿")
        fenced = _FENCE.match(text)
        if fenced:
            text = fenced.group(1).strip()
        if not text:
            raise BendJSONError("no bend JSON was given")
        if re.match(r"(https?://\S*)?#?tray=", text):
            raise BendJSONError("this is a navigator tray link: separate bends, each meant to be tried on its own. "
                                "Open it with open_tray (bendjson.py --tray LINK)")
        try:
            given = json.loads(text)
        except json.JSONDecodeError as e:
            raise BendJSONError(f"this is not valid JSON ({e.msg} at line {e.lineno}, column {e.colno}). A bend looks "
                                f'like {{"bends": [{{"path": "middle_block.1", "module_type": "rotate", '
                                f'"module_args": {{"angle_degrees": 90}}}}]}}') from None
    if isinstance(given, list):
        return {"bends": list(given)}
    if not isinstance(given, dict):
        raise BendJSONError("a bend document is an object with a 'bends' list, a list of bends, or one bend")
    if "path" in given and not (TOP_KEYS & set(given)):  # one bend on its own
        return {"bends": [given]}
    doc = dict(given)
    if "bends" not in doc and "bend" in doc:  # the older spelling
        doc["bends"] = doc.pop("bend")
    doc.pop("bend", None)
    if isinstance(doc.get("bends"), dict):
        doc["bends"] = list(doc["bends"].values())
    doc.setdefault("bends", [])
    return doc


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _coerce(typ, value):
    if typ is bool:
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
            raise ValueError(value)
        return bool(value)
    if typ is int:
        return int(float(value))
    if typ is float:
        return float(value)
    return str(value)


def _op(node: dict, where: str, problems: list[str], errors: list[str]) -> dict | None:
    """{module_type, module_args[, inner]} with the arguments that were given, typed; None if it cannot be used."""
    name = node.get("module_type")
    if name not in OPS:
        near = difflib.get_close_matches(str(name), list(OPS), n=1)
        errors.append(f"{where}unknown module_type {name!r}" + (f"; did you mean {near[0]!r}?" if near else
                                                                 f"; known: {', '.join(OPS)}"))
        return None
    spec, given = OPS[name], node.get("module_args") or {}
    if not isinstance(given, dict):
        problems.append(f"{where}module_args must be an object; the node uses the defaults")
        given = {}
    args = {}
    for arg, value in given.items():
        if arg not in spec:
            near = difflib.get_close_matches(arg, list(spec), n=1)
            problems.append(f"{where}unknown argument {arg!r} for {name}; the node ignores it"
                            + (f" (did you mean {near[0]!r}?)" if near else f" (accepted: {', '.join(spec) or 'none'})"))
            continue
        typ, default, limits = spec[arg]
        try:
            value = _coerce(typ, value)
        except (TypeError, ValueError):
            problems.append(f"{where}{name}.{arg}={value!r} is not a {typ.__name__}; the node uses {default!r}")
            continue
        if typ is str and limits and value not in limits:
            problems.append(f"{where}{name}.{arg}={value!r} must be one of {', '.join(limits)}; the node uses "
                            f"{default!r}")
            continue
        if typ in (int, float) and limits and not limits[0] <= value <= limits[1]:
            problems.append(f"{where}{name}.{arg}={value:g} is outside the node's own range {limits[0]:g} to "
                            f"{limits[1]:g}")
        args[arg] = int(value) if isinstance(value, float) and value == int(value) and typ is float else value
    op = {"module_type": name, "module_args": args}
    if name in INNER_OPS:
        inner = node.get("inner")
        if not isinstance(inner, dict) or not inner.get("module_type"):
            errors.append(f"{where}{name} needs an 'inner' bend with a module_type")
            return None
        inner_op = _op(inner, where + "inner ", problems, errors)
        if inner_op is None:
            return None
        op["inner"] = inner_op
    elif node.get("inner") is not None:
        problems.append(f"{where}'inner' is only used by {' / '.join(INNER_OPS)}; the node ignores it")
    return op


def _timing(b: dict, where: str, problems: list[str]) -> dict:
    out = {}
    if b.get("steps") is not None:
        if _STEPS.fullmatch(str(b["steps"])):
            out["steps"] = b["steps"]
        else:
            problems.append(f"{where}steps {b['steps']!r} is not a step list like '0-4,9' or '3-'; the node bends "
                            f"all steps")
    t = b.get("t")
    if t is not None:
        if isinstance(t, (list, tuple)) and len(t) == 2 and all(_number(x) for x in t):
            out["t"] = [max(t), min(t)]
            if not (0 <= min(t) and max(t) <= 1):
                problems.append(f"{where}t {list(t)} goes outside 0 to 1 (1 = pure noise, 0 = the finished picture)")
        else:
            problems.append(f"{where}'t' must be [hi, lo], two numbers between 1 and 0; the node refuses {t!r}")
    if b.get("blend") is not None:
        if _number(b["blend"]) and 0 <= b["blend"] <= 1:
            out["blend"] = b["blend"]
        else:
            problems.append(f"{where}blend {b['blend']!r} must be a number from 0 to 1")
    guard = b.get("guard")
    if isinstance(guard, dict):
        ok = {k: v for k, v in guard.items()
              if (k == "nan" and v in GUARD_NAN) or (k == "max_std_ratio" and _number(v) and v > 0)
              or (k == "preserve_norm" and isinstance(v, bool))}
        for k in guard:
            if k not in ok:
                problems.append(f"{where}guard.{k}={guard[k]!r} is not understood; the node ignores it (nan: "
                                f"{'|'.join(GUARD_NAN)}, max_std_ratio: a number above 0, preserve_norm: true|false)")
        if ok:
            out["guard"] = ok
    elif guard is not None:
        problems.append(f"{where}guard must be an object; the node ignores it")
    if isinstance(b.get("label"), str) and b["label"]:
        out["label"] = b["label"]
    return out


# --------------------------------------------------------------------------- plain words
def _when(b: dict) -> str:
    t, steps = b.get("t"), b.get("steps")
    if t and not (t[0] >= 1 and t[1] <= 0):
        hi, lo = t
        if lo >= 0.7:
            return "early in the painting, while the layout forms"
        if hi <= 0.7 and lo >= 0.2:
            return "in the middle of the painting, while the style forms"
        if hi <= 0.2:
            return "late in the painting, in the fine detail"
        return f"for part of the painting (t {hi:g} to {lo:g})"
    if steps not in (None, "", "*"):
        return f"on steps {steps}"
    return "through the whole painting"


def _how(op: dict) -> str:
    from surprise import NEUTRAL, _phrase
    name, args = op["module_type"], op["module_args"]
    if name in NEUTRAL and _number(args.get(NEUTRAL[name][0])):
        return _phrase(name, args)
    text = PLAIN_OP.get(name, name)
    if "inner" in op:
        text += f" ({_how(op['inner'])})"
    return text


def _where(path: str, family: str) -> tuple[str | None, str]:
    from surprise import PLAIN_GROUP
    group = kb.unet_group(path.split("*")[0].rstrip(".[") or path, family) if family in ("sd1", "sd2", "sdxl") else None
    if group in PLAIN_GROUP:
        return group, PLAIN_GROUP[group]
    if group == "embed":
        return group, "the model's sense of time (the step embedding)"
    if group == "out.conv":
        return group, "the very last layer, just before the picture"
    return None, f"the layer {path}"


# --------------------------------------------------------------------------- checking
def check(given, arch: str = "") -> dict:
    """Check a bend document the way Apply Bends from JSON would, and describe it.

    Returns ok (can it be rendered), document / bends_json (tidied: the bends that can be used, with the keys the
    node reads), bends (one entry per bend: where, how and when in plain words, and whether the amount is inside
    the measured safe range when the model type has a table), errors (bends the node would refuse), problems
    (things it would ignore or change), and summary (one plain sentence per bend). Raises BendJSONError when
    there is nothing to work with."""
    from surprise import NEUTRAL, TABLES, _range
    doc = parse(given)
    errors: list[str] = []
    problems: list[str] = [f"unknown top-level key {k!r}; the node ignores it" for k in doc if k not in TOP_KEYS]
    family = kb.family_of(arch) if arch else ""
    ranges = None
    if TABLES.get(family):
        from surprise import DATA
        ranges = json.loads((DATA / f"safe_ranges_{TABLES[family]}.json").read_text(encoding="utf-8"))
    raw = doc.get("bends")
    if not isinstance(raw, list):
        errors.append("'bends' must be a list")
        raw = []
    bends, described = [], []
    for i, b in enumerate(raw):
        if not isinstance(b, dict):
            errors.append(f"bend #{i} is not an object")
            continue
        path = b.get("path")
        if not path or not isinstance(path, str):
            errors.append(f"bend #{i} has no 'path' (the layer to bend, e.g. \"middle_block.1\")")
            continue
        where = f"bend #{i} ({path}): "
        problems += [f"{where}unknown key {k!r}; the node ignores it" for k in b if k not in BEND_KEYS]
        if _number(b.get("angle")) and not b.get("module_type"):  # the oldest form: a rotation
            node = {"module_type": "rotate", "module_args": {"angle_degrees": b["angle"]}}
        else:
            node = {"module_type": (b.get("module_type") or "").strip() if isinstance(b.get("module_type"), str)
                    else b.get("module_type"), "module_args": b.get("module_args"), "inner": b.get("inner")}
            if not node["module_type"]:
                errors.append(f"{where}no module_type (what to do to the layer, e.g. \"rotate\")")
                continue
        op = _op(node, where, problems, errors)
        if op is None:
            continue
        bend = {"path": path, **op, **_timing(b, where, problems)}
        ref = None
        if "kb" in b:  # where the bend came from; the node ignores it, the skill keeps it
            ref, kb_problems = kb.kb_ref(b["kb"])
            problems += [where + p for p in kb_problems]
            if ref:
                bend["kb"] = ref
        bends.append(bend)
        name, notes = op["module_type"], []
        group, where_plain = _where(path, family)
        safe = None
        if ranges and name in NEUTRAL and _number(op["module_args"].get(NEUTRAL[name][0])):
            rng, _ = _range(ranges, path, name)
            if rng:
                v = op["module_args"][NEUTRAL[name][0]]
                safe = rng[0] <= v <= rng[1]
                if not safe:
                    notes.append(f"{v:g} is outside the range that usually keeps a picture together here "
                                 f"({rng[0]:g} to {rng[1]:g}); it may fall apart, and clamp = safe would pull it back")
        if name in SPATIAL_OPS and family in ("sd1", "sd2", "sdxl") \
                and kb.unet_kind(path, family) in ("self_attn", "cross_attn", "ff", "norm", "time_embed"):
            notes.append(f"{name} moves or filters the picture plane, and this layer is not image-shaped: bend a "
                         f"whole block (e.g. {'.'.join(path.split('.')[:2])}) or a convolution instead")
        if name in VIDEO_OPS and family and family != "wan":
            notes.append(f"{name} acts across video frames; on an image model it has nothing to act on")
        if name in NEEDS_03 or (op.get("inner") or {}).get("module_type") in NEEDS_03:
            notes.append("needs ComfyUI-Model-Bending 0.3 or newer")
        if family and group is None and family in ("sd1", "sd2", "sdxl") and "*" not in path and "[" not in path:
            notes.append("this path is not one of the model's usual blocks: the run's report will say whether it "
                         "matched a layer")
        described.append({"index": i, **({"label": bend["label"]} if bend.get("label") else {}), "path": path,
                          "op": name, "args": op["module_args"], "group": group, "where": where_plain,
                          "how": _how(op), "when": _when(bend), "inside_safe_range": safe, "notes": notes,
                          **({"kb": ref} if ref else {}),
                          **({"link": kb.ref_link(ref)} if kb.ref_link(ref) else {})})
    attention = doc.get("attention_bends")
    if attention is not None:
        if not isinstance(attention, list):
            errors.append("'attention_bends' must be a list")
            attention = []
        for i, a in enumerate(attention):
            if not isinstance(a, dict) or a.get("module_type") not in OPS:
                errors.append(f"attention bend #{i}: unknown or missing module_type")
            elif a.get("attention") not in ATTENTIONS:
                errors.append(f"attention bend #{i}: 'attention' must be one of {', '.join(ATTENTIONS)}")
    tidy = {k: v for k, v in doc.items() if k in TOP_KEYS and k not in ("bends", "bend", "attention_bends")}
    tidy["bends"] = bends
    usable_attention = [a for a in attention or [] if isinstance(a, dict) and a.get("module_type") in OPS
                        and a.get("attention") in ATTENTIONS]
    if usable_attention:
        tidy["attention_bends"] = usable_attention
    if not bends and not usable_attention and not errors:
        errors.append("there are no bends in it")
    lines = [(f"{d['label']}: " if d.get("label") else "") + f"{d['where']}, {d['how']}, {d['when']}"
             for d in described]
    if usable_attention:
        lines.append(f"{len(usable_attention)} bend(s) on a video model's attention maps (references/video.md)")
    return {"ok": bool(bends or usable_attention), "document": tidy, "bends_json": json.dumps(tidy),
            "bends": described, "errors": errors, "problems": problems,
            "summary": [ln[0].upper() + ln[1:] for ln in lines],
            **({"arch": arch} if arch else {"note": "pass arch (sd15, sdxl, …) to place each bend in the model and "
                                                    "check its amount against the safe ranges"})}


def open_tray(link: str, arch: str = "") -> dict:
    """The bends in a navigator tray link, each checked on its own the way check() does. Returns ok, count, and
    bends: one entry per bend with ok, summary, bends_json (a one-bend document for bend_run), kb and link (where it
    came from), inside_safe_range, notes, errors and problems."""
    try:
        raw = kb.read_tray(link)
    except ValueError as e:
        raise BendJSONError(str(e)) from None
    items = []
    for i, b in enumerate(raw):
        r = check({"bends": [b]}, arch)
        d = (r["bends"] or [{}])[0]
        items.append({"index": i, "ok": r["ok"], "summary": (r["summary"] or [""])[0], "bends_json": r["bends_json"],
                      **{k: d[k] for k in ("kb", "link") if k in d},
                      "inside_safe_range": d.get("inside_safe_range"), "notes": d.get("notes", []),
                      "errors": [e.replace("bend #0", f"bend #{i}") for e in r["errors"]],
                      "problems": [p.replace("bend #0", f"bend #{i}") for p in r["problems"]]})
    return {"ok": any(it["ok"] for it in items), "count": len(items), "bends": items,
            "note": "each bend in a tray was tested on its own: try them one at a time (one bend_run per bend, each "
                    "with its own name) and combine them only if the user asks",
            **({"arch": arch} if arch else {})}


def main(argv=None) -> None:
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bends", help="a file with the JSON, the JSON text itself, or (with --tray) a navigator tray link")
    ap.add_argument("--arch", default="", help="sd15, sd14, sdxl, flux, …: places the bends and checks safe ranges")
    ap.add_argument("--tray", action="store_true", help="read a navigator tray link: separate bends, checked one by one")
    ap.add_argument("--to-tray", action="store_true", help="print a navigator tray link carrying these bends")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        text = a.bends
        if a.tray:
            r = open_tray(text, a.arch)
            print(json.dumps(r, indent=1, ensure_ascii=False))
            raise SystemExit(0 if r["ok"] else 1)
        if not text.lstrip().startswith(("{", "[", "`")) and Path(text).is_file():
            text = Path(text).read_text(encoding="utf-8")
        r = check(text, a.arch)
        if a.to_tray and r["ok"]:
            print(kb.tray_link(r["document"]["bends"]))
            return
    except BendJSONError as e:
        raise SystemExit(str(e))
    print(json.dumps(r, indent=1, ensure_ascii=False))
    if not r["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
