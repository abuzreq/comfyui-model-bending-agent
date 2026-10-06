#!/usr/bin/env python3
"""The bend knowledge base: what bending which part of which model does. Standard library only.

A knowledge base is a folder (the community one is a Hugging Face dataset; the user's own lives in the work folder).
Every record keeps facts apart from interpretation:

  records/<family>/<source>/<id>/record.json         facts: model, setup, bends, outputs. Written once.
  records/<family>/<source>/<id>/measurements.json   numbers, each with its method (and model, if a learned one)
  records/<family>/<source>/<id>/interpretations.jsonl  captions, keywords, effect tags, notes. Append-only; every
                                                     entry names its author: a human, or an AI and which model.
  records/<family>/<source>/<id>/output.webp         the output image (input.webp only with consent)
  baselines/<id>.webp                                the unbent render of the same setup
  findings/<id>.json                                 claims about many records (e.g. a paper's results)
  index/{records,cells,findings}.jsonl               image-free summaries that agents read

Prompts and input images are kept only with consent; otherwise a salted key stands in, so records can still be
counted per prompt without revealing it.

A cell groups single-bend records by family, layer group, kind, module type, op, amount bucket, step window and route.
Cells carry the evidence grade that find_recipes ranks by.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "kb"
SCHEMA_VERSION = "1.0"
SOURCES = ("paper", "paper_reproduction", "author_experiment", "sweep", "run", "session", "artist")
INTERPRETATION_KINDS = ("caption", "change", "keywords", "effect_tags", "concepts", "note", "verdict")
ARCH_FAMILY = {"sd14": "sd1", "sd15": "sd1", "sd1": "sd1", "sd2": "sd2", "sd21": "sd2", "sdxl": "sdxl",
               "sd3": "sd3", "flux": "flux", "wan21": "wan", "wan21_i2v": "wan", "wan22": "wan"}

# What each standard measurement means. The paper's four distances come first; new producers use the same ones so
# numbers stay comparable across sources.
METRICS = {
    "cosine_distance": {"method": "1 - cosine similarity of the final latent vs the baseline's"},
    "lpips_distance": {"method": "LPIPS vs the baseline at 256 px", "model": "lpips/alex"},
    "dinov2_distance": {"method": "1 - cosine of CLS embeddings vs the baseline's",
                        "model": "facebookresearch/dinov2:dinov2_vitb14"},
    "clip_distance": {"method": "1 - cosine of image embeddings vs the baseline's",
                      "model": "openai/clip-vit-base-patch32"},
    "mae_vs_baseline": {"method": "mean absolute RGB difference vs the baseline, 0-255"},
    "std": {"method": "luminance standard deviation at 256 px, 0-255"},
    "hf_ratio": {"method": "share of luminance FFT power above radius 0.25 at 256 px"},
    "prompt_retention": {"method": "CLIP image-prompt cosine of the output divided by the baseline's (1 = shows the "
                                   "prompt as well as the unbent image; low = the subject is lost)",
                         "model": "openai/clip-vit-base-patch32"},
    "clip_degenerate": {"method": "CLIP zero-shot probability mass on degenerate-image prompts (noise, static, blank, "
                                  "blobs...) vs content prompts (painting, photograph, abstract artwork...)",
                        "model": "openai/clip-vit-base-patch32"},
}

# "Broken" is strict: it should almost never mark an image an artist would keep. Thresholds were set from the KB
# owner's labels on 59 renders around the old cut-offs (2026-10-04): at these values none of the images they called
# fine or borderline is flagged, at the cost of missing most they called broken (4 of 11 caught). Losing the subject
# (low prompt_retention), flat textures and smooth blob fields were often judged fine, so they are recorded
# (prompt_retention, pixel_flags) but do not make a render broken on their own. Only static-like noise and
# all-black/all-white frames do.
HARD_PIXEL_FLAGS = ("noise", "extreme")
DEGENERATE_MASS_MIN, DEGENERATE_RETENTION_MAX = 0.90, 0.85
DEGENERATE_MASS_NO_PROMPT = 0.95
METRICS["pixel_flags"] = {"method": "pixel guard observations at 256 px: noise (high-frequency power), blob (smooth "
                                    "high-contrast field), flat (low contrast/entropy), clipped, extreme (near-black or "
                                    "near-white mean); recorded, only noise and extreme make a render broken"}
METRICS["degenerate"] = {"method": f"broken output: clip_degenerate >= {DEGENERATE_MASS_MIN} with prompt_retention < "
                                   f"{DEGENERATE_RETENTION_MAX} (>= {DEGENERATE_MASS_NO_PROMPT} when no prompt is "
                                   f"shared), or pixel_flags noise / extreme; calibrated on human labels"}


# Softer signals, recorded per cell to describe what a bend tends to do, never to exclude renders: the subject fading
# (prompt_retention below this) and a noise/blob look (clip_degenerate at or above this).
RISK_RETENTION, RISK_DEGENERATE = 0.75, 0.6


def at_risk(m: dict) -> bool:
    """A render that broke, or whose subject faded, or that CLIP sees as noise/blobs (measurement values)."""
    r, g = m.get("prompt_retention"), m.get("clip_degenerate")
    return bool(m.get("degenerate")) or (r is not None and r < RISK_RETENTION) or (g is not None and g >= RISK_DEGENERATE)


def degenerate_reasons(pixel_flags: list[str], prompt_retention: float | None = None,
                       clip_degenerate: float | None = None) -> list[str]:
    """Why a render counts as broken (empty = it does not). See METRICS["degenerate"]."""
    reasons = [r for r in pixel_flags if r in HARD_PIXEL_FLAGS]
    if clip_degenerate is not None:
        if prompt_retention is not None:
            hit = clip_degenerate >= DEGENERATE_MASS_MIN and prompt_retention < DEGENERATE_RETENTION_MAX
        else:
            hit = clip_degenerate >= DEGENERATE_MASS_NO_PROMPT
        if hit:
            reasons.append("clip_degenerate")
    return reasons


# --------------------------------------------------------------------------- ids and privacy
def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj, n: int = 20) -> str:
    s = obj if isinstance(obj, str) else canonical(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def private_key(text: str, salt: str = "") -> str:
    """Stands in for a prompt or input image that was not shared. With a salt (kept by the contributor), the key
    cannot be checked against guessed prompts; without one (shared data), equal prompts get equal keys."""
    return digest(salt + "\x00" + text, 16)


def local_salt(path: Path) -> str:
    """The contributor's secret salt for private keys, created on first use. Never shared."""
    path = Path(path)
    if not path.exists():
        import secrets
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(16), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- where a path sits in the model
def family_of(arch: str | None) -> str:
    return ARCH_FAMILY.get((arch or "").lower(), (arch or "unknown").lower())


def guess_arch(checkpoint: str) -> str | None:
    """A guess from the checkpoint name; callers that know the architecture pass it instead."""
    n = (checkpoint or "").lower()
    if "wan2.2" in n or "wan22" in n:
        return "wan22"
    if "wan" in n:
        return "wan21_i2v" if "i2v" in n else "wan21"
    if "flux" in n:
        return "flux"
    if re.search(r"sd3|stable-diffusion-3", n):
        return "sd3"
    if re.search(r"xl|sdxl|pony|illustrious", n):
        return "sdxl"
    if re.search(r"v1-4|sd-?1\.?4|sd14", n):
        return "sd14"
    if re.search(r"v2-|sd-?2|sd21", n):
        return "sd2"
    if re.search(r"v1-5|sd-?1\.?5|sd15|dreamshaper|realisticvision|analog|kohaku|anything|deliberate", n):
        return "sd15"
    return None


# Layer groups by U-Net block index (the same groups the model profiles and safe-range tables use).
_GROUPS = {
    "sd1": {"input_blocks": [(3, "in.hi"), (6, "in.mid"), (11, "in.lo")],
            "output_blocks": [(5, "out.lo"), (8, "out.mid"), (11, "out.hi")]},
    "sd2": {"input_blocks": [(3, "in.hi"), (6, "in.mid"), (11, "in.lo")],
            "output_blocks": [(5, "out.lo"), (8, "out.mid"), (11, "out.hi")]},
    "sdxl": {"input_blocks": [(3, "in.hi"), (6, "in.mid"), (8, "in.lo")],
             "output_blocks": [(2, "out.lo"), (5, "out.mid"), (8, "out.hi")]},
}
# Whole-block kinds where the path ends at a block (sd1/sd2: input 3, 6, 9 downsample; output x.2 or 2.1 upsample).
_DOWNSAMPLE = {"sd1": {3, 6, 9}, "sd2": {3, 6, 9}, "sdxl": {3, 6}}


def unet_group(path: str, family: str) -> str | None:
    if path.startswith("middle_block"):
        return "mid"
    if path.startswith(("time_embed", "label_emb")):
        return "embed"
    if path == "out" or path.startswith("out."):
        return "out.conv"
    m = re.match(r"(input_blocks|output_blocks)\.(\d+)", path)
    table = _GROUPS.get(family)
    if not m or not table:
        return None
    i = int(m.group(2))
    return next((g for last, g in table[m.group(1)] if i <= last), None)


def unet_kind(path: str, family: str = "sd1") -> str:
    """A sub-module kind: skip, res_branch, self_attn, cross_attn, ff, norm, proj, conv_in, downsample, upsample,
    time_embed, out_conv, or res_block/attn_block for a whole block."""
    p = "." + path + "."
    checks = [(".attn2.", "cross_attn"), (".attn1.", "self_attn"), (".ff.", "ff"), (".skip_connection.", "skip"),
              (".in_layers.", "res_branch"), (".out_layers.", "res_branch"), (".emb_layers.", "res_branch"),
              (".proj_in.", "proj"), (".proj_out.", "proj"), (".op.", "downsample"), (".h_upd.", "res_branch"),
              (".x_upd.", "res_branch")]
    for needle, kind in checks:
        if needle in p:
            return kind
    if re.search(r"\.norm\d?\.", p):
        return "norm"
    if path.startswith("time_embed") or path.startswith("label_emb"):
        return "time_embed"
    if path == "out" or path.startswith("out."):
        return "out_conv"
    if path.startswith("input_blocks.0"):
        return "conv_in"
    m = re.fullmatch(r"(input_blocks|output_blocks)\.(\d+)\.(\d+)(?:\.conv)?", path)
    if m:
        blk, i, j = m.group(1), int(m.group(2)), int(m.group(3))
        if blk == "input_blocks" and i in _DOWNSAMPLE.get(family, set()):
            return "downsample"
        if blk == "output_blocks" and (j == 2 or path.endswith(".conv")):
            return "upsample"
        if blk == "output_blocks" and j == 1 and family in ("sd1", "sd2") and i == 2:
            return "upsample"
        return "res_block" if j == 0 else "attn_block"
    m = re.fullmatch(r"middle_block\.(\d+)", path)
    if m:
        return "attn_block" if m.group(1) == "1" else "res_block"
    if re.fullmatch(r"(input_blocks|output_blocks)\.\d+", path):
        return "block"
    return "other"


# --------------------------------------------------------------------------- amounts and windows
_MAIN_ARG = {"multiply": "scalar", "add_scalar": "scalar", "add_noise": "noise_std", "rotate": "angle_degrees",
             "scale": "scale_factor", "threshold": "threshold", "fourier": "amp_factor", "subset": "percentage"}


def main_arg(op: str, args: dict) -> float | None:
    v = (args or {}).get(_MAIN_ARG.get(op, ""))
    return float(v) if isinstance(v, (int, float)) else None


def is_neutral(op: str, args: dict) -> bool:
    v = main_arg(op, args)
    if v is None:
        return False
    return {"multiply": v == 1, "add_scalar": v == 0, "add_noise": v == 0, "rotate": v % 360 == 0,
            "scale": v == 1}.get(op, False)


def amount_bucket(op: str, args: dict) -> str:
    """A coarse, model-independent amount label, so cells pool nearby values."""
    v = main_arg(op, args)
    if v is None:
        return "custom"
    if op == "multiply":
        return ("negative" if v < 0 else "zero" if v == 0 else "damp" if v < 0.75 else "neutral" if v <= 1.25
                else "boost" if v <= 2.0 else "strong")
    if op == "add_scalar":
        a = abs(v)
        return "neutral" if a <= 0.1 else "low" if a <= 0.6 else "mid" if a <= 1.5 else "high"
    if op == "add_noise":
        return "none" if v == 0 else "light" if v <= 0.1 else "medium" if v <= 0.5 else "heavy" if v <= 1.0 else "extreme"
    if op == "rotate":
        d = abs(v) % 360
        d = min(d, 360 - d)
        return "none" if d == 0 else "slight" if d <= 30 else "quarter" if d <= 120 else "half"
    if op == "scale":
        return "neutral" if abs(v - 1) <= 0.05 else "shrink" if v < 1 else "grow" if v <= 1.5 else "strong_grow"
    return f"{v:g}"


def step_fraction(bend: dict, steps: int | None) -> tuple[float, float]:
    """(start, end) as fractions of the denoising run, 0 = first step. Reads t windows, step windows, or the
    start/end fractions of the run."""
    if isinstance(bend.get("t"), (list, tuple)) and len(bend["t"]) == 2:
        hi, lo = bend["t"]
        return round(1 - float(hi), 3), round(1 - float(lo), 3)
    lo, hi = bend.get("steps_min"), bend.get("steps_max")
    if isinstance(bend.get("steps"), str) and steps:
        nums = [int(x) for x in re.findall(r"\d+", bend["steps"])]
        if nums:
            lo, hi = min(nums), (max(nums) if not bend["steps"].strip().endswith("-") else steps - 1)
    if (lo is not None or hi is not None) and steps:
        lo = 0 if lo is None else lo
        hi = steps - 1 if hi is None else hi
        return round(lo / steps, 3), round(min(hi + 1, steps) / steps, 3)
    if "start" in bend or "end" in bend:
        return float(bend.get("start", 0.0)), float(bend.get("end", 1.0))
    return 0.0, 1.0


def window_of(start: float, end: float) -> str:
    """all | early | middle | late, by fraction of the steps (early ≈ structure, late ≈ detail)."""
    if start <= 0.05 and end >= 0.95:
        return "all"
    if end <= 0.4:
        return "early"
    if start >= 0.6:
        return "late"
    if start <= 0.05:
        return "early_to_mid"
    if end >= 0.95:
        return "mid_to_end"
    return "middle"


# --------------------------------------------------------------------------- building records
KIND_ALIASES = {"res": "res_block", "attn": "attn_block"}  # model-profile block kinds -> kb kinds


def norm_bend(b: dict, family: str, steps: int | None) -> dict:
    """One bend in the record's vocabulary: path, op, args, where it sits, amount bucket and step window."""
    op = b.get("op") or b.get("module_type") or b.get("bend_module_type")
    args = b.get("args") or b.get("module_args") or {}
    if not args and "amount" in b and op in _MAIN_ARG:
        args = {_MAIN_ARG[op]: b["amount"]}
    path = b.get("path") or b.get("layer") or b.get("layer_path") or ""
    start, end = step_fraction(b, steps)
    out = {"path": path, "op": op, "args": args,
           "group": b.get("group") or unet_group(path, family),
           "kind": KIND_ALIASES.get(b.get("kind"), b.get("kind")) if b.get("kind") else unet_kind(path, family),
           "module_type": b.get("module_type_class") or b.get("layer_type") or None,
           "container": b.get("container") or b.get("container_type") or None,
           "start": start, "end": end, "window": window_of(start, end),
           "bucket": amount_bucket(op, args)}
    for k in ("t", "steps", "steps_min", "steps_max", "blend", "label"):
        if b.get(k) is not None:
            out[k] = b[k]
    return {k: v for k, v in out.items() if v is not None}


def make_record(*, source: str, model: dict, setup: dict, bends: list[dict], outputs: dict | None = None,
                consent: dict | None = None, license: str = "CC0-1.0", contributor: str | None = None,
                provenance: dict | None = None, salt: str = "", created: str | None = None) -> dict:
    """Facts only. setup may carry prompt / negative / input_image; they are kept only where consent says so."""
    consent = {"prompt": False, "input_image": False, **(consent or {})}
    model = dict(model)
    model.setdefault("arch", guess_arch(model.get("checkpoint", "")))
    model["family"] = model.get("family") or family_of(model.get("arch"))
    s = {k: v for k, v in setup.items() if v is not None}
    salt_used = "" if consent["prompt"] else salt
    if s.get("prompt"):
        s["prompt_key"] = private_key(s["prompt"], salt_used)
        if not consent["prompt"]:
            s.pop("prompt")
            s.pop("negative", None)
    if s.get("input_image"):
        s["input_key"] = private_key(str(s["input_image"]), "" if consent["input_image"] else salt)
        if not consent["input_image"]:
            s.pop("input_image")
    nb = [norm_bend(b, model["family"], s.get("steps")) for b in bends]
    rid = record_id(model, s, nb)
    rec = {"id": rid, "id_scheme": ID_SCHEME, "schema_version": SCHEMA_VERSION, "source": source,
           "created": created or now(),
           "license": license, "consent": consent, "model": model, "setup": s, "bends": nb,
           "outputs": outputs or {"image": "output.webp"}, "provenance": provenance or {}}
    if contributor:
        rec["contributor"] = contributor
    return rec


# Record ids are frozen: they hash exactly these fields and nothing else, so a published id never changes when the
# dataset grows or records gain new fields (loaders, notes, provenance...). A new scheme would need a new number and
# an alias map from the old ids. Scheme 1 (2026-10-05) added the negative prompt to scheme 0's fields and normalises
# numbers (7 and 7.0 are the same fact); an absent negative means "not recorded", "" means it was empty.
ID_SCHEME = 1
ID_MODEL_KEYS = ("family", "arch", "checkpoint")
ID_SETUP_KEYS = ("route", "seed", "sampler", "scheduler", "steps", "cfg", "width", "height", "denoise", "end_at_frac",
                 "prompt", "prompt_key", "negative", "input_key")
ID_BEND_KEYS = ("path", "op", "args", "start", "end", "blend")


def _id_setup(setup: dict) -> dict:
    return {k: setup[k] for k in ID_SETUP_KEYS if k in setup}


def _id_value(v):
    """Numbers as the same fact, however they were written: 7, 7.0 and 7.000000001 hash alike."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        f = round(float(v), 6)
        return int(f) if f.is_integer() else f
    if isinstance(v, dict):
        return {k: _id_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_id_value(x) for x in v]
    return v


def record_id(model: dict, setup: dict, bends: list[dict]) -> str:
    """The frozen id of a record (ID_SCHEME): its model, setup and bends, limited to the ID_* fields, with numbers
    normalised (_id_value)."""
    return digest(_id_value({"model": {k: model.get(k) for k in ID_MODEL_KEYS}, "setup": _id_setup(setup),
                             "bends": [{k: b.get(k) for k in ID_BEND_KEYS} for b in bends]}))


def baseline_id(rec: dict) -> str:
    """Id of the unbent render that shares this record's model and setup."""
    return record_id(rec["model"], rec["setup"], [])


def measure(name: str, value, **override) -> dict:
    out = {"value": value, **METRICS.get(name, {}), **override}
    return {k: v for k, v in out.items() if v is not None}


def make_measurements(record_id: str, values: dict) -> dict:
    """values: name -> number (standard metrics get their method filled in) or name -> measure(...)."""
    vals = {k: (v if isinstance(v, dict) else measure(k, v)) for k, v in values.items()
            if v is not None and not (isinstance(v, dict) and v.get("value") is None)}
    return {"record": record_id, "values": vals}


def human(name: str | None = None, role: str | None = None) -> dict:
    return {k: v for k, v in {"type": "human", "name": name, "role": role}.items() if v}


def ai(model: str, prompt_version: str | None = None, tool: str | None = None) -> dict:
    """Author block for anything a model wrote or labelled. model is the exact model id."""
    return {k: v for k, v in {"type": "ai", "model": model, "prompt_version": prompt_version,
                              "tool": tool}.items() if v}


def make_interpretation(target: dict, kind: str, value, author: dict, basis: str | None = None,
                        created: str | None = None) -> dict:
    """target: {"record": id} or {"cell": key}. kind: caption, change, keywords, effect_tags, concepts, note, verdict.
    basis: what the author looked at, e.g. "baseline|output sheet"."""
    entry = {"target": target, "kind": kind, "value": value, "author": author,
             "created": created or now()}
    if basis:
        entry["basis"] = basis
    entry["id"] = digest({k: entry[k] for k in ("target", "kind", "value", "author")}, 16)
    return entry


def make_finding(claim: str, *, scope: dict, author: dict, citation: dict | None = None,
                 evidence: dict | None = None, quote: bool = False, level: str = "cell",
                 interpretations: list[dict] | None = None) -> dict:
    """A claim about many records. scope keys (lists): family, arch, checkpoint, group, kind, module_type, op,
    bucket, window, route. quote = the claim is the source's own words. level: "cell" (it is about the cells its
    scope matches, which then count as study-backed) or "general" (study-wide context, e.g. "type matters more
    than location"). interpretations: e.g. effect tags someone mapped the claim to, each with its author."""
    f = {"claim": claim, "quote": quote, "level": level, "scope": scope, "author": author,
         "evidence": evidence or {}, "citation": citation or {}}
    if interpretations:
        f["interpretations"] = interpretations
    f["id"] = digest({"claim": claim, "scope": scope, "citation": citation or {}}, 16)
    return f


# --------------------------------------------------------------------------- validation (a JSON Schema subset)
_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _check(obj, schema: dict, where: str, errors: list[str]) -> None:
    t = schema.get("type")
    if t:
        ts = t if isinstance(t, list) else [t]
        ok = any((isinstance(obj, (int, float)) and not isinstance(obj, bool)) if x == "number"
                 else (isinstance(obj, int) and not isinstance(obj, bool)) if x == "integer"
                 else isinstance(obj, _TYPES[x]) for x in ts)
        if not ok:
            errors.append(f"{where}: expected {t}, got {type(obj).__name__}")
            return
    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{where}: {obj!r} not one of {schema['enum']}")
    if isinstance(obj, dict):
        for k in schema.get("required", []):
            if k not in obj:
                errors.append(f"{where}: missing {k!r}")
        props = schema.get("properties", {})
        for k, v in obj.items():
            if k in props:
                _check(v, props[k], f"{where}.{k}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{where}: unexpected key {k!r}")
    if isinstance(obj, list) and "items" in schema:
        for i, v in enumerate(obj):
            _check(v, schema["items"], f"{where}[{i}]", errors)


def schema(name: str) -> dict:
    return json.loads((DATA / "schema" / f"{name}.schema.json").read_text(encoding="utf-8"))


def validate(obj, name: str) -> list[str]:
    """Errors for obj against data/kb/schema/<name>.schema.json (record, measurements, interpretation, finding)."""
    errors: list[str] = []
    _check(obj, schema(name), name, errors)
    if name == "interpretation" and isinstance(obj, dict):
        a = obj.get("author") or {}
        if a.get("type") == "ai" and not a.get("model"):
            errors.append("interpretation.author: an AI author must name its model")
    if name == "record" and isinstance(obj, dict):
        c, s = obj.get("consent") or {}, obj.get("setup") or {}
        if "prompt" in s and not c.get("prompt"):
            errors.append("record.setup: prompt present without consent")
        if "input_image" in s and not c.get("input_image"):
            errors.append("record.setup: input_image present without consent")
    return errors


# --------------------------------------------------------------------------- reading and writing a knowledge base
def record_dir(root: Path, rec: dict) -> Path:
    return Path(root) / "records" / rec["model"]["family"] / rec["source"] / rec["id"]


def save_record(root: Path, rec: dict, files: dict[str, bytes] | None = None, measurements: dict | None = None,
                interpretations: list[dict] = ()) -> Path:
    """Write a record (record.json is never rewritten once present), its files, measurements, and any new
    interpretations (deduplicated by id)."""
    errs = validate(rec, "record")
    if errs:
        raise ValueError("; ".join(errs))
    d = record_dir(root, rec)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "record.json").exists():
        (d / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, data in (files or {}).items():
        if not (d / name).exists():
            (d / name).write_bytes(data)
    if measurements:
        m = load_measurements(d)
        m["record"] = rec["id"]
        m.setdefault("values", {}).update(measurements.get("values", {}))
        (d / "measurements.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    add_interpretations(d, interpretations)
    return d


def add_interpretations(d: Path, entries) -> int:
    entries = list(entries)
    if not entries:
        return 0
    for e in entries:
        errs = validate(e, "interpretation")
        if errs:
            raise ValueError("; ".join(errs))
    p = Path(d) / "interpretations.jsonl"
    have = {e["id"] for e in load_interpretations(d)}
    new = [e for e in entries if e["id"] not in have]
    with p.open("a", encoding="utf-8") as f:
        for e in new:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return len(new)


def save_baseline(root: Path, bid: str, data: bytes, ext: str = "webp") -> Path:
    p = Path(root) / "baselines" / f"{bid}.{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(data)
    return p


def load_measurements(d: Path) -> dict:
    p = Path(d) / "measurements.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_interpretations(d: Path) -> list[dict]:
    p = Path(d) / "interpretations.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_records(root: Path):
    """(folder, record) for every record under root."""
    for p in sorted(Path(root).glob("records/*/*/*/record.json")):
        yield p.parent, json.loads(p.read_text(encoding="utf-8"))


def load_findings(root: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(root).glob("findings/*.json"))]


# --------------------------------------------------------------------------- cells
CELL_FIELDS = ("family", "group", "kind", "module_type", "op", "bucket", "window", "route")


def cell_fields(rec: dict) -> dict | None:
    """The cell a single-bend record belongs to; None for unbent or multi-bend records."""
    if len(rec.get("bends") or []) != 1:
        return None
    b = rec["bends"][0]
    return {"family": rec["model"]["family"], "group": b.get("group") or "?", "kind": b.get("kind") or "?",
            "module_type": b.get("module_type") or "*", "op": b.get("op") or "?", "bucket": b.get("bucket") or "?",
            "window": b.get("window") or "all", "route": rec["setup"].get("route") or "txt2img"}


def cell_key(fields: dict) -> str:
    return "|".join(str(fields[k]) for k in CELL_FIELDS)


def evidence_grade(n_seeds: int, n_prompts: int) -> str:
    if n_seeds >= 2 and n_prompts >= 2:
        return "replicated"
    if n_prompts >= 2:
        return "multi-prompt"
    if n_seeds >= 2:
        return "multi-seed"
    return "anecdotal"


def finding_covers(f: dict, fields: dict) -> bool:
    scope = f.get("scope") or {}
    for k, allowed in scope.items():
        if k not in fields or not allowed:
            continue
        allowed = allowed if isinstance(allowed, list) else [allowed]
        if fields[k] not in allowed and "*" not in allowed:
            return False
    return any(k in fields for k in scope)


def _latest(entries: list[dict], kind: str) -> dict | None:
    xs = [e for e in entries if e.get("kind") == kind]
    return max(xs, key=lambda e: e.get("created", "")) if xs else None


def summarize_record(d: Path, rec: dict) -> dict:
    """One line of index/records.jsonl: the record plus its measurement values and its latest interpretation of each
    kind, each still carrying its author."""
    m = load_measurements(d).get("values", {})
    ints = load_interpretations(d)
    latest = {}
    for k in INTERPRETATION_KINDS:
        e = _latest(ints, k)
        if e:
            latest[k] = {"value": e["value"], "author": e["author"]}
    cf = cell_fields(rec)
    return {**rec, "dir": str(d.relative_to(d.parents[3])).replace("\\", "/"),
            "cell": cell_key(cf) if cf else None,
            "measurements": {**{k: v.get("value") for k, v in m.items()},
                             **({"degenerate_reasons": m["degenerate"].get("reasons", [])}
                                if (m.get("degenerate") or {}).get("value") else {})},
            "interpretations": latest, "n_interpretations": len(ints)}


def aggregate_cells(summaries: list[dict], findings: list[dict] = (), cell_interpretations: list[dict] = ()) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        if s.get("cell"):
            groups[s["cell"]].append(s)
    by_cell = defaultdict(list)
    for e in cell_interpretations:
        by_cell[(e.get("target") or {}).get("cell")].append(e)
    cells = []
    for key, rs in groups.items():
        fields = dict(zip(CELL_FIELDS, key.split("|")))
        seeds = {r["setup"].get("seed") for r in rs}
        prompts = {r["setup"].get("prompt_key") or r["setup"].get("input_key") for r in rs}
        args: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            for k, v in (r["bends"][0].get("args") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    args[k].append(float(v))
        # Broken renders say what can go wrong, not what the bend does: statistics and effect tags use intact ones.
        degen = [r for r in rs if (r.get("measurements") or {}).get("degenerate")]
        intact = [r for r in rs if not (r.get("measurements") or {}).get("degenerate")]
        meas: dict[str, list[float]] = defaultdict(list)
        for r in intact:
            for k, v in (r.get("measurements") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    meas[k].append(float(v))
        tags, tag_authors, keywords = Counter(), defaultdict(set), Counter()
        n_described = 0
        for r in intact:
            it = r.get("interpretations") or {}
            et = it.get("effect_tags")
            if et:
                n_described += 1
                for t in et["value"]:
                    tags[t] += 1
                    tag_authors[t].add(_author_label(et["author"]))
            kw = it.get("keywords")
            if kw:
                keywords.update(kw["value"])
        reasons = Counter(x for r in degen for x in (r.get("measurements") or {}).get("degenerate_reasons") or [])
        measured = [r["measurements"] for r in rs if "prompt_retention" in (r.get("measurements") or {})
                    or "clip_degenerate" in (r.get("measurements") or {})]
        signals = {}
        if measured:
            ret = [m["prompt_retention"] for m in measured if m.get("prompt_retention") is not None]
            deg = [m["clip_degenerate"] for m in measured if m.get("clip_degenerate") is not None]
            signals = {"n": len(measured),
                       "subject_fades": round(sum(x < RISK_RETENTION for x in ret) / len(ret), 3) if ret else None,
                       "noise_or_blob_look": round(sum(x >= RISK_DEGENERATE for x in deg) / len(deg), 3) if deg else None,
                       "prompt_retention_mean": round(sum(ret) / len(ret), 3) if ret else None}
        risky = [r for r in rs if at_risk(r.get("measurements") or {})]
        failure_notes = [{"record": r["id"], "broken": bool((r.get("measurements") or {}).get("degenerate")),
                          "value": r["interpretations"]["change"]["value"],
                          "author": r["interpretations"]["change"]["author"]}
                         for r in sorted(risky, key=lambda r: not (r.get("measurements") or {}).get("degenerate"))
                         if (r.get("interpretations") or {}).get("change")][:3]
        pins = sum(1 for r in rs if (r.get("interpretations") or {}).get("verdict", {}).get("value") == "pinned")
        vetoes = sum(1 for r in rs if (r.get("interpretations") or {}).get("verdict", {}).get("value") == "vetoed")
        covering = [f["id"] for f in findings if f.get("level", "cell") == "cell" and finding_covers(f, fields)]
        examples = sorted(intact or rs, key=lambda r: (-(r.get("n_interpretations") or 0), r["id"]))[:6]
        broken_examples = sorted(degen, key=lambda r: (-(r.get("n_interpretations") or 0), r["id"]))[:3]
        cell = {"key": key, **fields, "n": len(rs), "seeds": len(seeds), "prompts": len(prompts),
                "checkpoints": sorted({r["model"].get("checkpoint") or "?" for r in rs}),
                "archs": sorted({r["model"].get("arch") or "?" for r in rs}),
                "sources": dict(Counter(r["source"] for r in rs)),
                "args": {k: [min(v), max(v)] for k, v in args.items()},
                "n_intact": len(intact),
                # over intact renders only
                "measurements": {k: {"mean": round(sum(v) / len(v), 4), "sd": round(_sd(v), 4), "n": len(v)}
                                 for k, v in meas.items()},
                # over all renders: how often it broke and why, softer warning signs, and what failure looks like
                "degenerate_rate": round(len(degen) / len(rs), 3),
                "broken_reasons": dict(reasons),
                "signals": signals,
                "failure_notes": failure_notes,
                "broken_examples": {r["id"]: r["dir"] for r in broken_examples},
                # share = among the records someone described, not all records: cells are often described by
                # one representative record
                "n_described": n_described,
                "effect_tags": {t: {"count": c, "share": round(c / max(n_described, 1), 3),
                                    "by": sorted(tag_authors[t])} for t, c in tags.most_common()},
                "keywords": [k for k, _ in keywords.most_common(12)],
                "feedback": {"pinned": pins, "vetoed": vetoes},
                "grade": evidence_grade(len(seeds), len(prompts)), "study_backed": bool(covering),
                "findings": covering, "examples": [r["id"] for r in examples],
                "example_paths": {r["id"]: r["dir"] for r in examples},
                "example_baselines": {r["id"]: r["outputs"].get("baseline_id") for r in examples},
                "example_captions": {r["id"]: {k: (r.get("interpretations") or {})[k] for k in ("caption", "change")
                                               if k in (r.get("interpretations") or {})} for r in examples},
                "recipe": recipe_template(rs[0]),
                "interpretations": [{k: e[k] for k in ("kind", "value", "author")} for e in by_cell.get(key, [])]}
        cells.append(cell)
    return sorted(cells, key=lambda c: c["key"])


def _sd(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _author_label(a: dict) -> str:
    return f"ai:{a.get('model')}" if a.get("type") == "ai" else f"human:{a.get('name') or 'anonymous'}"


def recipe_template(rec: dict) -> dict:
    """An Apply Bends from JSON bend for this record (amount from this example; cells list the tested range)."""
    b = rec["bends"][0]
    out = {"path": b["path"], "module_type": b["op"], "module_args": b.get("args") or {}}
    if b.get("window") != "all":
        out["t"] = [round(1 - b.get("start", 0.0), 3), round(1 - b.get("end", 1.0), 3)]
    return out


PROMPT_PLACEHOLDER = "<your prompt: this record's prompt was not shared>"


def comfy_workflow(rec: dict) -> dict | None:
    """A ComfyUI API-format workflow that renders this record from its facts: the checkpoint (plus separate text
    encoder / VAE / sampling mode when the record names them), its sampler, steps, cfg, seed, size, prompt and
    negative, and its bends in one Apply Bends from JSON node with clamping off, so the amounts are used exactly.
    Prompts that were not shared become a placeholder. None for routes other than txt2img."""
    s, m = rec["setup"], rec["model"]
    if s.get("route", "txt2img") != "txt2img" or not m.get("checkpoint"):
        return None
    bends = []
    for b in rec["bends"]:
        bends.append({"path": b["path"], "module_type": b["op"], "module_args": b.get("args") or {},
                      **({"blend": b["blend"]} if b.get("blend") is not None else {})})
    doc: dict = {"bends": bends}
    lo = next((b.get("steps_min") for b in rec["bends"] if b.get("steps_min") is not None), None)
    hi = next((b.get("steps_max") for b in rec["bends"] if b.get("steps_max") is not None), None)
    if lo is not None or hi is not None:  # executed-step window, as the producer sent it
        doc.update({k: v for k, v in (("steps_min", lo), ("steps_max", hi)) if v is not None})
    elif any(b.get("window", "all") != "all" for b in rec["bends"]):
        for jb, b in zip(bends, rec["bends"]):
            jb["t"] = [round(1 - b.get("start", 0.0), 3), round(1 - b.get("end", 1.0), 3)]
    clip, vae = ["1", 1], ["1", 2]
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m["checkpoint"]}}}
    if m.get("clip"):
        wf["2"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": "stable_diffusion"}}
        clip = ["2", 0]
    if m.get("vae"):
        wf["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}}
        vae = ["3", 0]
    model = ["1", 0]
    if m.get("model_sampling"):
        wf["4"] = {"class_type": "ModelSamplingDiscrete", "inputs": {"model": model, "sampling": m["model_sampling"],
                                                                     "zsnr": False}}
        model = ["4", 0]
    if bends:
        wf["5"] = {"class_type": "ApplyBendsFromJSON", "_meta": {"title": f"Bend (knowledge base record {rec['id']})"},
                   "inputs": {"model": model, "bends_json": json.dumps(doc, indent=1), "strict": False,
                              "clamp": "none"}}
        model = ["5", 0]
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": s.get("prompt") or PROMPT_PLACEHOLDER}}
    wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": s.get("negative") or ""}}
    wf["8"] = {"class_type": "EmptyLatentImage", "inputs": {"width": s.get("width", 512), "height": s.get("height", 512),
                                                             "batch_size": 1}}
    wf["9"] = {"class_type": "KSampler", "inputs": {
        "model": model, "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["8", 0], "seed": s["seed"],
        "steps": s["steps"], "cfg": s["cfg"], "sampler_name": s["sampler"], "scheduler": s["scheduler"],
        "denoise": s.get("denoise", 1.0),
        # read by the ComfyUI frontend when the image is dropped (keeps the seed), ignored by the server
        "control_after_generate": "fixed"}}
    wf["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": vae}}
    wf["11"] = {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "model_bending_kb"}}
    return wf


def build_index(root: Path, out: Path | None = None) -> dict:
    """Validate everything and rebuild index/records.jsonl, cells.jsonl and findings.jsonl. Returns counts and
    validation errors."""
    root = Path(root)
    out = Path(out or root / "index")
    out.mkdir(parents=True, exist_ok=True)
    errors, summaries = [], []
    for d, rec in iter_records(root):
        errors += [f"{d.name}: {e}" for e in validate(rec, "record")]
        for e in load_interpretations(d):
            errors += [f"{d.name}: {x}" for x in validate(e, "interpretation")]
        summaries.append(summarize_record(d, rec))
    findings = load_findings(root)
    for f in findings:
        errors += [f"finding {f.get('id')}: {x}" for x in validate(f, "finding")]
    cell_ints = []
    p = root / "cells" / "interpretations.jsonl"
    if p.exists():
        cell_ints = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    cells = aggregate_cells(summaries, findings, cell_ints)
    aliases = id_aliases(root)
    if aliases["records"] or aliases["baselines"]:
        (out / "id_aliases.json").write_text(json.dumps(aliases, indent=1), encoding="utf-8")
    _write_jsonl(out / "records.jsonl", summaries)
    _write_jsonl(out / "cells.jsonl", cells)
    _write_jsonl(out / "findings.jsonl", findings)
    meta = {"built": now(), "schema_version": SCHEMA_VERSION, "records": len(summaries), "cells": len(cells),
            "findings": len(findings), "families": dict(Counter(s["model"]["family"] for s in summaries)),
            "sources": dict(Counter(s["source"] for s in summaries))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {**meta, "errors": errors}


def id_aliases(root: Path) -> dict:
    """Old id -> current id for records and baselines, from migrations/id_scheme_*.json, chained across schemes, so
    links made with an id from before a migration still resolve."""
    rec, base = {}, {}
    for p in sorted(Path(root).glob("migrations/id_scheme_*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        for src, dst in ((m.get("records", {}), rec), (m.get("baselines", {}), base)):
            for old, new in src.items():
                for k, v in list(dst.items()):
                    if v == old:
                        dst[k] = new
                dst[old] = new
    return {"records": rec, "baselines": base}


def resolve_id(rid: str, aliases: dict, kind: str = "records") -> str:
    """The current id for a record (or baseline) id that may predate a migration."""
    return (aliases or {}).get(kind, {}).get(rid, rid)


def _write_jsonl(p: Path, rows) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# --------------------------------------------------------------------------- queries (find_recipes)
def load_vocab() -> dict:
    return json.loads((DATA / "vocab" / "effects.json").read_text(encoding="utf-8"))


# The knowledge base's description prompt. The community captions were written with exactly this text (by the
# maintainer's captioner); agents describing a user's results use it too, so descriptions stay comparable. Change the
# wording only together with CAPTION_PROMPT_VERSION.
CAPTION_PROMPT_VERSION = "kb-caption-v1"
CAPTION_PROMPT = """You describe the visual effect of interventions inside an image-generation model, for a public knowledge
base that artists and agents consult. Each labelled pair shows the unchanged picture (left, "before") and the changed
one (right, "after"), made with the same seed and prompt. You are not told what was done; describe only what you see.

For every label give:
- caption: what the AFTER image shows, 8-20 plain words.
- change: what changed from before to after, one sentence an artist would understand (composition, subject,
  palette, light, texture, style, abstraction). Say "almost no visible change" when that is the case.
- keywords: 3-8 short lowercase phrases someone might search for.
- effect_tags: tags from this list only (use none that do not fit): {tags}"""


def caption_prompt() -> str:
    """CAPTION_PROMPT with the current effect-tag vocabulary filled in."""
    return CAPTION_PROMPT.replace("{tags}", ", ".join(load_vocab()["tags"]))


def goal_tags(goal: str, vocab: dict | None = None) -> list[str]:
    """Effect tags a plain-language goal asks for, e.g. "more abstract, keep the subject" -> abstract, subject_kept."""
    vocab = vocab or load_vocab()
    g = " " + re.sub(r"[^a-z0-9 ]+", " ", goal.lower()) + " "
    hits = []
    for tag, info in vocab["tags"].items():
        words = [tag.replace("_", " ")] + info.get("synonyms", [])
        if any(f" {w.lower()} " in g or (len(w) > 5 and w.lower() in g) for w in words):
            hits.append(tag)
    return hits


_GRADE_BONUS = {"replicated": 1.0, "multi-prompt": 0.6, "multi-seed": 0.4, "anecdotal": 0.0}


def risk_summary(c: dict) -> dict:
    """What can go wrong with a cell's bend, in words an agent can pass on: how often it broke (and why), how often
    the subject faded or the image read as noise/blobs, and descriptions of failing renders (with their author)."""
    s = c.get("signals") or {}
    parts = []
    if c.get("degenerate_rate"):
        why = ", ".join(c.get("broken_reasons") or {}) or "broken"
        parts.append(f"{round(100 * c['degenerate_rate'])}% of renders broke ({why})")
    if s.get("subject_fades"):
        parts.append(f"subject faded in {round(100 * s['subject_fades'])}%")
    if s.get("noise_or_blob_look"):
        parts.append(f"read as noise or blobs in {round(100 * s['noise_or_blob_look'])}%")
    return {"summary": "; ".join(parts) or "no failures seen", "broken_share": c.get("degenerate_rate", 0.0),
            "signals": s, "failure_notes": c.get("failure_notes", []), "broken_examples": c.get("broken_examples", {})}


def rank_cells(cells: list[dict], goal: str = "", *, family: str | None = None, arch: str | None = None,
               tags: list[str] | None = None, findings: list[dict] = (), limit: int = 8,
               vocab: dict | None = None) -> dict:
    """Cells ranked for a goal. Score = how often the goal's effect tags were seen among the cell's described records
    (any author) + goal words in keywords + evidence grade + study backing + same architecture - degeneracy (measured,
    or tagged by a describer). Unmatched cells are left out when the goal names known effects."""
    want = list(tags or []) or (goal_tags(goal, vocab) if goal else [])
    words = {w for w in re.findall(r"[a-z]{4,}", goal.lower())}
    fam = family or (family_of(arch) if arch else None)
    pool = [c for c in cells if not fam or c["family"] == fam]
    ranked = []
    for c in pool:
        if c.get("n_intact", c["n"]) == 0 and "degenerate" not in want:
            continue  # only broken renders: a known failure, not a recipe
        tag_score = sum(c["effect_tags"].get(t, {}).get("share", 0.0) for t in want)
        kw_score = 0.3 * len(words & {k.lower() for k in c.get("keywords", [])})
        if want and tag_score == 0 and kw_score == 0:
            continue
        score = (2.0 * tag_score + kw_score + 0.5 * _GRADE_BONUS.get(c["grade"], 0) + 0.3 * c["study_backed"]
                 - 1.5 * c["degenerate_rate"] + 0.2 * c["feedback"]["pinned"] - 0.4 * c["feedback"]["vetoed"])
        if "degenerate" in c["effect_tags"] and "degenerate" not in want:
            score -= 2.0 * c["effect_tags"]["degenerate"]["share"]
        if arch and arch in c.get("archs", []):
            score += 0.5
        ranked.append((round(score, 3), c))
    ranked.sort(key=lambda x: -x[0])
    fids = {f["id"]: f for f in findings}
    results = []
    for score, c in ranked[:limit]:
        results.append({"score": score, "cell": c["key"], "where": f"{c['group']} / {c['kind']}"
                        + (f" ({c['module_type']})" if c["module_type"] != "*" else ""),
                        "op": c["op"], "amount": c["bucket"], "args_tested": c["args"], "window": c["window"],
                        "route": c["route"], "recipe": c["recipe"],
                        "evidence": {"grade": c["grade"], "n": c["n"], "seeds": c["seeds"], "prompts": c["prompts"],
                                     "tested_on": c["checkpoints"], "archs": c["archs"], "sources": c["sources"],
                                     "study_backed": c["study_backed"]},
                        "effects": {t: v for t, v in list(c["effect_tags"].items())[:6]},
                        "keywords": c.get("keywords", [])[:8],
                        "degenerate_rate": c["degenerate_rate"], "examples": c["examples"][:3],
                        "risks": risk_summary(c),
                        "findings": [{"claim": fids[f]["claim"], "author": fids[f]["author"],
                                      "citation": fids[f].get("citation")} for f in c["findings"] if f in fids][:3],
                        "interpretations": c.get("interpretations", [])[:3]})
    note = None
    if arch and fam and results and all(arch not in r["evidence"]["archs"] for r in results):
        note = f"no records on {arch} itself; these were tested on other {fam} checkpoints"
    general = [{"claim": f["claim"], "author": f["author"], "citation": f.get("citation")} for f in findings
               if f.get("level") == "general" and (not fam or fam in (f.get("scope") or {}).get("family", [fam]))]
    return {"goal": goal, "effect_tags": want, "family": fam, "results": results, "note": note,
            "considered": len(pool), "general_findings": general}


# --------------------------------------------------------------------------- source references, navigator links
# A bend can say where it came from with a "kb" key: {"dataset": ..., "cell": ..., "record": ...}, each optional
# (no dataset = the community one). Apply Bends from JSON ignores the key; check_bends keeps and checks it.
COMMUNITY_DATASET = "abuzreq/model-bending-knowledge-base"
NAVIGATOR = "https://abuzreq-model-bending-navigator.static.hf.space"
KB_REF_KEYS = ("dataset", "cell", "record")
TRAY_VERSION = 1


def kb_ref(given) -> tuple[dict | None, list[str]]:
    """A bend's "kb" value, tidied: (the reference or None, problems)."""
    if not isinstance(given, dict):
        return None, ['kb must be an object such as {"cell": "sd1|mid|…"} or {"record": "<id>"}']
    ref = {k: given[k] for k in KB_REF_KEYS if isinstance(given.get(k), str) and given[k].strip()}
    problems = [f"kb.{k} is not understood (kb holds {', '.join(KB_REF_KEYS)}, each a string)"
                for k in given if k not in ref]
    if not ({"cell", "record"} & set(ref)):
        problems.append("kb names no cell or record, so it cannot be traced back")
    return (ref or None), problems


def cell_link(cell_key: str, base: str = NAVIGATOR) -> str:
    """The navigator page for one cell: its before/after examples, evidence and risks."""
    from urllib.parse import quote
    return f"{base}/?cell={quote(cell_key, safe='')}"


def record_link(record_id: str, base: str = NAVIGATOR) -> str:
    """The navigator page for one record: its before/after (with a wipe), captions, measurements and setup."""
    return f"{base}/?record={record_id}"


def ref_link(ref: dict | None) -> str | None:
    """The most specific navigator page for a kb reference (a record's page, else its cell's), when it is in the
    community knowledge base."""
    if not ref or ref.get("dataset", COMMUNITY_DATASET) != COMMUNITY_DATASET:
        return None
    if re.fullmatch(r"[0-9a-f]{8,40}", ref.get("record", "")):
        return record_link(ref["record"])
    return cell_link(ref["cell"]) if ref.get("cell") else None


def tray_link(bends: list[dict], base: str = NAVIGATOR) -> str:
    """A link that carries separate bends (each meant to be tried on its own) to the navigator, or to open_tray.
    The bends travel in the URL fragment, which browsers do not send to the server."""
    import base64
    raw = json.dumps({"v": TRAY_VERSION, "bends": bends}, separators=(",", ":"), ensure_ascii=False)
    return f"{base}/#tray=" + base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def read_tray(text: str) -> list[dict]:
    """The bends in a tray link (or in the bare `tray=…` part of one). Raises ValueError when there is none."""
    import base64
    m = re.search(r"tray=([A-Za-z0-9_-]+)", text or "")
    if not m:
        raise ValueError("this is not a navigator tray link (it has no '#tray=…' part)")
    data = m.group(1)
    try:
        tray = json.loads(base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8"))
    except ValueError:
        raise ValueError("the tray part of this link is damaged: copy the whole link again") from None
    if not isinstance(tray, dict) or not isinstance(tray.get("bends"), list):
        raise ValueError("the tray holds no list of bends")
    if not isinstance(tray.get("v", 1), int) or tray.get("v", 1) > TRAY_VERSION:
        raise ValueError(f"this tray was made by a newer navigator (version {tray.get('v')}); update the skill")
    return tray["bends"]


# --------------------------------------------------------------------------- reading ComfyUI prompts
_SAMPLERS = {"KSampler": ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"),
             "KSamplerAdvanced": ("noise_seed", "steps", "cfg", "sampler_name", "scheduler")}
_LOADERS = {"CheckpointLoaderSimple": "ckpt_name", "CheckpointLoader": "ckpt_name", "UNETLoader": "unet_name",
            "UnetLoaderGGUF": "unet_name"}
_DISCRETE_OPS = {"Rotate Module (Bending)": "rotate", "Scale Module (Bending)": "scale",
                 "Multiply Scalar Module (Bending)": "multiply", "Add Scalar Module (Bending)": "add_scalar",
                 "Add Noise Module (Bending)": "add_noise", "Threshold Module (Bending)": "threshold"}


def _link(v) -> str | None:
    return str(v[0]) if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int) else None


def _text_of(prompt: dict, link) -> str | None:
    """Follow a conditioning link back to its text (through simple pass-through nodes)."""
    seen = set()
    nid = _link(link)
    while nid and nid not in seen and nid in prompt:
        seen.add(nid)
        node = prompt[nid]
        ins = node.get("inputs", {})
        for k in ("text", "text_g", "prompt"):
            if isinstance(ins.get(k), str):
                return ins[k]
        nid = next((_link(v) for k, v in ins.items() if _link(v) and k in ("conditioning", "conditioning_to",
                                                                            "conditioning_1", "clip")), None)
    return None


def _module_from(prompt: dict, link) -> tuple[str | None, dict, dict]:
    """(op, args, timing) of a discrete bending module node, unwrapping Timestep Gated Bending."""
    nid = _link(link)
    timing = {}
    for _ in range(5):
        if not nid or nid not in prompt:
            return None, {}, timing
        node = prompt[nid]
        ct, ins = node["class_type"], node.get("inputs", {})
        if ct == "Timestep Gated Bending":
            timing = {"t": [ins.get("t_start"), ins.get("t_end")]}
            nid = _link(ins.get("bending_module"))
            continue
        op = _DISCRETE_OPS.get(ct)
        args = {k: v for k, v in ins.items() if not _link(v)}
        return op or ct, args, timing
    return None, {}, timing


def setup_from_api_prompt(prompt: dict, resolved_json: str | None = None) -> tuple[dict, dict, list[dict]]:
    """(model, setup, bends) read from a ComfyUI API prompt (e.g. /api/history/<id> ["prompt"][2]).
    resolved_json, when given (Apply Bends from JSON's output), is preferred over the node's bends_json, because
    it shows what actually ran after placeholders and clamping."""
    model, setup, bends = {}, {}, []
    for nid, node in prompt.items():
        ct, ins = node.get("class_type"), node.get("inputs", {})
        if ct in _LOADERS and isinstance(ins.get(_LOADERS[ct]), str):
            model["checkpoint"] = ins[_LOADERS[ct]]
        elif ct in _SAMPLERS:
            fields = _SAMPLERS[ct]
            for f in fields:
                if f in ins and not _link(ins[f]):
                    setup["seed" if f == "noise_seed" else f.replace("sampler_name", "sampler")] = ins[f]
            pos = _text_of(prompt, ins.get("positive"))
            neg = _text_of(prompt, ins.get("negative"))
            if pos is not None:
                setup["prompt"] = pos
            if neg:
                setup["negative"] = neg
            lat = _link(ins.get("latent_image"))
            ln = prompt.get(lat or "", {})
            if ln.get("class_type") == "RepeatLatentBatch":  # a starting picture repeated for several seeds
                ln = prompt.get(_link(ln["inputs"].get("samples")) or "", {})
            if ln.get("class_type") in ("EmptyLatentImage", "EmptySD3LatentImage"):
                setup["width"], setup["height"] = ln["inputs"].get("width"), ln["inputs"].get("height")
                setup["route"] = "txt2img"
            elif ln.get("class_type") in ("EmptyHunyuanLatentVideo", "WanImageToVideo"):
                setup["width"], setup["height"] = ln["inputs"].get("width"), ln["inputs"].get("height")
                setup["frames"] = ln["inputs"].get("length")
                setup["route"] = "img2video" if ln["class_type"] == "WanImageToVideo" else "txt2video"
                img = prompt.get(_link(ln["inputs"].get("start_image")) or "", {})
                if img.get("class_type") == "LoadImage":
                    setup["input_image"] = img["inputs"].get("image")
            elif ln.get("class_type") == "VAEEncode":
                setup["route"] = "img2img"
                img = prompt.get(_link(ln["inputs"].get("pixels")) or "", {})
                if img.get("class_type") == "LoadImage":
                    setup["input_image"] = img["inputs"].get("image")
        elif ct == "ApplyBendsFromJSON":
            src = resolved_json or ins.get("bends_json")
            try:
                data = json.loads(src) if isinstance(src, str) else (src or {})
            except json.JSONDecodeError:
                data = {}
            for b in (data.get("bends") if isinstance(data, dict) else data) or []:
                bends.append({"path": b.get("path"), "op": b.get("module_type"), "args": b.get("module_args") or {},
                              **{k: b[k] for k in ("t", "steps", "blend", "label") if k in b}})
            for a in (data.get("attention_bends") if isinstance(data, dict) else None) or []:
                bends.append(_attention_bend(a, a.get("module_type"), a.get("module_args") or {}))
        elif ct == "Attention Map Bending":
            op, args, timing = _module_from(prompt, ins.get("bending_module"))
            bends.append({**_attention_bend(ins, op, args), **timing})
        elif ct in ("Model Bending", "Model Bending (SD Layers)", "DiT Block Bending"):
            op, args, timing = _module_from(prompt, ins.get("bending_module"))
            paths = ins.get("path") or ins.get("blocks") or ""
            t = timing or ({"t": [ins["t_start"], ins["t_end"]]} if "t_start" in ins and "t_end" in ins else {})
            for p in [x.strip() for x in str(paths).split(",") if x.strip()]:
                bends.append({"path": p, "op": op, "args": args, **t,
                              **({"steps": ins["steps_to_bend_str"]} if ins.get("steps_to_bend_str") else {})})
    setup.setdefault("route", "txt2img")
    if setup.get("route") == "txt2img" and setup.get("denoise") == 1:
        setup.pop("denoise")
    model["arch"] = guess_arch(model.get("checkpoint", ""))
    return model, setup, bends


def _attention_bend(a: dict, op, args: dict) -> dict:
    """A video attention bend as a record bend: the path names the attention route and blocks."""
    blocks = a.get("blocks", "*")
    blocks = ",".join(map(str, blocks)) if isinstance(blocks, list) else str(blocks)
    out = {"path": f"attention.{a.get('attention', 'cross_text')}.blocks[{blocks}]", "op": op, "args": args,
           "group": "attention", "kind": a.get("attention", "cross_text")}
    if a.get("t"):
        out["t"] = a["t"]
    elif "t_start" in a and "t_end" in a:
        out["t"] = [a["t_start"], a["t_end"]]
    if a.get("steps") not in (None, "*", ""):
        out["steps"] = a["steps"]
    return out


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("index", help="validate a knowledge base folder and rebuild its index/")
    b.add_argument("root")
    f = sub.add_parser("find", help="rank cells for a goal")
    f.add_argument("goal")
    f.add_argument("--root", help="knowledge base folder (default: the bundled community snapshot)")
    f.add_argument("--arch")
    f.add_argument("--limit", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "index":
        r = build_index(Path(a.root))
        print(json.dumps({k: v for k, v in r.items() if k != "errors"}, indent=1))
        for e in r["errors"][:50]:
            print("ERROR", e)
    else:
        idx = Path(a.root) / "index" if a.root else DATA / "community"
        res = rank_cells(read_jsonl(idx / "cells.jsonl"), a.goal, arch=a.arch, limit=a.limit,
                         findings=read_jsonl(idx / "findings.jsonl"))
        print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
