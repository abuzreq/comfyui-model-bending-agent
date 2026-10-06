#!/usr/bin/env python3
"""Full runs for the community bend knowledge base: plan a systematic sweep of one model, render it on your ComfyUI,
check it, and send it as a pull request on the dataset (https://huggingface.co/datasets/abuzreq/model-bending-knowledge-base).

The knowledge base takes full runs, not single pictures: one model, every part of its U-Net (all seven regions), at
least three operations with at least three amounts each. One prompt and seed is enough; more seeds and prompts let
the results count as replicated. Measuring and describing are optional: the maintainer adds what you leave out.

  python kb_run.py init --arch sd15 --checkpoint v1-5-pruned-emaonly.safetensors --name "Your name" --out run.json
      edit run.json (prompts, seeds, layers, ops and amounts, windows), set "agree_cc0": true, then
  python kb_run.py render run.json --out my_run        # resumable; needs ComfyUI with ComfyUI-Model-Bending
  python kb_run.py check my_run                       # format + the full-run rule, with a coverage table
  python kb_run.py measure my_run                     # optional: LPIPS, DINOv2, CLIP (needs torch, see kb_measure.py)
  python kb_run.py sheets my_run                      # optional: contact sheets for your agent to describe ...
  python kb_run.py add-descriptions my_run descriptions.json --model <model id>   # ... saved as AI descriptions
  python kb_run.py submit my_run [--dry-run]          # a PR on the dataset, with your own Hugging Face token
  python kb_run.py layers --checkpoint X.safetensors  # list a model's bendable layers, for deeper plans

Everything in the records is a fact (model, settings, the bend as sent, the picture, pixel measurements); the
renders carry their ComfyUI workflow, so anyone can run the same bend. Your prompts are published with the run.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import math
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bendjson  # noqa: E402
import kb  # noqa: E402

PLAN_VERSION = 1
LICENSE = "CC0-1.0"
REGIONS = ("in.hi", "in.mid", "in.lo", "mid", "out.lo", "out.mid", "out.hi")
MIN_OPS, MIN_AMOUNTS = 3, 3
UNET_FAMILIES = ("sd1", "sd2", "sdxl")
# one block per region of the U-Net (its first ResBlock or attention block): the knowledge base's tier-1 layers
LAYERS = {
    "sd1": {"in.hi": "input_blocks.2.0", "in.mid": "input_blocks.5.0", "in.lo": "input_blocks.8.0",
            "mid": "middle_block.1", "out.lo": "output_blocks.3.1", "out.mid": "output_blocks.6.0",
            "out.hi": "output_blocks.10.1"},
    "sdxl": {"in.hi": "input_blocks.2.0", "in.mid": "input_blocks.5.0", "in.lo": "input_blocks.8.0",
             "mid": "middle_block.1", "out.lo": "output_blocks.1.1", "out.mid": "output_blocks.4.0",
             "out.hi": "output_blocks.7.0"},
}
LAYERS["sd2"] = LAYERS["sd1"]
OPS = {"multiply": [0.0, 0.5, 2.0], "rotate": [30, 90, 180], "scale": [0.7, 1.4, 2.0], "add_noise": [0.2, 0.8, 1.5]}
WINDOWS = {"all": [0.0, 1.0], "early": [0.0, 0.3], "middle": [0.3, 0.8], "late": [0.8, 1.0]}
SAMPLING = {"sd1": {"sampler": "dpmpp_2m", "scheduler": "karras", "steps": 20, "cfg": 7.0, "width": 512, "height": 512},
            "sd2": {"sampler": "dpmpp_2m", "scheduler": "karras", "steps": 25, "cfg": 7.0, "width": 768, "height": 768},
            "sdxl": {"sampler": "dpmpp_2m", "scheduler": "karras", "steps": 25, "cfg": 6.0, "width": 1024,
                     "height": 1024}}
PROMPTS = [{"prompt": "a lighthouse on a rocky coast at dusk, oil painting", "negative": ""},
           {"prompt": "portrait of an old fisherman, analog photograph", "negative": ""}]


class RunError(RuntimeError):
    pass


# --------------------------------------------------------------------------- plan
def init_plan(arch: str, checkpoint: str, name: str | None, clip: str | None = None, vae: str | None = None,
              model_sampling: str | None = None) -> dict:
    fam = kb.family_of(arch)
    if fam not in UNET_FAMILIES:
        raise RunError(f"{arch}: full runs are for U-Net models (SD1.x, SD2, SDXL) for now")
    slug = re.sub(r"[^a-z0-9]+", "-", Path(checkpoint).stem.lower()).strip("-")[:40]
    return {"kb_run": PLAN_VERSION, "run": f"{slug}-{datetime.date.today():%Y%m%d}",
            "contributor": {"name": name, "agree_cc0": False,
                            "_note": "name: shown with the run (null = 'not named'). agree_cc0: the renders and "
                                     "records are released under CC0 1.0 (public domain); set it to true to render."},
            "model": {"checkpoint": checkpoint, "arch": arch,
                      **{k: v for k, v in (("clip", clip), ("vae", vae), ("model_sampling", model_sampling)) if v}},
            "sampling": dict(SAMPLING[fam]),
            "prompts": PROMPTS[:1], "seeds": [42],
            "layers": {r: [p] for r, p in LAYERS[fam].items()},
            "ops": {k: list(v) for k, v in OPS.items()},
            "windows": ["all"],
            "_help": {"layers": "region -> layer paths (add more per region for a deeper run; `kb_run.py layers` lists "
                                "them). Every region must keep at least one.",
                      "ops": f"op -> amounts: at least {MIN_OPS} ops with {MIN_AMOUNTS} amounts each. Amounts past "
                             "the safe range are welcome: what breaks is knowledge too.",
                      "windows": f"which steps a bend acts on: {', '.join(WINDOWS)} (all = every step).",
                      "prompts": "each with its negative (\"\" if none). They are published with the run."}}


def plan_hash(plan: dict) -> str:
    return hashlib.sha256(kb.canonical({k: v for k, v in plan.items() if not k.startswith("_")}).encode()).hexdigest()[:12]


def spatial(path: str) -> bool:
    """Layers whose output is image-shaped (B, C, H, W) on an SD-style U-Net; token-shaped ones (time embedding,
    ResBlock embedding layers, everything inside transformer_blocks) are not, so spatial ops skip them."""
    return not (path.startswith(("time_embed", "label_emb")) or ".emb_layers" in path or ".transformer_blocks." in path)


def step_range(start: float, end: float, steps: int) -> tuple[int | None, int | None]:
    """A [start, end] fraction of the executed steps as inclusive step indices; (None, None) = every step."""
    lo = min(steps - 1, math.floor(start * steps))
    hi = max(lo, min(steps - 1, math.ceil(round(end * steps, 6)) - 1))
    return (None, None) if lo == 0 and hi == steps - 1 else (lo, hi)


def check_plan(plan: dict) -> list[str]:
    errs = []
    if plan.get("kb_run") != PLAN_VERSION:
        errs.append(f"not a kb_run plan (version {PLAN_VERSION})")
    m = plan.get("model") or {}
    fam = kb.family_of(m.get("arch"))
    if fam not in UNET_FAMILIES:
        errs.append(f"model.arch {m.get('arch')!r}: full runs are for U-Net models (sd15, sd14, sd21, sdxl) for now")
    if not m.get("checkpoint"):
        errs.append("model.checkpoint is missing")
    s = plan.get("sampling") or {}
    for k in ("sampler", "scheduler", "steps", "cfg", "width", "height"):
        if s.get(k) in (None, ""):
            errs.append(f"sampling.{k} is missing")
    if not plan.get("prompts") or any("prompt" not in p or "negative" not in p for p in plan["prompts"]):
        errs.append("prompts: each needs \"prompt\" and \"negative\" (\"\" if none)")
    if not plan.get("seeds"):
        errs.append("seeds: at least one")
    layers = plan.get("layers") or {}
    missing = [r for r in REGIONS if not layers.get(r)]
    if missing:
        errs.append(f"layers: every region needs at least one layer; missing {', '.join(missing)}")
    if fam in UNET_FAMILIES:
        for r, paths in layers.items():
            for p in paths:
                g = kb.unet_group(p, fam)
                if g != r:
                    errs.append(f"layers.{r}: {p} is in region {g}, not {r}")
    ops = plan.get("ops") or {}
    for op, amounts in ops.items():
        if op not in bendjson.OPS or op in bendjson.VIDEO_OPS or op in bendjson.INNER_OPS:
            errs.append(f"ops.{op}: not an op a run can sweep")
            continue
        arg, (typ, _, lim) = next(iter(bendjson.OPS[op].items()))
        if typ not in (int, float):
            errs.append(f"ops.{op}: its argument is not a number")
            continue
        for a in amounts:
            if lim and not lim[0] <= a <= lim[1]:
                errs.append(f"ops.{op}: {a} is outside the node's limits {lim}")
    full = [op for op, a in ops.items() if len(set(a)) >= MIN_AMOUNTS]
    if len(full) < MIN_OPS:
        errs.append(f"ops: at least {MIN_OPS} ops with {MIN_AMOUNTS} different amounts each (have {len(full)})")
    bad = [w for w in plan.get("windows") or [] if w not in WINDOWS]
    if bad or not plan.get("windows"):
        errs.append(f"windows: choose from {', '.join(WINDOWS)}")
    return errs


def bends_of(plan: dict) -> list[tuple[str, str, str, float, str]]:
    """(region, path, op, amount, window) for every render the plan asks for, spatial ops only on image layers."""
    out = []
    for r in REGIONS:
        for path in plan["layers"].get(r, []):
            for op, amounts in plan["ops"].items():
                if op in bendjson.SPATIAL_OPS and not spatial(path):
                    continue
                for a in amounts:
                    for w in plan["windows"]:
                        out.append((r, path, op, float(a), w))
    return out


# --------------------------------------------------------------------------- records
def model_of(plan: dict) -> dict:
    m = plan["model"]
    return {k: m[k] for k in ("checkpoint", "arch", "clip", "vae", "model_sampling") if m.get(k)}


def setup_of(plan: dict, prompt: dict, seed: int) -> dict:
    s = plan["sampling"]
    return {"route": "txt2img", "seed": int(seed), "sampler": s["sampler"], "scheduler": s["scheduler"],
            "steps": int(s["steps"]), "cfg": float(s["cfg"]), "width": int(s["width"]), "height": int(s["height"]),
            "prompt": prompt["prompt"], "negative": prompt["negative"]}


def contribution(plan: dict, via: str = "script") -> dict:
    c = plan["contributor"]
    return {"by": {"type": "human", "name": c.get("name") or None}, "via": via, "run": plan["run"],
            "agreed": {"license": LICENSE, "at": c.get("agreed_at") or kb.now()}}


def webp_with_workflow(img, workflow: dict | None) -> bytes:
    """A lossless WebP with the ComfyUI workflow embedded the way ComfyUI saves it (EXIF 0x0110 "prompt:"), so
    dropping the picture on ComfyUI opens the graph that made it."""
    from PIL import Image
    buf = io.BytesIO()
    kw = {}
    if workflow:
        exif = Image.Exif()
        exif[0x0110] = "prompt:" + json.dumps(workflow, separators=(",", ":"), ensure_ascii=False)
        kw["exif"] = exif.tobytes()
    img.convert("RGB").save(buf, "WEBP", lossless=True, quality=80, method=4, **kw)
    return buf.getvalue()


def pixel_values(img, base) -> dict:
    import metrics
    import numpy as np
    m = metrics.pixel_metrics(img)
    flags = metrics.kb_flags(m)
    b = base.resize(img.size) if base.size != img.size else base
    mae = float(np.abs(np.asarray(img, np.float32) - np.asarray(b, np.float32)).mean())
    reasons = kb.degenerate_reasons(flags)
    return {"std": round(m["std"], 3), "hf_ratio": round(m["hf_ratio"], 5), "mae_vs_baseline": round(mae, 3),
            "pixel_flags": flags, "degenerate": kb.measure("degenerate", bool(reasons), reasons=reasons)}


# --------------------------------------------------------------------------- ComfyUI
def _render(workflow: dict, timeout: float = 900):
    import comfy_canvas as cc
    from PIL import Image
    r = cc.queue_prompt(workflow, wait=True, timeout=timeout)
    if r.get("errors"):
        e = r["errors"][0]
        raise RunError(f"{e.get('node_type')}: {e.get('exception_message', '').strip()[:400]}")
    imgs = [i for i in r.get("images") or [] if i.get("type") == "output"] or r.get("images") or []
    if not imgs:
        raise RunError("the run produced no picture")
    return Image.open(io.BytesIO(cc.fetch_bytes(imgs[0]["url"]))).convert("RGB")


def render(plan: dict, out: Path, limit: int = 0, via: str = "script", log=print) -> dict:
    errs = check_plan(plan)
    if errs:
        raise RunError("the plan has problems:\n  " + "\n  ".join(errs))
    if plan["contributor"].get("agree_cc0") is not True:
        raise RunError('set "contributor": {"agree_cc0": true} in the plan: the run is released under CC0 1.0')
    plan["contributor"].setdefault("agreed_at", kb.now())
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    bends, model, ph = bends_of(plan), model_of(plan), plan_hash(plan)
    total = len(bends) * len(plan["prompts"]) * len(plan["seeds"])
    n = defaultdict(int)
    t0 = time.monotonic()
    log(f"{plan['run']}: {len(bends)} bends x {len(plan['prompts'])} prompt(s) x {len(plan['seeds'])} seed(s) = "
        f"{total} renders, plus one unbent render per prompt and seed")
    for prompt in plan["prompts"]:
        for seed in plan["seeds"]:
            setup = setup_of(plan, prompt, seed)
            base_rec = kb.make_record(source="sweep", model=model, setup=setup, bends=[], consent={"prompt": True})
            bid = kb.baseline_id(base_rec)
            bpath = out / "baselines" / f"{bid}.webp"
            if not bpath.exists():
                wf = kb.comfy_workflow({**base_rec, "id": bid})
                img = _render(wf)
                kb.save_baseline(out, bid, webp_with_workflow(img, wf))
                n["baselines"] += 1
            base = Image.open(bpath).convert("RGB")
            for region, path, op, amount, window in bends:
                if limit and n["rendered"] >= limit:
                    break
                arg = next(iter(bendjson.OPS[op]))
                start, end = WINDOWS[window]
                lo, hi = step_range(start, end, setup["steps"])
                bend = {"path": path, "op": op, "args": {arg: amount},
                        **({"steps_min": lo, "steps_max": hi} if lo is not None else {})}
                rec = kb.make_record(source="sweep", model=model, setup=setup, bends=[bend], consent={"prompt": True},
                                     license=LICENSE, provenance={"producer": "kb_run.py", "run": plan["run"],
                                                                  "plan": ph, "cell_request": [region, op, amount,
                                                                                               window]})
                if (kb.record_dir(out, rec) / "record.json").exists():
                    n["already"] += 1
                    continue
                rec["contribution"] = contribution(plan, via)
                rec["outputs"]["baseline_id"] = bid
                wf = kb.comfy_workflow(rec, strict=True)
                try:
                    img = _render(wf)
                except RunError as e:
                    n["failed"] += 1
                    with (out / "failures.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"bend": bend, "seed": seed, "prompt": prompt["prompt"],
                                            "error": str(e)}) + "\n")
                    log(f"  failed: {path} {op} {amount} ({e})")
                    continue
                rec["outputs"]["workflow"] = "workflow.json"
                files = {"output.webp": webp_with_workflow(img, wf),
                         "workflow.json": json.dumps(wf, indent=1, ensure_ascii=False).encode("utf-8")}
                kb.save_record(out, rec, files, kb.make_measurements(rec["id"], pixel_values(img, base)))
                n["rendered"] += 1
                if n["rendered"] % 10 == 0:
                    rate = n["rendered"] / max(time.monotonic() - t0, 1e-6)
                    left = total - n["rendered"] - n["already"] - n["failed"]
                    log(f"  {n['rendered']} rendered, {left} to go (~{left / max(rate, 1e-6) / 60:.0f} min)")
    log(json.dumps(dict(n)))
    return dict(n)


def layers(checkpoint: str, max_depth: int = 4, filt: str = "") -> list[dict]:
    """The model's bendable layers, from ComfyUI-Model-Bending's Bendable Layer Catalogue node."""
    import comfy_canvas as cc
    wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
          "2": {"class_type": "Bendable Layer Catalogue", "inputs": {"model": ["1", 0], "filter": filt,
                                                                     "max_depth": max_depth}},
          "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}}}
    r = cc.queue_prompt(wf, wait=True)
    if r.get("errors") or not r.get("texts"):
        raise RunError(f"the catalogue did not run: {r.get('errors')}")
    cat = json.loads(r["texts"][0]["text"])
    return cat.get("layers", cat) if isinstance(cat, dict) else cat


# --------------------------------------------------------------------------- check
def _records(root: Path):
    for f in sorted((root / "records").glob("*/*/*/record.json")):
        yield f.parent, json.loads(f.read_text(encoding="utf-8"))


def check(root: Path) -> dict:
    """Problems that would stop a review, plus the run's coverage. A run passes when problems is empty."""
    problems, cov = [], defaultdict(lambda: defaultdict(set))
    checkpoints, seeds, prompts, regions, n, broken = set(), set(), set(), set(), 0, 0
    if not (root / "records").is_dir():
        return {"ok": False, "problems": [f"{root}: no records/ folder"], "records": 0}
    for d, rec in _records(root):
        n += 1
        where = f"{d.relative_to(root).as_posix()}"
        problems += [f"{where}: {e}" for e in kb.validate(rec, "record")]
        if rec.get("source") != "sweep":
            problems.append(f"{where}: source should be 'sweep'")
        if d.name != rec.get("id") or kb.record_id(rec["model"], rec["setup"], rec["bends"]) != rec.get("id"):
            problems.append(f"{where}: the id does not match the record's facts (folder {d.name})")
        if rec.get("license") != LICENSE:
            problems.append(f"{where}: license must be {LICENSE}")
        if not (rec.get("consent") or {}).get("prompt") or not rec["setup"].get("prompt"):
            problems.append(f"{where}: the prompt must be shared")
        if "negative" not in rec["setup"]:
            problems.append(f"{where}: setup.negative is missing (\"\" if none)")
        if not rec.get("contribution"):
            problems.append(f"{where}: the contribution block is missing")
        if len(rec.get("bends") or []) != 1:
            problems.append(f"{where}: a run renders one bend per picture")
        img = d / (rec.get("outputs") or {}).get("image", "output.webp")
        if not img.exists():
            problems.append(f"{where}: {img.name} is missing")
        if not (d / "workflow.json").exists():
            problems.append(f"{where}: workflow.json is missing")
        bid = (rec.get("outputs") or {}).get("baseline_id")
        if not bid or not any((root / "baselines" / f"{bid}.{e}").exists() for e in ("webp", "png")):
            problems.append(f"{where}: its unbent picture baselines/{bid}.webp is missing")
        elif bid != kb.baseline_id(rec):
            problems.append(f"{where}: baseline_id does not match the record's setup")
        m = kb.load_measurements(d).get("values", {})
        broken += bool((m.get("degenerate") or {}).get("value"))
        fam = rec["model"].get("family")
        checkpoints.add(rec["model"].get("checkpoint"))
        seeds.add(rec["setup"].get("seed"))
        prompts.add(rec["setup"].get("prompt"))
        for b in rec.get("bends") or []:
            g = b.get("group") or kb.unet_group(b["path"], fam)
            regions.add(g)
            amt = kb.main_arg(b["op"], b.get("args") or {})
            cov[b["op"]][g].add(amt)
    if not n:
        problems.append("no records")
    if len(checkpoints) > 1:
        problems.append(f"one model per run; found {len(checkpoints)}: {', '.join(sorted(map(str, checkpoints)))}")
    fams = {f.parent.name for f in (root / "records").glob("*/sweep")}
    if fams - set(UNET_FAMILIES):
        problems.append(f"model family {', '.join(sorted(fams))}: full runs are for U-Net models for now")
    miss = [r for r in REGIONS if r not in regions]
    if n and miss:
        problems.append(f"the run does not reach every region of the U-Net; missing {', '.join(miss)}")
    full_ops = [op for op, by in cov.items() if len({a for s in by.values() for a in s}) >= MIN_AMOUNTS]
    if n and len(full_ops) < MIN_OPS:
        problems.append(f"at least {MIN_OPS} ops with {MIN_AMOUNTS} amounts each; have {len(full_ops)} "
                        f"({', '.join(full_ops) or 'none'})")
    table = {op: {r: sorted(by.get(r, set())) for r in REGIONS} for op, by in sorted(cov.items())}
    return {"ok": not problems, "problems": problems, "records": n, "checkpoint": sorted(map(str, checkpoints)),
            "seeds": sorted(seeds), "prompts": len(prompts), "broken": broken, "coverage": table}


def coverage_text(rep: dict) -> str:
    lines = [f"{rep['records']} records, model {', '.join(rep.get('checkpoint') or ['?'])}, "
             f"{len(rep.get('seeds') or [])} seed(s), {rep.get('prompts', 0)} prompt(s), {rep.get('broken', 0)} broken",
             "", "op \\ region".ljust(14) + "".join(r.ljust(9) for r in REGIONS)]
    for op, by in (rep.get("coverage") or {}).items():
        lines.append(op.ljust(14) + "".join(str(len(by[r]) or "-").ljust(9) for r in REGIONS))
    lines.append("(cells: how many amounts of that op were rendered in that region)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- descriptions (optional)
def sheets(root: Path, per_sheet: int = 9) -> dict:
    """Contact sheets (unbent | bent) of one typical render per cell, for an agent to describe with the knowledge
    base's prompt (kb_local.py describe-prompt). Writes sheets/sheet_NNN.jpg and sheets/index.json (label -> record)."""
    from PIL import Image, ImageDraw
    cells = defaultdict(list)
    for d, rec in _records(root):
        m = kb.load_measurements(d).get("values", {})
        if (m.get("degenerate") or {}).get("value"):
            continue
        f = kb.cell_fields(rec)
        key = (kb.cell_key(f) if f else rec["id"], rec["setup"].get("seed"), rec["setup"].get("prompt"))
        cells[key].append(((m.get("mae_vs_baseline") or {}).get("value") or 0.0, d, rec))
    picks = []
    for rs in cells.values():
        rs.sort(key=lambda x: x[0])
        picks.append(rs[len(rs) // 2])
    picks.sort(key=lambda x: x[2]["id"])
    out = root / "sheets"
    out.mkdir(exist_ok=True)
    index, T = {}, 256
    labels = [chr(65 + i) for i in range(per_sheet)]
    for s in range(0, len(picks), per_sheet):
        chunk = picks[s:s + per_sheet]
        name = f"sheet_{s // per_sheet + 1:03d}"
        cols = 3
        rows = (len(chunk) + cols - 1) // cols
        im = Image.new("RGB", (cols * (2 * T + 24), rows * (T + 26)), "white")
        dr = ImageDraw.Draw(im)
        for i, (_, d, rec) in enumerate(chunk):
            x, y = (i % cols) * (2 * T + 24), (i // cols) * (T + 26)
            dr.text((x + 4, y + 4), f"{labels[i]}   before | after", fill="black")
            bid = rec["outputs"]["baseline_id"]
            bp = next(p for p in (root / "baselines" / f"{bid}.webp", root / "baselines" / f"{bid}.png") if p.exists())
            im.paste(Image.open(bp).convert("RGB").resize((T, T)), (x, y + 22))
            im.paste(Image.open(d / rec["outputs"]["image"]).convert("RGB").resize((T, T)), (x + T + 4, y + 22))
            index[f"{name}:{labels[i]}"] = rec["id"]
        im.save(out / f"{name}.jpg", quality=88)
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return {"sheets": len(range(0, len(picks), per_sheet)), "pairs": len(picks), "folder": str(out),
            "prompt_version": kb.CAPTION_PROMPT_VERSION}


def add_descriptions(root: Path, path: Path, model: str) -> dict:
    """Save an agent's descriptions ({"sheet_001:A": {caption, change, keywords, effect_tags}, ...} or keyed by
    record id) as AI interpretations naming the model and the knowledge base's prompt version."""
    if not model:
        raise RunError("--model: the exact id of the model that wrote the descriptions")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    index = json.loads((root / "sheets" / "index.json").read_text(encoding="utf-8")) \
        if (root / "sheets" / "index.json").exists() else {}
    dirs = {rec["id"]: d for d, rec in _records(root)}
    vocab = kb.load_vocab()["tags"]
    author = kb.ai(model, kb.CAPTION_PROMPT_VERSION, "kb_run.py")
    basis = "baseline|output pair on a contact sheet; recipe not shown"
    n, unknown = 0, []
    for key, desc in data.items():
        rid = index.get(key, key)
        d = dirs.get(rid)
        if d is None:
            unknown.append(key)
            continue
        entries = []
        for kind in ("caption", "change", "keywords", "effect_tags"):
            v = desc.get(kind)
            if kind == "effect_tags":
                v = [t for t in v or [] if t in vocab]
            if kind == "keywords":
                v = [str(k).strip().lower() for k in v or [] if str(k).strip()][:8]
            if v not in (None, "", []):
                entries.append(kb.make_interpretation({"record": rid}, kind, v, author, basis))
        n += kb.add_interpretations(d, entries) and 1
    return {"described": n, "unknown": unknown}


# --------------------------------------------------------------------------- submit
def submit(root: Path, dry_run: bool = False, repo: str = kb.COMMUNITY_DATASET, batch: int = 1000) -> dict:
    """Open a pull request on the dataset with the run's records/ and baselines/, using your own Hugging Face login
    (`hf auth login`). Large runs go in several commits to the same PR."""
    rep = check(root)
    if not rep["ok"]:
        raise RunError("fix these first (kb_run.py check):\n  " + "\n  ".join(rep["problems"][:20]))
    files = sorted(p for sub in ("records", "baselines") for p in (root / sub).rglob("*") if p.is_file())
    rels = [p.relative_to(root).as_posix() for p in files]
    run = json.loads((root / "run.json").read_text(encoding="utf-8")) if (root / "run.json").exists() else {}
    title = f"Full run: {run.get('run') or root.name} ({rep['records']} renders)"
    body = (f"A full bending run made with kb_run.py.\n\n```\n{coverage_text(rep)}\n```\n\n"
            "Released under CC0 1.0 by the contributor.")
    if dry_run:
        return {"would_upload": len(rels), "files": rels[:10], "title": title, "repo": repo}
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        raise RunError("submit needs huggingface_hub: pip install huggingface_hub, then hf auth login") from None
    api = HfApi()
    who = api.whoami()["name"]
    pr_ref, pr_url = None, None
    for i in range(0, len(files), batch):
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for r, p in
               zip(rels[i:i + batch], files[i:i + batch])]
        info = api.create_commit(repo, repo_type="dataset", operations=ops,
                                 commit_message=title if pr_ref is None else f"{title}: part {i // batch + 1}",
                                 commit_description=body if pr_ref is None else None,
                                 create_pr=pr_ref is None, revision=pr_ref)
        if pr_ref is None:
            pr_ref, pr_url = info.pr_revision, info.pr_url
    return {"pull_request": pr_url, "files": len(rels), "by": who}


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init", help="write a plan to edit")
    p.add_argument("--arch", required=True, help="sd15, sd14, sd21 or sdxl")
    p.add_argument("--checkpoint", required=True, help="the checkpoint's file name in ComfyUI")
    p.add_argument("--name", default=None, help="your name as shown with the run (leave out: 'not named')")
    p.add_argument("--clip", default=None, help="separate text encoder file, if the checkpoint has none")
    p.add_argument("--vae", default=None, help="separate VAE file")
    p.add_argument("--model-sampling", default=None, help="ModelSamplingDiscrete mode, e.g. lcm")
    p.add_argument("--out", type=Path, default=Path("run.json"))
    p = sp.add_parser("layers", help="list a model's bendable layers (ComfyUI)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--filter", default="")
    p.add_argument("--out", type=Path, default=None)
    p = sp.add_parser("render", help="render the plan on ComfyUI (resumable)")
    p.add_argument("plan", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="stop after this many new renders (a test)")
    p.add_argument("--via", choices=("script", "agent"), default="script", help="agent: an AI agent drove the run")
    for name in ("check", "measure", "sheets"):
        p = sp.add_parser(name)
        p.add_argument("dir", type=Path)
        if name == "measure":
            p.add_argument("--device", default=None)
    p = sp.add_parser("add-descriptions")
    p.add_argument("dir", type=Path)
    p.add_argument("descriptions", type=Path)
    p.add_argument("--model", required=True)
    p = sp.add_parser("submit", help="open a pull request on the dataset")
    p.add_argument("dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "init":
            plan = init_plan(a.arch, a.checkpoint, a.name, a.clip, a.vae, a.model_sampling)
            a.out.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
            n = len(bends_of(plan))
            print(f"wrote {a.out}: {n} renders per prompt and seed as planned. Edit it, set "
                  f"\"agree_cc0\": true, then: kb_run.py render {a.out} --out {plan['run']}")
        elif a.cmd == "layers":
            ls = layers(a.checkpoint, a.max_depth, a.filter)
            text = json.dumps(ls, indent=1)
            (a.out.write_text(text, encoding="utf-8") if a.out else print(text))
        elif a.cmd == "render":
            plan = json.loads(a.plan.read_text(encoding="utf-8"))
            render(plan, a.out, a.limit, a.via)
            rep = check(a.out)
            print(coverage_text(rep))
            print("ready to submit" if rep["ok"] else "not ready yet:\n  " + "\n  ".join(rep["problems"][:10]))
        elif a.cmd == "check":
            rep = check(a.dir)
            print(coverage_text(rep))
            print("\nOK: ready to submit" if rep["ok"] else "\nproblems:\n  " + "\n  ".join(rep["problems"][:40]))
            sys.exit(0 if rep["ok"] else 1)
        elif a.cmd == "measure":
            import kb_measure
            print(json.dumps(kb_measure.measure_dir(a.dir, device=a.device)))
        elif a.cmd == "sheets":
            r = sheets(a.dir)
            print(json.dumps(r))
            print("Describe each pair with the knowledge base's prompt (python kb_local.py describe-prompt), as "
                  "{\"sheet_001:A\": {\"caption\", \"change\", \"keywords\", \"effect_tags\"}, ...}, then "
                  "kb_run.py add-descriptions DIR descriptions.json --model <exact model id>.")
        elif a.cmd == "add-descriptions":
            print(json.dumps(add_descriptions(a.dir, a.descriptions, a.model)))
        elif a.cmd == "submit":
            print(json.dumps(submit(a.dir, a.dry_run), indent=1))
    except RunError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
