#!/usr/bin/env python3
"""A surprise round: a few random bends, one per part of the model, drawn inside the measured safe ranges
(data/safe_ranges_<arch>.json), plus an optional wild card that goes past them on purpose. Standard library only.

  python surprise.py --arch sd15 [-n 3] [--no-wild] [--seed 7] [--avoid "input_blocks.4|rotate"]

Each candidate is one bend, ready for Apply Bends from JSON, with the `clamp` to run it with and a plain line saying
what was bent. The same seed gives the same draw, so a round can be repeated. Tables exist for SD1.5-type and SDXL
models; for other models there is nothing to draw from yet.
"""

from __future__ import annotations

import fnmatch
import json
import random
import sys
from pathlib import Path

import kb

DATA = Path(__file__).resolve().parent.parent / "data"
TABLES = {"sd1": "sd15", "sdxl": "sdxl"}  # model family -> safe-range table
BLOCKS = {"sd1": {"input_blocks": 12, "middle_block": 3, "output_blocks": 12},
          "sdxl": {"input_blocks": 9, "middle_block": 3, "output_blocks": 9}}
# op -> (argument, the value that leaves the layer unchanged)
NEUTRAL = {"multiply": ("scalar", 1.0), "add_scalar": ("scalar", 0.0), "add_noise": ("noise_std", 0.0),
           "rotate": ("angle_degrees", 0.0), "scale": ("scale_factor", 1.0)}
REGIONS = {"early": ("in.hi", "in.mid", "in.lo"), "core": ("mid",), "late": ("out.lo", "out.mid", "out.hi")}
PLAIN_GROUP = {"in.hi": "the early stages (fine detail)", "in.mid": "the early stages (layout)",
               "in.lo": "the early stages (overall composition)", "mid": "the model's core (what the scene is)",
               "out.lo": "the late stages (rebuilding the composition)",
               "out.mid": "the late stages (style and handling)", "out.hi": "the late stages (texture and fine detail)"}
# (t window, plain words, weight)
WINDOWS = ((None, "through the whole painting", 2),
           ([1.0, 0.7], "early in the painting, while the layout forms", 1),
           ([0.7, 0.2], "in the middle of the painting, while the style forms", 1))
# A range that barely leaves "no change" (rotate 0-5°) is safe but shows nothing: skip ops whose reach at a path is
# below this share of the op's general reach.
MIN_REACH = 0.25
# Wild-card ops outside the safe-range tables (they need the 4-D output of a whole block).
OFF_TABLE = {
    "sobel": (lambda r: {"normalized": True}, "reduced to its edges"),
    "gradient": (lambda r: {"kernel_size": r.choice((3, 5))}, "reduced to its outlines"),
    "dilation": (lambda r: {"kernel_size": r.choice((3, 5, 7))}, "its strong areas swollen"),
    "erosion": (lambda r: {"kernel_size": r.choice((3, 5, 7))}, "its strong areas eaten away"),
    "fourier": (lambda r: {"cutoff_freq": round(r.uniform(1, 4), 1), "amp_factor": round(r.uniform(2, 5), 1)},
                "its fine patterns amplified"),
}


def table(arch: str) -> tuple[str, dict]:
    """(model family, safe ranges) for an architecture, or a ValueError saying there is no table."""
    family = kb.family_of(arch)
    name = TABLES.get(family)
    if not name:
        raise ValueError(f"no safe-range table for {arch or 'this model'} (tables: sd15, sdxl), so there is nothing "
                         f"to draw a surprise from. Pick two or three bends near 'no change' by hand, say that they "
                         f"are your picks, and widen from what you see")
    return family, json.loads((DATA / f"safe_ranges_{name}.json").read_text(encoding="utf-8"))


def _range(ranges: dict, path: str, op: str) -> tuple[list[float], bool]:
    """([lo, hi], measured at this path?) the way clamp = safe resolves it: the op's general range, then every
    matching path pattern, most specific last."""
    arg = NEUTRAL[op][0]
    out, measured = (ranges.get(op) or {}).get(arg), False
    for pattern in sorted(ranges.get("paths") or {}, key=len):
        r = (ranges["paths"][pattern].get(op) or {}).get(arg)
        if r and fnmatch.fnmatchcase(path, pattern):
            out, measured = r, True
    return out, measured


def _reach(rng: list[float], neutral: float) -> float:
    return max(abs(rng[0] - neutral), abs(rng[1] - neutral))


def options(arch: str) -> dict[str, list[dict]]:
    """Every (path, op) a surprise may draw, by region: {region: [{path, group, op, arg, range, measured}]}."""
    family, ranges = table(arch)
    out: dict[str, list[dict]] = {r: [] for r in REGIONS}
    for stack, count in BLOCKS[family].items():
        for i in range(count):
            path = f"{stack}.{i}"
            group = kb.unet_group(path, family)
            region = next((r for r, gs in REGIONS.items() if group in gs), None)
            if not region or path == "input_blocks.0":  # the first convolution: every later stage depends on it
                continue
            for op, (arg, neutral) in NEUTRAL.items():
                rng, measured = _range(ranges, path, op)
                general = (ranges.get(op) or {}).get(arg)
                if not rng or not general or _reach(rng, neutral) < MIN_REACH * _reach(general, neutral):
                    continue
                out[region].append({"path": path, "group": group, "op": op, "arg": arg, "range": rng,
                                    "measured": measured})
    return out


def _round(op: str, v: float) -> float:
    return float(round(v)) if op == "rotate" else round(v, 2)


def _sides(rng: list[float], neutral: float) -> list[tuple[float, float]]:
    """(near, far) on each side of neutral that the range reaches: values run from near (closest to no change) to
    far (the limit)."""
    lo, hi = rng
    sides = []
    if hi > neutral:
        sides.append((max(lo, neutral), hi))
    if lo < neutral:
        sides.append((min(hi, neutral), lo))
    return sides


def _amount(r: random.Random, o: dict, wild: bool) -> float | None:
    """An amount away from "no change": the outer part of the safe range, or (wild) past its limit."""
    neutral = NEUTRAL[o["op"]][1]
    sides = _sides(o["range"], neutral)
    near, far = r.choices(sides, weights=[abs(f - n) for n, f in sides])[0]
    if not wild:
        return _round(o["op"], near + r.uniform(0.4, 1.0) * (far - near))
    v = far + r.uniform(0.3, 1.0) * (far - neutral)
    if o["op"] == "rotate":
        if far >= 180:
            return None  # a half turn is already the limit of turning
        v = min(v, 180.0)
    return _round(o["op"], v)


def _phrase(op: str, args: dict) -> str:
    v = args.get(NEUTRAL[op][0]) if op in NEUTRAL else None
    if op == "rotate":
        return f"turned by {v:g}°"
    if op == "scale":
        return f"zoomed {'in' if v > 1 else 'out'} (×{v:g})"
    if op == "multiply":
        return f"{'strengthened' if v > 1 else 'weakened' if v >= 0 else 'inverted'} (×{v:g})"
    if op == "add_scalar":
        return f"shifted {'up' if v > 0 else 'down'} by {abs(v):g}"
    if op == "add_noise":
        return f"shaken up (noise {v:g})"
    return OFF_TABLE[op][1]


def draw(arch: str, n: int = 3, wild: bool = True, seed: int | None = None, avoid: list[str] | None = None) -> dict:
    """n single-bend candidates, one per region of the model where possible, with different ops; with wild, the last
    one goes past the safe range (or uses an op outside the table) and is labelled. avoid: "path|op" keys from
    earlier draws, so a new round brings new bends."""
    if not 1 <= n <= 6:
        raise ValueError("n must be between 1 and 6")
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    r = random.Random(seed)
    pool = options(arch)
    skip = set(avoid or [])
    regions = list(REGIONS)
    r.shuffle(regions)
    used_ops: set[str] = set()
    out = []
    for i in range(n):
        is_wild = wild and i == n - 1
        region = regions[i % len(regions)]
        cands = [o for o in pool[region] if f"{o['path']}|{o['op']}" not in skip] or pool[region]
        fresh = [o for o in cands if o["op"] not in used_ops] or cands
        bend = key = None
        if is_wild and r.random() < 0.5:
            o = r.choice(fresh)
            op = r.choice(sorted(OFF_TABLE))
            bend = {"path": o["path"], "module_type": op, "module_args": OFF_TABLE[op][0](r)}
            key, group = f"{o['path']}|{op}", o["group"]
        while bend is None:
            o = r.choices(fresh, weights=[3 if c["measured"] else 1 for c in fresh])[0]
            v = _amount(r, o, is_wild)
            if v is None:  # nothing past this limit: draw another
                fresh = [c for c in fresh if c is not o] or [c for c in cands if c["op"] != "rotate"]
                continue
            args = {o["arg"]: v, **({"seed": r.randrange(10_000)} if o["op"] == "add_noise" else {})}
            bend = {"path": o["path"], "module_type": o["op"], "module_args": args}
            key, group = f"{o['path']}|{o['op']}", o["group"]
        used_ops.add(bend["module_type"])
        skip.add(key)
        t, when, _ = r.choices(WINDOWS, weights=[w for *_, w in WINDOWS])[0]
        letter = "ABCDEF"[i]
        bend["label"] = f"{letter} wild" if is_wild else letter
        if t:
            bend["t"] = t
        if is_wild:
            bend["guard"] = {"nan": "zero"}
        what = f"{PLAIN_GROUP[group]}, {_phrase(bend['module_type'], bend['module_args'])}, {when}"
        out.append({"label": letter, "wild": is_wild, "region": region, "group": group, "key": key, "bend": bend,
                    "bends_json": json.dumps({"bends": [bend]}),
                    # a wild card past the safe range must not be pulled back into it
                    "clamp": "none" if is_wild else "safe",
                    "what_was_bent": what[0].upper() + what[1:] + (". The wild card: past the safe limits on purpose"
                                                                  if is_wild else "")})
    return {"arch": arch, "seed": seed, "safe_ranges": f"safe_ranges_{TABLES[kb.family_of(arch)]}.json",
            "candidates": out,
            "note": "Render one unbent picture, then bend a copy of that run once per candidate (bend_run, or "
                    "comfy_canvas.py bend-run) with the candidate's bends_json and clamp: same model, prompt and "
                    "seed. Show the wild card labelled as such, even if it fell apart. Captions say what is seen; "
                    "say what each one was (what_was_bent) once the artist has picked or asks."}


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", required=True, help="sd15, sd14 or sdxl")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--no-wild", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--avoid", default="", help='comma-separated "path|op" keys from earlier draws')
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        r = draw(a.arch, a.n, not a.no_wild, a.seed, [k for k in a.avoid.split(",") if k])
    except ValueError as e:
        raise SystemExit(str(e))
    print(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
