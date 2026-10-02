#!/usr/bin/env python3
"""Cheap image and video diagnostics and contact sheets for bend exploration. Needs numpy and Pillow.

  compare BASE CAND [CAND ...] [--json]      effect size vs the unbent baseline + degeneracy flags
  sheet OUT.png IMG [IMG ...] --labels "A|B|C" [--cols 3] [--thumb 320] [--title TEXT] [--frames 6]
  diff BASE CAND -o OUT.png                  where did the bend act? |delta| heat overlaid on the baseline

Images may be local paths or ComfyUI /api/view URLs (as printed by comfy_canvas.py last-run). Videos work too:
animated WEBP / GIF (what video workflows save with SaveAnimatedWEBP) need nothing extra; .mp4 / .webm need ffmpeg.
- compare on videos matches frames by position in the clip (8 evenly spaced frames) and adds motion (mean change
  between consecutive frames, 0-255 scale) and motion_ratio (candidate / baseline). A still baseline against a video
  candidate (image-to-video) measures how far each frame drifts from the starting picture.
- sheet puts each video on its own row as a filmstrip of --frames evenly spaced frames.
- diff averages |delta| over the matched frames and overlays it on the baseline's middle frame.
Run with any Python that has numpy + Pillow, e.g.  uv run --with numpy --with pillow python metrics.py ...
or ComfyUI's own python_embeded/python.exe.

Flags (thresholds calibrated on SD1.5/SDXL bend sweeps):
  noop     MAE < 0.5 vs baseline: nothing changed; read the bend report (skipped path? window outside the
           executed steps?) before concluding 'no effect'
  subtle   MAE < 4:   fired, but barely visible at this resolution
  blob     hf_ratio < 0.006 and std > 50: degenerate smooth blob field (rotate/scale on hi-res groups)
  noise    hf_ratio > 0.35: activation blow-out into colour noise
  flat     std < 6 or entropy < 3: collapsed to a flat field
  clipped  > 40 % of pixels at 0 or 255 in some channel: saturation / NaN-like blow-out
Video only (provisional thresholds, not yet calibrated on real video models):
  frozen   motion < 0.5 while the baseline moves (motion > 2): the bend stopped the motion
  flicker  motion_ratio > 2.5: frames jump much more than in the baseline (temporal blow-out)
  static   the baseline itself barely moves (motion < 0.5): judge motion-related bends on another seed or prompt
"""

from __future__ import annotations

import argparse
import functools
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import media

VIDEO_SAMPLES = 8
KEEP_FRAMES, KEEP_SIDE = 32, 512  # what is kept of a video for measuring (the MCP server is long-running)


@functools.lru_cache(maxsize=8)
def _clip(src: str) -> tuple[tuple[Image.Image, ...], float]:
    """(frames, motion): a still as is; a video as at most 32 evenly spaced frames of at most 512 px, with its
    motion measured on every frame."""
    seq = media.frames(src)[0]
    if len(seq) == 1:
        return (seq[0],), 0.0
    mot = _motion(seq)
    kept = []
    for f in media.sample(seq, KEEP_FRAMES):
        f = f.copy()
        f.thumbnail((KEEP_SIDE, KEEP_SIDE))
        kept.append(f)
    return tuple(kept), mot


def frames(src: str) -> tuple[Image.Image, ...]:
    """The frames of a source (one for a still)."""
    return _clip(src)[0]


def load(src: str) -> Image.Image:
    """A still, or the middle frame of a video."""
    f = frames(src)
    return f[len(f) // 2]


def _matched(base: tuple, cand: tuple, n: int = VIDEO_SAMPLES) -> list[tuple[Image.Image, Image.Image]]:
    """Pairs of frames at the same position in each clip (a still pairs with every frame of the other)."""
    k = max(min(n, max(len(base), len(cand))), 1)
    pos = [i / (k - 1) if k > 1 else 0.5 for i in range(k)]
    return [(base[round(p * (len(base) - 1))], cand[round(p * (len(cand) - 1))]) for p in pos]


def motion(src: str) -> float:
    """Mean absolute change between consecutive frames (0-255 scale, at 128 px); 0 for a still."""
    return _clip(src)[1]


def _motion(seq: list) -> float:
    if len(seq) < 2:
        return 0.0
    a = [np.asarray(f.resize((128, 128), Image.BILINEAR), np.float32) for f in seq]
    return float(np.mean([np.abs(x - y).mean() for x, y in zip(a, a[1:])]))


def pixel_metrics(img: Image.Image) -> dict[str, float]:
    small = img.resize((256, 256), Image.BILINEAR)
    rgb = np.asarray(small, np.uint8)
    lum = np.asarray(small.convert("L"), np.float32)
    hist = np.bincount(lum.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    clipped = float(((rgb == 0) | (rgb == 255)).any(axis=2).mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(lum - lum.mean()))) ** 2
    yy, xx = np.mgrid[-128:128, -128:128] / 256.0
    hf = float(power[np.sqrt(xx**2 + yy**2) > 0.25].sum() / max(power.sum(), 1e-9))
    return {"std": float(lum.std()), "entropy": entropy, "mean": float(lum.mean()), "clipped": clipped,
            "hf_ratio": hf}


def flags(m: dict, mae: float | None) -> list[str]:
    out = []
    if mae is not None:
        if mae < 0.5:
            out.append("noop")
        elif mae < 4:
            out.append("subtle")
    if m["hf_ratio"] < 0.006 and m["std"] > 50:
        out.append("blob")
    if m["hf_ratio"] > 0.35:
        out.append("noise")
    if m["std"] < 6 or m["entropy"] < 3:
        out.append("flat")
    if m["clipped"] > 0.40:
        out.append("clipped")
    return out


def _mae(a: Image.Image, b: Image.Image) -> float:
    b = b.resize(a.size, Image.BILINEAR) if b.size != a.size else b
    return float(np.abs(np.asarray(b, np.float32) - np.asarray(a, np.float32)).mean())


def compare(base_src: str, cands: list[str]) -> list[dict]:
    if len(frames(base_src)) > 1 or any(len(frames(c)) > 1 for c in cands):
        return _compare_video(base_src, cands)
    base = load(base_src)
    b = np.asarray(base, np.float32)
    rows = [{"image": base_src, "role": "baseline", **_r(pixel_metrics(base)), "flags": flags(pixel_metrics(base), None)}]
    for c in cands:
        img = load(c)
        if img.size != base.size:
            img_cmp = img.resize(base.size, Image.BILINEAR)
        else:
            img_cmp = img
        mae = float(np.abs(np.asarray(img_cmp, np.float32) - b).mean())
        m = pixel_metrics(img)
        rows.append({"image": c, "role": "candidate", "mae": round(mae, 2), **_r(m), "flags": flags(m, mae)})
    return rows


def _compare_video(base_src: str, cands: list[str]) -> list[dict]:
    bseq = frames(base_src)
    bm, bpm = motion(base_src), pixel_metrics(load(base_src))
    base_row = {"image": base_src, "role": "baseline", "frames": len(bseq), "motion": round(bm, 2), **_r(bpm),
                "flags": flags(bpm, None)}
    if len(bseq) > 1 and bm < 0.5:
        base_row["flags"].append("static")
    rows = [base_row]
    for c in cands:
        cseq = frames(c)
        per = [_mae(b, x) for b, x in _matched(bseq, cseq)]
        mae = float(np.mean(per))
        m = pixel_metrics(load(c))
        row = {"image": c, "role": "candidate", "frames": len(cseq), "mae": round(mae, 2),
               "mae_by_frame": [round(v, 1) for v in per], **_r(m), "flags": flags(m, mae)}
        if len(cseq) > 1:
            cm = motion(c)
            row["motion"] = round(cm, 2)
            if len(bseq) > 1:
                row["motion_ratio"] = round(cm / max(bm, 1e-6), 2)
                if cm < 0.5 and bm > 2:
                    row["flags"].append("frozen")
                if cm / max(bm, 1e-6) > 2.5:
                    row["flags"].append("flicker")
        rows.append(row)
    return rows


def _r(m: dict) -> dict:
    return {k: round(v, 4 if k == "hf_ratio" else 2) for k, v in m.items()}


def sheet(out: str, srcs: list[str], labels: list[str], cols: int, thumb: int, title: str | None,
          n_frames: int = 6):
    """A labelled contact sheet. If any source is a video, each source gets its own row: a filmstrip of n_frames
    evenly spaced frames (a still fills the first cell)."""
    seqs = [frames(s) for s in srcs]
    if any(len(q) > 1 for q in seqs):
        return _film_sheet(out, seqs, labels, min(thumb, 240), title, n_frames)
    imgs = [q[0] for q in seqs]
    rows = (len(imgs) + cols - 1) // cols
    cap, head = 44, (48 if title else 0)
    W, H = cols * (thumb + 12) + 12, head + rows * (thumb + cap + 12) + 12
    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(canvas)
    font = _font(18)
    if title:
        d.text((14, 12), title, fill=(235, 235, 235), font=_font(22))
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        x, y = 12 + c * (thumb + 12), head + 12 + r * (thumb + cap + 12)
        t = img.copy()
        t.thumbnail((thumb, thumb))
        canvas.paste(t, (x + (thumb - t.width) // 2, y + (thumb - t.height) // 2))
        label = labels[i] if i < len(labels) else chr(65 + i)
        d.text((x, y + thumb + 8), label[:48], fill=(230, 230, 230), font=font)
    canvas.save(out)
    return f"wrote {out} ({len(imgs)} images, {cols} columns)"


def _film_sheet(out: str, seqs: list[tuple], labels: list[str], thumb: int, title: str | None, n: int):
    n = max(1, n)
    cap, head, gap = 30, (48 if title else 0), 8
    W = 12 + n * (thumb + gap) + 4
    H = head + 12 + len(seqs) * (thumb + cap + 12)
    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((14, 12), title, fill=(235, 235, 235), font=_font(22))
    font = _font(18)
    for r, seq in enumerate(seqs):
        y = head + 12 + r * (thumb + cap + 12)
        label = labels[r] if r < len(labels) else chr(65 + r)
        d.text((12, y), (label if len(seq) == 1 else f"{label}   ({len(seq)} frames)")[:80], fill=(230, 230, 230),
               font=font)
        for c, f in enumerate(media.sample(list(seq), n)):
            x = 12 + c * (thumb + gap)
            t = f.copy()
            t.thumbnail((thumb, thumb))
            canvas.paste(t, (x + (thumb - t.width) // 2, y + cap + (thumb - t.height) // 2))
    canvas.save(out)
    return f"wrote {out} ({len(seqs)} rows of up to {n} frames)"


def diff(base_src: str, cand_src: str, out: str):
    base = load(base_src)
    b = np.asarray(base, np.float32)
    deltas = []
    for bf, cf in _matched(frames(base_src), frames(cand_src)):
        bf = bf.resize(base.size, Image.BILINEAR) if bf.size != base.size else bf
        cf = cf.resize(base.size, Image.BILINEAR) if cf.size != base.size else cf
        deltas.append(np.abs(np.asarray(cf, np.float32) - np.asarray(bf, np.float32)).mean(axis=2))
    d = np.mean(deltas, axis=0)
    scale = max(float(np.percentile(d, 99)), 1e-6)
    heat = np.clip(d / scale, 0, 1)[..., None]
    red = np.zeros_like(b); red[..., 0] = 255; red[..., 1] = 60
    img = (b * 0.35 * (1 - heat) + red * heat).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(out)
    q = [float(d[: d.shape[0] // 2].mean()), float(d[d.shape[0] // 2:].mean())]
    return {"mean_abs_delta": round(float(d.mean()), 2), "p99": round(scale, 1),
            "top_half": round(q[0], 1), "bottom_half": round(q[1], 1)}


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare"); p.add_argument("base"); p.add_argument("cands", nargs="+")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("sheet"); p.add_argument("out"); p.add_argument("images", nargs="+")
    p.add_argument("--labels", default=""); p.add_argument("--cols", type=int, default=3)
    p.add_argument("--thumb", type=int, default=320); p.add_argument("--title")
    p.add_argument("--frames", type=int, default=6, help="frames per video in the filmstrip rows")
    p = sub.add_parser("diff"); p.add_argument("base"); p.add_argument("cand"); p.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        rows = compare(a.base, a.cands)
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                mae = f"MAE {r['mae']:6.2f}" if "mae" in r else "baseline  "
                mot = f"  motion {r['motion']:5.2f}" if "motion" in r else ""
                print(f"{mae}{mot}  std {r['std']:6.2f}  hf {r['hf_ratio']:.4f}  clip {r['clipped']:.2f}  "
                      f"{','.join(r['flags']) or 'ok':<14} {r['image']}")
    elif a.cmd == "diff":
        print(f"wrote {a.out}  {diff(a.base, a.cand, a.out)}")
    else:
        print(sheet(a.out, a.images, [x.strip() for x in a.labels.split("|")] if a.labels else [], a.cols,
                    a.thumb, a.title, a.frames))


if __name__ == "__main__":
    sys.exit(main())
