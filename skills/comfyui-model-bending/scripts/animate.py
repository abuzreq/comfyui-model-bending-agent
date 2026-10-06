#!/usr/bin/env python3
"""Animate bends: render a workflow frame by frame while bend amounts (or any numeric input) move in small
increments, then encode a looping video. Same seed, prompt and sampling in every frame, so only the bend moves.

Source (one of):
  --api FILE.json          an API-format workflow (comfy_canvas.py build --api …)
  --png IMAGE.png          a ComfyUI output image (its embedded prompt)
  --prompt-id ID           a run from ComfyUI's history
  --last-run               the newest finished run, e.g. one the user just made on the canvas

Tracks (what moves; repeatable). Selector@name=FROM:TO. FROM defaults to the neutral value, TO to the current value:
  --bend  "recompose@angle_degrees=0:180"   a JSON bend in Apply Bends from JSON, by its label, its path
  --bend  "A bends#0@blend=0:1"             (e.g. middle_block.1), or NODE#index (node id or title). name is a
                                            module_args key, or `blend` (0 = unbent … 1 = full; works for any op)
  --input "Bends@a=0:180"                   any numeric input of any node, by node id or title: the a–d slider
  --input "12@angle_degrees=0:90"           placeholders, a Rotate Module's angle, ApplySteeringVector's strength…
With no tracks, every JSON bend moves from its neutral value to its current amount (or blend 0 → 1 for ops with no
neutral: threshold, sobel, gradient, subset).

Timing:
  --increment 2            step of the first moving track in its own units (2 → 0, 2, 4, …); sets the frame count
  --frames 24              or a fixed number of rendered frames (default 24; at most --max-frames, 240)
  --easing ease|linear     ease = slow at both ends (default)
  --loop boomerang|restart|none   there and back (default), jump back to the start, or play once
  --hold 0  --repeats 2  --fps 12  --reverse
Output:
  (default)                saved into ComfyUI's output folder, inside agent_bending/, next to the pictures ComfyUI
                           saves; --format mp4 (needs ffmpeg; falls back to gif) | gif | webp, --name for the file
  --out anim.mp4           or a file of your own: .mp4, .gif or .webp. The frames and anim.json (tracks, values per
                           frame, prompt ids) go in a folder next to it (by default in the skill's work folder).
  --output-node NODE       which image output to film, when the workflow has several downstream of the tracks
  --dry-run                show the plan (frames, values, duration) and render nothing

  python animate.py --last-run --bend "recompose@angle_degrees=0:180" --increment 5 --name recompose
Environment: COMFYUI_URL (default http://127.0.0.1:8188), COMFY_BENDING_WORKDIR (see comfy_canvas.work_dir).
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_canvas as cc  # noqa: E402

BEND_NODE = "ApplyBendsFromJSON"
IMAGE_OUTPUTS = {"SaveImage", "PreviewImage"}
# 240 frames = 20 s at 12 fps, one render each; for longer ranges raise the increment instead
MAX_FRAMES = 240
# the value of each op argument that leaves the layer unchanged; ops missing here animate `blend` instead
NEUTRAL = {("multiply", "scalar"): 1.0, ("add_scalar", "scalar"): 0.0, ("add_noise", "noise_std"): 0.0,
           ("rotate", "angle_degrees"): 0.0, ("scale", "scale_factor"): 1.0, ("erosion", "kernel_size"): 1,
           ("dilation", "kernel_size"): 1, ("fourier", "amp_factor"): 1.0}
MAIN_ARG = {"multiply": "scalar", "add_scalar": "scalar", "add_noise": "noise_std", "rotate": "angle_degrees",
            "scale": "scale_factor", "erosion": "kernel_size", "dilation": "kernel_size", "fourier": "amp_factor"}
TRACK_RE = re.compile(r"^(?P<sel>.+)@(?P<name>[A-Za-z_][\w]*)=(?P<a>[^:]*):(?P<b>[^:]*)$")


class AnimationError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ sources
def load_source(api: str | None = None, png: str | None = None, prompt_id: str | None = None,
                last_run: bool = False) -> tuple[dict, dict | None]:
    """(API prompt, canvas graph or None)."""
    if api:
        return json.loads(Path(api).read_text(encoding="utf-8")), None
    if png:
        from PIL import Image
        with Image.open(png) as im:
            info = dict(im.info)
        if "prompt" not in info:
            raise AnimationError(f"{png} carries no ComfyUI prompt (saved outside ComfyUI, or metadata stripped)")
        wf = info.get("workflow")
        return json.loads(info["prompt"]), json.loads(wf) if wf else None
    if last_run or prompt_id:
        if prompt_id:
            h = cc._req("GET", f"/api/history/{prompt_id}")
            if prompt_id not in h:
                raise AnimationError(f"no run {prompt_id} in ComfyUI's history")
            entry = h[prompt_id]
        else:
            items = cc._history(1)
            if not items:
                raise AnimationError("ComfyUI's history is empty")
            entry = items[0][1]
        prompt = entry["prompt"]
        extra = prompt[3] if len(prompt) > 3 and isinstance(prompt[3], dict) else {}
        return copy.deepcopy(prompt[2]), (extra.get("extra_pnginfo") or {}).get("workflow")
    raise AnimationError("give a source: --api, --png, --prompt-id or --last-run")


# ------------------------------------------------------------------------------------------------ selectors
def _title(node: dict) -> str:
    return (node.get("_meta") or {}).get("title") or node.get("class_type", "")


def find_node(api: dict, sel: str, class_type: str | None = None) -> str:
    if sel in api:
        return sel
    hits = [nid for nid, n in api.items() if _title(n).lower() == sel.lower()
            and (class_type is None or n.get("class_type") == class_type)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise AnimationError(f"no node with id or title {sel!r}; titles: {sorted({_title(n) for n in api.values()})}")
    raise AnimationError(f"title {sel!r} matches nodes {hits}; use the node id")


def _bends(api: dict, nid: str) -> dict | None:
    """Parsed bends JSON of a node, or None when it holds {{a}}-style placeholders (animate those with --input)."""
    text = api[nid]["inputs"].get("bends_json") or '{"bends": []}'
    if not isinstance(text, str) or "{{" in text:
        return None
    return json.loads(text)


def find_bend(api: dict, sel: str) -> tuple[str, int]:
    nodes = [nid for nid, n in api.items() if n.get("class_type") == BEND_NODE]
    if "#" in sel:
        node_sel, idx = sel.rsplit("#", 1)
        return find_node(api, node_sel, BEND_NODE), int(idx)
    hits = []
    for nid in nodes:
        data = _bends(api, nid) or {"bends": []}
        hits += [(nid, i) for i, b in enumerate(data["bends"]) if sel in (b.get("label"), b.get("path"))]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise AnimationError(f"no bend labelled or at path {sel!r}")
    raise AnimationError(f"{sel!r} matches several bends {hits}; use NODE#index")


# ------------------------------------------------------------------------------------------------ tracks
def _num(text: str | None):
    return None if text is None or text.strip() == "" else float(text)


def parse_track(api: dict, kind: str, text: str) -> dict:
    m = TRACK_RE.match(text)
    if not m:
        raise AnimationError(f"bad track {text!r}; expected SELECTOR@name=FROM:TO (FROM or TO may be empty)")
    sel, name, a, b = m["sel"], m["name"], _num(m["a"]), _num(m["b"])
    if kind == "input":
        nid = find_node(api, sel)
        cur = api[nid]["inputs"].get(name)
        if isinstance(cur, list):
            raise AnimationError(f"{sel}@{name} is linked to another node; animate that node's value instead")
        if cur is not None and not isinstance(cur, (int, float)) or isinstance(cur, bool):
            raise AnimationError(f"{sel}@{name} is not numeric ({cur!r})")
        if a is None and cur is None:
            raise AnimationError(f"{sel}@{name}: give FROM (the input has no current value)")
        return {"kind": "input", "node": nid, "name": name, "label": f"{_title(api[nid])}@{name}",
                "from": a if a is not None else float(cur), "to": b if b is not None else float(cur or 0),
                "integer": isinstance(cur, int)}
    nid, idx = find_bend(api, sel)
    data = _bends(api, nid)
    if data is None:
        raise AnimationError(f"bends in node {nid} use {{{{a}}}} placeholders; animate them with --input "
                             f"\"{_title(api[nid])}@a=FROM:TO\"")
    bend = data["bends"][idx]
    op = bend.get("module_type", "rotate")
    if name == "blend":
        cur, neutral = float(bend.get("blend", 1.0)), 0.0
    else:
        cur = (bend.get("module_args") or {}).get(name)
        if cur is None and a is None:
            raise AnimationError(f"bend {sel} has no argument {name!r}; its args: {sorted(bend.get('module_args', {}))}")
        neutral = NEUTRAL.get((op, name))
        if a is None and neutral is None:
            raise AnimationError(f"{op}.{name} has no neutral value; give FROM, or animate `blend` instead")
    return {"kind": "bend", "node": nid, "index": idx, "name": name, "op": op,
            "label": bend.get("label") or f"{bend.get('path')}", "path": bend.get("path"),
            "from": a if a is not None else neutral, "to": b if b is not None else float(cur),
            "integer": name == "kernel_size"}


def default_tracks(api: dict) -> list[dict]:
    tracks, templated = [], []
    for nid, n in api.items():
        if n.get("class_type") != BEND_NODE:
            continue
        data = _bends(api, nid)
        if data is None:
            templated.append(_title(n))
            continue
        for i, b in enumerate(data["bends"]):
            op = b.get("module_type", "rotate")
            arg = MAIN_ARG.get(op)
            if arg and arg in (b.get("module_args") or {}):
                tracks.append(parse_track(api, "bend", f"{nid}#{i}@{arg}=:"))
            else:
                tracks.append(parse_track(api, "bend", f"{nid}#{i}@blend=0:"))
    if not tracks and templated:
        raise AnimationError(f"the bends in {templated} use {{{{a}}}}-style placeholders; animate the value that "
                             f"feeds them, e.g. --input \"<slider node id or title>@value=FROM:TO\" "
                             f"(or --input \"{templated[0]}@a=FROM:TO\" when a is not linked)")
    if not tracks:
        raise AnimationError("no JSON bends to animate; add --bend / --input tracks")
    return tracks


# ------------------------------------------------------------------------------------------------ plan
def times(n: int, easing: str) -> list[float]:
    lin = [i / (n - 1) for i in range(n)]
    return [(1 - math.cos(math.pi * x)) / 2 if easing == "ease" else x for x in lin]


def frame_count(tracks: list[dict], frames: int, increment: float | None, max_frames: int) -> int:
    if increment:
        span = next((abs(t["to"] - t["from"]) for t in tracks if t["to"] != t["from"]), 0.0)
        if span == 0:
            raise AnimationError("nothing moves (every track has FROM = TO)")
        frames = round(span / increment) + 1
    if not 2 <= frames <= max_frames:
        raise AnimationError(f"{frames} frames; allowed 2-{max_frames} (use a larger increment or --max-frames)")
    return frames


def seamless(tracks: list[dict]) -> bool:
    moving = [t for t in tracks if t["to"] != t["from"]]
    return bool(moving) and all(t.get("name") == "angle_degrees" and abs(t["to"] - t["from"]) % 360 == 0
                                for t in moving)


def play_order(n: int, tracks: list[dict], loop: str, hold: int, repeats: int, reverse: bool) -> list[int]:
    fwd = list(range(n))
    if loop == "restart" and seamless(tracks):
        fwd = fwd[:-1]
    if reverse:
        fwd.reverse()
    one = [fwd[0]] * hold + fwd + [fwd[-1]] * hold + (fwd[-2:0:-1] if loop == "boomerang" else [])
    return one * (1 if loop == "none" else repeats)


def value_at(track: dict, t: float):
    v = track["from"] + (track["to"] - track["from"]) * t
    return int(round(v)) if track["integer"] else round(v, 6)


def frame_prompt(api: dict, tracks: list[dict], t: float, keep: set[str], prefix: str) -> tuple[dict, dict]:
    wf = {nid: copy.deepcopy(n) for nid, n in api.items() if nid in keep}
    values, parsed = {}, {}
    for tr in tracks:
        v = value_at(tr, t)
        values[tr["label"] + (f"@{tr['name']}" if tr["kind"] == "bend" else "")] = v
        if tr["kind"] == "input":
            wf[tr["node"]]["inputs"][tr["name"]] = v
        else:
            data = parsed.setdefault(tr["node"], json.loads(wf[tr["node"]]["inputs"]["bends_json"]))
            bend = data["bends"][tr["index"]]
            if tr["name"] == "blend":
                bend["blend"] = v
            else:
                bend.setdefault("module_args", {})[tr["name"]] = v
    for nid, data in parsed.items():
        wf[nid]["inputs"]["bends_json"] = json.dumps(data, separators=(",", ":"))
    for n in wf.values():
        if n.get("class_type") == "SaveImage":
            n["inputs"]["filename_prefix"] = prefix
    return wf, values


def downstream(api: dict, sources: set[str]) -> set[str]:
    out, changed = set(sources), True
    while changed:
        changed = False
        for nid, n in api.items():
            if nid not in out and any(isinstance(v, list) and len(v) == 2 and str(v[0]) in out
                                      for v in n["inputs"].values()):
                out.add(nid)
                changed = True
    return out


def ancestors(api: dict, nid: str) -> set[str]:
    out, stack = set(), [nid]
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack += [str(v[0]) for v in api[cur]["inputs"].values() if isinstance(v, list) and len(v) == 2]
    return out


def pick_output(api: dict, tracks: list[dict], sel: str | None) -> str:
    if sel:
        nid = find_node(api, sel)
        if api[nid].get("class_type") not in IMAGE_OUTPUTS:
            raise AnimationError(f"{sel} is a {api[nid].get('class_type')}, not SaveImage/PreviewImage")
        return nid
    moved = downstream(api, {t["node"] for t in tracks})
    outs = [nid for nid in moved if api[nid].get("class_type") in IMAGE_OUTPUTS]
    if len(outs) == 1:
        return outs[0]
    if not outs:
        video = sorted({n.get("class_type") for n in api.values()
                        if n.get("class_type") in ("SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveVideo", "SaveWEBM")})
        if video:
            raise AnimationError(f"this workflow makes a video ({', '.join(video)}), so it already moves: to make a "
                                 f"bend change over the clip, wrap it in frame_ramp (one render) instead of animating")
        raise AnimationError("no SaveImage/PreviewImage downstream of the animated nodes")
    raise AnimationError("several image outputs are downstream of the tracks; choose one with --output-node: "
                         + ", ".join(f"{nid} ({_title(api[nid])})" for nid in sorted(outs)))


def plan(api: dict, tracks: list[dict], frames: int = 24, increment: float | None = None, easing: str = "ease",
         loop: str = "boomerang", hold: int = 0, repeats: int = 2, fps: int = 12, reverse: bool = False,
         output_node: str | None = None, max_frames: int = MAX_FRAMES) -> dict:
    n = frame_count(tracks, frames, increment, max_frames)
    ts = times(n, easing)
    for tr in tracks:
        if tr.get("name") == "scale_factor" and any(value_at(tr, t) == 0 for t in ts):
            raise AnimationError(f"scale reaches 0 on {tr['label']} (a singular zoom); start from e.g. 0.01")
    out = pick_output(api, tracks, output_node)
    order = play_order(n, tracks, loop, hold, repeats, reverse)
    return {"tracks": tracks, "times": ts, "output_node": out, "keep": ancestors(api, out) | {out},
            "order": order, "fps": fps, "duration_s": round(len(order) / fps, 2),
            "values": [{k: v for k, v in frame_prompt(api, tracks, t, set(api), "")[1].items()} for t in ts]}


# ------------------------------------------------------------------------------------------------ render
def render(api: dict, ui: dict | None, p: dict, out_dir: Path, progress=None, timeout: float = 900) -> list[Path]:
    """Queue every frame (ComfyUI pipelines them), then collect each frame's image in order."""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir.name
    pids = []
    for i, t in enumerate(p["times"]):
        wf, _ = frame_prompt(api, p["tracks"], t, p["keep"], f"{cc.INPUT_SUBFOLDER}/animation_frames/{stamp}/f{i:03d}")
        pids.append(cc.queue_prompt(wf, ui)["prompt_id"])
    paths = []
    deadline = time.time() + timeout * max(1, len(pids) // 4)
    for i, pid in enumerate(pids):
        while True:
            h = cc._req("GET", f"/api/history/{pid}")
            entry = h.get(pid)
            status = (entry or {}).get("status") or {}
            if entry and status.get("status_str") == "error":
                err = next((m[1] for m in status.get("messages") or [] if m[0] == "execution_error"), {})
                raise AnimationError(f"frame {i} failed in {err.get('node_type')}: {err.get('exception_message')}")
            if entry and status.get("completed"):
                break
            if time.time() > deadline:
                raise AnimationError(f"frame {i} did not finish in time")
            time.sleep(0.5)
        imgs = (entry.get("outputs") or {}).get(p["output_node"], {}).get("images") or []
        if not imgs:
            raise AnimationError(f"frame {i}: the output node produced no image")
        im = imgs[0]
        q = urllib.parse.urlencode({k: im.get(k, "") for k in ("filename", "subfolder", "type")})
        path = frames_dir / f"f{i:03d}.png"
        path.write_bytes(_get(f"/api/view?{q}"))
        paths.append(path)
        if progress:
            progress(i + 1, len(pids))
    return paths


def _get(path: str) -> bytes:
    import urllib.request
    with cc._OPENER.open(cc.BASE + path, timeout=120) as r:
        return r.read()


def encode(frame_paths: list[Path], order: list[int], out: Path, fps: int) -> Path:
    seq = [frame_paths[i] for i in order]
    ext = out.suffix.lower()
    if ext == ".mp4":
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise AnimationError("ffmpeg not found for .mp4; install it, or write .gif / .webp instead")
        return _mp4(exe, seq, out, fps)
    if ext in (".gif", ".webp"):
        from PIL import Image
        cache: dict[Path, Image.Image] = {}
        imgs = [cache.setdefault(p, Image.open(p).convert("RGB")) for p in seq]
        kw = {"loop": 0, "duration": round(1000 / fps)}
        if ext == ".webp":
            kw.update(quality=90, method=4)
        imgs[0].save(out, save_all=True, append_images=imgs[1:], **kw)
        return out
    raise AnimationError("output must end in .mp4, .gif or .webp")


def _mp4(exe: str, seq: list[Path], out: Path, fps: int) -> Path:
    from PIL import Image
    with Image.open(seq[0]) as im:
        w, h = im.size
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cache: dict[Path, bytes] = {}
    try:
        for p in seq:
            if p not in cache:
                with Image.open(p) as im:
                    cache[p] = im.convert("RGB").resize((w, h)).tobytes()
            proc.stdin.write(cache[p])
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise AnimationError(f"ffmpeg failed: {err.strip()[:500]}")
    return out


def filmstrip(frame_paths: list[Path], n: int = 6, thumb: int = 256) -> bytes:
    """n evenly spaced frames side by side (JPEG), to look at the motion without playing the video."""
    from PIL import Image
    idx = sorted({round(i * (len(frame_paths) - 1) / max(1, n - 1)) for i in range(n)})
    ims = []
    for i in idx:
        im = Image.open(frame_paths[i]).convert("RGB")
        im.thumbnail((thumb, thumb))
        ims.append(im)
    strip = Image.new("RGB", (sum(i.width for i in ims) + 4 * (len(ims) - 1), max(i.height for i in ims)), (20, 20, 24))
    x = 0
    for im in ims:
        strip.paste(im, (x, 0))
        x += im.width + 4
    buf = io.BytesIO()
    strip.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def animate(api: dict, ui: dict | None, tracks: list[dict], out: Path, progress=None, **plan_kw) -> dict:
    """Plan, render, encode; returns the manifest (also written as anim.json next to the frames)."""
    p = plan(api, tracks, **plan_kw)
    out = Path(out)
    work = out.with_suffix("")
    work = work.parent / f"{work.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    work.mkdir(parents=True, exist_ok=True)
    frames = render(api, ui, p, work, progress)
    encode(frames, p["order"], out, p["fps"])
    (work / "filmstrip.jpg").write_bytes(filmstrip(frames))
    manifest = {"video": str(out.resolve()), "frames_dir": str((work / "frames").resolve()),
                "filmstrip": str((work / "filmstrip.jpg").resolve()), "rendered_frames": len(frames),
                "video_frames": len(p["order"]), "fps": p["fps"], "duration_s": p["duration_s"],
                "output_node": p["output_node"], "tracks": p["tracks"], "values_per_frame": p["values"],
                "created": datetime.now().isoformat(timespec="seconds")}
    (work / "anim.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def video_name(name: str, fmt: str) -> str:
    """e.g. recompose_animation_20261004-153012.mp4"""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:50] or "bend"
    return f"{stem}_animation_{datetime.now().strftime('%Y%m%d-%H%M%S')}.{fmt}"


def deliver(video: Path) -> dict:
    """Save the encoded video into ComfyUI's output folder (agent_bending/). With ComfyUI on this computer the local
    copy is removed, so there is one file; with ComfyUI elsewhere a copy stays here too. Returns `where`, a sentence
    for the artist, plus `file` and `view_url`, or the local `video` path if ComfyUI did not take it."""
    try:
        saved = cc.save_output(video.read_bytes(), video.name)
    except (cc.CanvasError, OSError) as e:
        return {"video": str(video), "where": f"Saved on this computer at {video} (ComfyUI did not take it: {e})."}
    where = (f"Saved in ComfyUI's output folder, inside agent_bending, as {Path(saved['file']).name}. That is the "
             f"folder where ComfyUI saves every picture it makes. It also plays in the browser: {saved['view_url']}")
    if cc.is_local():
        video.unlink(missing_ok=True)
        return {**saved, "video": None, "where": where}
    return {**saved, "video": str(video),
            "where": where.replace("ComfyUI's output folder", "the output folder of the ComfyUI on the other machine",
                                   1) + f". A copy is on this computer at {video}"}


def build_tracks(api: dict, bends: list[str], inputs: list[str]) -> list[dict]:
    tracks = [parse_track(api, "bend", b) for b in bends] + [parse_track(api, "input", s) for s in inputs]
    return tracks or default_tracks(api)


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--api"); src.add_argument("--png"); src.add_argument("--prompt-id")
    src.add_argument("--last-run", action="store_true")
    ap.add_argument("--bend", action="append", default=[]); ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--frames", type=int, default=24); ap.add_argument("--increment", type=float)
    ap.add_argument("--easing", choices=["ease", "linear"], default="ease")
    ap.add_argument("--loop", choices=["boomerang", "restart", "none"], default="boomerang")
    ap.add_argument("--hold", type=int, default=0); ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--fps", type=int, default=12); ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--output-node"); ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--out"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--name", default=""); ap.add_argument("--format", choices=["mp4", "gif", "webp"], default="mp4")
    a = ap.parse_args(argv)
    try:
        api, ui = load_source(a.api, a.png, a.prompt_id, a.last_run)
        tracks = build_tracks(api, a.bend, a.input)
        kw = dict(frames=a.frames, increment=a.increment, easing=a.easing, loop=a.loop, hold=a.hold,
                  repeats=a.repeats, fps=a.fps, reverse=a.reverse, output_node=a.output_node, max_frames=a.max_frames)
        if a.dry_run:
            p = plan(api, tracks, **kw)
            print(json.dumps({"rendered_frames": len(p["times"]), "video_frames": len(p["order"]),
                              "duration_s": p["duration_s"], "output_node": p["output_node"],
                              "tracks": [{k: t[k] for k in ("label", "name", "from", "to")} for t in tracks],
                              "first_values": p["values"][:3], "last_value": p["values"][-1]},
                             indent=1, ensure_ascii=False))
            return
        if a.out:
            out = Path(a.out)
        else:
            fmt = "gif" if a.format == "mp4" and shutil.which("ffmpeg") is None else a.format
            out = cc.work_dir() / "animations" / video_name(a.name, fmt)
            out.parent.mkdir(parents=True, exist_ok=True)
        m = animate(api, ui, tracks, out, progress=lambda d, n: print(f"\rframe {d}/{n}", end="", flush=True), **kw)
        where = f"wrote {m['video']}" if a.out else deliver(out)["where"]
        print(f"\n{where}\n({m['rendered_frames']} rendered frames, {m['duration_s']} s; frames and anim.json in "
              f"{Path(m['frames_dir']).parent})")
    except (AnimationError, cc.CanvasError, OSError, ValueError) as e:
        sys.exit(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
