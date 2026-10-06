"""Stills and videos alike: read the frames of a ComfyUI output, and make small previews. Needs Pillow; ffmpeg (on
the PATH) only for .mp4 / .webm outputs.

Video models (WAN) usually save with SaveAnimatedWEBP, which Pillow reads directly. A still is a one-frame video
here, so callers can treat both the same way.

What may be opened (`read`): ComfyUI /api/view URLs on the configured server (COMFYUI_URL), the bend knowledge
base's own example images on Hugging Face, and, for the command-line scripts only, local files. The MCP server turns
local files off (ALLOW_LOCAL_PATHS = False), so a prompt cannot make it read files or call other addresses.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageSequence, UnidentifiedImageError

import comfy_canvas as cc

VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
# ffmpeg demuxer per extension: forcing it stops ffmpeg from probing the data as a playlist (hls, concat), which
# could make it open other files or URLs
DEMUXER = {".mp4": "mov", ".mov": "mov", ".m4v": "mov", ".webm": "matroska", ".mkv": "matroska", ".avi": "avi"}
ALLOW_LOCAL_PATHS = True  # the MCP server sets this to False
KB_IMAGES = ("huggingface.co",
             f"/datasets/{os.environ.get('BEND_KB_DATASET', 'abuzreq/model-bending-knowledge-base')}/resolve/")


class MediaError(ValueError):
    pass


def name_of(src: str) -> str:
    """The file name of a path or a ComfyUI /api/view URL (its `filename` query)."""
    if src.startswith(("http://", "https://")):
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(src).query)
        return (q.get("filename") or [Path(urllib.parse.urlsplit(src).path).name])[0]
    return Path(src).name


def is_video_name(src: str) -> bool:
    return Path(name_of(src)).suffix.lower() in VIDEO_EXT


def source_kind(src: str) -> str:
    """'comfyui', 'kb' or 'file' for a source that may be opened; MediaError otherwise."""
    if src.startswith(("http://", "https://")):
        u = urllib.parse.urlsplit(src)
        if u.scheme == "https" and u.netloc == KB_IMAGES[0] and u.path.startswith(KB_IMAGES[1]) \
                and ".." not in u.path.split("/"):
            return "kb"
        base = urllib.parse.urlsplit(cc.BASE)
        if (u.scheme, u.netloc) == (base.scheme, base.netloc) and u.path in ("/api/view", "/view"):
            return "comfyui"
        raise MediaError(f"only ComfyUI /api/view URLs on {cc.BASE} (from a run) or knowledge-base example images can "
                         f"be opened, not {src!r}")
    if not ALLOW_LOCAL_PATHS:
        raise MediaError("local files cannot be opened through the tools; use the ComfyUI /api/view URL from a run "
                         "(to start from a picture on this computer, use upload_image)")
    return "file"


def read(src: str, timeout: float = 120) -> bytes:
    kind = source_kind(src)
    if kind == "comfyui":
        return cc.fetch_bytes(src, timeout)
    if kind == "kb":
        with urllib.request.urlopen(src, timeout=timeout) as r:  # Hugging Face redirects to its file CDN
            return r.read()
    return Path(src).read_bytes()


def decode(data: bytes, name: str = "") -> tuple[list[Image.Image], int]:
    """(RGB frames, total duration in ms; 0 for a still). Animated WEBP/GIF/APNG through Pillow, other video files
    through ffmpeg."""
    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_EXT:
        try:
            im = Image.open(io.BytesIO(data))
            frames, total = [], 0
            for f in ImageSequence.Iterator(im):
                frames.append(f.convert("RGB"))
                total += int(f.info.get("duration") or 0)
            if frames:
                return frames, (total if len(frames) > 1 else 0)
        except UnidentifiedImageError:
            pass
        if suffix:
            raise MediaError(f"{name}: not a readable image")
    return _ffmpeg_frames(data, suffix or ".mp4")


def _ffmpeg_frames(data: bytes, suffix: str) -> tuple[list[Image.Image], int]:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise MediaError("this video needs ffmpeg to be read, and ffmpeg is not installed. Save previews with "
                         "SaveAnimatedWEBP instead (it needs nothing extra), or install ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in{suffix}"
        src.write_bytes(data)
        safe = ["-protocol_whitelist", "file", "-f", DEMUXER.get(suffix, "mov")]
        r = subprocess.run([exe, "-loglevel", "error", *safe, "-i", str(src), "-vf", "scale='min(832,iw)':-2",
                            str(Path(tmp) / "f%05d.png")], capture_output=True, text=True)
        files = sorted(Path(tmp).glob("f*.png"))
        if r.returncode or not files:
            raise MediaError(f"ffmpeg could not read the video: {r.stderr.strip()[:300]}")
        frames = [Image.open(p).convert("RGB") for p in files]
        probe = subprocess.run([exe, *safe, "-i", str(src)], capture_output=True, text=True).stderr
    fps = _fps(probe)
    return frames, int(1000 * len(frames) / fps) if fps else 0


def _fps(ffmpeg_banner: str) -> float | None:
    m = re.search(r"([\d.]+) fps", ffmpeg_banner)
    return float(m.group(1)) if m else None


def frames(src: str) -> tuple[list[Image.Image], int]:
    """Frames and duration of a path or URL."""
    return decode(read(src), name_of(src))


def sample(seq: list, n: int) -> list:
    """n evenly spaced items, first and last included (all of them if there are fewer)."""
    if n <= 0 or len(seq) <= n:
        return list(seq)
    if n == 1:
        return [seq[len(seq) // 2]]
    return [seq[round(i * (len(seq) - 1) / (n - 1))] for i in range(n)]


def preview(data: bytes, name: str = "", max_side: int = 640, limit: int = 90_000) -> tuple[bytes, str]:
    """A small picture of an output for the in-chat board: (bytes, "jpeg" | "webp"). A still becomes a JPEG; a
    video becomes a short looping animated WEBP. Both stay under `limit` bytes so a tool result stays inline
    (hosts offload results over ~150 KB, and base64 adds a third)."""
    seq, total = decode(data, name)
    if len(seq) == 1:
        return _jpeg(seq[0], max_side, limit), "jpeg"
    side, count = min(max_side, 384), min(len(seq), 24)
    while True:
        picked = sample(seq, count)
        small = []
        for f in picked:
            t = f.copy()
            t.thumbnail((side, side))
            small.append(t)
        ms = max(40, int((total or 62 * len(seq)) / len(picked)))
        for q in (60, 45, 30):
            buf = io.BytesIO()
            small[0].save(buf, "WEBP", save_all=True, append_images=small[1:], duration=ms, loop=0, quality=q,
                          method=4)
            if buf.tell() <= limit:
                return buf.getvalue(), "webp"
        if side <= 160 and count <= 8:  # cannot shrink further: show the middle frame instead
            return _jpeg(seq[len(seq) // 2], max_side, limit), "jpeg"
        side, count = max(160, side * 3 // 4), max(8, count * 3 // 4)


def _jpeg(im: Image.Image, max_side: int, limit: int) -> bytes:
    im = im.copy()
    im.thumbnail((max_side, max_side))
    while True:
        for q in (82, 70, 55):
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() <= limit:
                return buf.getvalue()
        if max(im.size) <= 128:
            return buf.getvalue()
        im = im.resize((max(1, im.width * 3 // 4), max(1, im.height * 3 // 4)), Image.LANCZOS)

