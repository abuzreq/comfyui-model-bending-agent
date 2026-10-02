"""The in-chat board: a pick board the MCP server shows inside the Claude app (an MCP Apps view), so the artist chooses
between rendered versions without opening ComfyUI.

The model calls `show_board` with the versions (ComfyUI image or video URLs plus a plain caption each). The view
asks the server for each picture (`board_image`, one downscaled JPEG per call, or a short looping WEBP for a video,
never through the model), and on Send stores
the artist's answer (`submit_board`) and posts a plain sentence into the chat. The model reads the exact answer with
`board_feedback`. Clients without MCP Apps get the same versions as one labelled contact sheet instead.

Boards are JSON files under <workdir>/boards/. Standard library plus Pillow.
"""

from __future__ import annotations

import io
import json
import re
import secrets
import time
import urllib.parse
from pathlib import Path

import comfy_canvas as cc
import media

UI_DIR = Path(__file__).resolve().parent / "ui"
BUNDLE_MARK = "/*__EXT_APPS_BUNDLE__*/"
MAX_CANDIDATES = 6
EXTRAS = ("votes", "keep_change", "strength", "direction")
CHIPS = ("composition", "subject", "palette", "lighting", "texture", "style")
NOT_SHOWN_AFTER_S = 20  # a displayed board fetches its first picture within seconds
_ID = re.compile(r"^board_[0-9a-f]{6,32}$")


class BoardError(ValueError):
    pass


# ------------------------------------------------------------------ store

def _path(root: Path, board_id: str) -> Path:
    if not _ID.match(board_id or ""):
        raise BoardError(f"bad board id {board_id!r}")
    return root / f"{board_id}.json"


def create(root: Path, *, title: str, question: str, candidates: list[dict], original_url: str = "",
           session: str = "", round: int | None = None, ask: list[str] | None = None) -> dict:
    """Validate and store a board; returns it (without feedback)."""
    if not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise BoardError(f"a board shows 1-{MAX_CANDIDATES} versions, got {len(candidates)}")
    cands = []
    for i, c in enumerate(candidates):
        url = str(c.get("image_url") or "")
        _check_url(url)
        cands.append({"label": str(c.get("label") or chr(65 + i))[:24], "image_url": url,
                      "caption": str(c.get("caption") or "")[:400], "recommended": bool(c.get("recommended")),
                      "details": str(c.get("details") or "")[:300]})
    if len({c["label"] for c in cands}) != len(cands):
        raise BoardError("version labels must be unique")
    if original_url:
        _check_url(original_url)
    unknown = [a for a in ask or [] if a not in EXTRAS]
    if unknown:
        raise BoardError(f"unknown extras {unknown}; choose from {list(EXTRAS)}")
    board = {"board_id": f"board_{secrets.token_hex(6)}", "title": title[:120], "question": question[:400],
             "original_url": original_url, "candidates": cands, "session": session, "round": round,
             "ask": list(ask if ask is not None else EXTRAS), "chips": list(CHIPS),
             "created": time.time(), "feedback": None}
    root.mkdir(parents=True, exist_ok=True)
    _path(root, board["board_id"]).write_text(json.dumps(board, indent=1), encoding="utf-8")
    return board


def load(root: Path, board_id: str) -> dict:
    p = _path(root, board_id)
    if not p.exists():
        raise BoardError(f"no board {board_id!r}")
    return json.loads(p.read_text(encoding="utf-8"))


def view(board: dict) -> dict:
    """What the in-chat view needs (no image data, no feedback history)."""
    return {k: board[k] for k in ("board_id", "title", "question", "ask", "chips", "round")} | {
        "has_original": bool(board["original_url"]),
        "candidates": [{k: c[k] for k in ("label", "caption", "recommended", "details")} for c in board["candidates"]]}


def submit(root: Path, board_id: str, feedback: dict) -> dict:
    """Store the artist's answer (a later submit replaces an earlier one); returns the cleaned answer."""
    board = load(root, board_id)
    labels = [c["label"] for c in board["candidates"]]
    picks = [label for label in labels if label in (feedback.get("picks") or [])]  # in board order
    clean = {"picks": picks, "none": bool(feedback.get("none")) and not picks,
             "votes": {k: v for k, v in (feedback.get("votes") or {}).items() if k in labels and v in ("up", "down")},
             "keep": [c for c in feedback.get("keep") or [] if c in CHIPS],
             "change": [c for c in feedback.get("change") or [] if c in CHIPS],
             "rating": max(0, min(5, int(feedback.get("rating") or 0))),
             "push": max(-1.0, min(1.0, float(feedback.get("push") or 0.0))),
             "direction": str(feedback.get("direction") or "")[:1000], "submitted": time.time()}
    clean["revision"] = int((board.get("feedback") or {}).get("revision", 0)) + 1
    board["feedback"] = clean
    _path(root, board_id).write_text(json.dumps(board, indent=1), encoding="utf-8")
    return clean


def wait_feedback(root: Path, board_id: str, max_wait: float, since_revision: int = 0) -> dict:
    """The stored answer once it is newer than `since_revision`, waiting up to max_wait seconds."""
    deadline = time.time() + max(0.0, max_wait)
    while True:
        board = load(root, board_id)
        fb = board.get("feedback")
        if fb and fb.get("revision", 0) > since_revision:
            return {"status": "answered", "board_id": board_id, **fb}
        if not board.get("opened") and time.time() - board["created"] > NOT_SHOWN_AFTER_S:
            return {"status": "not_displayed", "board_id": board_id, "opened": False,
                    "hint": "the board never asked for its pictures, so the artist's app did not display it. "
                            "Show the versions with view_images (one plain caption each) and ask in chat."}
        if time.time() >= deadline:
            return {"status": "waiting", "board_id": board_id, "opened": bool(board.get("opened")),
                    "hint": "the artist has the board open but has not sent an answer yet; call board_feedback again"}
        time.sleep(0.5)


def mark_opened(root: Path, board_id: str) -> None:
    """Record that the board was displayed (it fetched its first picture)."""
    board = load(root, board_id)
    if not board.get("opened"):
        board["opened"] = time.time()
        _path(root, board_id).write_text(json.dumps(board, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ images

def _check_url(url: str) -> None:
    u, base = urllib.parse.urlsplit(url), urllib.parse.urlsplit(cc.BASE)
    if (u.scheme, u.netloc) != (base.scheme, base.netloc) or u.path not in ("/api/view", "/view"):
        raise BoardError(f"board images must be ComfyUI /api/view URLs on {cc.BASE} (from a run), got {url!r}")


def image_url(board: dict, index: int) -> str:
    """The source of image `index` in view order (the original first, when there is one)."""
    urls = ([board["original_url"]] if board["original_url"] else []) + [c["image_url"] for c in board["candidates"]]
    if not 0 <= index < len(urls):
        raise BoardError(f"no image {index} on this board ({len(urls)} images)")
    return urls[index]


def picture(data: bytes, name: str = "", max_side: int = 640, limit: int = 90_000) -> tuple[bytes, str]:
    """What the board shows for one source: (bytes, "jpeg" | "webp"); a video becomes a short loop."""
    return media.preview(data, name, max_side, limit)


def jpeg(data: bytes, max_side: int = 640, quality: int = 82, limit: int = 90_000) -> bytes:
    """A downscaled JPEG of at most `limit` bytes, so one tool result stays under the host's ~150 KB inline limit
    after base64 (+33 %). Quality drops first, then the size, until it fits."""
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_side, max_side))
    while True:
        for q in (quality, 70, 55):
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=q, optimize=True)
            if buf.tell() <= limit:
                return buf.getvalue()
        if max(im.size) <= 128:
            return buf.getvalue()
        im = im.resize((max(1, im.width * 3 // 4), max(1, im.height * 3 // 4)), Image.LANCZOS)


# ------------------------------------------------------------------ the view

def html() -> str:
    """board.html with the vendored MCP Apps client inlined (hosts render CDN imports blank)."""
    page = (UI_DIR / "board.html").read_text(encoding="utf-8")
    bundle = (UI_DIR / "vendor" / "ext-apps-app-with-deps.js").read_text(encoding="utf-8")
    return page.replace(BUNDLE_MARK, as_global(bundle), 1)


def as_global(bundle: str) -> str:
    """Turn the ES module's trailing `export{a as B,...}` into `globalThis.ExtApps={B:a,...}`, so the bundle can sit
    in a classic inline script and the page's own script can use it."""
    m = re.search(r"export\s*\{([^}]*)\}\s*;?\s*$", bundle)
    if not m:
        raise BoardError("unexpected MCP Apps bundle format (no trailing export list)")
    pairs = []
    for part in m.group(1).split(","):
        local, _, name = part.strip().partition(" as ")
        pairs.append(f"{(name or local).strip()}:{local.strip()}")
    code = bundle[:m.start()] + "globalThis.ExtApps={" + ",".join(pairs) + "};"
    return code.replace("</script", "<\\/script")  # it sits inside an inline <script> element
