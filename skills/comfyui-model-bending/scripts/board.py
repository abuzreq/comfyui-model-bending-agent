"""The in-chat board: a pick board the MCP server shows inside the Claude app (an MCP Apps view), so the artist chooses
between rendered versions without opening ComfyUI.

The model calls `show_board` with the versions (ComfyUI image or video URLs plus a plain caption each). The view
asks the server for each picture (`board_image`, one downscaled JPEG per call, or a short looping WEBP for a video,
never through the model), and on Send stores
the artist's answer (`submit_board`) and posts a plain sentence into the chat. The model reads the exact answer with
`board_feedback`. Clients without MCP Apps get the same versions as one labelled contact sheet instead.

The picture box is the same idea in the other direction: `ask_for_picture` shows a box in the chat where the artist
drops a picture (one attached to the chat is not a file the server can open); the view sends it to the server in parts
(`add_picture`), which uploads it to ComfyUI, and posts a sentence into the chat. The model reads the result with
`picture_received`.

Boards and picture boxes are JSON files under <workdir>/boards/. Standard library plus Pillow.
"""

from __future__ import annotations

import base64
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
MAX_PICTURE_BYTES = 30_000_000
PART_BYTES = 400_000  # raw bytes per add_picture call (about 530 KB as base64)
_ID = re.compile(r"^(board|picture)_[0-9a-f]{6,32}$")


class BoardError(ValueError):
    pass


# ------------------------------------------------------------------ store

def _path(root: Path, board_id: str) -> Path:
    if not _ID.match(board_id or ""):
        raise BoardError(f"bad board or picture box id {board_id!r}")
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
        cand = {"label": str(c.get("label") or chr(65 + i))[:24], "image_url": url,
                "caption": str(c.get("caption") or "")[:400], "recommended": bool(c.get("recommended")),
                "details": str(c.get("details") or "")[:400]}
        if isinstance(c.get("run"), dict):  # the run behind the picture (server-side only, never shown)
            cand["run"] = {k: c["run"][k] for k in ("prompt_id", "bent", "bends") if k in c["run"]}
        cands.append(cand)
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
        raise BoardError(f"no board or picture box {board_id!r}")
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


# ------------------------------------------------------------------ picture box

def create_request(root: Path, *, purpose: str = "", session: str = "") -> dict:
    """Store a picture box (the artist drops a picture into it); returns it."""
    req = {"board_id": f"picture_{secrets.token_hex(6)}", "purpose": purpose[:300], "session": session,
           "created": time.time(), "picture": None}
    root.mkdir(parents=True, exist_ok=True)
    _path(root, req["board_id"]).write_text(json.dumps(req, indent=1), encoding="utf-8")
    return req


def request_view(req: dict) -> dict:
    return {"request_id": req["board_id"], "purpose": req["purpose"], "max_bytes": MAX_PICTURE_BYTES,
            "part_bytes": PART_BYTES}


def add_part(root: Path, request_id: str, data_b64: str, part: int, parts: int) -> bytes | None:
    """Collect one part of a dropped picture; returns the whole file after the last part, None before it. Part 0
    starts over (the artist chose another picture)."""
    load(root, request_id)  # it exists
    if not 1 <= parts <= MAX_PICTURE_BYTES // PART_BYTES + 1 or not 0 <= part < parts:
        raise BoardError(f"bad part {part} of {parts}")
    try:
        chunk = base64.b64decode(data_b64, validate=True)
    except ValueError as e:
        raise BoardError(f"part {part} is not base64 data") from e
    tmp = root / f"{request_id}.part"
    have = tmp.stat().st_size if part and tmp.exists() else 0
    if part and not tmp.exists():
        raise BoardError("send the picture from its first part")
    if have + len(chunk) > MAX_PICTURE_BYTES:
        tmp.unlink(missing_ok=True)
        raise BoardError(f"the picture is larger than {MAX_PICTURE_BYTES // 1_000_000} MB")
    with tmp.open("ab" if part else "wb") as f:
        f.write(chunk)
    if part < parts - 1:
        return None
    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    return data


def record_picture(root: Path, request_id: str, uploaded: dict) -> dict:
    """Store what the dropped picture became in ComfyUI (a later picture replaces an earlier one)."""
    req = load(root, request_id)
    pic = {k: uploaded[k] for k in ("image", "view_url", "bytes", "width", "height", "name") if k in uploaded}
    pic["revision"] = int((req.get("picture") or {}).get("revision", 0)) + 1
    pic["received"] = time.time()
    req["picture"] = pic
    _path(root, request_id).write_text(json.dumps(req, indent=1), encoding="utf-8")
    return pic


def wait_picture(root: Path, request_id: str, max_wait: float, since_revision: int = 0) -> dict:
    """The dropped picture once it is newer than `since_revision`, waiting up to max_wait seconds."""
    deadline = time.time() + max(0.0, max_wait)
    while True:
        req = load(root, request_id)
        pic = req.get("picture")
        if pic and pic.get("revision", 0) > since_revision:
            return {"status": "received", "request_id": request_id, **pic}
        if not req.get("opened") and time.time() - req["created"] > NOT_SHOWN_AFTER_S:
            return {"status": "not_displayed", "request_id": request_id, "opened": False,
                    "hint": "the artist's app did not display the picture box. Offer the other two ways in plain "
                            "words: paste the picture's full path into the chat (then upload_image), or drag it into "
                            "a Load Image box in ComfyUI and say its name (then list_input_images)."}
        if time.time() >= deadline:
            return {"status": "waiting", "request_id": request_id, "opened": bool(req.get("opened")),
                    "hint": "the picture box is open but no picture has arrived yet; call picture_received again"}
        time.sleep(0.5)


def picture_size(data: bytes) -> tuple[int, int]:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return im.size


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

def html(page_name: str = "board.html") -> str:
    """A view (board.html, picture.html) with the vendored MCP Apps client inlined (hosts render CDN imports blank)."""
    page = (UI_DIR / page_name).read_text(encoding="utf-8")
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
