"""The picture box: the artist drops a picture into the chat and it lands in ComfyUI. The MCP server runs over stdio
against a stub HTTP server standing in for ComfyUI.

    uv run --no-project --with "mcp>=2.2" --with pytest --with pillow --with numpy pytest tests
"""

from __future__ import annotations

import base64
import http.server
import io
import json
import re
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

from test_board import APPS, SCRIPTS, noise_png, run, text

URI = "ui://comfyui-bending/picture.html"


@pytest.fixture(scope="module")
def comfy():
    """Accepts /api/upload/image (multipart) and serves the uploads back on /api/view."""
    uploads = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            from urllib.parse import parse_qs, urlsplit
            name = parse_qs(urlsplit(self.path).query).get("filename", [""])[0]
            if name not in uploads:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(uploads[name])

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            boundary = re.search(r"boundary=(.+)", self.headers["Content-Type"]).group(1).encode()
            fields, image = {}, None
            for part in raw.split(b"--" + boundary)[1:-1]:
                head, _, value = part[2:].partition(b"\r\n\r\n")
                value = value[:-2]
                fn = re.search(rb'filename="([^"]+)"', head)
                if fn:
                    image = (fn.group(1).decode(), value)
                else:
                    fields[re.search(rb'name="([^"]+)"', head).group(1).decode()] = value.decode()
            uploads[image[0]] = image[1]
            out = json.dumps({"name": image[0], "subfolder": fields.get("subfolder", ""), "type": "input"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", uploads
    srv.shutdown()


def data(result) -> dict:
    return result.structured_content if isinstance(result.structured_content, dict) else json.loads(text(result))


def jpeg(side: int = 300) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (side, side // 2), (40, 80, 160)).save(buf, "JPEG")
    return buf.getvalue()


def test_picture_box_is_declared(comfy, tmp_path):
    async def body(s):
        tools = {t.name: t for t in (await s.list_tools()).tools}
        return tools, (await s.read_resource(URI)).contents[0]

    tools, page = run(comfy[0], tmp_path, body)
    assert tools["ask_for_picture"].meta["ui"]["resourceUri"] == URI == tools["ask_for_picture"].meta["ui/resourceUri"]
    assert tools["add_picture"].meta["ui"]["resourceUri"] == URI  # visible: relayed hosts pass on only those
    assert tools["picture_received"].meta is None
    assert page.mime_type == "text/html;profile=mcp-app"
    assert "globalThis.ExtApps={" in page.text and "__EXT_APPS_BUNDLE__" not in page.text and 'src="http' not in page.text


def test_dropped_picture_reaches_comfyui_in_parts(comfy, tmp_path):
    base, uploads = comfy
    big = noise_png(1, 700)  # about 1.5 MB: several parts

    async def body(s):
        shown = await s.call_tool("ask_for_picture", {"purpose": "The painting we start from"})
        req = shown.structured_content["request"]
        rid, size = req["request_id"], req["part_bytes"]
        opened = await s.call_tool("add_picture", {"request_id": rid})
        waiting = await s.call_tool("picture_received", {"request_id": rid, "max_wait": 0})
        parts = [big[i:i + size] for i in range(0, len(big), size)]
        replies = [await s.call_tool("add_picture", {"request_id": rid, "name": "Starry Night.png", "part": i,
                                                     "parts": len(parts), "data": base64.b64encode(p).decode()})
                   for i, p in enumerate(parts)]
        got = await s.call_tool("picture_received", {"request_id": rid, "max_wait": 0})
        # the artist drops another picture, a JPEG whose name says .png
        await s.call_tool("add_picture", {"request_id": rid, "name": "scan.png", "data": base64.b64encode(jpeg()).decode()})
        again = await s.call_tool("picture_received", {"request_id": rid, "since_revision": 1, "max_wait": 5})
        return shown, req, opened, waiting, parts, replies, got, again

    shown, req, opened, waiting, parts, replies, got, again = run(base, tmp_path, body, APPS)
    assert not shown.is_error and "picture_received" in text(shown) and req["purpose"] == "The painting we start from"
    assert data(opened)["ready"] is True
    assert data(waiting)["status"] == "waiting" and data(waiting)["opened"] is True
    assert len(parts) > 2 and [data(r).get("received_part") for r in replies[:-1]] == list(range(len(parts) - 1))
    fb = data(got)
    assert fb["status"] == "received" and fb["image"].startswith("agent_bending/Starry_Night_") and fb["revision"] == 1
    assert (fb["width"], fb["height"]) == (700, 700) and fb["view_url"].startswith(f"{base}/api/view?")
    assert uploads[fb["image"].split("/", 1)[1]] == big
    second = data(again)
    assert second["revision"] == 2 and second["image"].endswith(".jpg") and (second["width"], second["height"]) == (300, 150)


def test_without_the_box_it_offers_the_other_ways(comfy, tmp_path):
    async def body(s):
        return await s.call_tool("ask_for_picture", {})

    r = run(comfy[0], tmp_path, body)  # a client without MCP Apps
    t = text(r)
    assert not r.is_error and "cannot show the picture box" in t
    assert "Copy as path" in t and "upload_image" in t and "Load Image" in t and "list_input_images" in t
    assert "PICTURE_DIRS" not in t and "Picture folder" not in t  # a scope setting, not a way to load a picture


def test_a_remote_comfyui_gets_no_pictures_from_this_computer(tmp_path):
    async def body(s):
        asked = await s.call_tool("ask_for_picture", {})
        sent = await s.call_tool("add_picture", {"request_id": "picture_0123456789ab", "data": "aGk="})
        return asked, sent

    asked, sent = run("http://192.0.2.10:8188", tmp_path, body, APPS)
    assert "another machine" in text(asked) and "Load Image" in text(asked) and "Copy as path" not in text(asked)
    assert sent.is_error and "another machine" in text(sent)


def test_only_pictures_are_taken(comfy, tmp_path):
    async def body(s):
        rid = (await s.call_tool("ask_for_picture", {})).structured_content["request"]["request_id"]
        notes = await s.call_tool("add_picture", {"request_id": rid, "name": "notes.png",
                                                  "data": base64.b64encode(b"not a picture at all").decode()})
        garbage = await s.call_tool("add_picture", {"request_id": rid, "data": "%%%"})
        late = await s.call_tool("add_picture", {"request_id": rid, "data": "aGk=", "part": 1, "parts": 2})
        bad_id = await s.call_tool("add_picture", {"request_id": "../../x", "data": "aGk="})
        return notes, garbage, late, bad_id

    notes, garbage, late, bad_id = run(comfy[0], tmp_path, body, APPS)
    assert notes.is_error and "not a picture" in text(notes)
    assert garbage.is_error and "base64" in text(garbage)
    assert late.is_error and "first part" in text(late)
    assert bad_id.is_error and "bad board or picture box id" in text(bad_id)


def test_picture_box_that_was_never_displayed(tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import board
    req = board.create_request(tmp_path, purpose="p")
    assert board.wait_picture(tmp_path, req["board_id"], 0)["status"] == "waiting"
    monkeypatch.setattr(board, "NOT_SHOWN_AFTER_S", -1)
    gone = board.wait_picture(tmp_path, req["board_id"], 30)
    assert gone["status"] == "not_displayed" and "full path" in gone["hint"] and "Load Image" in gone["hint"]
    board.mark_opened(tmp_path, req["board_id"])
    assert board.wait_picture(tmp_path, req["board_id"], 0)["status"] == "waiting"
