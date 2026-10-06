"""The in-chat board: the MCP server over stdio, against a stub HTTP server standing in for ComfyUI.

    uv run --no-project --with "mcp>=2.2" --with pytest --with pillow --with numpy pytest tests
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import io
import json
import os
import sys
import threading
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "comfyui-model-bending" / "scripts"
APPS = {"io.modelcontextprotocol/ui": {"mimeTypes": ["text/html;profile=mcp-app"]}}
URI = "ui://comfyui-bending/board.html"


def noise_png(seed: int, side: int = 768) -> bytes:
    """Random pixels: the worst case for JPEG size."""
    rnd = __import__("random").Random(seed)
    im = Image.frombytes("RGB", (side, side), bytes(rnd.getrandbits(8) for _ in range(side * side * 3)))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def comfy():
    """A stand-in for ComfyUI that serves /api/view?filename=... as PNGs."""
    pngs = {f"c{i}.png": noise_png(i) for i in range(4)}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            from urllib.parse import parse_qs, urlsplit
            u = urlsplit(self.path)
            name = parse_qs(u.query).get("filename", [""])[0]
            if u.path != "/api/view" or name not in pngs:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(pngs[name])

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def view(base: str, i: int) -> str:
    return f"{base}/api/view?filename=c{i}.png&subfolder=&type=output"


def run(base: str, workdir: Path, body, extensions=None):
    env = {**os.environ, "COMFYUI_URL": base, "COMFY_BENDING_WORKDIR": str(workdir)}
    params = StdioServerParameters(command=sys.executable, args=[str(SCRIPTS / "mcp_server.py")], env=env)

    async def main():
        async with stdio_client(params) as (r, w), ClientSession(r, w, extensions=extensions) as s:
            await s.initialize()
            return await body(s)

    return asyncio.run(main())


def candidates(base: str, n: int = 3) -> list[dict]:
    return [{"label": chr(65 + i), "image_url": view(base, i + 1), "caption": f"version {chr(65 + i)}",
             "recommended": i == 1, "details": f"rotate mid {30 * (i + 1)}"} for i in range(n)]


def text(result) -> str:
    return " ".join(c.text for c in result.content if c.type == "text")


def data(result) -> dict:
    return result.structured_content if isinstance(result.structured_content, dict) else json.loads(text(result))


def test_tools_and_view_are_declared(comfy, tmp_path):
    async def body(s):
        tools = {t.name: t for t in (await s.list_tools()).tools}
        resource = (await s.read_resource(URI)).contents[0]
        return tools, resource

    tools, resource = run(comfy, tmp_path, body)
    assert tools["show_board"].meta["ui"]["resourceUri"] == URI == tools["show_board"].meta["ui/resourceUri"]
    for name in ("board_image", "submit_board"):  # relayed hosts pass on only model-visible tools
        assert tools[name].meta["ui"]["resourceUri"] == URI and "visibility" not in tools[name].meta["ui"]
    assert tools["board_feedback"].meta is None and "comfy_status" in tools
    assert resource.mime_type == "text/html;profile=mcp-app"
    page = resource.text
    assert "globalThis.ExtApps={" in page and "__EXT_APPS_BUNDLE__" not in page
    assert "import " not in page.split("<script>", 1)[0] and 'src="http' not in page  # nothing loaded from the web


def test_board_round_trip_in_an_app(comfy, tmp_path):
    async def body(s):
        shown = await s.call_tool("show_board", {"title": "Round 2", "question": "Which way?",
                                                  "candidates": candidates(comfy), "original_url": view(comfy, 0),
                                                  "round": 2, "ask": ["keep_change", "direction"]})
        board = shown.structured_content["board"]
        images = [await s.call_tool("board_image", {"board_id": board["board_id"], "index": i}) for i in range(4)]
        missing = await s.call_tool("board_image", {"board_id": board["board_id"], "index": 4})
        waiting = await s.call_tool("board_feedback", {"board_id": board["board_id"], "max_wait": 0})  # opened
        first = await s.call_tool("submit_board", {"board_id": board["board_id"], "feedback": {
            "picks": ["B", "C", "Z"], "keep": ["palette", "nonsense"], "change": ["lighting"], "push": 3,
            "direction": "calmer sea"}})
        got = await s.call_tool("board_feedback", {"board_id": board["board_id"], "max_wait": 0})
        await s.call_tool("submit_board", {"board_id": board["board_id"], "feedback": {"picks": ["A"]}})
        changed = await s.call_tool("board_feedback", {"board_id": board["board_id"], "since_revision": 1,
                                                        "max_wait": 5})
        return shown, board, images, missing, waiting, first, got, changed

    shown, board, images, missing, waiting, first, got, changed = run(comfy, tmp_path, body, APPS)
    assert not shown.is_error and "board_feedback" in text(shown) and not any(c.type == "image" for c in shown.content)
    assert board["has_original"] and [c["label"] for c in board["candidates"]] == ["A", "B", "C"]
    assert board["ask"] == ["keep_change", "direction"] and "image_url" not in str(board)  # no sources to the view
    for r in images:
        (img,) = [c for c in r.content if c.type == "image"]
        raw = base64.b64decode(img.data)
        assert img.mime_type == "image/jpeg" and len(img.data) < 150_000
        assert max(Image.open(io.BytesIO(raw)).size) <= 640  # noise shrinks further to fit
    assert missing.is_error and "no image 4" in text(missing)
    assert data(waiting)["status"] == "waiting" and data(waiting)["opened"] is True
    assert data(first) == {"saved": True, "revision": 1}
    fb = data(got)
    assert fb["status"] == "answered" and fb["picks"] == ["B", "C"] and fb["keep"] == ["palette"]
    assert fb["change"] == ["lighting"] and fb["push"] == 1.0 and fb["direction"] == "calmer sea"
    assert data(changed)["picks"] == ["A"] and data(changed)["revision"] == 2


def test_contact_sheet_when_the_app_cannot_show_boards(comfy, tmp_path):
    async def body(s):
        return await s.call_tool("show_board", {"title": "Round 1", "question": "Which?",
                                                 "candidates": candidates(comfy, 2)})

    r = run(comfy, tmp_path, body)
    assert not r.is_error and [c.type for c in r.content] == ["image", "text"]
    assert "ask in chat" in text(r) and "A, B" in text(r)


def test_only_comfyui_images_can_be_shown(comfy, tmp_path):
    async def body(s):
        bad_host = await s.call_tool("show_board", {"title": "t", "question": "q", "candidates": [
            {"label": "A", "image_url": "https://example.com/api/view?filename=x.png"}]})
        bad_path = await s.call_tool("show_board", {"title": "t", "question": "q", "candidates": [
            {"label": "A", "image_url": f"{comfy}/api/userdata/secret"}]})
        bad_id = await s.call_tool("board_image", {"board_id": "../../etc", "index": 0})
        return bad_host, bad_path, bad_id

    for r in run(comfy, tmp_path, body, APPS):
        assert r.is_error
    assert "ComfyUI /api/view URLs" in text(run(comfy, tmp_path, lambda s: body(s), APPS)[0])


def bends_doc(*bends) -> str:
    return json.dumps({"bends": list(bends)})


HISTORY = {  # c1: the unbent run; c2: a bend the safe ranges pulled back, from a knowledge-base cell; c3: another bend
    "p-base": {"prompt": [1, "p-base", {"9": {"class_type": "SaveImage", "inputs": {}}}, {}, []],
               "outputs": {"9": {"images": [{"filename": "c1.png", "subfolder": "", "type": "output"}]}},
               "status": {"status_str": "success", "completed": True}},
    "p-b": {"prompt": [2, "p-b", {"13": {"class_type": "ApplyBendsFromJSON", "inputs": {"bends_json": bends_doc(
        {"path": "output_blocks.7", "module_type": "add_noise", "module_args": {"noise_std": 3.0}, "label": "B",
         "kb": {"cell": "sd1|out.hi|resblock|ResBlock|add_noise|high|all|img2img"}})}}}, {}, []],
        "outputs": {"9": {"images": [{"filename": "c2.png", "subfolder": "", "type": "output"}]},
                    "14": {"text": [json.dumps({"clamp": "safe", "clamped": [
                        {"path": "output_blocks.7", "op": "add_noise", "arg": "noise_std", "requested": 3.0,
                         "applied": 1.5}]})]}},
        "status": {"status_str": "success", "completed": True}},
    "p-c": {"prompt": [3, "p-c", {"13": {"class_type": "ApplyBendsFromJSON", "inputs": {"bends_json": bends_doc(
        {"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 30}})}}}, {}, []],
        "outputs": {"9": {"images": [{"filename": "c3.png", "subfolder": "", "type": "output"}]}},
        "status": {"status_str": "success", "completed": True}},
}


@pytest.fixture()
def comfy_runs():
    """Like `comfy`, plus /api/history with the runs that made c1-c3."""
    pngs = {f"c{i}.png": noise_png(i, 64) for i in range(4)}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            from urllib.parse import parse_qs, urlsplit
            u = urlsplit(self.path)
            if u.path == "/api/history":
                body, kind = json.dumps(HISTORY).encode(), "application/json"
            elif u.path == "/api/view" and parse_qs(u.query).get("filename", [""])[0] in pngs:
                body, kind = pngs[parse_qs(u.query)["filename"][0]], "image/png"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_board_shows_applied_amounts_and_asks_for_logs_and_links(comfy_runs, tmp_path):
    base = comfy_runs
    cands = [{"label": "Unbent", "image_url": view(base, 1), "caption": "no bend"},
             {"label": "B", "image_url": view(base, 2), "caption": "grainy", "details": "out.hi add_noise 3.0"},
             {"label": "C", "image_url": view(base, 3), "caption": "rotated"}]

    async def body(s):
        shown = await s.call_tool("show_board", {"title": "Round 2", "question": "Which?", "candidates": cands,
                                                  "session": "starry-1006", "round": 2})
        bid = shown.structured_content["board"]["board_id"]
        await s.call_tool("submit_board", {"board_id": bid, "feedback": {"picks": ["B"], "votes": {"C": "down"},
                                                                         "direction": "more grain"}})
        return shown, await s.call_tool("board_feedback", {"board_id": bid, "max_wait": 0})

    shown, got = run(base, tmp_path, body, APPS)
    b = {c["label"]: c for c in shown.structured_content["board"]["candidates"]}
    assert b["B"]["details"].startswith("Applied after the safe-range clamp: add_noise noise_std 1.5 (asked 3.0)")
    assert "out.hi add_noise 3.0" in b["B"]["details"] and "say the applied one" in text(shown)
    assert "Applied" not in b["C"]["details"] and "prompt_id" not in str(b)  # the run stays on the server
    fb = data(got)
    assert fb["log_round"]["todo"] == [{"label": "B", "prompt_id": "p-b", "verdict": "kept"},
                                       {"label": "C", "prompt_id": "p-c", "verdict": "rejected"}]
    assert fb["log_round"]["baseline_prompt_id"] == "p-base" and fb["log_round"]["session"] == "starry-1006"
    (liked,) = fb["liked"]["versions"]
    assert liked["label"] == "B" and liked["kb_links"] == [
        "https://abuzreq-model-bending-navigator.static.hf.space/?cell=sd1%7Cout.hi%7Cresblock%7CResBlock%7Cadd_noise"
        "%7Chigh%7Call%7Cimg2img"]
    assert fb["liked"]["tray_link"].startswith("https://abuzreq-model-bending-navigator.static.hf.space/#tray=")
    sys.path.insert(0, str(SCRIPTS))
    import kb
    (bend,) = kb.read_tray(fb["liked"]["tray_link"])
    assert bend["path"] == "output_blocks.7" and bend["kb"]["cell"].startswith("sd1|out.hi")


def test_bundle_export_becomes_a_global():
    sys.path.insert(0, str(SCRIPTS))
    import board
    assert board.as_global("var a=1,b=2;export{a as App,b};") == "var a=1,b=2;globalThis.ExtApps={App:a,b:b};"
    assert board.as_global('x="</script>";export{x as X};').startswith('x="<\\/script>"')


def test_board_that_was_never_displayed(comfy, tmp_path, monkeypatch):
    sys.path.insert(0, str(SCRIPTS))
    import board
    monkeypatch.setattr(board.cc, "BASE", comfy)
    b = board.create(tmp_path, title="t", question="q", candidates=candidates(comfy, 2))
    assert board.wait_feedback(tmp_path, b["board_id"], 0)["status"] == "waiting"  # just shown: give it time
    monkeypatch.setattr(board, "NOT_SHOWN_AFTER_S", -1)
    gone = board.wait_feedback(tmp_path, b["board_id"], 30)  # returns at once, does not wait the 30 s
    assert gone["status"] == "not_displayed" and "view_images" in gone["hint"]
    board.mark_opened(tmp_path, b["board_id"])
    assert board.wait_feedback(tmp_path, b["board_id"], 0)["status"] == "waiting"
