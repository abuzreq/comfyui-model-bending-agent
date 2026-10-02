"""Video outputs (frames, metrics, filmstrips, board previews) and starting pictures (upload, image-to-image specs),
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
from PIL import Image, ImageDraw

from test_board import APPS, SCRIPTS, data, run, text

sys.path.insert(0, str(SCRIPTS))
import comfy_canvas as cc  # noqa: E402
import media  # noqa: E402
import metrics  # noqa: E402

PRESETS = SCRIPTS.parent / "presets"


def clip(kind: str, n: int = 12, side: int = 160) -> bytes:
    """An animated WEBP: a square moving across ("moving"), the same frame over and over ("frozen"), or frames that
    jump around ("flicker")."""
    rnd = __import__("random").Random(1)
    frames = []
    for i in range(n):
        im = Image.new("RGB", (side, side), (40, 60, 90))
        d = ImageDraw.Draw(im)
        if kind == "moving":
            x = 10 + i * (side - 50) // n
        elif kind == "frozen":
            x = 10
        else:
            x = rnd.randrange(0, side - 40)
            im.paste((rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)), (0, 0, side, side // 2))
        d.rectangle((x, 60, x + 40, 100), fill=(240, 200, 60))
        im.putpixel((0, 0), (i, i, i))  # never two identical frames (the encoder would merge them)
        frames.append(im)
    buf = io.BytesIO()
    frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:], duration=62, loop=0, lossless=True)
    return buf.getvalue()


def still() -> bytes:
    im = Image.new("RGB", (160, 160), (40, 60, 90))
    ImageDraw.Draw(im).rectangle((10, 60, 50, 100), fill=(240, 200, 60))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def comfy():
    """Serves /api/view (stills and clips), accepts /api/upload/image (multipart) and answers /api/history."""
    files = {"moving.webp": clip("moving"), "frozen.webp": clip("frozen"), "flicker.webp": clip("flicker"),
             "long.webp": clip("moving", n=40, side=480), "start.png": still()}
    uploads = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            from urllib.parse import parse_qs, urlsplit
            u = urlsplit(self.path)
            name = parse_qs(u.query).get("filename", [""])[0]
            body = files.get(name) or uploads.get(name)
            if u.path != "/api/view" or body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            boundary = re.search(r"boundary=(.+)", self.headers["Content-Type"]).group(1).encode()
            fields, image = {}, None
            for part in raw.split(b"--" + boundary)[1:-1]:
                head, _, value = part.strip(b"\r\n").partition(b"\r\n\r\n")
                name = re.search(rb'name="([^"]+)"', head).group(1).decode()
                fn = re.search(rb'filename="([^"]+)"', head)
                if fn:
                    image = (fn.group(1).decode(), value)
                else:
                    fields[name] = value.decode()
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
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    srv.uploads = uploads
    yield base, uploads
    srv.shutdown()


def url(base: str, name: str) -> str:
    return f"{base}/api/view?filename={name}&subfolder=&type=output"


# ---------------------------------------------------------------- frames and metrics
def test_frames_of_stills_and_clips(comfy):
    base, _ = comfy
    seq, ms = media.frames(url(base, "moving.webp"))
    assert len(seq) == 12 and ms > 0
    seq, ms = media.frames(url(base, "start.png"))
    assert len(seq) == 1 and ms == 0
    assert media.sample(list(range(10)), 4) == [0, 3, 6, 9] and media.sample([1, 2], 5) == [1, 2]
    assert media.name_of(url(base, "x.mp4")) == "x.mp4" and media.is_video_name(url(base, "x.mp4"))


def test_video_compare_flags_frozen_and_flicker(comfy):
    base, _ = comfy
    rows = metrics.compare(url(base, "moving.webp"), [url(base, "moving.webp"), url(base, "frozen.webp"),
                                                      url(base, "flicker.webp")])
    b, same, frozen, flicker = rows
    assert b["role"] == "baseline" and b["frames"] == 12 and b["motion"] > 2 and "static" not in b["flags"]
    assert same["mae"] < 0.5 and "noop" in same["flags"] and same["motion_ratio"] == 1.0
    assert "frozen" in frozen["flags"] and frozen["motion"] < 0.5 and len(frozen["mae_by_frame"]) == 8
    assert "flicker" in flicker["flags"] and flicker["motion_ratio"] > 2.5


def test_still_against_a_video_measures_drift(comfy):
    base, _ = comfy
    _, row = metrics.compare(url(base, "start.png"), [url(base, "moving.webp")])
    assert row["mae_by_frame"][0] < row["mae_by_frame"][-1]  # starts on the picture, moves away from it
    assert "motion_ratio" not in row and row["motion"] > 2


def test_filmstrip_sheet_and_video_diff(comfy, tmp_path):
    base, _ = comfy
    out = tmp_path / "sheet.jpg"
    msg = metrics.sheet(str(out), [url(base, "start.png"), url(base, "moving.webp")], ["start", "A"], 3, 320,
                        "t", 5)
    im = Image.open(out)
    assert "2 rows of up to 5 frames" in msg and im.width == 12 + 5 * (240 + 8) + 4
    stats = metrics.diff(url(base, "moving.webp"), url(base, "frozen.webp"), str(tmp_path / "d.png"))
    assert stats["mean_abs_delta"] > 1


def test_long_clips_are_kept_small_for_measuring(comfy):
    base, _ = comfy
    kept = metrics.frames(url(base, "long.webp"))
    assert len(kept) == metrics.KEEP_FRAMES and max(kept[0].size) <= metrics.KEEP_SIDE
    assert metrics.motion(url(base, "long.webp")) > 0


def test_board_preview_of_a_video_is_a_small_loop(comfy):
    base, _ = comfy
    data_, fmt = media.preview(media.read(url(base, "long.webp")), "long.webp", 640)
    im = Image.open(io.BytesIO(data_))
    assert fmt == "webp" and getattr(im, "n_frames", 1) > 1 and len(data_) <= 90_000
    data_, fmt = media.preview(media.read(url(base, "start.png")), "start.png", 640)
    assert fmt == "jpeg"


def test_board_shows_videos_through_the_server(comfy, tmp_path):
    base, _ = comfy

    async def body(s):
        shown = await s.call_tool("show_board", {"title": "Round 1", "question": "Which?",
                                                  "original_url": url(base, "start.png"), "candidates": [
                                                      {"label": "A", "image_url": url(base, "moving.webp")},
                                                      {"label": "B", "image_url": url(base, "flicker.webp")}]})
        bid = shown.structured_content["board"]["board_id"]
        return [await s.call_tool("board_image", {"board_id": bid, "index": i}) for i in range(3)]

    imgs = run(base, tmp_path, body, APPS)
    types = [[c for c in r.content if c.type == "image"][0].mime_type for r in imgs]
    assert types == ["image/jpeg", "image/webp", "image/webp"]
    assert all(len([c for c in r.content if c.type == "image"][0].data) < 150_000 for r in imgs)


def test_view_images_makes_filmstrips(comfy, tmp_path):
    base, _ = comfy

    async def body(s):
        return await s.call_tool("view_images", {"urls": [url(base, "moving.webp"), url(base, "frozen.webp")],
                                                  "labels": ["0", "A"], "frames": 4})

    r = run(base, tmp_path, body)
    (img,) = [c for c in r.content if c.type == "image"]
    w, h = Image.open(io.BytesIO(base64.b64decode(img.data))).size
    assert w > h  # two filmstrip rows of 4 frames


def test_runs_mark_video_outputs():
    entry = {"prompt": [0, "p", {"9": {"class_type": "SaveAnimatedWEBP", "inputs": {}}}, {}],
             "outputs": {"9": {"images": [{"filename": "a_00001_.webp", "subfolder": "", "type": "output"}],
                               "animated": [True]},
                         "5": {"images": [{"filename": "b.png", "subfolder": "", "type": "temp"}]}},
             "status": {"status_str": "success", "messages": []}}
    import unittest.mock as um
    with um.patch.object(cc, "object_info", return_value={}):
        imgs = {i["filename"]: i for i in cc.describe_run("p", entry)["images"]}
    assert imgs["a_00001_.webp"].get("video") is True and "video" not in imgs["b.png"]


# ---------------------------------------------------------------- starting pictures
def test_upload_from_a_file_and_from_a_run(comfy, tmp_path, monkeypatch):
    base, uploads = comfy
    monkeypatch.setattr(cc, "BASE", base)
    pic = tmp_path / "My Picture.png"
    pic.write_bytes(still())
    a = cc.upload_image(str(pic))
    assert re.fullmatch(r"agent_bending/My_Picture_[0-9a-f]{8}\.png", a["image"]) and a["bytes"] == len(still())
    assert cc.upload_image(f'"{pic}"')["image"] == a["image"]  # quoted "Copy as path" works; same picture, same name
    b = cc.upload_image(url(base, "moving.webp"), name="version B")
    assert b["image"].startswith("agent_bending/version_B_") and b["image"].endswith(".webp")
    assert uploads[b["image"].split("/", 1)[1]] == clip("moving")
    with pytest.raises(cc.CanvasError, match="no file"):
        cc.upload_image(str(tmp_path / "missing.png"))
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(cc.CanvasError, match="not a picture"):
        cc.upload_image(str(tmp_path / "notes.txt"))
    with pytest.raises(cc.CanvasError, match="not a ComfyUI image URL"):
        cc.upload_image("https://example.com/api/view?filename=x.png")


def test_start_image_turns_a_text_to_image_spec_into_image_to_image():
    spec = json.loads((PRESETS / "switchboard_sd15.spec.json").read_text(encoding="utf-8"))
    new, notes = cc.use_start_image(spec, "agent_bending/pic_1234abcd.png")
    n = new["nodes"]
    assert n["start_image"] == {"class_type": "LoadImage", "title": "Starting picture",
                                "inputs": {"image": "agent_bending/pic_1234abcd.png"}, "group": "base"}
    assert n["lat"]["class_type"] == "VAEEncode" and n["lat"]["inputs"] == {"pixels": ["lat_fit", 0], "vae": ["vae", 0]}
    assert n["lat_fit"]["inputs"]["width"] == 512 and n["lat_fit"]["inputs"]["image"] == ["start_image", 0]
    assert n["latP"]["class_type"] == "RepeatLatentBatch" and n["latP"]["inputs"]["amount"] == 4  # the 4-seed lane
    assert {n[k]["inputs"]["denoise"] for k in ("ks0", "ksA", "ksB", "ksC", "ksP")} == {0.6}
    assert new["meta"]["start_image"] == "agent_bending/pic_1234abcd.png" and new["meta"]["denoise"] == 0.6
    assert any("structure window" in x for x in notes)
    assert spec["nodes"]["lat"]["class_type"] == "EmptyLatentImage"  # the input spec is left alone
    links = [v for node in n.values() for v in (node.get("inputs") or {}).values() if cc._is_link(v)]
    assert all(src in n for src, _ in links)
    high, notes = cc.use_start_image(spec, "x.png", 0.85)
    assert high["nodes"]["ksA"]["inputs"]["denoise"] == 0.85 and not any("structure" in x for x in notes)


def test_start_image_in_video_presets():
    i2v = json.loads((PRESETS / "video_sweep_wan21_i2v.spec.json").read_text(encoding="utf-8"))
    new, notes = cc.use_start_image(i2v, "agent_bending/pic.png")
    assert new["nodes"]["start_image"]["inputs"]["image"] == "agent_bending/pic.png" and "existing" in notes[0]
    assert new["nodes"]["ksA"]["inputs"]["denoise"] == 1.0  # image-to-video starts from the picture, no denoise
    t2v = json.loads((PRESETS / "video_sweep_wan21_t2v.spec.json").read_text(encoding="utf-8"))
    with pytest.raises(cc.CanvasError, match="image-to-video"):
        cc.use_start_image(t2v, "pic.png")
    with pytest.raises(cc.CanvasError, match="denoise"):
        cc.use_start_image(json.loads((PRESETS / "switchboard_sd15.spec.json").read_text(encoding="utf-8")), "p.png", 1.5)


def test_video_presets_bend_attention_with_reports():
    for name in ("video_sweep_wan21_t2v", "video_sweep_wan21_i2v"):
        spec = json.loads((PRESETS / f"{name}.spec.json").read_text(encoding="utf-8"))
        bends = [n for n in spec["nodes"].values() if n["class_type"] == "ApplyBendsFromJSON"]
        assert len(bends) == 3 and spec["meta"]["experimental"] is True
        for b in bends:
            doc = json.loads(b["inputs"]["bends_json"])
            assert doc["bends"] == [] and len(doc["attention_bends"]) == 1 and b["inputs"]["strict"] is True
        saves = [n for n in spec["nodes"].values() if n["class_type"] == "SaveAnimatedWEBP"]
        assert len(saves) == 4  # baseline + 3, readable without ffmpeg
        reports = [n for n in spec["nodes"].values() if n["class_type"] == "PreviewAny"]
        assert len(reports) == 3


def test_wan_time_windows_and_animation_of_a_video():
    import animate
    import timesteps
    t = timesteps.table("wan", 10)
    assert t["t_per_step"][0] == 1.0 and t["windows"]["structure"]["steps"] == "0-7"  # shift 8: 8 of 10 steps sit above t 0.7
    api = {"1": {"class_type": "ApplyBendsFromJSON", "inputs": {"bends_json": json.dumps(
        {"bends": [{"path": "blocks.15", "module_type": "rotate", "module_args": {"angle_degrees": 10}}]})}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["2", 0]}}}
    with pytest.raises(animate.AnimationError, match="frame_ramp"):
        animate.plan(api, animate.build_tracks(api, [], []), frames=4)


def test_new_tools_are_listed(comfy, tmp_path):
    base, _ = comfy

    async def body(s):
        tools = {t.name: t for t in (await s.list_tools()).tools}
        return tools

    tools = run(base, tmp_path, body)
    assert {"upload_image", "list_input_images"} <= set(tools)
    assert "start_image" in tools["build_workflow"].input_schema["properties"]
    assert "frames" in tools["view_images"].input_schema["properties"]
    assert text  # noqa: B018  (helpers shared with test_board)
    assert data


def test_video_runs_become_knowledge_base_records():
    import kb
    spec = json.loads((PRESETS / "video_sweep_wan21_i2v.spec.json").read_text(encoding="utf-8"))
    spec, _ = cc.use_start_image(spec, "agent_bending/pic_1234abcd.png")
    lane = {k: n for k, n in spec["nodes"].items() if n.get("group") in ("base", "candA")}
    model, setup, bends = kb.setup_from_api_prompt(lane)
    assert model["arch"] == "wan21_i2v" and kb.family_of(model["arch"]) == "wan"
    assert setup["route"] == "img2video" and setup["frames"] == 25 and setup["input_image"].endswith("pic_1234abcd.png")
    (b,) = bends
    assert b["path"] == "attention.cross_image.blocks[17-24]" and b["op"] == "rotate" and b["steps"] == "0-2"
    rec = kb.make_record(source="session", model=model, setup=setup, bends=bends, salt="s")
    assert kb.validate(rec, "record") == [] and "input_image" not in rec["setup"]  # kept private
    assert rec["bends"][0]["group"] == "attention" and rec["bends"][0]["kind"] == "cross_image"
