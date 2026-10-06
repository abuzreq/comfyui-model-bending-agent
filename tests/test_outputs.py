"""Where things are saved: the skill's own files in the system's app-data folder (no setting needed), animations in
ComfyUI's output folder next to the pictures ComfyUI makes.

    uv run --no-project --with "mcp>=2.2" --with pytest --with pillow --with numpy pytest tests
"""

from __future__ import annotations

import http.server
import json
import re
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "comfyui-model-bending" / "scripts"))

import animate  # noqa: E402
import comfy_canvas as cc  # noqa: E402


@pytest.fixture()
def comfy(monkeypatch):
    """Accepts /api/upload/image; `fail` makes it answer 500."""
    got = {"files": {}, "fail": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if got["fail"]:
                self.send_error(500)
                return
            boundary = re.search(r"boundary=(.+)", self.headers["Content-Type"]).group(1).encode()
            fields, image = {}, None
            for part in raw.split(b"--" + boundary)[1:-1]:
                head, _, value = part[2:].partition(b"\r\n\r\n")
                fn = re.search(rb'filename="([^"]+)"', head)
                if fn:
                    image = (fn.group(1).decode(), value[:-2])
                else:
                    fields[re.search(rb'name="([^"]+)"', head).group(1).decode()] = value[:-2].decode()
            got["files"][(fields["type"], fields["subfolder"], image[0])] = image[1]
            out = json.dumps({"name": image[0], "subfolder": fields["subfolder"], "type": fields["type"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(cc, "BASE", f"http://127.0.0.1:{srv.server_address[1]}")
    yield got
    srv.shutdown()


def test_work_folder_needs_no_setting(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(cc.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("COMFY_BENDING_WORKDIR", raising=False)
    expected = {"win32": tmp_path / "localappdata", "darwin": home / "Library" / "Application Support"}.get(
        sys.platform, tmp_path / "xdg") / "comfyui-model-bending"
    assert cc.work_dir() == expected
    monkeypatch.setenv("COMFY_BENDING_WORKDIR", "${user_config.workdir}")  # an unfilled setting from an old install
    assert cc.work_dir() == expected
    (home / ".comfyui-model-bending").mkdir()  # made by an earlier version: keep using it
    assert cc.work_dir() == home / ".comfyui-model-bending"
    monkeypatch.setenv("COMFY_BENDING_WORKDIR", str(tmp_path / "mine"))
    assert cc.work_dir() == tmp_path / "mine"


def test_animation_lands_in_comfyui_output_folder(comfy, tmp_path):
    video = tmp_path / animate.video_name("re compose!", "mp4")
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42 a video")
    out = animate.deliver(video)
    name = video.name
    assert re.fullmatch(r"re_compose_animation_\d{8}-\d{6}\.mp4", name)
    assert comfy["files"][("output", "agent_bending", name)] == b"\x00\x00\x00\x18ftypmp42 a video"
    assert out["file"] == f"agent_bending/{name}" and "type=output" in out["view_url"]
    assert not video.exists() and out["video"] is None  # ComfyUI is on this computer: one copy, in its folder
    assert "ComfyUI's output folder, inside agent_bending" in out["where"] and "every picture it makes" in out["where"]
    assert out["view_url"] in out["where"]


def test_animation_for_a_comfyui_elsewhere_keeps_a_copy_here(comfy, tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "is_local", lambda base="": False)
    video = tmp_path / "loop.gif"
    video.write_bytes(b"GIF89a")
    out = animate.deliver(video)
    assert video.exists() and out["video"] == str(video)
    assert "ComfyUI on the other machine" in out["where"] and str(video) in out["where"]


def test_animation_stays_here_when_comfyui_refuses_it(comfy, tmp_path):
    comfy["fail"] = True
    video = tmp_path / "loop.gif"
    video.write_bytes(b"GIF89a")
    out = animate.deliver(video)
    assert video.exists() and out == {"video": str(video), "where": out["where"]}
    assert out["where"].startswith(f"Saved on this computer at {video}")


def test_anim_json_names_where_the_video_ended_up(comfy, tmp_path, monkeypatch):
    video = tmp_path / animate.video_name("d3 on d25", "mp4")
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42 a video")
    frames = tmp_path / video.stem / "frames"
    frames.mkdir(parents=True)
    m = {"video": str(video), "frames_dir": str(frames), "rendered_frames": 2}
    done = animate.finish(m)
    on_disk = json.loads((frames.parent / "anim.json").read_text(encoding="utf-8"))
    assert done["view_url"] and on_disk["file"] == f"agent_bending/{video.name}" and on_disk["video"] is None
    assert "where" not in on_disk and done["where"].startswith("Saved in ComfyUI's output folder")


def test_animation_work_folder_is_dated_once(tmp_path, monkeypatch):
    monkeypatch.setattr(animate, "render", lambda api, ui, p, work, progress: [])
    monkeypatch.setattr(animate, "encode", lambda frames, order, out, fps: out)
    monkeypatch.setattr(animate, "filmstrip", lambda frames: b"")
    monkeypatch.setattr(animate, "plan", lambda api, tracks, **kw: {"order": [], "fps": 12, "duration_s": 0,
                                                                     "output_node": "9", "tracks": [], "values": []})
    m = animate.animate({}, None, [], tmp_path / animate.video_name("loop", "mp4"))
    assert re.fullmatch(r"loop_animation_\d{8}-\d{6}", Path(m["frames_dir"]).parent.name)
    own = animate.animate({}, None, [], tmp_path / "mine.gif")  # a file name of the user's own still gets a date
    assert re.fullmatch(r"mine_\d{8}-\d{6}", Path(own["frames_dir"]).parent.name)


def test_extension_asks_for_no_folder():
    m = json.loads((ROOT / "packaging" / "mcpb" / "manifest.json").read_text(encoding="utf-8"))
    assert set(m["user_config"]) == {"comfyui_url", "picture_dir", "bridge_token"}
    assert "COMFY_BENDING_WORKDIR" not in m["server"]["mcp_config"]["env"]
