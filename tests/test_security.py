"""Trust boundaries: the bridge token, what the tools may open, where workflows are written, caches, uploads, ffmpeg.

    uv run --no-project --with "mcp>=2.2" --with pytest --with pillow --with numpy pytest tests
"""

from __future__ import annotations

import http.server
import io
import os
import secrets
import shutil
import sys
import threading
import urllib.parse
from pathlib import Path

import pytest
from PIL import Image

from test_board import comfy, run, text  # noqa: F401  (the stub ComfyUI fixture)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "comfyui-model-bending" / "scripts"))
import comfy_canvas as cc  # noqa: E402
import media  # noqa: E402

KB = "https://huggingface.co/datasets/abuzreq/model-bending-knowledge-base/resolve/main"


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- 1. the bridge token
@pytest.fixture
def bridge(monkeypatch, tmp_path):
    """A local ComfyUI whose bridge reports `token_file`; returns a setter for that path."""
    monkeypatch.delenv("AGENT_BRIDGE_TOKEN", raising=False)
    monkeypatch.setattr(cc, "BASE", "http://127.0.0.1:8188")
    reported = {}
    monkeypatch.setattr(cc, "bridge_info", lambda: {"version": "0.1.0", "token_file": reported.get("path")})
    return lambda p: reported.update(path=str(p))


def test_the_real_token_file_is_read(bridge, tmp_path):
    tok = secrets.token_urlsafe(32)
    (tmp_path / "agent_bridge").mkdir()
    (tmp_path / "agent_bridge" / "token").write_text(tok + "\n")
    bridge(tmp_path / "agent_bridge" / "token")
    assert cc._bridge_token() == tok


@pytest.mark.parametrize("rel", ["id_ed25519", "agent_bridge/credentials", "aws/token", "token"])
def test_other_files_named_by_the_server_are_never_read(bridge, tmp_path, rel):
    secret = tmp_path / rel
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("x" * 40)  # would pass the token check if it were read
    bridge(secret)
    with pytest.raises(cc.CanvasError, match="not the bridge's token file"):
        cc._bridge_token()


def test_a_link_named_like_the_token_file_is_followed_and_refused(bridge, tmp_path):
    (tmp_path / "keys").write_text("x" * 40)
    (tmp_path / "agent_bridge").mkdir()
    link = tmp_path / "agent_bridge" / "token"
    try:
        os.symlink(tmp_path / "keys", link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    bridge(link)
    with pytest.raises(cc.CanvasError, match="not the bridge's token file"):
        cc._bridge_token()


def test_a_token_file_without_a_token_is_refused(bridge, tmp_path):
    (tmp_path / "agent_bridge").mkdir()
    (tmp_path / "agent_bridge" / "token").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n")
    bridge(tmp_path / "agent_bridge" / "token")
    with pytest.raises(cc.CanvasError, match="does not hold a token"):
        cc._bridge_token()


def test_a_remote_comfyui_needs_the_token_from_the_environment(bridge, tmp_path, monkeypatch):
    (tmp_path / "agent_bridge").mkdir()
    (tmp_path / "agent_bridge" / "token").write_text(secrets.token_urlsafe(32))
    bridge(tmp_path / "agent_bridge" / "token")
    monkeypatch.setattr(cc, "BASE", "http://192.168.1.20:8188")
    with pytest.raises(cc.CanvasError, match="not on this computer.*AGENT_BRIDGE_TOKEN"):
        cc._bridge_token()
    monkeypatch.setenv("AGENT_BRIDGE_TOKEN", "from-the-user")
    assert cc._bridge_token() == "from-the-user"


@pytest.mark.parametrize("base,local", [("http://127.0.0.1:8188", True), ("http://localhost:8188", True),
                                        ("http://[::1]:8188", True), ("http://127.2.3.4:9000", True),
                                        ("http://192.168.1.20:8188", False), ("https://comfy.example.com", False),
                                        ("http://localhost.evil.com", False)])
def test_what_counts_as_this_computer(base, local):
    assert cc.is_local(base) is local


# ---------------------------------------------------------------- 2. what the tools may open
@pytest.mark.parametrize("src,kind", [
    ("http://127.0.0.1:8188/api/view?filename=a.png&type=output", "comfyui"),
    ("http://127.0.0.1:8188/view?filename=a.png", "comfyui"),
    (f"{KB}/records/sd1/paper/abc/output.webp", "kb"),
])
def test_allowed_sources(monkeypatch, src, kind):
    monkeypatch.setattr(cc, "BASE", "http://127.0.0.1:8188")
    assert media.source_kind(src) == kind


@pytest.mark.parametrize("src", [
    "http://127.0.0.1:8188/api/userdata/workflows%2Fmine.json",   # ComfyUI, but not an image route
    "http://127.0.0.1:9999/api/view?filename=a.png",              # another port
    "http://169.254.169.254/latest/meta-data/",                   # cloud metadata
    "http://192.168.1.1/admin",                                    # the local network
    "https://attacker.example/log?d=secret",                      # exfiltration
    "http://huggingface.co/datasets/abuzreq/model-bending-knowledge-base/resolve/main/x.webp",  # not https
    f"{KB}/../../other/resolve/main/x.webp",                      # out of the dataset
    "https://huggingface.co/datasets/someone/else/resolve/main/x.webp",
])
def test_other_addresses_are_refused(monkeypatch, src):
    monkeypatch.setattr(cc, "BASE", "http://127.0.0.1:8188")
    with pytest.raises(media.MediaError):
        media.source_kind(src)


def test_local_files_only_for_the_scripts(monkeypatch, tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(png())
    assert media.read(str(f)) == png()  # command-line scripts
    monkeypatch.setattr(media, "ALLOW_LOCAL_PATHS", False)  # what the MCP server sets
    with pytest.raises(media.MediaError, match="local files"):
        media.read(str(f))


def test_mcp_tools_refuse_files_and_other_addresses(comfy, tmp_path):  # noqa: F811
    pic = tmp_path / "private.png"
    pic.write_bytes(png())

    async def body(s):
        return [await s.call_tool("compare_images", {"baseline_url": str(pic), "candidate_urls": [str(pic)]}),
                await s.call_tool("view_images", {"urls": ["https://attacker.example/x.png?d=1"]}),
                await s.call_tool("diff_image", {"baseline_url": f"{comfy}/api/userdata/x",
                                                 "candidate_url": f"{comfy}/api/userdata/x"})]

    files, web, other_route = run(comfy, tmp_path / "work", body)
    assert files.is_error and "local files" in text(files)
    assert web.is_error and "only ComfyUI /api/view URLs" in text(web)
    assert other_route.is_error


def test_requests_to_comfyui_do_not_follow_redirects(monkeypatch):
    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        monkeypatch.setattr(cc, "BASE", base)
        with pytest.raises(cc.CanvasError, match=r"redirect \(HTTP 302\).*final address"):
            cc.fetch_bytes(f"{base}/api/view?filename=a.png")
        with pytest.raises(cc.CanvasError, match=r"redirect \(HTTP 302\).*final address"):
            cc._req("GET", "/api/system_stats")
    finally:
        srv.shutdown()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_does_not_follow_a_playlist_disguised_as_a_video(tmp_path):
    target = tmp_path / "secret.png"
    target.write_bytes(png())
    playlist = (f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{target.as_uri()}\n#EXT-X-ENDLIST\n").encode()
    with pytest.raises(media.MediaError):
        media.decode(playlist, "clip.mp4")
    with pytest.raises(media.MediaError, match="not a readable image"):
        media.decode(b"not a picture", "a.png")  # unknown data never goes to ffmpeg


# ---------------------------------------------------------------- 3. workflows are written only in the agent's folder
@pytest.mark.parametrize("name,saved", [("r2_board", "agent_bending/r2_board"),
                                        ("agent_bending/s1/r2.json", "agent_bending/s1/r2"),
                                        ("agent_bridge/r3", "agent_bridge/r3"),
                                        ("/my workflow (2)", "agent_bending/my workflow (2)")])
def test_workflow_names_land_in_the_agent_folder(name, saved):
    assert cc.agent_workflow_name(name) == saved


@pytest.mark.parametrize("name", ["../mine", "agent_bending/../mine", "a//b", "C:\\x", "", "x\x00y", "a/./b"])
def test_unsafe_workflow_names_are_refused(name):
    with pytest.raises(cc.CanvasError):
        cc.agent_workflow_name(name)


def test_push_never_writes_outside_the_agent_folder(monkeypatch):
    calls = []
    monkeypatch.setattr(cc, "_req", lambda method, path, body=None, **kw: calls.append(path))
    assert cc.push_workflow({"nodes": []}, "my_portrait") == "workflows/agent_bending/my_portrait.json"
    assert urllib.parse.unquote(calls[0]).startswith("/api/userdata/workflows/agent_bending/my_portrait.json")


# ---------------------------------------------------------------- 4. caches and scratch files
def test_node_schema_cache_lives_in_the_work_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(cc, "_req", lambda method, path, **kw: {"KSampler": {}})
    assert cc.object_info(fresh=True) == {"KSampler": {}}
    (cached,) = (tmp_path / "cache").glob("object_info_*.json")
    assert cc.object_info() == {"KSampler": {}} and cached.parent == tmp_path / "cache"


def test_no_shared_temp_files_in_the_tools():
    scripts = Path(cc.__file__).parent
    for name in ("comfy_canvas.py", "mcp_server.py"):
        assert "gettempdir" not in (scripts / name).read_text(encoding="utf-8"), name


# ---------------------------------------------------------------- uploads are pictures only
def test_only_real_pictures_are_uploaded(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_req", lambda *a, **k: pytest.fail("nothing should be sent"))
    fake = tmp_path / "keys.png"
    fake.write_text("-----BEGIN OPENSSH PRIVATE KEY-----")
    with pytest.raises(cc.CanvasError, match="not a picture"):
        cc.upload_image(str(fake))
    assert cc._looks_like_image(png())


# ---------------------------------------------------------------- second review: session ids, uploads, the community index
import json  # noqa: E402

import kb_local  # noqa: E402


@pytest.mark.parametrize("session", ["../../home/victim/config", "..", "a/b", r"a\b", "", "x" * 65, ".hidden",
                                     r"C:\Users\x", "s" + chr(0)])
def test_session_ids_cannot_leave_the_sessions_folder(monkeypatch, tmp_path, session):
    monkeypatch.setattr(kb_local, "SESSIONS", tmp_path / "work" / "sessions")
    victim = tmp_path / "home" / "victim" / "config.json"
    victim.parent.mkdir(parents=True)
    victim.write_text('{"mcpServers": {}}')
    with pytest.raises(ValueError, match="unusable session id"):
        kb_local.session_sources(session, ["community"])
    assert victim.read_text() == '{"mcpServers": {}}'


def test_ordinary_session_ids_still_work(monkeypatch, tmp_path):
    monkeypatch.setattr(kb_local, "SESSIONS", tmp_path / "sessions")
    assert kb_local.session_sources("lighthouse-0929", ["community", "mine"]) == ["community", "mine"]
    assert kb_local.session_sources("lighthouse-0929") == ["community", "mine"]
    assert [p.name for p in (tmp_path / "sessions").iterdir()] == ["lighthouse-0929.json"]


def test_safe_range_tables_by_name_only(comfy, tmp_path):  # noqa: F811
    async def body(s):
        return [await s.call_tool("get_safe_ranges", {"arch": a}) for a in ("sd15", "../kb/vocab/effects", "flux")]

    ok, traversal, missing = run(comfy, tmp_path, body)
    assert not ok.is_error and traversal.is_error and missing.is_error


def test_local_pictures_go_only_to_a_local_comfyui(monkeypatch, tmp_path):
    pic = tmp_path / "me.png"
    pic.write_bytes(png())
    monkeypatch.setattr(cc, "_req", lambda *a, **k: pytest.fail("nothing should be sent"))
    monkeypatch.setattr(cc, "BASE", "http://192.168.1.20:8188")
    with pytest.raises(cc.CanvasError, match="another machine"):
        cc.upload_image(str(pic))


def test_picture_folders_limit_uploads(monkeypatch, tmp_path):
    allowed, other = tmp_path / "Pictures", tmp_path / "Documents"
    allowed.mkdir(), other.mkdir()
    (allowed / "a.png").write_bytes(png())
    (other / "b.png").write_bytes(png())
    monkeypatch.setattr(cc, "BASE", "http://127.0.0.1:8188")
    monkeypatch.setenv("COMFY_BENDING_PICTURE_DIRS", str(allowed))
    sent = []
    monkeypatch.setattr(cc, "_req", lambda *a, **k: sent.append(a) or {"name": "a.png", "subfolder": "agent_bending"})
    assert cc.upload_image(str(allowed / "a.png"))["image"] == "agent_bending/a.png"
    with pytest.raises(cc.CanvasError, match="outside the picture folders"):
        cc.upload_image(str(other / "b.png"))
    assert len(sent) == 1


SHA = "0123456789abcdef0123456789abcdef01234567"


def fake_hub(files: dict[str, str], calls: list):
    def get(url: str) -> bytes:
        calls.append(url)
        if "/api/datasets/" in url:
            return json.dumps({"sha": SHA}).encode()
        name = url.rsplit("/", 1)[1]
        return files[name].encode()
    return get


def index_files(cell_note: str) -> dict[str, str]:
    return {"cells.jsonl": json.dumps({"key": "k", "note": cell_note}) + "\n",
            "findings.jsonl": json.dumps({"claim": "c"}) + "\n",
            "meta.json": json.dumps({"built": "2099-01-01T00:00:00Z", "cells": 1})}


def test_refresh_fetches_one_commit_and_cleans_text(monkeypatch, tmp_path):
    monkeypatch.setattr(kb_local, "CACHE", tmp_path / "kb_community")
    calls = []
    note = "ok\u200bhidden\u202eflip\x07" + "x" * 3000
    monkeypatch.setattr(kb_local, "_get", fake_hub(index_files(note), calls))
    src = kb_local.refresh_community()
    assert src["commit"] == SHA and all(f"/resolve/{SHA}/index/" in u for u in calls[1:])  # all from one commit
    cell = json.loads((tmp_path / "kb_community" / "cells.jsonl").read_text(encoding="utf-8"))
    assert cell["note"].startswith("okhiddenflip") and len(cell["note"]) <= kb_local.MAX_TEXT + 1
    idx, where = kb_local.community_index()
    assert idx == tmp_path / "kb_community" and SHA[:10] in where


def test_a_bad_download_keeps_the_previous_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(kb_local, "CACHE", tmp_path / "kb_community")
    monkeypatch.setattr(kb_local, "_get", fake_hub(index_files("first"), []))
    kb_local.refresh_community()
    broken = index_files("second")
    broken["findings.jsonl"] = "{not json\n"
    monkeypatch.setattr(kb_local, "_get", fake_hub(broken, []))
    assert kb_local.refresh_community() is None
    assert json.loads((tmp_path / "kb_community" / "cells.jsonl").read_text(encoding="utf-8"))["note"] == "first"


def test_unchecked_old_caches_are_ignored(monkeypatch, tmp_path):
    old = tmp_path / "kb_community"  # written by an earlier version: no source.json
    old.mkdir()
    (old / "cells.jsonl").write_text(json.dumps({"key": "k", "note": "unchecked"}) + "\n")
    (old / "meta.json").write_text(json.dumps({"built": "2099-01-01T00:00:00Z"}))
    monkeypatch.setattr(kb_local, "CACHE", old)
    assert kb_local.community_index()[0] == kb_local.SNAPSHOT


def test_a_commit_pins_the_revision(monkeypatch):
    monkeypatch.setattr(kb_local, "REVISION", SHA)
    monkeypatch.setattr(kb_local, "_get", lambda url: pytest.fail("no lookup needed for a commit"))
    assert kb_local._revision_commit() == SHA
    monkeypatch.setattr(kb_local, "REVISION", "main; rm -rf")
    with pytest.raises(ValueError):
        kb_local._revision_commit()


def test_extension_settings_reach_the_server_and_empty_ones_are_ignored(monkeypatch, tmp_path):
    """The Desktop extension passes its settings as environment variables; an empty optional setting arrives as the
    unfilled placeholder and must count as not set."""
    manifest = json.loads((Path(__file__).resolve().parents[1] / "packaging" / "mcpb" / "manifest.json")
                          .read_text(encoding="utf-8"))
    env, cfg = manifest["server"]["mcp_config"]["env"], manifest["user_config"]
    assert env["AGENT_BRIDGE_TOKEN"] == "${user_config.bridge_token}" and cfg["bridge_token"]["sensitive"] is True
    assert env["COMFY_BENDING_PICTURE_DIRS"] == "${user_config.picture_dir}"
    assert cfg["picture_dir"]["type"] == "directory" and not cfg["picture_dir"].get("multiple")
    # Claude Desktop checks required settings against saved values only (not defaults) and keeps Save disabled until
    # something changes, so a required setting with a default blocks the install: nothing may be required.
    assert not any(c.get("required") for c in cfg.values())
    assert cfg["comfyui_url"]["default"] == "http://127.0.0.1:8188"
    monkeypatch.setenv("COMFY_BENDING_PICTURE_DIRS", "${user_config.picture_dir}")
    monkeypatch.setenv("AGENT_BRIDGE_TOKEN", "${user_config.bridge_token}")
    assert cc._env("COMFY_BENDING_PICTURE_DIRS") is None and cc._env("AGENT_BRIDGE_TOKEN") is None
    pic = tmp_path / "anywhere.png"
    pic.write_bytes(png())
    monkeypatch.setattr(cc, "BASE", "http://127.0.0.1:8188")
    monkeypatch.setattr(cc, "_req", lambda *a, **k: {"name": "anywhere.png", "subfolder": "agent_bending"})
    assert cc.upload_image(str(pic))["image"] == "agent_bending/anywhere.png"  # no folder limit when left empty
