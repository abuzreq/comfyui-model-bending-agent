"""Where a bend came from ("kb"), navigator links and tray links (scripts/kb.py, bendjson.py, kb_local.find).

    uv run --no-project --with pytest pytest tests/test_tray.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "comfyui-model-bending" / "scripts"))
import bendjson  # noqa: E402
import kb  # noqa: E402

CELL = "sd1|out.lo|norm|LayerNorm|multiply|zero|all|txt2img"
ROT = {"path": "middle_block.1", "module_type": "rotate", "module_args": {"angle_degrees": 90}, "label": "recompose",
       "kb": {"cell": "sd1|mid|attn_block|SpatialTransformer|rotate|quarter|all|txt2img", "record": "0372734c25ed"}}
OFF = {"path": "output_blocks.5.1.transformer_blocks.0.norm1", "module_type": "multiply",
       "module_args": {"scalar": 0.0}, "t": [1.0, 0.67], "kb": {"cell": CELL}}


def test_the_kb_key_is_kept_checked_and_linked():
    r = bendjson.check({"bends": [ROT]}, "sd15")
    assert r["ok"] and r["problems"] == []
    assert r["document"]["bends"][0]["kb"] == ROT["kb"]
    assert json.loads(r["bends_json"])["bends"][0]["kb"] == ROT["kb"]  # bend_run gets it unchanged
    d = r["bends"][0]
    assert d["kb"] == ROT["kb"] and d["link"] == kb.NAVIGATOR + "/?record=0372734c25ed"  # the record, more specific
    r = bendjson.check({"bends": [{**ROT, "kb": {"cell": ROT["kb"]["cell"]}}]})
    assert r["bends"][0]["link"].startswith(kb.NAVIGATOR + "/?cell=sd1%7Cmid%7C")
    assert kb.ref_link({"record": "../../index"}) is None  # only record ids make record links


def test_a_bad_kb_is_a_problem_not_an_error():
    r = bendjson.check({"bends": [{**OFF, "kb": {"cell": CELL, "mood": "stormy"}}]})
    assert r["ok"] and any("kb.mood" in p for p in r["problems"])
    assert r["document"]["bends"][0]["kb"] == {"cell": CELL}
    r = bendjson.check({"bends": [{**OFF, "kb": "somewhere"}]})
    assert r["ok"] and "kb" not in r["document"]["bends"][0] and any("kb must be an object" in p for p in r["problems"])
    r = bendjson.check({"bends": [{**OFF, "kb": {"dataset": "someone/else"}}]})
    assert any("no cell or record" in p for p in r["problems"])


def test_own_records_get_no_public_link():
    r = bendjson.check({"bends": [{**OFF, "kb": {"dataset": "local", "cell": CELL}}]})
    assert "link" not in r["bends"][0]


def test_a_tray_link_round_trips_and_each_bend_is_checked_on_its_own():
    link = kb.tray_link([ROT, OFF])
    assert link.startswith(kb.NAVIGATOR + "/#tray=") and "=" not in link.split("#tray=")[1]
    assert kb.read_tray(link) == [ROT, OFF]
    assert kb.read_tray("look at this: " + link + " thanks") == [ROT, OFF]  # pasted inside a sentence
    t = bendjson.open_tray(link, "sd15")
    assert t["ok"] and t["count"] == 2 and "one at a time" in t["note"]
    assert [json.loads(b["bends_json"])["bends"][0]["path"] for b in t["bends"]] == [ROT["path"], OFF["path"]]
    assert t["bends"][1]["kb"] == {"cell": CELL} and t["bends"][1]["link"] == kb.cell_link(CELL)
    assert "weakened" in t["bends"][1]["summary"] and "t 1 to 0.67" in t["bends"][1]["summary"]


def test_a_tray_with_a_broken_bend_names_it_by_its_place():
    t = bendjson.open_tray(kb.tray_link([ROT, {"path": "input_blocks.4", "module_type": "multipy"}]))
    assert t["ok"] and t["bends"][0]["ok"] and not t["bends"][1]["ok"]
    assert t["bends"][1]["errors"] and all("bend #1" in e for e in t["bends"][1]["errors"])


@pytest.mark.parametrize("bad, words", [
    ("https://example.org/no-tray-here", "not a navigator tray link"),
    (kb.NAVIGATOR + "/#tray=%%%", "not a navigator tray link"),
    (kb.NAVIGATOR + "/#tray=bm90IGpzb24", "damaged"),
    (kb.tray_link([ROT]).replace(kb.tray_link([ROT]).split("tray=")[1], "eyJ2Ijo5LCJiZW5kcyI6W119"), "newer navigator"),
])
def test_bad_tray_links_say_what_is_wrong(bad, words):
    with pytest.raises(ValueError, match=words):
        kb.read_tray(bad)


def test_check_bends_points_a_tray_link_to_open_tray():
    with pytest.raises(bendjson.BendJSONError, match="open_tray"):
        bendjson.check(kb.tray_link([ROT]))


def test_the_command_line_reads_and_makes_tray_links(capsys):
    with pytest.raises(SystemExit) as done:
        bendjson.main(["--tray", kb.tray_link([OFF])])
    assert done.value.code == 0 and json.loads(capsys.readouterr().out)["count"] == 1
    bendjson.main([json.dumps({"bends": [OFF]}), "--to-tray"])
    assert kb.read_tray(capsys.readouterr().out.strip())[0]["kb"] == {"cell": CELL}


def test_find_recipes_results_carry_their_source_and_a_link(monkeypatch):
    import kb_local
    monkeypatch.setattr(kb_local, "community_index", lambda refresh=False: (kb_local.SNAPSHOT, "bundled snapshot"))
    out = kb_local.find("more abstract", "sd15", ["community"], limit=3)
    assert out["results"]
    for r in out["results"]:
        assert r["recipe"]["kb"] == {"dataset": kb_local.DATASET, "cell": r["cell"]}
        assert r["link"] == kb.cell_link(r["cell"])
        assert r["example_links"] == [kb.record_link(x) for x in r["examples"]]
        assert bendjson.check({"bends": [r["recipe"]]})["problems"] == []


# ---------------------------------------------------------------- "kb" and the plugin version (comfy_canvas.fit_bends)
import comfy_canvas as cc  # noqa: E402

KB_DOC = json.dumps({"bends": [ROT, OFF], "version": 1.2}, indent=1)
OLD_NODE = {"ok": True, "change_hash": "x", "warnings": ["bend #0 (probe): unknown key 'kb' (ignored)"]}


def _graphs(text=KB_DOC):
    api = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
           "5": {"class_type": "ApplyBendsFromJSON", "inputs": {"model": ["1", 0], "bends_json": text, "strict": True}}}
    ui = {"nodes": [{"id": 5, "type": "ApplyBendsFromJSON", "widgets_values": [text, True, 0, 0, 0, 0, "safe", ""]}]}
    return api, ui


def _probe(monkeypatch, answer):
    calls = []

    def req(method, path, body=None, **kw):
        calls.append((method, path, json.loads(body) if body else None))
        if isinstance(answer, Exception):
            raise answer
        return answer
    monkeypatch.setattr(cc, "_KB_OK", {})
    monkeypatch.setattr(cc, "_req", req)
    return calls


def test_bends_without_kb_are_sent_as_they_are_without_asking(monkeypatch):
    calls = _probe(monkeypatch, OLD_NODE)
    api, ui = _graphs(json.dumps({"bends": [{k: v for k, v in ROT.items() if k != "kb"}]}))
    assert cc.fit_bends(api, ui) == (api, ui, 0) and calls == []


def test_a_plugin_that_knows_kb_gets_it_and_is_asked_once(monkeypatch):
    calls = _probe(monkeypatch, {"ok": True, "change_hash": "x", "warnings": []})
    api, ui = _graphs()
    assert cc.fit_bends(api, ui) == (api, ui, 0)
    assert cc.fit_bends(api, ui)[2] == 0 and len(calls) == 1
    method, path, body = calls[0]
    assert (method, path) == ("POST", "/web_bend_demo/selection")
    assert body["session_id"] == cc.KB_PROBE_SESSION and "kb" in body["bends"][0]  # never the user's own session


@pytest.mark.parametrize("answer", ["warning", "no route"])
def test_older_plugins_get_the_bends_without_kb(monkeypatch, answer):
    _probe(monkeypatch, OLD_NODE if answer == "warning" else
           cc.CanvasError("POST /web_bend_demo/selection -> HTTP 404: Not Found"))
    api, ui = _graphs()
    api2, ui2, n = cc.fit_bends(api, ui)
    assert n == 2
    sent = json.loads(api2["5"]["inputs"]["bends_json"])
    assert [b.get("kb") for b in sent["bends"]] == [None, None]
    assert sent["bends"][0]["label"] == "recompose" and sent["bends"][1]["t"] == [1.0, 0.67]  # nothing else changes
    assert api2["5"]["inputs"]["strict"] is True and api2["1"] is api["1"]
    assert "kb" not in json.loads(ui2["nodes"][0]["widgets_values"][0])["bends"][0]
    assert ui2["nodes"][0]["widgets_values"][1:] == ui["nodes"][0]["widgets_values"][1:]
    assert api["5"]["inputs"]["bends_json"] == KB_DOC  # the caller's graphs (and the skill's results) keep kb


def test_queueing_on_an_older_plugin_sends_no_kb(monkeypatch):
    sent = []

    def req(method, path, body=None, **kw):
        if path == "/web_bend_demo/selection":
            return OLD_NODE
        sent.append(json.loads(body))
        return {"prompt_id": "p1"}
    monkeypatch.setattr(cc, "_KB_OK", {})
    monkeypatch.setattr(cc, "_req", req)
    api, ui = _graphs()
    assert cc.queue_prompt(api, ui) == {"prompt_id": "p1", "status": "queued"}
    assert '"kb"' not in json.dumps(sent[0])


def test_an_unreachable_comfyui_is_an_error_not_an_answer(monkeypatch):
    _probe(monkeypatch, cc.CanvasError("cannot reach ComfyUI at http://127.0.0.1:8188: refused"))
    with pytest.raises(cc.CanvasError, match="cannot reach"):
        cc.fit_bends(*_graphs())
    assert cc._KB_OK == {}  # asked again next time
