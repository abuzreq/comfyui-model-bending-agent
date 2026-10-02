#!/usr/bin/env python3
"""The user's own bend knowledge base and the community one, as the skill uses them.

- log_round: turn a finished ComfyUI run into a record in the user's own knowledge base (the work folder's kb/). The
  prompt and input image are kept in a private sidecar (private.json) that is never shared; the record itself follows
  the community format, so it can be shared later only if the user agrees (sharing.json).
- find: rank cells for a goal from the community knowledge base, the user's own, or both.
- community: the snapshot bundled with the skill (data/kb/community/), or the latest index from the Hugging Face
  dataset, fetched without a token and cached in the work folder.

Script use (Claude Code):
  python kb_local.py log <prompt_id> --session S [--verdict kept] [--words "..."] [--baseline <prompt_id>]
  python kb_local.py find "more abstract" --arch sd15 --sources community,mine
  python kb_local.py status
"""

from __future__ import annotations

import io
import json
import os
import urllib.request
from pathlib import Path

import comfy_canvas as cc
import kb

WORK = Path(cc._env("COMFY_BENDING_WORKDIR") or Path.home() / ".comfyui-model-bending").expanduser()
OWN = WORK / "kb"
CACHE = WORK / "kb_community"
SESSIONS = WORK / "sessions"
SNAPSHOT = kb.DATA / "community"
DATASET = os.environ.get("BEND_KB_DATASET", "abuzreq/model-bending-knowledge-base")
HF = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
SOURCES = ("community", "mine")
DEGENERATE_FLAGS = {"blob", "noise", "flat", "clipped"}


# --------------------------------------------------------------------------- logging a round
def _history(pid: str) -> dict:
    h = cc._req("GET", f"/api/history/{pid}")
    if pid not in h:
        raise cc.CanvasError(f"no finished run {pid!r} in ComfyUI's history")
    if (h[pid].get("status") or {}).get("status_str") == "error":
        raise cc.CanvasError(f"run {pid!r} failed in ComfyUI; only finished runs can be recorded")
    return h[pid]


def _resolved_json(entry: dict) -> str | None:
    """Apply Bends from JSON's resolved_json, if a PreviewAny / report sink showed it."""
    for out in (entry.get("outputs") or {}).values():
        for t in out.get("text") or []:
            try:
                d = json.loads(t)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(d, dict) and isinstance(d.get("bends"), list):
                return t
    return None


def _first_image_url(pid: str, entry: dict) -> str | None:
    imgs = cc.describe_run(pid, entry)["images"]
    return imgs[0]["url"] if imgs else None


def _webp(data: bytes, max_side: int = 768) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=85)
    return buf.getvalue()


def log_round(prompt_id: str, session: str, verdict: str = "", words: str = "", caption: str = "",
              change: str = "", effect_tags: list[str] | None = None, agent_model: str = "",
              baseline_prompt_id: str = "", arch: str = "", contributor: str = "") -> dict:
    """Record a finished bent run in the user's own knowledge base. verdict/words are the user's (human);
    caption/change/effect_tags are the agent's (AI: agent_model names the model and is then required)."""
    if (caption or change or effect_tags) and not agent_model:
        raise ValueError("agent_model is required when the agent adds a caption, change or effect tags")
    entry = _history(prompt_id)
    model, setup, bends = kb.setup_from_api_prompt(entry["prompt"][2], _resolved_json(entry))
    if arch:
        model["arch"] = arch
    if not bends:
        raise ValueError("this run has no bends (an unbent run is a baseline: pass it as baseline_prompt_id)")
    private = {k: setup.get(k) for k in ("prompt", "negative", "input_image") if setup.get(k)}
    rec = kb.make_record(source="session", model=model, setup=setup, bends=bends,
                         consent={"prompt": False, "input_image": False}, salt=kb.local_salt(OWN / "salt"),
                         contributor=contributor or None,
                         provenance={"producer": "comfyui-model-bending skill", "session": session,
                                     "prompt_id": prompt_id})
    url = _first_image_url(prompt_id, entry)
    if not url:
        raise ValueError("the run produced no image")
    out_bytes = cc.fetch_bytes(url)
    measurements = {}
    if baseline_prompt_id:
        import metrics
        b_entry = _history(baseline_prompt_id)
        b_url = _first_image_url(baseline_prompt_id, b_entry)
        if b_url:
            bid = kb.baseline_id(rec)
            kb.save_baseline(OWN, bid, _webp(cc.fetch_bytes(b_url)))
            rec["outputs"]["baseline_id"] = bid
            row = metrics.compare(b_url, [url])[1]
            measurements = {"mae_vs_baseline": row["mae"], "std": row["std"], "hf_ratio": row["hf_ratio"],
                            "degenerate": kb.measure("degenerate", bool(DEGENERATE_FLAGS & set(row["flags"])),
                                                     flags=row["flags"])}
    ints = []
    target = {"record": rec["id"]}
    who = kb.human(contributor or "the user")
    if verdict:
        ints.append(kb.make_interpretation(target, "verdict", verdict, who, "the user's choice in the session"))
    if words:
        ints.append(kb.make_interpretation(target, "note", words, who, "the user's own words"))
    agent = kb.ai(agent_model, "skill-log-round-v1", "comfyui-model-bending skill") if agent_model else None
    vocab = kb.load_vocab()["tags"]
    for kind, val in (("caption", caption), ("change", change),
                      ("effect_tags", [t for t in (effect_tags or []) if t in vocab])):
        if val:
            ints.append(kb.make_interpretation(target, kind, val, agent, "output (and baseline, when given)"))
    d = kb.save_record(OWN, rec, {"output.webp": _webp(out_bytes)},
                       kb.make_measurements(rec["id"], measurements) if measurements else None, ints)
    if private:
        (d / "private.json").write_text(json.dumps(private, indent=1, ensure_ascii=False), encoding="utf-8")
    if not (d / "sharing.json").exists():
        (d / "sharing.json").write_text(json.dumps({"share_ok": False, "prompt": False, "input_image": False}),
                                        encoding="utf-8")
    skipped = [t for t in (effect_tags or []) if t not in vocab]
    return {"record": rec["id"], "folder": str(d), "cell": kb.cell_key(kb.cell_fields(rec)) if kb.cell_fields(rec)
            else None, "bends": [{k: b.get(k) for k in ("path", "op", "args", "window", "bucket")} for b in rec["bends"]],
            "interpretations": len(ints), "shared": False,
            **({"unknown_tags_dropped": skipped} if skipped else {})}


# --------------------------------------------------------------------------- sources
def session_sources(session: str, sources: list[str] | None = None) -> list[str] | None:
    """Read (or, with sources, store) which knowledge the user chose for a session: community, mine, both, none."""
    p = SESSIONS / f"{session}.json"
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    if sources is not None:
        bad = [s for s in sources if s not in SOURCES]
        if bad:
            raise ValueError(f"unknown source(s) {bad}; use community, mine, or an empty list for neither")
        data.update(kb_sources=list(sources), decided=kb.now())
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data.get("kb_sources")


def _fetch(name: str) -> Path | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(f"{HF}/index/{name}", timeout=20) as r:
            data = r.read()
    except Exception:  # noqa: BLE001 - offline or not published yet: the snapshot serves
        return None
    (CACHE / name).write_bytes(data)
    return CACHE / name


def _meta(folder: Path) -> dict:
    p = folder / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def community_index(refresh: bool = False) -> tuple[Path, str]:
    """Folder holding cells.jsonl / findings.jsonl / meta.json, and where it came from: the latest index from the
    dataset (refresh, or a newer cached copy), else the snapshot bundled with the skill."""
    if refresh and all(_fetch(n) for n in ("cells.jsonl", "findings.jsonl", "meta.json")):
        return CACHE, f"latest from huggingface.co/datasets/{DATASET}"
    if (CACHE / "cells.jsonl").exists() and _meta(CACHE).get("built", "") >= _meta(SNAPSHOT).get("built", ""):
        return CACHE, f"cached from huggingface.co/datasets/{DATASET} ({_meta(CACHE).get('built')})"
    return SNAPSHOT, f"snapshot bundled with the skill ({_meta(SNAPSHOT).get('built', 'empty')})"


def own_index() -> Path:
    """The user's own knowledge base, re-aggregated (cheap: it is small)."""
    if not (OWN / "records").exists():
        return OWN / "index"
    kb.build_index(OWN)
    return OWN / "index"


def find(goal: str, arch: str = "", sources: list[str] | None = None, limit: int = 6, refresh: bool = False) -> dict:
    sources = list(sources or [])
    if not sources:
        return {"sources": [], "results": [], "note": "the user chose not to use stored knowledge; explore fresh"}
    out = {"goal": goal, "sources": {}, "results": []}
    for src in sources:
        if src == "community":
            idx, where = community_index(refresh)
        else:
            idx, where = own_index(), f"your own records in {OWN}"
        cells, findings = kb.read_jsonl(idx / "cells.jsonl"), kb.read_jsonl(idx / "findings.jsonl")
        res = kb.rank_cells(cells, goal, arch=arch or None, findings=findings, limit=limit)
        out["sources"][src] = {"from": where, "cells": len(cells), "considered": res["considered"],
                               "note": res.get("note"), "general_findings": res.get("general_findings", [])}
        out["effect_tags"] = res["effect_tags"]
        for r in res["results"]:
            r["source"] = src
            if src == "community":
                r["example_urls"] = [example_url(cells, r["cell"], rid) for rid in r["examples"]]
        out["results"] += res["results"]
    out["results"].sort(key=lambda r: -r["score"])
    out["results"] = out["results"][:limit]
    if not out["results"]:
        out["note"] = ("nothing matched; the knowledge base has no records for this goal on this model family yet. "
                       "Explore with the general heuristics and log what works (log_round).")
    return out


def example_url(cells: list[dict], cell_key: str, rid: str) -> str | None:
    """A public image URL for a community example record."""
    c = next((c for c in cells if c["key"] == cell_key), None)
    path = ((c or {}).get("example_paths") or {}).get(rid)
    return f"{HF}/{path}/output.webp" if path else None


def describe_cell(cell_key: str, sources: list[str] | None = None) -> dict:
    for src in sources or list(SOURCES):
        idx = community_index()[0] if src == "community" else own_index()
        c = next((c for c in kb.read_jsonl(idx / "cells.jsonl") if c["key"] == cell_key), None)
        if c:
            return {"source": src, **c}
    raise ValueError(f"no cell {cell_key!r} in {sources or list(SOURCES)}")


def status(session: str = "") -> dict:
    idx, where = community_index()
    meta = json.loads((idx / "meta.json").read_text(encoding="utf-8")) if (idx / "meta.json").exists() else {}
    own = sum(1 for _ in kb.iter_records(OWN)) if (OWN / "records").exists() else 0
    shared = sum(1 for d, _ in kb.iter_records(OWN)
                 if json.loads((d / "sharing.json").read_text(encoding="utf-8")).get("share_ok")) if own else 0
    return {"community": {"from": where, **{k: meta.get(k) for k in ("built", "records", "cells", "findings",
                                                                     "families")}},
            "mine": {"folder": str(OWN), "records": own, "marked_for_sharing": shared},
            **({"session_sources": session_sources(session)} if session else {})}


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("prompt_id")
    lg.add_argument("--session", required=True)
    for k in ("verdict", "words", "caption", "change", "agent-model", "baseline", "arch", "contributor"):
        lg.add_argument(f"--{k}", default="")
    lg.add_argument("--tags", default="", help="comma-separated effect tags (the agent's)")
    fd = sub.add_parser("find")
    fd.add_argument("goal")
    fd.add_argument("--arch", default="")
    fd.add_argument("--sources", default="community,mine")
    fd.add_argument("--limit", type=int, default=6)
    fd.add_argument("--refresh", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("--session", default="")
    so = sub.add_parser("sources", help="store a session's choice: community, mine, both, none")
    so.add_argument("session")
    so.add_argument("choice", choices=("community", "mine", "both", "none"))
    a = ap.parse_args(argv)
    if a.cmd == "log":
        r = log_round(a.prompt_id, a.session, a.verdict, a.words, a.caption, a.change,
                      [t for t in a.tags.split(",") if t], a.agent_model, a.baseline, a.arch, a.contributor)
    elif a.cmd == "find":
        r = find(a.goal, a.arch, [s for s in a.sources.split(",") if s], a.limit, a.refresh)
    elif a.cmd == "sources":
        r = session_sources(a.session, {"both": ["community", "mine"], "none": []}.get(a.choice, [a.choice]))
    else:
        r = status(a.session)
    print(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
