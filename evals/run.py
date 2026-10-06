#!/usr/bin/env python3
"""Scenario suite for the comfyui-model-bending skill: how does the agent behave in realistic conversations?

Each scenario in evals/scenarios/ is a small play with three roles, all headless `claude -p` sessions:

  the agent    a fresh Claude Code session in an empty folder that holds only the BUILT skill (dist/*.zip), with
               the BUILT comfyui-bending MCP server (dist/_mcpb) as its only way to reach ComfyUI. No shell: this is
               the Claude app's situation. It talks to your real ComfyUI and renders real pictures.
  the artist   a simulated person with a persona and a goal, who answers whatever the agent says. They see what a
               person would see: the agent's messages, the in-chat pick board and the pictures.
  the judge    reads the finished conversation and marks each expected behaviour met / partly / not met, with
               evidence. You have the last word: everything it read is in the report.

The Claude app's in-chat views (pick board, picture box) cannot be drawn by Claude Code, so this runner plays the
artist's side of them: it opens the board, shows its pictures to the artist and stores their pick, as the view would.

  python evals/run.py --list
  python evals/run.py                         every scenario
  python evals/run.py --tag smoke             the short ones
  python evals/run.py --only hello-unclear,surprise-again
  python evals/run.py --report evals/runs/<run>      rebuild the report of an earlier run
  python evals/run.py --rejudge evals/runs/<run>     judge an earlier run again

Needs: ComfyUI running with ComfyUI-Model-Bending, the `claude` CLI signed in, `uv`, and a build
(`python tools/build_dist.py`; --build runs it first). Results: evals/runs/<run>/report.html.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
SCENARIOS = EVALS / "scenarios"
DIST = ROOT / "dist"
STAGE = DIST / "_mcpb"  # the unpacked extension: what the .mcpb installs
SKILL_ZIP = DIST / "comfyui-model-bending.zip"
SERVER = "comfyui-bending"
PREFIX = f"mcp__{SERVER}__"
AGENT_TOOLS = "Read,Glob,Grep,Write,Edit,Skill,ToolSearch"  # no shell, no web: the Claude app's sandbox
USER_CLIENT = "scenario-artist"  # runs queued for the artist are "their own" runs (not the agent's)
# Claude Code tells its model that it runs in a terminal. The scenarios play the Claude app, so the agent is told that.
APP_NOTE = ("Environment for this session: you are the assistant in the Claude desktop app's chat, not in a terminal "
            "and not in an IDE. The person you talk to is not a developer. They see your chat messages, the in-chat "
            "views that MCP tools open (such as a pick board or a picture box), and pictures that tools return. They "
            "cannot open file paths on this computer, so never link to or embed local files. You have no shell. "
            "Files you write yourself, such as a session log, go in the current folder.")
VIEW_ID = re.compile(r"\b((?:board|picture)_[0-9a-f]{6,32})\b")
VERDICTS = ("met", "partly", "not_met", "not_reached")

ARTIST_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "what you type in the chat (may be empty when you only click on a board)"},
        "board": {"type": ["object", "null"], "description": "your answer on the open pick board, or null",
                  "properties": {"picks": {"type": "array", "items": {"type": "string"}},
                                 "none": {"type": "boolean"},
                                 "keep": {"type": "array", "items": {"type": "string"}},
                                 "change": {"type": "array", "items": {"type": "string"}},
                                 "push": {"type": ["integer", "null"]},
                                 "direction": {"type": "string"}}},
        "drop_picture": {"type": "boolean", "description": "true to drop your picture into the open picture box"},
        "done": {"type": "boolean", "description": "true when the conversation has reached its end for you"},
        "why": {"type": "string", "description": "one line for the test report: why you answered this way"}},
    "required": ["message", "done", "why"]}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {
            "type": "object",
            "properties": {"n": {"type": "integer"}, "verdict": {"type": "string", "enum": list(VERDICTS)},
                           "evidence": {"type": "string"}, "note": {"type": "string"}},
            "required": ["n", "verdict", "evidence"]}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"}},
    "required": ["items", "observations", "summary"]}


class SuiteError(RuntimeError):
    pass


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ the skill's code
_skill = None


def skill(state: Path):
    """(comfy_canvas, board) from the BUILT extension, for setup runs and for playing the in-chat views."""
    global _skill
    if _skill is None:
        os.environ.setdefault("COMFYUI_URL", "http://127.0.0.1:8188")
        os.environ["COMFY_BENDING_WORKDIR"] = str(state)
        sys.path.insert(0, str(STAGE / "scripts"))
        import board as boards  # noqa: PLC0415
        import comfy_canvas as cc  # noqa: PLC0415
        _skill = (cc, boards)
    return _skill


# ------------------------------------------------------------------------------------------------ claude -p
def claude_bin() -> str:
    exe = shutil.which("claude")
    if not exe:
        raise SuiteError("the `claude` command was not found: install Claude Code and sign in")
    return exe


def run_claude(args: list[str], prompt: str, cwd: Path, raw: Path | None = None, on_event=None,
               timeout: float = 2400, env: dict | None = None) -> list[dict]:
    """One headless call. The prompt goes in on stdin; stream-json events come back one per line."""
    cmd = [claude_bin(), "-p", *args]
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ, **(env or {})})
    killed = threading.Event()

    def watchdog():
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            killed.set()
            proc.kill()

    err: list[bytes] = []
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True).start()
    proc.stdin.write(prompt.encode("utf-8"))
    proc.stdin.close()
    events: list[dict] = []
    out = raw.open("w", encoding="utf-8") if raw else None
    try:
        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if out:
                out.write(text + "\n")
            try:
                ev = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict):
                events.append(ev)
                if on_event:
                    on_event(ev)
    finally:
        if out:
            out.close()
    proc.wait()
    if killed.is_set():
        raise SuiteError(f"timed out after {int(timeout)} s")
    if not events:
        raise SuiteError(f"claude gave no output (exit {proc.returncode}): {b''.join(err).decode('utf-8', 'replace')[-600:]}")
    return events


def ask_json(prompt: str, schema: dict, cwd: Path, model: str, system: str, raw: Path | None = None,
             timeout: float = 600) -> tuple[dict, float]:
    """A role-play call (artist, judge) that answers with one JSON object. It may only read files (pictures)."""
    empty = cwd / "no_mcp.json"
    empty.write_text('{"mcpServers": {}}', encoding="utf-8")
    args = ["--output-format", "json", "--model", model, "--mcp-config", str(empty), "--strict-mcp-config",
            "--setting-sources", "project", "--tools", "Read", "--no-session-persistence",
            "--disable-slash-commands", "--system-prompt", system, "--json-schema", json.dumps(schema)]
    last = None
    for _ in range(2):
        events = run_claude(args, prompt, cwd, raw, timeout=timeout)
        res = events[-1]
        cost = float(res.get("total_cost_usd") or 0)
        got = res.get("structured_output")
        if not isinstance(got, dict):
            m = re.search(r"\{.*\}", str(res.get("result") or ""), re.S)
            try:
                got = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                got = None
        if isinstance(got, dict):
            return got, cost
        last = res
    raise SuiteError(f"no JSON answer: {str((last or {}).get('result'))[:300]}")


# ------------------------------------------------------------------------------------------------ setup
def _base_spec(prompt: str, seed: int, prefix: str) -> dict:
    spec = json.loads((STAGE / "presets" / "base_sd15.spec.json").read_text(encoding="utf-8"))
    spec["meta"] = {"note": "a picture the artist made themselves"}
    spec["groups"] = [{"id": "base", "title": "My picture", "color": "#444444"}]
    n = spec["nodes"]
    n["pos"]["inputs"]["text"] = prompt
    n["ks"]["inputs"]["seed"] = seed
    n["save"]["title"] = "Save Image"
    n["save"]["inputs"]["filename_prefix"] = prefix
    return spec


def _queue(cc, spec: dict, client: str) -> dict:
    ui, api = cc.build(spec)
    payload = {"prompt": api, "client_id": client, "extra_data": {"extra_pnginfo": {"workflow": ui}}}
    r = cc._req("POST", "/api/prompt", json.dumps(payload).encode())
    if r.get("node_errors"):
        raise SuiteError(f"setup run refused: {json.dumps(r['node_errors'])[:600]}")
    run = cc._wait_for(r["prompt_id"], 600)
    if not run or run.get("status") != "success" or not run.get("images"):
        raise SuiteError(f"setup run failed: {json.dumps(run)[:600]}")
    return run


def do_setup(sc: dict, sdir: Path, state: Path) -> list[str]:
    """Put the artist's world in place: a run of their own in ComfyUI's history, a picture file of theirs."""
    cc, _ = skill(state)
    notes = []
    for step in sc.get("setup") or []:
        do = step["do"]
        if do == "user_run":
            spec = _base_spec(step["prompt"], int(step.get("seed", 1234)), step.get("prefix", "scenario_artist/piece"))
            if step.get("kind") == "bent":  # like a run made in the bending web UI: the model is already bent
                spec["nodes"]["mybend"] = {
                    "class_type": "ApplyBendsFromJSON", "group": "base", "title": "My bends",
                    "inputs": {"model": ["msd", 0], "bends_json": json.dumps(step.get("bends") or {"bends": [
                        {"path": "output_blocks.8", "module_type": "multiply", "module_args": {"scalar": 1.3}}]})}}
                spec["nodes"]["ks"]["inputs"]["model"] = ["mybend", 0]
            run = _queue(cc, spec, USER_CLIENT)
            (sdir / "artist").mkdir(exist_ok=True)
            pic = sdir / "artist" / "my_last_render.png"
            pic.write_bytes(cc.fetch_bytes(run["images"][0]["url"]))
            notes.append(f"the artist's own run is in ComfyUI's history ({step.get('kind', 'plain')}; "
                         f"prompt: {step['prompt'][:90]}); their picture: {pic.name}")
        elif do == "picture":
            run = _queue(cc, _base_spec(step["prompt"], int(step.get("seed", 77)), "agent_bending/scenario_fixture"),
                         "agent-model-bending")
            (sdir / "artist").mkdir(exist_ok=True)
            pic = sdir / "artist" / step.get("name", "my_painting.png")
            pic.write_bytes(cc.fetch_bytes(run["images"][0]["url"]))
            notes.append(f"the artist has a picture file of their own: {pic.name}")
        else:
            raise SuiteError(f"unknown setup step {do!r}")
    return notes


# ------------------------------------------------------------------------------------------------ the agent
class Turn:
    """One user message and everything the agent did in answer to it."""

    def __init__(self, n: int, user: str, sdir: Path, state: Path, why: str = ""):
        self.n, self.user, self.why = n, user, why
        self.sdir, self.state = sdir, state
        self.blocks: list[dict] = []  # in order: {"text": …} or {"call": id}
        self.calls: dict[str, dict] = {}
        self.boards: list[dict] = []
        self.boxes: list[dict] = []
        self.session = ""
        self.cost = 0.0
        self.seconds = 0.0
        self.error = ""
        self.denied: list[str] = []
        self.init: dict = {}
        self._pics = 0

    # -- stream events
    def on_event(self, ev: dict) -> None:
        t = ev.get("type")
        if t == "system" and ev.get("subtype") == "init":
            self.session = ev.get("session_id") or self.session
            self.init = {"model": ev.get("model"), "skills": ev.get("skills"), "mcp": ev.get("mcp_servers"),
                         "tools": len(ev.get("tools") or [])}
        elif t == "assistant":
            for b in (ev.get("message") or {}).get("content") or []:
                if b.get("type") == "text" and (b.get("text") or "").strip():
                    self.blocks.append({"text": b["text"].strip()})
                elif b.get("type") == "tool_use":
                    name = b.get("name") or ""
                    self.calls[b["id"]] = {"name": name.removeprefix(PREFIX), "mcp": name.startswith(PREFIX),
                                           "input": b.get("input") or {}, "result": "", "images": [], "error": False}
                    self.blocks.append({"call": b["id"]})
        elif t == "user":
            content = (ev.get("message") or {}).get("content")
            for b in content if isinstance(content, list) else []:
                if b.get("type") == "tool_result" and b.get("tool_use_id") in self.calls:
                    self._result(self.calls[b["tool_use_id"]], b)
        elif t == "result":
            self.session = ev.get("session_id") or self.session
            self.cost = float(ev.get("total_cost_usd") or 0)
            self.seconds = float(ev.get("duration_ms") or 0) / 1000
            if ev.get("is_error"):
                self.error = str(ev.get("result") or ev.get("subtype") or "error")[:500]
            self.denied = sorted({str(d.get("tool_name")) for d in ev.get("permission_denials") or []})

    def _result(self, call: dict, block: dict) -> None:
        call["error"] = bool(block.get("is_error"))
        content, texts = block.get("content"), []
        for part in content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]:
            if part.get("type") == "text":
                texts.append(part.get("text") or "")
            elif part.get("type") == "image":
                src = part.get("source") or {}
                if src.get("type") == "base64" and src.get("data"):
                    self._pics += 1
                    ext = {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(src.get("media_type"), ".png")
                    p = self.sdir / "pictures" / f"t{self.n}_{call['name']}_{self._pics}{ext}"
                    p.parent.mkdir(exist_ok=True)
                    p.write_bytes(base64.b64decode(src["data"]))
                    call["images"].append(p.name)
            elif part.get("type") == "tool_reference":
                texts.append(f"[loaded {part.get('tool_name', '').removeprefix(PREFIX)}]")
        call["result"] = "\n".join(t for t in texts if t)
        if call["name"] in ("show_board", "ask_for_picture") and not call["error"]:
            self._view(call)

    def _view(self, call: dict) -> None:
        """Play the Claude app's side of an in-chat view: it opens at once (it fetches its first picture)."""
        m = VIEW_ID.search(call["result"])
        if not m:
            return  # the tool said the view cannot be shown: the agent has to use its fallback
        _, boards = skill(self.state)
        try:
            boards.mark_opened(self.state / "boards", m.group(1))
        except Exception as e:  # noqa: BLE001
            call["result"] += f"\n[runner: could not open the view: {e}]"
            return
        if call["name"] == "show_board":
            self.boards.append({"id": m.group(1), "input": call["input"], "pictures": {}})
        else:
            self.boxes.append({"id": m.group(1), "purpose": call["input"].get("purpose", "")})

    # -- after the turn
    def fetch_board_pictures(self) -> None:
        cc, _ = skill(self.state)
        for b in self.boards:
            items = [("original", b["input"].get("original_url"))] + [
                (str(c.get("label") or chr(65 + i)), c.get("image_url")) for i, c in enumerate(b["input"].get("candidates") or [])]
            for label, url in items:
                if not url:
                    continue
                try:
                    data = cc.fetch_bytes(url)
                    name = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("filename", ["x.png"])[0]
                    p = self.sdir / "pictures" / f"t{self.n}_board_{re.sub(r'[^A-Za-z0-9]+', '_', label)}{Path(name).suffix or '.png'}"
                    p.parent.mkdir(exist_ok=True)
                    p.write_bytes(data)
                    b["pictures"][label] = p.name
                except Exception as e:  # noqa: BLE001
                    b["pictures"][label] = ""
                    b.setdefault("problems", []).append(f"{label}: {e}")

    @property
    def text(self) -> str:
        return "\n\n".join(b["text"] for b in self.blocks if "text" in b)

    def data(self) -> dict:
        return {"n": self.n, "user": self.user, "why": self.why, "blocks": self.blocks, "calls": self.calls,
                "boards": self.boards, "boxes": self.boxes, "cost": self.cost, "seconds": self.seconds,
                "error": self.error, "denied": self.denied, "init": self.init}


def agent_turn(turn: Turn, adir: Path, mcp: Path, model: str | None, effort: str | None, timeout: float) -> None:
    args = ["--output-format", "stream-json", "--verbose", "--mcp-config", str(mcp), "--strict-mcp-config",
            "--setting-sources", "project", "--permission-mode", "acceptEdits", "--permission-prompts", "none",
            "--tools", AGENT_TOOLS, "--allowedTools", f"mcp__{SERVER}", "--append-system-prompt", APP_NOTE]
    if model:
        args += ["--model", model]
    if effort:
        args += ["--effort", effort]
    if turn.session:
        args += ["--resume", turn.session]
    raw = turn.sdir / "raw" / f"agent_turn{turn.n}.jsonl"
    raw.parent.mkdir(exist_ok=True)
    try:
        run_claude(args, turn.user, adir, raw, turn.on_event, timeout, env={"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"})
    except SuiteError as e:
        turn.error = turn.error or str(e)
    if turn.init and not any(s.get("name") == SERVER and s.get("status") == "connected" for s in turn.init.get("mcp") or []):
        turn.error = turn.error or f"the {SERVER} MCP server did not connect: {turn.init.get('mcp')}"
    turn.fetch_board_pictures()


# ------------------------------------------------------------------------------------------------ the artist
ARTIST_SYSTEM = ("You are playing a person in a test conversation. Stay in character at all times. You are the "
                 "human user, never the assistant. Answer with the JSON object you are asked for.")


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def board_sentence(fb: dict, board_id: str, push) -> str:
    """The chat message the pick board sends with an answer (scripts/ui/board.html, sentence())."""
    parts = []
    if fb["picks"]:
        parts.append(f"I pick {fb['picks'][0]}." if len(fb["picks"]) == 1 else f"I like {_and(fb['picks'])}.")
    elif fb["none"]:
        parts.append("None of these.")
    if fb["keep"]:
        parts.append(f"Keep the {_and(fb['keep'])}.")
    if fb["change"]:
        parts.append(f"Change the {_and(fb['change'])}.")
    if push in (-1, 0, 1):
        parts.append({-1: "Pull it back.", 0: "Keep the strength about the same.", 1: "Push it further."}[push])
    if fb["direction"]:
        parts.append(f'"{fb["direction"]}"')
    return f"{' '.join(parts)} (my answer on board {board_id})"


def _labels(picks: list, labels: list[str]) -> list[str]:
    """The board's own labels for what the artist picked ("C" for a version labelled "C · wild card")."""
    out = []
    for p in (str(x).strip() for x in picks):
        hit = next((l for l in labels if l == p), None) \
            or next((l for l in labels if l.lower() == p.lower()), None) \
            or next((l for l in labels if p and re.split(r"[^A-Za-z0-9]+", l.strip())[0].lower() == re.split(r"[^A-Za-z0-9]+", p)[0].lower()), None)
        if hit and hit not in out:
            out.append(hit)
    return out


def artist_prompt(sc: dict, turns: list[Turn], sdir: Path, left: int) -> str:
    a, last = sc["artist"], turns[-1]
    lines = ["You are playing an artist who is chatting with an AI studio assistant. The assistant helps people "
             '"bend" image models in ComfyUI (it nudges the inside of the model while it paints). This is a test of '
             "the assistant: play your part as a real person would.", "",
             f"Who you are: {a['who']}", f"What you want: {a['wants']}", f"How you talk: {a.get('style', 'short, plain chat messages')}"]
    if a.get("rules"):
        lines += ["Your rules for this conversation:"] + [f"- {r}" for r in a["rules"]]
    lines += [f"You are done when: {a.get('ends_when', 'you got what you came for, or the assistant has nothing more for you')}", "",
              "How to play:",
              "- Write what this person would type: short chat messages, their words, their level of knowledge. Do not "
              "be more helpful, patient or technical than they would be. Never mention tests, tools or JSON.",
              "- Before you judge pictures, LOOK at them: use the Read tool on each picture path listed below. Decide "
              "from what you see, with this person's taste.",
              "- If a pick board is open, answer on it through `board`, as you would click: `picks` holds the labels you "
              "choose (one or several), written exactly as listed below, or set `none` true. The rest is optional, "
              "and a person in a hurry leaves it alone: `keep` / `change` take words from: composition, subject, "
              "palette, lighting, texture, style; `push` is -1 (pull it back), 0 (about the same) or 1 (push it "
              "further), and null when you do not touch it; `direction` is a short note typed on the board. "
              "`message` may then stay empty. You may also ignore the board and just type, if that is what this "
              "person would do.",
              "- If a picture box is open and you want to give your picture, set `drop_picture` true.",
              "- If the assistant asks several things at once, answer what this person would bother to answer.",
              "- You are not sitting at ComfyUI during this conversation. You cannot open, run or check anything "
              "there, and you must not claim that you did. If asked to, say you will do it later, unless your rules "
              "above say otherwise.",
              "- Set `done` true when you are done (see above), or when the assistant has finished and you have nothing "
              "to add. A last short message is fine.",
              "- `why` is one line for the test report: why you answered this way. It is not shown to the assistant.", "",
              "The conversation so far:"]
    for t in turns:
        lines += [f"[you] {t.user}", f"[assistant] {t.text or '(no text)'}"]
    seen = []
    for b in last.boards:
        i = b["input"]
        seen.append(f'A pick board is open in the chat: "{i.get("title", "")}". It asks: "{i.get("question", "")}"')
        if b["pictures"].get("original"):
            seen.append(f"  - original: {sdir / 'pictures' / b['pictures']['original']}")
        for k, c in enumerate(i.get("candidates") or []):
            label = str(c.get("label") or chr(65 + k))
            pic = b["pictures"].get(label)
            mark = " (marked as the assistant's pick)" if c.get("recommended") else ""
            where = str(sdir / "pictures" / pic) if pic else "(picture missing)"
            seen.append(f"  - {label}{mark}: {where} — caption: \"" + str(c.get("caption", "")) + '"')
    for box in last.boxes:
        seen.append(f'A picture box is open in the chat: "{box["purpose"]}" (you can drop a picture into it)')
    loose = [(c["name"], img) for c in last.calls.values() if c["name"] != "show_board" for img in c["images"]]
    if loose:
        seen.append("Pictures that appeared in the assistant's working output this turn (you can open them):")
        seen += [f"  - {sdir / 'pictures' / img}" for _, img in loose]
    mine = sorted((sdir / "artist").glob("*.*")) if (sdir / "artist").is_dir() else []
    mine = [p for p in mine if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    lines += ["", "What you can see now:"] + (seen or ["(only the assistant's message; no board, no pictures)"])
    if mine:
        lines += ["", "Your own files on this computer (you can paste a path into the chat if asked):"] + [f"  - {p}" for p in mine]
    lines += ["", f"Turns you have left: {left}." + (" This is your last message: wrap up." if left <= 1 else "")]
    return "\n".join(lines)


def artist_reply(sc: dict, turns: list[Turn], sdir: Path, state: Path, model: str, left: int) -> tuple[str, str, bool, float]:
    """(the next user message, the artist's private reason, done, cost). Board picks and dropped pictures are stored
    as the in-chat views would store them."""
    cc, boards = skill(state)
    last = turns[-1]
    raw = sdir / "raw" / f"artist_after_turn{last.n}.json"
    ans, cost = ask_json(artist_prompt(sc, turns, sdir, left), ARTIST_SCHEMA, sdir, model, ARTIST_SYSTEM, raw)
    parts = []
    fb = ans.get("board")
    if last.boards and isinstance(fb, dict) and (fb.get("picks") or fb.get("none") or fb.get("direction")):
        b = last.boards[-1]
        push = fb.get("push") if fb.get("push") in (-1, 0, 1) else None
        labels = [str(c.get("label") or chr(65 + i)) for i, c in enumerate(b["input"].get("candidates") or [])]
        fb = {**fb, "picks": _labels(fb.get("picks") or [], labels)}
        clean = boards.submit(state / "boards", b["id"], {**fb, "push": push or 0})
        b["answer"] = {**clean, "push": push}
        parts.append(board_sentence(clean, b["id"], push))
    if last.boxes and ans.get("drop_picture"):
        box = last.boxes[-1]
        pics = sorted(p for p in (sdir / "artist").glob("*.*") if p.name != "my_last_render.png") or sorted((sdir / "artist").glob("*.*"))
        if pics:
            data = pics[0].read_bytes()
            up = cc.upload_bytes(data, pics[0].name, pics[0].stem)
            w, h = boards.picture_size(data)
            boards.record_picture(state / "boards", box["id"], {**up, "width": w, "height": h, "name": pics[0].name})
            box["dropped"] = pics[0].name
            parts.append(f'Here\'s my picture: "{pics[0].name}" (picture box {box["id"]}).')
    if (ans.get("message") or "").strip():
        parts.append(ans["message"].strip())
    return "\n".join(parts), str(ans.get("why") or ""), bool(ans.get("done")), cost


# ------------------------------------------------------------------------------------------------ the judge
JUDGE_SYSTEM = ("You review a recorded conversation between an AI assistant and a (simulated) artist against a list "
                "of expected behaviours. Be exact and fair: judge only from the record. Answer with the JSON object "
                "you are asked for.")


def clip(s, n: int) -> str:
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f" …[{len(s) - n} more characters]"


def transcript(rec: dict, sdir: Path, calls: bool = True) -> str:
    out = []
    for t in rec["turns"]:
        out.append(f"=== Turn {t['n']} ===\n[artist] {t['user']}")
        for b in t["blocks"]:
            if "text" in b:
                out.append(f"[assistant] {b['text']}")
            elif calls:
                c = t["calls"][b["call"]]
                line = f"    <call {c['name']} {clip(c['input'], 700)}>"
                res = clip(c["result"], 700) if c["result"] else ""
                pics = "".join(f" [picture: {sdir / 'pictures' / p}]" for p in c["images"])
                out.append(f"{line}\n    <result{' ERROR' if c['error'] else ''}: {res}{pics}>")
        for b in t["boards"]:
            shown = ", ".join(f"{k}: {sdir / 'pictures' / v}" for k, v in b["pictures"].items() if v)
            out.append(f"    <the pick board {b['id']} was displayed to the artist; pictures: {shown}>")
            if b.get("answer"):
                out.append(f"    <the artist answered on the board: {json.dumps(b['answer'])}>")
        for box in t["boxes"]:
            out.append(f"    <the picture box {box['id']} was displayed" + (f"; the artist dropped {box['dropped']}>" if box.get("dropped") else "; nothing was dropped>"))
        if t["denied"]:
            out.append(f"    <tools the assistant tried but may not use here: {', '.join(t['denied'])}>")
        if t["error"]:
            out.append(f"    <this turn ended with an error: {t['error']}>")
    return "\n".join(out)


def judge(sc: dict, rec: dict, sdir: Path, model: str) -> dict:
    exp = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(sc["expected"]))
    prompt = f"""An AI studio assistant uses a skill for "model bending" in ComfyUI (nudging the inside of an image model while it
paints). Below is one recorded test conversation with a simulated artist. The assistant could reach ComfyUI only
through MCP tools (shown as <call …>), as in the Claude app; the artist saw the assistant's messages, the pick boards
and the pictures, never the calls.

Scenario: {sc['title']}
What it tests: {sc.get('why', '')}
The artist: {sc['artist']['who']} They want: {sc['artist']['wants']}
Setup before the conversation: {'; '.join(rec.get('setup_notes') or []) or 'none'}
The conversation ended because: {rec.get('ended', '')}

Expected behaviours:
{exp}

For EACH expected behaviour give a verdict:
- "met": the record clearly shows it.
- "partly": some of it, or with a flaw worth knowing.
- "not_met": the record shows the opposite, or it should have happened and did not.
- "not_reached": the conversation never got to the point where it applies.
`evidence` is a short quote or the name of a call from the record (with its turn number). `note` says, in one
sentence, what was missing or off; leave it empty for a clean "met". Use `n` for the behaviour's number.

Then give up to six `observations`: things the skill's author would want to know that the list does not cover
(for example: jargon or numbers in chat with a non-technical artist, long messages, several questions at once, a
guess presented as a fact, a wasted render, a picture described wrongly, something done notably well). Where a
judgement depends on what a picture shows, open it with the Read tool before you judge. `summary` is one plain
sentence on how the assistant did.

The record:
{transcript(rec, sdir)}
"""
    ans, cost = ask_json(prompt, JUDGE_SCHEMA, sdir, model, JUDGE_SYSTEM, sdir / "raw" / "judge.json", timeout=900)
    by_n = {int(i.get("n", 0)): i for i in ans.get("items") or []}
    items = []
    for n, e in enumerate(sc["expected"], 1):
        i = by_n.get(n) or {}
        v = i.get("verdict") if i.get("verdict") in VERDICTS else "not_reached"
        items.append({"n": n, "expected": e, "verdict": v, "evidence": str(i.get("evidence") or ""), "note": str(i.get("note") or "")})
    return {"items": items, "observations": [str(o) for o in ans.get("observations") or []][:8],
            "summary": str(ans.get("summary") or ""), "cost": cost}


# ------------------------------------------------------------------------------------------------ one scenario
def run_scenario(sc: dict, run_dir: Path, opts) -> dict:
    sdir = run_dir / sc["id"]
    adir, state = sdir / "agent", sdir / "state"
    for d in (adir / ".claude" / "skills", state, sdir / "raw", sdir / "pictures"):
        d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SKILL_ZIP) as z:  # the built skill, and nothing else, is what the agent finds
        z.extractall(adir / ".claude" / "skills")
    mcp = sdir / "mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {SERVER: {
        "command": "uv", "args": ["run", "--directory", str(STAGE), "python", str(EVALS / "app_server.py")],
        "env": {"COMFYUI_URL": opts.comfyui, "COMFY_BENDING_WORKDIR": str(state)}}}}, indent=1), encoding="utf-8")
    rec = {"id": sc["id"], "title": sc["title"], "tags": sc.get("tags") or [], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "turns": [], "setup_notes": [], "ended": "", "error": "", "cost": {"agent": 0.0, "artist": 0.0, "judge": 0.0}}
    t0 = time.time()
    try:
        rec["setup_notes"] = do_setup(sc, sdir, state)
        turns: list[Turn] = []
        user, why, session, closing = sc["first_message"], "", "", False
        max_turns = min(opts.max_turns or 99, int(sc.get("max_turns", 4)))
        for n in range(1, max_turns + 1):
            turn = Turn(n, user, sdir, state, why)
            turn.session = session
            say(f"    turn {n}: artist: {clip(user, 110)}")
            agent_turn(turn, adir, mcp, opts.agent_model, opts.effort, opts.turn_timeout)
            session = turn.session
            turns.append(turn)
            rec["turns"] = [t.data() for t in turns]
            rec["cost"]["agent"] += turn.cost
            (sdir / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            mcp_calls = [c["name"] for c in turn.calls.values() if c["mcp"]]
            say(f"             agent: {len(mcp_calls)} tool calls, {len(turn.boards)} board(s), {int(turn.seconds)} s"
                + (f"  ERROR: {clip(turn.error, 160)}" if turn.error else ""))
            if turn.error:
                rec["ended"] = f"the agent's turn {n} failed: {turn.error}"
                rec["error"] = turn.error
                break
            if n == max_turns:
                rec["ended"] = (f"the artist was done after turn {n - 1} and the agent answered their last message"
                                if closing else f"the scenario's limit of {max_turns} turns was reached")
                break
            user, why, done, cost = artist_reply(sc, turns, sdir, state, opts.artist_model, max_turns - n)
            rec["cost"]["artist"] += cost
            if done and not user.strip():
                rec["ended"] = f"the artist was done after turn {n}" + (f" ({why})" if why else "")
                break
            if done:
                # a closing message still deserves an answer, then the play ends
                max_turns, closing = n + 1, True
    except SuiteError as e:
        rec["error"] = str(e)
        rec["ended"] = f"the run stopped: {e}"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["ended"] = f"the run stopped: {rec['error']}"
    rec["seconds"] = round(time.time() - t0)
    if not opts.no_judge and rec["turns"]:
        try:
            rec["judge"] = judge(sc, rec, sdir, opts.judge_model)
            rec["cost"]["judge"] = rec["judge"].pop("cost", 0.0)
        except Exception as e:  # noqa: BLE001
            rec["judge_error"] = f"{type(e).__name__}: {e}"
    (sdir / "record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


# ------------------------------------------------------------------------------------------------ the report
CSS = """
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--mut:#6b6b70;--line:#e2e0da;--me:#e8f0fe;--ai:#fff;--met:#1a7f4b;
--partly:#b7791f;--not:#c0392b;--nr:#7a7a80;--code:#f0efe9}
@media (prefers-color-scheme:dark){:root{--bg:#17171a;--card:#202024;--ink:#ececf0;--mut:#a0a0a8;--line:#34343a;
--me:#243352;--ai:#26262b;--code:#2c2c32}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 80px}h1{font-size:24px;margin:0 0 4px}h2{font-size:20px;margin:0}
h3{font-size:15px;margin:22px 0 8px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
a{color:inherit}.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:18px 0}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.chip{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;font-weight:600;color:#fff;white-space:nowrap}
.met{background:var(--met)}.partly{background:var(--partly)}.not_met{background:var(--not)}.not_reached{background:var(--nr)}
.tag{display:inline-block;border:1px solid var(--line);border-radius:99px;padding:0 8px;font-size:12px;color:var(--mut);margin-right:4px}
.msg{border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin:10px 0;white-space:pre-wrap;overflow-wrap:anywhere}
.me{background:var(--me);margin-left:14%}.ai{background:var(--ai);margin-right:8%}
.who{font-size:12px;font-weight:700;color:var(--mut);margin-bottom:2px;white-space:normal}
.why{font-size:12px;color:var(--mut);font-style:italic;margin-top:6px;white-space:normal}
details{margin:6px 8% 6px 0;border-left:3px solid var(--line);padding-left:10px}summary{cursor:pointer;color:var(--mut);font-size:13px}
.call{font:12px/1.45 ui-monospace,Consolas,monospace;background:var(--code);border-radius:8px;padding:8px 10px;margin:6px 0;white-space:pre-wrap;overflow-wrap:anywhere}
.call b{font-weight:700}.err{color:var(--not)}
.pics{display:flex;flex-wrap:wrap;gap:10px;margin:8px 8% 8px 0}.pics figure{margin:0;width:190px}
.pics img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}.pics figcaption{font-size:12px;color:var(--mut);margin-top:3px}
.board{border:1px dashed var(--line);border-radius:12px;padding:10px 12px;margin:8px 8% 8px 0}.sheet figure{width:420px;max-width:100%}
.bar{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--mut);margin-top:4px}code{background:var(--code);padding:0 4px;border-radius:4px}
"""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def rich(s: str) -> str:
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`([^`\n]+)`", r"<code>\1</code>", s)


def tally(rec: dict) -> dict:
    items = (rec.get("judge") or {}).get("items") or []
    return {v: sum(1 for i in items if i["verdict"] == v) for v in VERDICTS}


def chips(t: dict) -> str:
    names = {"met": "met", "partly": "partly", "not_met": "not met", "not_reached": "not reached"}
    return " ".join(f'<span class="chip {v}">{t[v]} {names[v]}</span>' for v in VERDICTS if t[v]) or '<span class="mut">not judged</span>'


def scenario_html(sc: dict, rec: dict) -> str:
    sid, a = rec["id"], sc.get("artist") or {}
    pic = lambda name: f"{sid}/pictures/{urllib.parse.quote(name)}"  # noqa: E731
    cost = rec.get("cost") or {}
    h = [f'<div class="card" id="{esc(sid)}"><h2>{esc(rec["title"])}</h2>',
         f'<div class="bar"><span><code>{esc(sid)}</code></span><span>{len(rec["turns"])} turns</span>'
         f'<span>{rec.get("seconds", 0) // 60} min {rec.get("seconds", 0) % 60} s</span>'
         f'<span>${sum(cost.values()):.2f} (agent ${cost.get("agent", 0):.2f})</span>'
         f'<span>{"".join(f"<span class=tag>{esc(t)}</span>" for t in rec.get("tags") or [])}</span></div>',
         f'<p>{chips(tally(rec))}</p>']
    if rec.get("error"):
        h.append(f'<p class="err"><b>The run did not finish:</b> {esc(rec["error"])}</p>')
    h.append(f'<p><b>What it tests.</b> {esc(sc.get("why", ""))}</p>')
    h.append(f'<p><b>The artist.</b> {esc(a.get("who", ""))} <b>Wants:</b> {esc(a.get("wants", ""))}</p>')
    if rec.get("setup_notes"):
        h.append(f'<p class="mut">Before the conversation: {esc("; ".join(rec["setup_notes"]))}</p>')
    j = rec.get("judge")
    if j:
        h.append("<h3>Checklist (the judge's reading: check it against the conversation)</h3>")
        if j.get("summary"):
            h.append(f"<p>{esc(j['summary'])}</p>")
        h.append("<table><tr><th>Verdict</th><th>Expected</th><th>Evidence and note</th></tr>")
        for i in j["items"]:
            h.append(f'<tr><td><span class="chip {i["verdict"]}">{i["verdict"].replace("_", " ")}</span></td><td>{esc(i["expected"])}</td>'
                     f'<td>{esc(i["evidence"])}{"<br><i>" + esc(i["note"]) + "</i>" if i["note"] else ""}</td></tr>')
        h.append("</table>")
        if j.get("observations"):
            h.append("<h3>The judge also noticed</h3><ul>" + "".join(f"<li>{esc(o)}</li>" for o in j["observations"]) + "</ul>")
    elif rec.get("judge_error"):
        h.append(f'<p class="err">The judge failed: {esc(rec["judge_error"])}</p>')
    if sc.get("watch_for"):
        h.append("<h3>For you to look at (not judged)</h3><ul>" + "".join(f"<li>{esc(w)}</li>" for w in sc["watch_for"]) + "</ul>")
    h.append("<h3>The conversation</h3>")
    for t in rec["turns"]:
        h.append(f'<div class="msg me"><div class="who">Artist · turn {t["n"]}</div>{esc(t["user"])}'
                 + (f'<div class="why">why (not shown to the agent): {esc(t["why"])}</div>' if t.get("why") else "") + "</div>")
        pending: list[str] = []

        def flush():
            if pending:
                h.append(f"<details><summary>{len(pending)} step(s) behind the scenes</summary>{''.join(pending)}</details>")
                pending.clear()

        for b in t["blocks"]:
            if "text" in b:
                flush()
                h.append(f'<div class="msg ai"><div class="who">Agent</div>{rich(b["text"])}</div>')
                continue
            c = t["calls"][b["call"]]
            arg = c["input"].get("file_path") or c["input"].get("skill") or c["input"].get("query") if not c["mcp"] else None
            pending.append(f'<div class="call"><b>{esc(c["name"])}</b> {esc(clip(arg if arg else c["input"], 900))}\n'
                           f'<span class="{"err" if c["error"] else "mut"}">→ {esc(clip(c["result"], 900))}</span></div>')
            if c["images"] and c["name"] != "show_board":
                flush()
                h.append('<div class="pics sheet">' + "".join(
                    f'<figure><a href="{pic(i)}"><img loading="lazy" src="{pic(i)}"></a><figcaption>from <code>{esc(c["name"])}</code> '
                    f"(the agent's own look)</figcaption></figure>" for i in c["images"]) + "</div>")
            if c["name"] == "show_board":
                bd = next((x for x in t["boards"] if x["input"] is c["input"] or x["input"] == c["input"]), None)
                if bd:
                    flush()
                    i = bd["input"]
                    figs = []
                    if bd["pictures"].get("original"):
                        figs.append(f'<figure><a href="{pic(bd["pictures"]["original"])}"><img loading="lazy" src="{pic(bd["pictures"]["original"])}"></a><figcaption><b>original</b></figcaption></figure>')
                    for k, cand in enumerate(i.get("candidates") or []):
                        label = str(cand.get("label") or chr(65 + k))
                        p = bd["pictures"].get(label)
                        img = f'<a href="{pic(p)}"><img loading="lazy" src="{pic(p)}"></a>' if p else '<div class="mut">(no picture)</div>'
                        figs.append(f'<figure>{img}<figcaption><b>{esc(label)}</b>{" ★ agent&#39;s pick" if cand.get("recommended") else ""}: '
                                    f'{esc(cand.get("caption", ""))}'
                                    + (f'<br><span class="mut">details: {esc(cand["details"])}</span>' if cand.get("details") else "") + "</figcaption></figure>")
                    ans = bd.get("answer") or {}
                    kept = {k: v for k, v in ans.items()
                            if k not in ("submitted", "revision", "votes", "rating") and v not in ([], "", None, False)}
                    answered = f'<div class="why">the artist answered on the board: {esc(json.dumps(kept))}</div>' if ans else ""
                    h.append('<div class="board"><div class="who">Pick board shown to the artist: ' + esc(i.get("title", ""))
                             + "</div><div>" + esc(i.get("question", "")) + '</div><div class="pics">' + "".join(figs)
                             + "</div>" + answered + "</div>")
        flush()
        for box in t["boxes"]:
            h.append(f'<div class="board"><div class="who">Picture box shown to the artist</div>{esc(box["purpose"])}'
                     + (f'<div class="why">the artist dropped {esc(box["dropped"])}</div>' if box.get("dropped") else "") + "</div>")
        if t["denied"]:
            h.append(f'<p class="err">Tried but not allowed here: {esc(", ".join(t["denied"]))}</p>')
        if t["error"]:
            h.append(f'<p class="err">This turn failed: {esc(t["error"])}</p>')
    h.append(f'<p class="mut">The conversation ended because {esc(rec.get("ended", ""))}.</p></div>')
    return "\n".join(h)


def write_report(run_dir: Path, scenarios: dict[str, dict]) -> Path:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("*/record.json"))]
    recs.sort(key=lambda r: r.get("started", ""))
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").exists() else {}
    total = {v: sum(tally(r)[v] for r in recs) for v in VERDICTS}
    rows = "".join(
        f'<tr><td><a href="#{esc(r["id"])}">{esc(r["title"])}</a><br><span class="mut"><code>{esc(r["id"])}</code></span></td>'
        f'<td>{chips(tally(r))}{"<br><span class=err>did not finish</span>" if r.get("error") else ""}</td>'
        f'<td>{len(r["turns"])}</td><td>{r.get("seconds", 0) // 60} min</td><td>${sum((r.get("cost") or {}).values()):.2f}</td>'
        f'<td>{esc((r.get("judge") or {}).get("summary", ""))}</td></tr>' for r in recs)
    body = [f"<h1>Model-bending skill: scenario run</h1><p class='mut'>{esc(run_dir.name)} · skill {esc(meta.get('version', '?'))} · "
            f"agent model: {esc(meta.get('agent_model') or 'the CLI default')} · artist: {esc(meta.get('artist_model', ''))} · "
            f"judge: {esc(meta.get('judge_model', ''))} · ComfyUI {esc(meta.get('comfyui_version', '?'))}</p>",
            f"<p>{chips(total)}</p>",
            "<p class='mut'>The agent had the built skill and the built MCP server, no shell (the Claude app's "
            "situation), and your real ComfyUI. The artist is simulated; the pick board and picture box were played by "
            "the runner. The checklist is an AI judge's reading: the conversation below it is the evidence.</p>",
            f'<div class="card"><table><tr><th>Scenario</th><th>Checklist</th><th>Turns</th><th>Time</th><th>Cost</th><th>In one line</th></tr>{rows}</table></div>']
    body += [scenario_html(scenarios.get(r["id"], {"artist": {}}), r) for r in recs]
    out = run_dir / "report.html"
    out.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                   f"<title>Scenario run {esc(run_dir.name)}</title><style>{CSS}</style></head><body><main>{''.join(body)}</main></body></html>",
                   encoding="utf-8")
    md = ["# Scenario run " + run_dir.name, "",
          "| scenario | met | partly | not met | not reached | turns | min | cost | in one line |", "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        t = tally(r)
        md.append(f"| {r['id']}{' (did not finish)' if r.get('error') else ''} | {t['met']} | {t['partly']} | {t['not_met']} | {t['not_reached']} | "
                  f"{len(r['turns'])} | {r.get('seconds', 0) // 60} | ${sum((r.get('cost') or {}).values()):.2f} | "
                  f"{(r.get('judge') or {}).get('summary', '').replace('|', '/')} |")
    md += ["", "Not met or partly:", ""]
    for r in recs:
        for i in (r.get("judge") or {}).get("items") or []:
            if i["verdict"] in ("not_met", "partly"):
                md.append(f"- **{r['id']}** ({i['verdict'].replace('_', ' ')}): {i['expected']}  \n  {i['note'] or i['evidence']}")
    (run_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------------ main
def load_scenarios() -> dict[str, dict]:
    out = {}
    for p in sorted(SCENARIOS.glob("*.json")):
        sc = json.loads(p.read_text(encoding="utf-8"))
        for key in ("id", "title", "artist", "first_message", "expected"):
            if key not in sc:
                raise SuiteError(f"{p.name}: missing {key!r}")
        if sc["id"] != p.stem:
            raise SuiteError(f"{p.name}: its id is {sc['id']!r}; file name and id must match")
        out[sc["id"]] = sc
    return out


def preflight(opts) -> dict:
    if opts.build or not SKILL_ZIP.exists() or not (STAGE / "scripts" / "mcp_server.py").exists():
        say("building the skill and the extension …")
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_dist.py")], check=True, cwd=str(ROOT))
    newest = max(p.stat().st_mtime for p in (ROOT / "skills").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if newest > SKILL_ZIP.stat().st_mtime + 1:
        say("NOTE: the skill's files are newer than the build in dist/. Run with --build to test the current files.")
    claude_bin()
    if not shutil.which("uv"):
        raise SuiteError("`uv` was not found: the MCP server is started with it")
    # the server's environment is made on first use: do that now, not inside the first scenario's start-up time
    warm = subprocess.run(["uv", "run", "--directory", str(STAGE), "python", "-c", "import mcp, PIL, numpy"],
                          capture_output=True, text=True)
    if warm.returncode:
        raise SuiteError(f"the MCP server's environment could not be set up: {warm.stderr[-400:]}")
    os.environ["COMFYUI_URL"] = opts.comfyui
    cc, _ = skill(opts.run_dir / "_state")
    try:
        st = cc.status()
    except Exception as e:  # noqa: BLE001
        raise SuiteError(f"ComfyUI does not answer at {opts.comfyui}: start it first ({e})") from e
    if not (st.get("model_bending") or {}).get("installed"):
        raise SuiteError("ComfyUI-Model-Bending is not installed in that ComfyUI")
    manifest = json.loads((STAGE / "manifest.json").read_text(encoding="utf-8"))
    return {"version": manifest.get("version"), "comfyui_version": st.get("comfyui_version"),
            "agent_ready": (st.get("model_bending") or {}).get("agent_ready"), "bridge": bool((st.get("bridge") or {}).get("installed"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="list the scenarios and stop")
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--tag", default="", help="only scenarios with this tag (e.g. smoke)")
    ap.add_argument("--agent-model", default=None, help="model of the agent under test (default: your CLI default)")
    ap.add_argument("--effort", default=None, help="effort level of the agent under test (low … max)")
    ap.add_argument("--artist-model", default="sonnet")
    ap.add_argument("--judge-model", default="sonnet")
    ap.add_argument("--max-turns", type=int, default=0, help="cap the turns of every scenario")
    ap.add_argument("--turn-timeout", type=float, default=2400, help="seconds one agent turn may take")
    ap.add_argument("--comfyui", default=os.environ.get("COMFYUI_URL") or "http://127.0.0.1:8188")
    ap.add_argument("--out", default="", help="run folder (default: evals/runs/<date-time>)")
    ap.add_argument("--build", action="store_true", help="run tools/build_dist.py first")
    ap.add_argument("--no-judge", action="store_true", help="record the conversations only")
    ap.add_argument("--report", default="", metavar="RUN", help="rebuild the report of an earlier run and stop")
    ap.add_argument("--rejudge", default="", metavar="RUN", help="judge an earlier run again, then rebuild its report")
    opts = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    try:
        scenarios = load_scenarios()
        if opts.list:
            for sc in scenarios.values():
                say(f"{sc['id']:<24} {','.join(sc.get('tags') or []):<22} {sc.get('max_turns', 4)} turns  {sc['title']}")
            return 0
        if opts.report or opts.rejudge:
            run_dir = Path(opts.report or opts.rejudge).resolve()
            if opts.rejudge:
                for p in sorted(run_dir.glob("*/record.json")):
                    rec = json.loads(p.read_text(encoding="utf-8"))
                    if rec["id"] in scenarios and rec["turns"]:
                        say(f"judging {rec['id']} …")
                        rec["judge"] = judge(scenarios[rec["id"]], rec, p.parent, opts.judge_model)
                        rec.setdefault("cost", {})["judge"] = rec["judge"].pop("cost", 0.0)
                        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            say(f"report: {write_report(run_dir, scenarios)}")
            return 0
        chosen = [s for s in scenarios.values()
                  if (not opts.only or s["id"] in {x.strip() for x in opts.only.split(",")})
                  and (not opts.tag or opts.tag in (s.get("tags") or []))]
        if not chosen:
            raise SuiteError("no scenario matches; see --list")
        opts.run_dir = Path(opts.out).resolve() if opts.out else EVALS / "runs" / time.strftime("%Y%m%d-%H%M%S")
        opts.run_dir.mkdir(parents=True, exist_ok=True)
        meta = preflight(opts)
        meta.update(agent_model=opts.agent_model, artist_model=opts.artist_model, judge_model=opts.judge_model,
                    scenarios=[s["id"] for s in chosen])
        (opts.run_dir / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        say(f"run folder: {opts.run_dir}\nskill {meta['version']} · ComfyUI {meta['comfyui_version']} · {len(chosen)} scenario(s)\n")
        failed = 0
        for k, sc in enumerate(chosen, 1):
            say(f"[{k}/{len(chosen)}] {sc['id']}: {sc['title']}")
            rec = run_scenario(sc, opts.run_dir, opts)
            t = tally(rec)
            say(f"    → {rec['ended']}\n    → met {t['met']}, partly {t['partly']}, not met {t['not_met']}, not reached {t['not_reached']}"
                f" · {rec['seconds'] // 60} min · ${sum(rec['cost'].values()):.2f}\n")
            write_report(opts.run_dir, scenarios)
            failed = failed + 1 if rec.get("error") else 0
            if failed >= 2:
                say("two scenarios in a row did not finish (a usage limit, or ComfyUI stopped?): stopping here.")
                break
        say(f"report: {write_report(opts.run_dir, scenarios)}")
        return 0
    except SuiteError as e:
        say(f"error: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
