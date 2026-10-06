# Changelog

## 0.4.0

The first minutes of a session, redesigned after a workshop with artists: people do not all arrive with a goal.

- **Four ways in.** The agent reads the first message and starts in the way that fits: an introduction, a surprise
  round, the user's own project, or a goal. When it cannot tell, it asks one question. A first result comes before
  the questions about depth, autonomy and the knowledge base (`references/first-session.md`).
- **Introduction with examples:** a short map of what can be bent (where, when, how, what) and a few example
  pictures from the community knowledge base, one per part of the model, shown next to their unbent pictures
  (`intro_examples`, `kb_local.py intro`). It needs no renders, so it works while ComfyUI is not running.
- **Surprise round:** `surprise_bends` (`scripts/surprise.py`) draws one random bend per part of the model inside
  the measured safe ranges, plus a labelled wild card past them. SD1.5-type and SDXL models. The same seed repeats
  a draw. The round is rendered as bent copies of one unbent run; the new `base_sd15` preset makes that run.
- **Bending a workflow you already have:** `inspect_run` reads your last ComfyUI run (model, LoRAs, sampler,
  prompt) and `bend_run` adds the bend to a copy of it, before its samplers. Your settings and seed are kept, so
  your own picture is the unbent original, and the copy opens with your layout plus one new group. Your own
  workflow is never changed (`references/own-workflow.md`; scripts: `comfy_canvas.py run-setup`, `bend-run`).
  Where the graph cannot take the bend (a run queued by a script, a sampler inside a subgraph), you get the pictures
  and instructions for adding the box by hand.
- **A bend you bring as JSON:** paste a bend in the format of Apply Bends from JSON (what the web UI's Copy Bends
  puts on the clipboard) and the agent makes it. `check_bends` (`scripts/bendjson.py`) checks it the way the node
  would, tidies it and says in plain words what it does; bends the node would refuse are raised with you instead
  of dropped. `bend_run` takes the JSON as given, and with `replace_bends` it stands in for the bends a run
  already has (a run made in the web UI), or renders that run unbent.
- **Bends remember where they came from.** A bend may carry `"kb": {"dataset", "cell", "record"}`, naming the
  knowledge-base cell or record it came from. Model-Bending 0.3.1+ (bends JSON 1.2) keeps the key in its
  `resolved_json`. For older plugins, which call it an unknown key (an error with `strict` on), the tools ask
  ComfyUI once and drop `kb` only from what they send (`comfy_canvas.fit_bends`; `comfy_status` reports
  `keeps_kb`). `check_bends` keeps it, checks
  it and adds the navigator link (the record's own page when it names a record, else its cell's). `find_recipes`
  results now carry it in their `recipe`, plus a `link` to the cell's page in the Model Bending Navigator
  (before/after examples, evidence, risks) and `example_links` to each example render's page.
- **Contributing a full run.** The knowledge base takes full runs, not single pictures: one model, all seven regions
  of its U-Net, at least three operations with three amounts each (one prompt and seed is enough).
  `scripts/kb_run.py` plans the run (`init`, `layers`), renders it on your ComfyUI with each picture's workflow
  embedded (`render`, resumable), checks the format and the full-run rule with a coverage table (`check`), and opens
  a pull request on the dataset with your own Hugging Face login (`submit`). Measuring (`measure`: LPIPS, DINOv2,
  CLIP and the broken-render check, from the new `scripts/kb_measure.py`) and describing (`sheets`,
  `add-descriptions`) are optional; the maintainer adds what a run leaves out. Records gain a `contribution` block:
  who made the run (a name, or "not named"), by script or by an agent, the run, and the CC0 agreement.
- **Request a model** in the dataset's Discussions (a pinned post explains how); the README links it.
- **Your own rounds stay local.** `log_round` no longer writes a `sharing.json`, and `kb_status` no longer counts
  rounds "marked for sharing": single rounds are not contributed.
- **Describing results the knowledge base's way.** `description_prompt` (`kb_local.py describe-prompt`) gives the
  exact prompt the community captions were written with (`kb-caption-v1`), with the current effect tags. When the
  user wants the agent to describe their results, the skill now has it use this prompt and save the result with
  `log_round(..., keywords, prompt_version)`, so everyone's descriptions stay comparable. Such descriptions are
  labelled as written in the session, with the recipe known.
- **Tray links.** The navigator can share the bends someone collected as one link (`…/#tray=…`; the bends travel in
  the URL fragment, never sent to a server). `open_tray` (`bendjson.py --tray LINK`) reads it and checks each bend
  on its own, to render one at a time on the user's run; `tray_link` (`bendjson.py BENDS --to-tray`) makes one, to
  hand a round's bends back.
- **The skill's instructions re-cut** to read as one text: a session overview, the five ways in side by side, one
  place for the depth and autonomy dials (formerly "sliders") and for the knowledge-source choice, two named
  render routes in the core loop, the script and tool table and the file list at the end, and the steps for
  starting from a picture in `references/starting-picture.md`.
- **Hand-picked intro examples** (`data/kb/intro.json`), chosen from renders the dataset's owner labelled as fine.
  Examples from the LCM sweep appear once those records are published.

## 0.3.1

- **Picture box:** when the artist attaches a picture to the chat or says they will bring one, the agent opens a box in
  the chat (`ask_for_picture`); the artist drops, chooses or pastes the picture and it goes straight into ComfyUI
  (`picture_received` gives its name). Where the box cannot be shown, the agent asks for the file's path or for the
  picture in a Load Image box.
- **No folder to set up:** the extension no longer asks for a Working folder. Animations are saved in ComfyUI's
  output folder, inside `agent_bending`, where ComfyUI saves the pictures it makes (also for a ComfyUI on another
  machine), and the agent tells the artist where to find them. The skill's own files (caches, boards, sessions, your
  knowledge base) live in the system's app-data folder; an existing `~/.comfyui-model-bending` keeps being used.
  If you had chosen another Working folder, move its contents to the app-data folder or set `COMFY_BENDING_WORKDIR`.
  `animate.py` saves into ComfyUI's output folder by default too (`--name`, `--format`; `--out` still works).
- **Extension settings save without changes:** the ComfyUI address is no longer marked required. Claude Desktop
  treated a required setting as missing until it was saved, but kept Save disabled while nothing had changed, so the
  extension could not be finished with the defaults. Every setting is now optional; the address still defaults to
  `http://127.0.0.1:8188`.

Security hardening (from an external review; nothing malicious was found):
- The Agent-Bridge token is read from a file only when ComfyUI runs on this computer, and only from the bridge's own
  token file (`…/agent_bridge/token`, a token inside). For a remote ComfyUI, set `AGENT_BRIDGE_TOKEN`.
- Image, video and metrics tools open only ComfyUI `/api/view` URLs on `COMFYUI_URL` and the knowledge base's example
  images; through the MCP server they never open local files. Requests to ComfyUI do not follow redirects.
- ffmpeg reads videos with a fixed format and file access only. `upload_image` uploads only real pictures.
- `push` writes only under `agent_bending/` (or `agent_bridge/`) in the Workflows sidebar.
- The node-schema cache and scratch images live in the user's work folder, not the shared temp folder.
- Install commands are pinned to the release tag, and dependencies are capped below their next major version
  (the script commands in the docs too).
- SKILL.md tells the agent that text from the knowledge base, the canvas, run reports and logs is material to read,
  never instructions to follow.
- `BEND_KB_DATASET` is documented with the other settings.
- Session ids are checked (letters, digits, `-`, `_`, `.`), so a session name can never write a file outside the work
  folder. The safe-range tool takes only its table names.
- `upload_image` sends pictures from this computer only to a ComfyUI on this computer, and only from
  `COMFY_BENDING_PICTURE_DIRS` when that is set.
- Refreshing the community knowledge base fetches all index files from one dataset commit (`BEND_KB_REVISION` pins
  it), checks them, cleans their text and replaces the cached copy in one step; older unchecked caches are ignored.
- The Claude Desktop extension has two new optional settings: a **Picture folder** and an **Agent-Bridge token** (for
  a ComfyUI on another machine, stored as a secret).
- A redirect from ComfyUI (a proxy or tunnel) gives an error that says what to change; the README has a
  troubleshooting table.

## 0.3.0

- **Video bending (experimental)** for WAN 2.1 / 2.2 with ComfyUI-Model-Bending 0.3+: attention-map bends
  (`attention_bends` in the bends JSON), temporal operations, `frame_ramp`, and two presets
  (`video_sweep_wan21_t2v`, `video_sweep_wan21_i2v`). Videos are measured for motion and flicker, shown as
  filmstrips, and play as loops on the pick board. `references/video.md` is the guide.
- **Starting pictures:** `upload_image` (script: `comfy_canvas.py upload`) copies a picture from the user's computer or
  an earlier round into ComfyUI, and `build_workflow(start_image, denoise)` (script: `build --start-image`) turns any
  text-to-image spec into image-to-image, or sets the picture an image-to-video preset starts from.
- The knowledge base records video runs (routes `txt2video` and `img2video`).
- WAN time windows in `timesteps.py` (shift 8).

## 0.2.2

- The in-chat board notices when the app could not show it and falls back to pictures in the chat.
- The board's helper tools are visible to every client, so apps that relay tools to a local extension pass them on.

## 0.2.1

- The board also declares the older `ui/resourceUri` key, for hosts that read only that.

## 0.2.0

- **In-chat pick board** (MCP Apps) in the Claude app: the artist picks one or several versions, marks what to keep
  or change, and sends a note without opening ComfyUI. Clients without MCP Apps get a contact sheet.
- Plain-language guidance for talking with artists.

## 0.1.0

- First release: the skill (three depths, three levels of involvement, canvas boards, bend animations), its MCP
  server and Claude Desktop extension, the bend knowledge base, and measured safe ranges for SD1.5 and SDXL.
