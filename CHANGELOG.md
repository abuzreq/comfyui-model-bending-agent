# Changelog

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
