# Starting from a picture

Tool names are the MCP tools; the scripts are named next to them.

The artist may give a picture alongside the prompt. ComfyUI must have it in its input folder first:
1. **Get it into ComfyUI.** A picture attached to the chat reaches you but not ComfyUI. As soon as the artist
   attaches one, or says they will bring a picture:
   - **Open the picture box** when `ask_for_picture` is available (the Claude app): call `ask_for_picture(purpose)`
     at once, before asking for anything else, and tell them in one line to drop the picture into it. It goes straight
     into ComfyUI; their next message says it is there, and `picture_received(request_id)` gives `image` and
     `view_url`.
   - **Otherwise offer the two other ways together**, worded as in `references/plain-language.md`: paste the file's
     full path (then `upload_image(path)`, script: `comfy_canvas.py upload …`), or drag the picture into a Load Image
     box in ComfyUI and tell you its name (`list_input_images`, script: `inputs`). That is the case when the tool says
     the box cannot be shown, when `picture_received` reports `not_displayed`, and in Claude Code. Do not bring up
     the Picture folder setting (`COMFY_BENDING_PICTURE_DIRS`): it only limits which folders may be read and does not
     help a picture load. Mention it only when `upload_image` says the picture is outside it.
   - A version from an earlier round needs no box: `upload_image` its `/api/view` URL ("start the next round from B").

   Uploads return `image` (e.g. `agent_bending/lighthouse_1a2b3c4d.png`); the same picture always gets the same
   name. Upload only a path the user gave you in the chat. A ComfyUI on another machine gets no files from this one:
   there, `ask_for_picture` asks for the Load Image way only.
2. **Describe it, and write the prompt from it.** Look at the picture (`view_images` with its `view_url`) and write
   a prompt that names what is in it: subject, layout ("a tall dark cypress on the left, a swirling sky, a village
   below"), colours and medium. The prompt steers every step the sampler repaints; a prompt written from their goal
   alone ("a glitchy night scene") pulls the picture toward a different one.
3. **Build from it.** `build_workflow(spec, name, start_image=image, denoise=…)` (script: `build … --start-image`)
   turns any text-to-image spec or preset into image-to-image: the empty latent becomes the picture, scaled to the
   model's pixel count with its own shape (never cropped), and the samplers repaint part of it. In an
   image-to-video preset it sets the picture the video starts from.
4. **Ask how closely to follow it**, in their terms, and map it to `denoise`:

   | they say | denoise |
   |---|---|
   | "bend my picture", "keep my picture, just restyle it" | 0.3–0.45 |
   | "keep the composition, repaint it" | 0.5–0.65 (default 0.6) |
   | "use it as a loose starting point" | 0.7–0.85 |

   Distilled models (LCM, Turbo) repaint more at the same denoise: on LCM, 0.25–0.4 keeps a painting recognisable.
   **Render the unbent version first and compare it with their picture** before any bend. If it has already lost
   their picture, lower denoise (or improve the prompt) first; bends judged against it would tell them nothing.
5. **Say what the bend can still reach.** At denoise ≤ 0.7 no step reaches the structure window (SKILL.md §5): bends restyle
   the picture but cannot re-compose it. For new compositions, raise denoise (it then follows the picture loosely).
6. Show the picture as the **original** on boards and sheets (`original_url` = the upload's `view_url`), and describe
   each version against it. The knowledge base keeps it private.

## Unsampling

Unsampling runs the picture backwards to partial noise, then samples forward from there, with the bends on the
forward pass only. It keeps more of the picture than img2img at the same depth. Rules that keep the comparison fair:
- **Same model setup and sampler as the other versions.** The backward and forward passes use the model the bends
  are applied to, with the same model-sampling node: an LCM checkpoint keeps `ModelSamplingDiscrete` (lcm) and the
  `lcm` sampler. Running euler on an LCM model (or 20 steps where the others ran 6) changes the result more than
  the bend does, and the bend's time window then covers many more steps.
- **Show the unbent round trip** (backward then forward, no bend) next to the bent versions: it is their baseline.
- Depth works like denoise: unsampling to a later step keeps more. Say which step, in plain words ("about a third
  of the way back to noise").
