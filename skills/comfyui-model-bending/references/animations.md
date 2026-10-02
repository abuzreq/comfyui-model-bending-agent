# Bend animations (`scripts/animate.py`, tools `start_animation` / `animation_status`)

An animation renders the same workflow frame by frame while one or more values move in small increments. Seed,
prompt and sampling stay fixed, so only the bend moves. The frames are then encoded as a looping video.

**Video models** (WAN) already make moving pictures, so `animate.py` refuses their workflows. To make a bend change
over the clip, wrap it in `frame_ramp` (`references/video.md`): one render, the bend's strength follows a curve
over the frames.

**When to use one:**
- to show the user what a bend *does* along its whole range, rather than at three sampled points
- to find where an effect "snaps". A sudden jump between two frames marks a threshold worth a finer sweep, and is
  evidence for L3.
- as the deliverable of a creative session

**Source.** Any of:
- an API workflow file (`--api`)
- a ComfyUI PNG, via its embedded prompt (`--png`)
- a run from history (`--prompt-id`)
- the newest run (`--last-run`), such as one the user just made on the canvas

**Tracks.** Each is `"SELECTOR@name=FROM:TO"`. FROM defaults to the value that leaves the layer unchanged, and TO
to the current value.
- `--bend "recompose@angle_degrees=0:180"`: a JSON bend, selected by label, by path, or as `NODE#index`. The name is
  a `module_args` key, or `blend` (0 → 1 fades any op in, including sobel and threshold, which have no neutral
  amount).
- `--input "NODE@name=FROM:TO"`: any numeric input of any node, by node id or title. Examples: a slider primitive's
  `value`, `a`–`d` on `Apply Bends from JSON`, a `Rotate Module`'s `angle_degrees`, `ApplySteeringVector`'s
  `strength`.
- No tracks: every JSON bend moves from neutral to its current amount.

**Timing.**
- `--increment 5` sets the step of the first moving track in its own units. Use small increments for smooth motion.
- `--frames 24` sets a fixed frame count instead.
- `--easing ease|linear`, `--loop boomerang|restart|none`, `--hold`, `--repeats`, `--fps`.
- A 0 → 360° rotation loops seamlessly with `--loop restart`.
- Always `--dry-run` first. It shows the frame count, the values and the duration. Expect one render per frame, so
  check the cost against the budget.

**Output.**
- `--out x.mp4` needs ffmpeg. `.gif` and `.webp` need only Pillow.
- A folder next to the video holds the frames, a `filmstrip.jpg` and `anim.json` (tracks, values per frame).
- Look at the filmstrip yourself before showing the video.
- If several image outputs are downstream of the tracks, choose one with `--output-node` (the error lists them).

```
uv run --no-project --with pillow python scripts/animate.py --last-run \
    --bend "recompose@angle_degrees=0:180" --increment 5 --out bending_sessions/<session>/recompose.mp4
```

With the MCP tools: `start_animation(name | prompt_id | last_run, bends=[…], inputs=[…], increment, …,
dry_run=true)` returns the plan. Call it again without `dry_run`, then poll `animation_status(job)`. When done it
returns the video path on the user's machine and the filmstrip image.
