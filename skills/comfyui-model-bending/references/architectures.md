# Architecture-aware bending: layer maps, routing rules, timestep windows

## Contents
1. UNet family: block map, sub-module kinds, hookability rules, measured safe ranges
2. Diffusion transformers: Flux.1, SD3 / SD3.5, WAN video (2.1 / 2.2)
3. Timestep gating: structure / style / detail windows and step tables
4. L2 steering primitives

All paths are **relative to `diffusion_model`**, which is what `ApplyBendsFromJSON`, `Model Bending`,
`ActivationProbe` and `Visualize Feature Map` resolve against. Only modules whose own `forward()` is called during
sampling can be bent.

The agent-ready Model-Bending handles the traps:
- containers expand to their last child
- lists and typos are skipped with hints
- tuples bend `output[0]` with a warning

`strict` turns all of these into errors. For a new architecture, run `Bendable Layer Catalogue` once instead of
guessing paths.

Resolution names below: `hi` = the latent's own size, `mid` = ½, `lo` = ¼ or ⅛. Figures are for SD1.5 at 512 px
and SDXL at 1024 px.

---

## 1. UNet family (SD1.5, SD2.x, SDXL, and LCM/Turbo/Lightning distils)

### Block map

| group | SD1.5 paths | SD1.5 res × ch | SDXL paths | SDXL res × ch | role (routing rule) |
|---|---|---|---|---|---|
| conv_in | `input_blocks.0.0` | 64² × 320 | `input_blocks.0.0` | 128² × 320 | acts directly on the latent. Keep multiply within 0.8–1.2 |
| in.hi | `input_blocks.1–2` (`.0` res, `.1` attn), `3.0` down | 64² × 320 | `input_blocks.1–2` (res only), `3.0` down | 128² × 320 | **edges and local geometry.** Fragile: rotate/scale here gives blob fields |
| in.mid | `input_blocks.4–5`, `6.0` down | 32² × 640 | `input_blocks.4–5` (attn depth 2), `6.0` down | 64² × 640 | **layout and framing.** scale 0.7 gives "painting in a frame" |
| in.lo | `input_blocks.7–8`, `9.0` down, `10–11` (res only, 8²) | 16² × 1280 | `input_blocks.7–8` (attn depth **10**) | 32² × 1280 | **composition and viewpoint.** rotate here makes new compositions |
| mid | `middle_block.0` res, `.1` attn, `.2` res | 8² × 1280 | same (attn depth 10) | 32² × 1280 | **semantic core / h-space.** Rotate re-stages the scene. HSpace directions live here |
| out.lo | `output_blocks.0–2` (`2.1` up), `3–5` (res+attn, `5.2` up) | 8–16² × 1280 | `output_blocks.0–2` (attn depth 10, `2.2` up) | 32² × 1280 | **re-composition and object identity** |
| out.mid | `output_blocks.6–8` (`8.2` up) | 32² × 640 | `output_blocks.3–5` (`5.2` up) | 64² × 640 | **style, handling and painterly abstraction.** SDXL: any rotate/scale here is colour noise |
| out.hi | `output_blocks.9–11` | 64² × 320 | `output_blocks.6–8` (res only) | 128² × 320 | **texture, detail and colour fidelity.** multiply >1.5 gives a poster look |

Routing summary: the **down path sets geometry and layout**, the **middle block holds dense semantics and style**,
and the **up path handles texture, detail and colour**. Skip connections carry high-frequency detail from the
down path into the up path. That is why out-block `skip_connection` bends are so strong.

### Sub-module kinds (inside each block)

| kind | path pattern | tensor | character (measured on SD1.5 with `multiply 0`) |
|---|---|---|---|
| `res` | `X.Y` where Y is the ResBlock | 4-D | the whole residual block's output |
| `skip` | `X.Y.skip_connection` | 4-D | **strongest.** On output blocks it carries the U-Net skip input. Zeroing it on in.hi, out.mid or out.hi wipes the image. Stay near 1 |
| `res_branch` | `X.Y.out_layers` | 4-D | redraws in a new style but keeps the subject. This is the "residual bias" site |
| `attn` | `X.1` (SpatialTransformer) | 4-D | the whole attention block |
| `self_attn` | `X.1.transformer_blocks.K.attn1` | **3-D tokens** | re-stages the composition moderately |
| `cross_attn` | `X.1.transformer_blocks.K.attn2` | **3-D tokens** | the prompt link. **Gentlest**, weakest on out.hi |
| `ff` | `X.1.transformer_blocks.K.ff` | **3-D tokens** | re-stages moderately |
| `upsample` / `downsample` | `output_blocks.N.{1,2}` / `input_blocks.N.0` | 4-D | resolution change; spatial ops act at the new size |

Deeper paths are hookable too, for Level 3 work: `…attn2.to_k`, `…attn2.to_v`, `…attn2.to_q`, `…attn2.to_out.0`,
`…attn1.to_q`, and so on. **Cross-attention K/V routing**: bending `attn2.to_k` or `attn2.to_v` changes what the
text tokens contribute (V), or where they attend (K), for that block only. The output is 3-D:
`(batch, 77 text tokens, inner_dim)`.

### Hookability rules (UNet)
- **Containers** such as `input_blocks.4`, `middle_block` and `output_blocks.4` (`TimestepEmbedSequential`) are
  never called by ComfyUI itself.
  - Model-Bending bends their **last child** and logs `[model-bending] expanded container …`.
  - `Model Bending (SD Blocks)` therefore has a visible effect. Measured: `middle_block` ×0 gives MAE 40.
  - To address a specific child, name it: `input_blocks.4.0` is the ResBlock and `.1` the transformer.
- `nn.ModuleList` paths (`input_blocks`, `…transformer_blocks`) are skipped, with a hint listing the children.
  Typos are skipped with a close-match hint.
- Wildcards per segment (`output_blocks.*.1`, `input_blocks.[4-8].0`) select families of layers in one JSON bend.
  `report.patterns` shows the matches.
- **Spatial ops** (rotate, scale, erosion, dilation, gradient, sobel, fourier) must target 4-D layers on a UNet. On a
  3-D token tensor, the generic hook still treats the tensor as `(1, B, L, C)`, so a "rotation" mixes token and
  channel axes. Use multiply, add_scalar, add_noise or threshold on `attn1`/`attn2`/`ff`.

### Measured safe ranges (sweeps on LCM Dreamshaper v7 and CommonCanvas XL; machine-readable in `data/safe_ranges_*.json`)
- **rotate**: `mid`, `in.lo` and `out.lo` give new compositions (SD1.5 0–180°, SDXL 30–180°). `in.mid` and `out.mid`
  dissolve into abstraction (SD1.5 5–45°; SDXL `in.mid` 30° already dissolves). `in.hi` and `out.hi` stay within 0–5°.
- **scale**: 0.7 on in.mid or in.lo frames the image. Above 1 on mid zooms in. hi groups must stay within 0.95–1.05.
- **multiply**: 0.5–2.0 is coherent almost everywhere. in.hi breaks around 1.5–2.5, and output groups above 1.5
  saturate.
- **add_noise**: strong on input groups and nearly invisible on mid/out groups (safe up to 3.0 there). On SDXL
  `in.hi`, 0.6 re-stages the scene.
- **add_scalar**: subtle, except on `in.hi` where +1.5 gives a painterly blur.
- SDXL is **much more fragile** than SD1.5 at `in.hi`, `out.mid` and `out.hi`. Use only multiply, add_scalar or light
  add_noise there.

---

## 2. Diffusion transformers / flow matching

### Flux.1 (dev / schnell) — ComfyUI `comfy/ldm/flux`
19 `double_blocks` then 38 `single_blocks`. Hidden size 3072, 24 heads. The latent has 16 channels and is packed
into 2×2 patches, so 1024 px gives 64×64 = 4096 image tokens. There are 256 T5 tokens on schnell and 512 on dev.
Guidance is distilled: use cfg 1 plus `FluxGuidance`, so there is no uncond batch.

| stream | hookable paths | tensor | routing rule |
|---|---|---|---|
| joint / double | `double_blocks.i.img_attn.qkv`, `.img_attn.proj`, `.img_mlp` | `(B, L_img, 3072)` (qkv: ×3) | **spatial framing and cross-modal binding** on the image side |
| joint / double | `double_blocks.i.txt_attn.qkv`, `.txt_attn.proj`, `.txt_mlp` | `(B, L_txt, 3072)` | the **text-side** representation. Bending here rewrites what the prompt means to later blocks |
| unified / single | `single_blocks.i` (whole block), `.linear1`, `.linear2` | `(B, L_txt + L_img, 3072)`, **text first** | **global style, contrast and colour balance.** The whole block also hits the text tokens |
| embed / out | `img_in`, `txt_in`, `final_layer` | tokens | `img_in` is like conv_in (keep it within ±20 %). `final_layer` is a straight velocity bias |

- **Use `DiT Block Bending` for block-level bends.** For example `blocks: "double:0-6"` with `stream: img`, or
  `"single:25-37"` with `stream: both`.
  - It bends the block's output stream explicitly, through ComfyUI's own block-patch mechanism.
  - `spatial: true` maps image tokens to the latent grid, so rotate and scale act spatially. On single blocks it
    splits the text and image tokens for you.
  - A raw hook on `double_blocks.i` only bends `output[0]` (the image stream) and warns.
- **Early double blocks (0–6)** decide layout and binding. **Late single blocks (~25–37)** act like SD's out.hi,
  controlling texture and contrast. The middle of the stack carries the "style" band. Treat these as priors and
  verify each one with a 3-point sweep.
- Sub-module paths (`double_blocks.i.img_attn.qkv`, `.img_mlp`, `single_blocks.i.linear2`, …) still suit L3
  work, such as K/V routing on `img_attn.qkv`. Use channel-wise ops there; `spatial` exists only on
  `DiT Block Bending`.
- Hardware: Flux needs 12 GB or more of VRAM in fp8. On a 6 GB card it needs GGUF Q4 or smaller. Check free
  VRAM with `comfy_canvas.py status` first.

### SD3 / SD3.5 (MMDiT) — `comfy/ldm/modules/diffusionmodules/mmdit.py`
There is **no double/single split**: every block is joint. SD3 Medium has 24 `joint_blocks`, and SD3.5 Large has 38.
SD3.5 Medium adds an extra x-only self-attention in its early blocks (MMDiT-X).

| hookable paths | tensor | notes |
|---|---|---|
| `joint_blocks.i.x_block.attn.qkv`, `.x_block.attn.proj`, `.x_block.mlp` | `(B, L_img, C)` | image stream. Use this where you would use Flux's `img_*` |
| `joint_blocks.i.context_block.attn.qkv`, `.attn.proj`, `.mlp` | `(B, L_ctx, C)` | text stream. The last block's context side is `pre_only` (no proj/mlp) |
| `x_embedder.proj`, `context_embedder`, `final_layer` | | embeddings and output |

For block-level bends, use `DiT Block Bending` with `blocks: "joint:0-11"` and `stream: img | txt`. As raw hook
targets, `x_block` and `context_block` never fire, because ComfyUI calls `pre_attention` and `post_attention`
directly. Only the Linear and Mlp children fire. Early joint blocks set structure and late ones set style. Real CFG
applies here (cfg 4–7), so the uncond batch is bent too; the probe measures the cond half only.

### WAN 2.1 / 2.2 video (experimental) — `comfy/ldm/wan`
A stack of `blocks` (30 in WAN 2.1 1.3B with 12 heads, 40 in 14B with 40 heads), each with `self_attn`,
`cross_attn` and `ffn`. The latent is 16 channels × latent frames × h × w, patched 1×2×2; one latent frame = 4 video
frames after the first. WAN 2.1 I2V adds 257 CLIP image tokens, attended in a separate cross-attention call.

| route | how | routing rule (upstream tips, to verify) |
|---|---|---|
| attention maps | `attention_bends` in the bends JSON, or `Attention Map Bending` | `cross_text`: where each word lands, frame by frame; `cross_image`: how an I2V video reads its picture; `self_query` / `self_key`: move or drag content |
| block outputs | `DiT Block Bending blocks: "double:13-18"` (stream img), or `bends` paths `blocks.N`, `blocks.N.ffn` | laid out on frames × h × w, so rotate, translate and blur act on each frame |
| over time | `temporal_shift`, `temporal_blur`, `frame_reverse`, `frame_ramp` | timing and motion |

The middle blocks (13–18 of 30; ~17–24 of 40) and the first steps change the most. WAN 2.2 14B chains two models
(high noise for layout and motion, then low noise for detail): bend each separately. Full guide:
`references/video.md`.

---

## 3. Timestep gating — structure / style / detail

The agent-ready Model-Bending gates directly by **diffusion time t**: `"t": [hi, lo]` per JSON bend, `t_start`/`t_end`
on `Model Bending`, `DiT Block Bending` and `ApplySteeringVector`, and `Timestep Gated Bending` for any module (with
ramps). t is the model's own normalised timestep (≈ σ for flow models), so **prefer t-windows**: they stay correct
when steps, scheduler, shift or denoise change. They also fail closed, so a window is skipped rather than widened
if t is unknown.

Step windows (`"steps"`, `steps_to_bend_str`) still exist, and `scripts/timesteps.py` converts between the two.
Use it to explain a result in steps, or to target a specific step. The windows:

| window | t range | what it controls |
|---|---|---|
| structure | 1.0 → 0.7 | layout, pose, composition |
| style | 0.7 → 0.2 | palette, contrast, handling, material |
| detail | 0.2 → 0.0 | high-frequency texture and micro-detail |

What the same t-windows cover in executed steps. This is why step fractions mislead:

| setup | structure | style | detail |
|---|---|---|---|
| Flux dev, 20 steps, 1024 px (μ = 1.15) | 0–11 | 12–18 | 19 |
| SD3, 28 steps, shift 3 | 0–15 | 16–25 | 26–27 |
| SDXL, 20 steps, karras | 0–6 | 7–11 | 12–19 |
| SD1.5 LCM, 6 steps, sgm_uniform | 0–1 | 2–4 | 5 |
| SD1.5, 20 steps, img2img at denoise 0.6 | **none** | 0–12 | 13–19 |
| WAN 2.1, 10 steps, shift 8 | 0–7 | 8–9 | **none** |
| WAN 2.1, 20 steps, shift 8 | 0–15 | 16–19 | **none** |

Two consequences:
- An img2img or unsample→resample carrier at denoise ≤ 0.7 **cannot change structure**, whatever the bend.
  Raise denoise or the unsample depth to move composition.
- On 6-step LCM, structure is fixed by step 0. A bend limited to step 0 changes the image by MAE 11–26, and
  one limited to step 5 by only 1.3.
- On WAN (shift 8) almost every step is in the structure window. "Early" means the first 2–3 steps
  (`"steps": "0-2"`) or a narrow window such as `[1.0, 0.9]`.

---

## 4. L2 steering primitives

| intent | node(s) | notes |
|---|---|---|
| h-space semantic direction | `Compute PCA` → `HSpace Bending` (direction 0–9, scale) | PCA of `middle_block.1` activations over a small batch. Applies to UNets only |
| guidance-space latent op at step k | `Latent Operation (…)` → `LatentApplyOperationCFGToStep` | acts on (cond − uncond) at exactly one step |
| conditioning arithmetic | `Latent Operation (…)` → `ConditioningApplyOperation`; core `ConditioningAverage`, `ConditioningCombine`, `ConditioningSetTimestepRange` | `ConditioningSetTimestepRange` is the core way to gate *prompts* by time (it takes t fractions directly) |
| residual bias | `add_scalar` / `multiply` on `…out_layers` (res_branch) | a uniform bias only |
| **activation steering vector** | `ActivationProbe` ×2 (prompt A run, prompt B run) → `ReadActivationProbe` ×2 → `SteeringVectorFromActivations` (site, channel \| spatial) → `ApplySteeringVector` (strength, t window, `apply_to: cond`) | a difference of means, added at one site. Name the direction by what it does. `workflows/activation_probe_steering.json` in Model-Bending is a worked example |
| any latent op as a bend | `Latent Operation To Module` → `Model Bending` | lets a latent op act on an internal activation |
