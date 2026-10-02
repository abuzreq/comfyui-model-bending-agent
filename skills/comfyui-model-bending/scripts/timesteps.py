#!/usr/bin/env python3
"""Show which executed sampler steps a diffusion-time window covers. Standard library only.

The agent-ready Model-Bending gates natively by t ("t": [hi, lo], t_start/t_end, Timestep Gated Bending): prefer that.
Use this script to explain a result in steps, to pick a step for Visualize Feature Map, or on older installs that
only gate by step index (steps_min/steps_max, steps_to_bend_str). Windows on normalised time t (1 = pure noise):
    structure  t in [1.0, 0.7]    layout, pose, composition
    style      t in [0.7, 0.2]    palette, contrast, handling
    detail     t in [0.2, 0.0]    high-frequency texture
The mapping from step index to t depends on the architecture, scheduler, shift and denoise, which is why
"first 30 % of steps" is wrong for shifted flow models (Flux spends ~57 % of its steps above t = 0.7 at 1024 px).

  python timesteps.py --arch flux --steps 20 --res 1024
  python timesteps.py --arch sdxl --steps 20 --scheduler karras
  python timesteps.py --arch sd3 --steps 28 --window 0.8 0.3        # custom window
  python timesteps.py --arch sd15 --steps 20 --denoise 0.6          # img2img: indices of the executed steps
  python timesteps.py --arch wan --steps 10                         # WAN video (ModelSamplingSD3, shift 8)

t for eps models (SD1.5 / SDXL) is the model timestep / 999; for flow models (Flux, SD3, WAN) it is the shifted
sigma. WAN workflows usually set ModelSamplingSD3 shift 8 (pass --shift if yours differs).
LCM-distilled models (ModelSamplingDiscrete lcm) take large jumps; use --arch sd15 --scheduler sgm_uniform as an
approximation and trust measurements (on 6-step LCM, structure is fixed by step 0).
"""

from __future__ import annotations

import argparse
import math

WINDOWS = {"structure": (1.0, 0.7), "style": (0.7, 0.2), "detail": (0.2, 0.0)}
ARCHS = ("sd15", "sdxl", "flux", "sd3", "wan")


# ---------------------------------------------------------------- eps (SD1.5 / SDXL) discrete schedule
def _discrete_sigmas():
    b0, b1 = math.sqrt(0.00085), math.sqrt(0.012)
    ac, sig = 1.0, []
    for i in range(1000):
        beta = (b0 + (b1 - b0) * i / 999) ** 2
        ac *= 1 - beta
        sig.append(math.sqrt((1 - ac) / ac))
    return sig


SIGMAS = _discrete_sigmas()
LOG_SIGMAS = [math.log(s) for s in SIGMAS]


def _sigma_of_t(t: float) -> float:
    lo = max(0, min(998, int(math.floor(t))))
    w = t - lo
    return math.exp((1 - w) * LOG_SIGMAS[lo] + w * LOG_SIGMAS[lo + 1])


def _t_of_sigma(s: float) -> float:
    ls = math.log(max(s, 1e-10))
    if ls <= LOG_SIGMAS[0]:
        return 0.0
    if ls >= LOG_SIGMAS[-1]:
        return 999.0
    for i in range(999):
        if LOG_SIGMAS[i] <= ls <= LOG_SIGMAS[i + 1]:
            return i + (ls - LOG_SIGMAS[i]) / (LOG_SIGMAS[i + 1] - LOG_SIGMAS[i])
    return 999.0


def eps_schedule(n: int, scheduler: str) -> list[float]:
    """Model timesteps (0..999) at which each of n steps evaluates the UNet."""
    if scheduler in ("normal", "sgm_uniform"):
        k = n + 1 if scheduler == "sgm_uniform" else n
        ts = [999 - 999 * i / (k - 1) for i in range(k)] if k > 1 else [999.0]
        return ts[:n]
    if scheduler == "simple":
        ss = 1000 / n
        return [999 - int(x * ss) for x in range(n)]
    if scheduler == "karras":
        rho, smin, smax = 7.0, SIGMAS[0], SIGMAS[-1]
        a, b = smax ** (1 / rho), smin ** (1 / rho)
        return [_t_of_sigma((a + i / max(n - 1, 1) * (b - a)) ** rho) for i in range(n)]
    raise ValueError(f"unsupported scheduler {scheduler!r} (normal, simple, sgm_uniform, karras)")


# ---------------------------------------------------------------- flow (Flux / SD3)
def flow_schedule(n: int, arch: str, res: int, mu: float | None, shift: float | None) -> list[float]:
    ts = [1 - i / n for i in range(n)]
    if arch == "flux":
        if mu is None:  # ComfyUI ModelSamplingFlux: base_shift 0.5 @ 256 tokens, max_shift 1.15 @ 4096 tokens
            tokens = res * res / 256
            mu = 0.5 + (1.15 - 0.5) / (4096 - 256) * (tokens - 256)
        e = math.exp(mu)
        return [e / (e + (1 / t - 1)) if t > 0 else 0.0 for t in ts]
    s = (8.0 if arch == "wan" else 3.0) if shift is None else shift  # SD3 default 3; WAN workflows use 8
    return [s * t / (1 + (s - 1) * t) for t in ts]


def schedule(a) -> list[float]:
    """Normalised t per executed step, honouring denoise the way KSampler does (total = steps / denoise)."""
    total = a.steps if a.denoise >= 0.9999 else int(a.steps / a.denoise)
    if a.arch in ("flux", "sd3", "wan"):
        full = flow_schedule(total, a.arch, a.res, a.mu, a.shift)
    else:
        full = [t / 999 for t in eps_schedule(total, a.scheduler)]
    return full[-a.steps:]


def window_steps(ts: list[float], hi: float, lo: float) -> tuple[int, int] | None:
    idx = [i for i, t in enumerate(ts) if lo <= t <= hi]
    return (idx[0], idx[-1]) if idx else None


def table(arch: str, steps: int, scheduler: str = "normal", res: int = 1024, mu: float | None = None,
          shift: float | None = None, denoise: float = 1.0, window: tuple[float, float] | None = None) -> dict:
    """t of every executed step, and the executed steps each window covers (None if it misses them all)."""
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {', '.join(ARCHS)}")
    ts = schedule(argparse.Namespace(arch=arch, steps=steps, scheduler=scheduler, res=res, mu=mu, shift=shift,
                                     denoise=denoise))
    wins = {**WINDOWS, **({"custom": tuple(window)} if window else {})}
    out = {}
    for name, (hi, lo) in wins.items():
        r = window_steps(ts, hi, lo)
        out[name] = {"t": [hi, lo], "steps": None if r is None else f"{r[0]}-{r[1]}"}
    return {"t_per_step": [round(t, 3) for t in ts], "windows": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--scheduler", default="normal", help="eps models: normal | simple | sgm_uniform | karras")
    ap.add_argument("--res", type=int, default=1024, help="flux: image side in px (sets the shift)")
    ap.add_argument("--mu", type=float, help="flux: override the ModelSamplingFlux shift exponent")
    ap.add_argument("--shift", type=float, help="sd3 / wan: ModelSamplingSD3 shift (default 3.0; wan 8.0)")
    ap.add_argument("--denoise", type=float, default=1.0)
    ap.add_argument("--window", nargs=2, type=float, metavar=("T_HI", "T_LO"), help="custom window, e.g. 0.8 0.3")
    a = ap.parse_args()
    ts = schedule(a)
    print("step  t     " + "  ".join(WINDOWS))
    for i, t in enumerate(ts):
        marks = "  ".join(("  x" if lo <= t <= hi else "  .").ljust(len(n)) for n, (hi, lo) in WINDOWS.items())
        print(f"{i:>4}  {t:.3f} {marks}")
    wins = {**WINDOWS, **({"custom": tuple(a.window)} if a.window else {})}
    print()
    for name, (hi, lo) in wins.items():
        r = window_steps(ts, hi, lo)
        if r is None:
            print(f"{name:<9} t in [{hi}, {lo}]: no executed step falls in this window")
        else:
            print(f'{name:<9} t in [{hi}, {lo}]: steps {r[0]}-{r[1]}  ->  "steps_min": {r[0]}, "steps_max": {r[1]}'
                  f'   |  steps_to_bend_str "{r[0]}-{r[1]}"')


if __name__ == "__main__":
    main()
