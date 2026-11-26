"""Agent term of GatedSpec vs approach speed of two agents: current barrier vs the proposed reformulation.
Usage: PYTHONPATH=src uv run python barrier_plot.py  ->  barrier_vs_speed.png"""
import matplotlib.pyplot as plt
import torch
import yaml

from config import Config
from fmai.sfo import GatedSpec

cfg = Config(**yaml.safe_load(open("src/experiments/IPPO_zones.yaml"))["config"])
s = GatedSpec.from_config(cfg)
v2 = 2 * s.v_max                                              # head-on, both agents at top speed
c = torch.linspace(-v2, v2, 401)                              # gap shrink per step (> 0 = approaching)


def current(d):                                               # the code in sfo.py, two agents on the x-axis
    prev = torch.stack([torch.zeros(len(c), 2), torch.stack([torch.full_like(c, d), torch.zeros_like(c)], -1)], 1)
    pos = prev.clone(); pos[:, 1, 0] -= c
    return s.terms(pos[None], pos[None], prev[None])["agent"][0, :, 0]


def proposed(d):                                              # min(distance, barrier normalised by head-on speed)
    h = d - s.d_safe
    dist = s._pw(d - c, s.d_crit, s.d_safe, s.d_far)
    bar = ((s.gamma * h - c) / v2).clamp(-1, 1)
    return torch.minimum(dist, bar)


GAPS = [(0.6, "#84b9f3"), (0.45, "#338ced"), (0.35, "#1b64be"), (0.3, "#0e3d7e")]   # ordinal blue ramp
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, f, title in [(axes[0], current, "Current: barrier only"),
                     (axes[1], proposed, "Proposed: min(distance, barrier / head-on speed)")]:
    for d, col in GAPS:
        ax.plot(c / v2, f(d), color=col, lw=2, label=f"gap d = {d}")
    contact = f(s.d_crit)[len(c) // 2].item()                 # standing still in contact (d = 0.2, no motion)
    ax.axhline(contact, color="#d6261f", lw=2, ls="--")
    ax.text(-0.98, contact + 0.04, f"standing still in contact: {contact:+.2f}", color="#191919", fontsize=10)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_title(title, fontsize=12, loc="left")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.supxlabel("closing speed / head-on top speed   (< 0: moving apart, 1: both at full speed towards each other)", fontsize=11)
axes[0].set_ylabel(r"$\rho_{Agent}$")
axes[0].set_ylim(-1.1, 1.1)
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle(rf"Agent term for two agents ($\gamma$ = {s.gamma:.3f}, d_safe = {s.d_safe}, contact at {s.d_crit})",
             fontsize=13)
fig.tight_layout()
fig.savefig("barrier_vs_speed.png", dpi=160)
print("wrote barrier_vs_speed.png")
