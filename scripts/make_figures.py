"""Generate every figure used in the report (success-rate comparisons + per-task
training-dynamics dashboards) in one pass.

Outputs (all under results/figures/, dark TensorBoard-ish theme):
  fig1_hero_pickandplace.png        — KL win vs Ensemble freeze (PickAndPlace)
  fig2_kl_cliff_pickandplace.png    — KL beta over-constraint cliff
  fig3_physics_taskdependence.png   — physics: PickAndPlace vs Slide
  fig_dynamics_<TASK>.png           — 3-panel critic-loss / proxy reward /
                                      success-rate dashboard per task

Honesty notes baked into the dynamics dashboard:
  * `rollout/ep_rew_mean` is each method's *own* training reward (KL subtracts
    beta*KL, Physics subtracts c*penalty, Ensemble is min-of-3) — compare the
    *shape* (is the agent climbing its reward?), not absolute height across
    methods.
  * Critic loss is plotted on a log axis; the KL early spike is real.
  * Vanilla has no dense log (deleted) and shows up as 3 sparse eval markers
    on the panels where it has data.

Run:  conda run -n cs224r_project python scripts/make_figures.py
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

# ── TensorBoard-ish dark theme ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1f1f1f",
    "axes.facecolor": "#1f1f1f",
    "savefig.facecolor": "#1f1f1f",
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#dddddd",
    "text.color": "#eeeeee",
    "xtick.color": "#bbbbbb",
    "ytick.color": "#bbbbbb",
    "grid.color": "#3a3a3a",
    "font.size": 12,
})

# ── Palette ──────────────────────────────────────────────────────────────────
# Neutral / vanilla / eureka / kl / ensemble / physics / extra KL beta tones
GRAY, VANILLA, CYAN, GREEN, RED, ORANGE, PURPLE = (
    "#cccccc", "#9ca3af", "#22d3ee", "#4ade80", "#f87171", "#fb923c", "#c084fc")

_kfmt = FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x else "0")


# ── Shared helpers ──────────────────────────────────────────────────────────
def _scalars(sub: str, tag: str, logger: str = "SAC_0"):
    """Return (steps, values) for a tag under logs/<sub>/<logger>, or None."""
    p = f"logs/{sub}/{logger}"
    if not os.path.isdir(p):
        return None
    ea = EventAccumulator(p, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    s = ea.Scalars(tag)
    return np.array([x.step for x in s]), np.array([x.value for x in s], dtype=float)


def ema(v, weight=0.9):
    """TensorBoard-style debiased EMA smoothing."""
    out = np.zeros_like(v, dtype=float)
    last = 0.0
    debias = 0.0
    for i, x in enumerate(v):
        last = last * weight + (1 - weight) * x
        debias = debias * weight + (1 - weight)
        out[i] = last / debias
    return out


def _line(ax, sub, tag, color, label, weight=0.9):
    """Draw raw+EMA dense curve; return True if data was found."""
    got = _scalars(sub, tag)
    if got is None:
        return False
    st, v = got
    ax.plot(st, v, color=color, alpha=0.13, linewidth=1)
    ax.plot(st, ema(v, weight), color=color, linewidth=2.4, label=label)
    return True


def _markers(ax, task, tag, color, label):
    """Draw vanilla's 3 sparse eval points (no dense log exists)."""
    got = _scalars(f"vanilla/{task}_vanilla", tag, logger="eval")
    if got is None:
        return False
    st, v = got
    ax.plot(st, v, color=color, linewidth=1.6, linestyle="--", alpha=0.7, zorder=5)
    ax.scatter(st, v, color=color, s=55, zorder=6, edgecolors="#1f1f1f",
               linewidths=1.2, label=label)
    return True


# ── Success-rate comparison figures (formerly make_figures.py) ───────────────
def _success_plot(series, title, fname, weight=0.9):
    """series: list of (label, sub, color). Single-panel success-rate chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, sub, color in series:
        _line(ax, sub, "rollout/success_rate", color, label, weight=weight)
    ax.set_title(title, color="#ffffff", fontsize=14, pad=12, loc="left")
    ax.set_xlabel("training steps")
    ax.set_ylabel("success rate")
    ax.set_xlim(0, 500_000)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(_kfmt)
    ax.grid(True, linewidth=0.7)
    ax.legend(loc="upper left", facecolor="#2a2a2a", edgecolor="#555555",
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/{fname}")


def fig1_hero():
    _success_plot(
        [("Eureka (baseline)", "eureka/FetchPickAndPlace-v4_eureka", GRAY),
         ("+ KL  β=0.01", "kl/FetchPickAndPlace-v4_kl_b0.01", CYAN),
         ("+ Ensemble (min)", "ensemble/FetchPickAndPlace-v4_ensemble", RED)],
        "FetchPickAndPlace — soft KL wins, min-ensemble freezes",
        "fig1_hero_pickandplace.png",
    )


def fig2_kl_cliff():
    _success_plot(
        [("KL β=0.01", "kl/FetchPickAndPlace-v4_kl_b0.01", CYAN),
         ("KL β=0.1", "kl/FetchPickAndPlace-v4_kl_b0.1", PURPLE),
         ("KL β=1.0", "kl/FetchPickAndPlace-v4_kl_b1.0", RED)],
        "FetchPickAndPlace — KL strength: the over-constraint cliff",
        "fig2_kl_cliff_pickandplace.png",
    )


def fig3_physics_pair():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    panels = [
        (axes[0], "FetchPickAndPlace — physics collapses it",
         [("Eureka", "eureka/FetchPickAndPlace-v4_eureka", GRAY),
          ("+ Physics c=1.0", "physics/FetchPickAndPlace-v4_physics_c1.0", GREEN)]),
        (axes[1], "FetchSlide — physics preserves success",
         [("Eureka", "eureka/FetchSlide-v4_eureka", GRAY),
          ("+ Physics c=1.0", "physics/FetchSlide-v4_physics_c1.0", GREEN)]),
    ]
    for ax, title, series in panels:
        for label, sub, color in series:
            _line(ax, sub, "rollout/success_rate", color, label)
        ax.set_title(title, color="#ffffff", fontsize=13, pad=10, loc="left")
        ax.set_xlabel("training steps")
        ax.set_xlim(0, 500_000); ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_formatter(_kfmt)
        ax.grid(True, linewidth=0.7)
        ax.legend(loc="upper left", facecolor="#2a2a2a", edgecolor="#555555",
                  framealpha=0.9)
    axes[0].set_ylabel("success rate")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_physics_taskdependence.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/fig3_physics_taskdependence.png")


# ── Per-task training-dynamics dashboards (formerly make_training_dynamics.py)
def _dyn_methods(task):
    # PHYS uses ORANGE here; GREEN is reserved for the physics-vs-eureka pair plot.
    return [
        ("Eureka", f"eureka/{task}_eureka", CYAN),
        ("KL  β=0.01", f"kl/{task}_kl_b0.01", GREEN),
        ("Ensemble (min)", f"ensemble/{task}_ensemble", RED),
        ("Physics c=1.0", f"physics/{task}_physics_c1.0", ORANGE),
    ]


def dashboard(task, pretty):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ms = _dyn_methods(task)

    # Panel 1 — critic loss (log y). Vanilla absent (no dense log).
    ax = axes[0]
    for label, sub, color in ms:
        _line(ax, sub, "train/critic_loss", color, label)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e6)
    ax.set_title("critic loss  (TD error, log scale)", color="#ffffff",
                 fontsize=13, pad=10, loc="left")
    ax.set_ylabel("critic loss")

    # Panel 2 — proxy reward (each method's own training reward).
    ax = axes[1]
    for label, sub, color in ms:
        _line(ax, sub, "rollout/ep_rew_mean", color, label)
    _markers(ax, task, "eval/mean_proxy_reward", VANILLA, "Vanilla (eval)")
    ax.set_title("proxy reward  (per-method training reward ↑ = learning)",
                 color="#ffffff", fontsize=13, pad=10, loc="left")
    ax.set_ylabel("episode proxy reward")

    # Panel 3 — task success rate (the comparable axis).
    ax = axes[2]
    for label, sub, color in ms:
        _line(ax, sub, "rollout/success_rate", color, label)
    _markers(ax, task, "eval/success_rate", VANILLA, "Vanilla (eval)")
    ax.set_ylim(0, 1.0)
    ax.set_title("task success rate  (true objective)", color="#ffffff",
                 fontsize=13, pad=10, loc="left")
    ax.set_ylabel("success rate")

    for ax in axes:
        ax.set_xlabel("training steps")
        ax.set_xlim(0, 500_000)
        ax.xaxis.set_major_formatter(_kfmt)
        ax.grid(True, linewidth=0.7)
        ax.legend(loc="best", fontsize=10, facecolor="#2a2a2a",
                  edgecolor="#555555", framealpha=0.9)

    fig.suptitle(f"{pretty} — training dynamics across reward designs",
                 color="#ffffff", fontsize=16, x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{OUT}/fig_dynamics_{task}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    fig1_hero()
    fig2_kl_cliff()
    fig3_physics_pair()
    for task, pretty in [
        ("FetchPickAndPlace-v4", "FetchPickAndPlace"),
        ("FetchSlide-v4", "FetchSlide"),
    ]:
        dashboard(task, pretty)
    print("done")


if __name__ == "__main__":
    main()
