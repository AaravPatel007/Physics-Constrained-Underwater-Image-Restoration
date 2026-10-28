"""Plan-of-action timeline (weeks after the intermediate review)."""
from pathlib import Path

from vizstyle import AXIS, INK, INK2, SERIES, plt

ROOT = Path(__file__).resolve().parents[1]

TASKS = [  # (label, start week, length in weeks, done?)
    ("Physics model, synthetic water, BetaNet v1→v2, baselines, real-data checks", -2, 2, True),
    ("Realism: forward-scatter blur + marine snow (De-water engine), outdoor RGB-D", 0, 1, False),
    ("Range-dependent β_D(z) head (Sea-thru's 2-exponential form)", 0.5, 1, False),
    ("Self-supervised fine-tune on 658 real dive frames (done early; collapse found + fixed)", -0.3, 0.3, True),
    ("Choose teacher weight per use; metric-depth fix before any UIEB fine-tune", 1, 1, False),
    ("Hybrid: physics output + small learned refiner (De-water BridgeNet)", 1.5, 1, False),
    ("Full real evaluation: all Sea-thru chart frames + SQUID, vs Sea-thru / DeepSeeColor", 2, 1.2, False),
    ("Uncertainty head + water-type (Jerlov) read-out", 2.5, 1, False),
    ("Stretch: detection mAP on restored images (original proposal)", 3, 1, False),
    ("Final report, demo, slides", 3.5, 0.8, False),
]


def main():
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_axisbelow(True)
    for i, (lab, s, L, done) in enumerate(TASKS):
        y = len(TASKS) - 1 - i
        ax.barh(y, L, left=s, height=0.55, color=SERIES[0] if done else "#86b6ef")
        ax.text(s + L + 0.06, y, lab, va="center", fontsize=10, color=INK if not done else INK2)
    ax.axvline(0, color=SERIES[1], lw=2)
    ax.text(0, len(TASKS) - 0.35, " intermediate review (29 Oct)", color=SERIES[1], fontsize=10, va="bottom")
    ax.set_yticks([])
    ax.set_xticks(range(-2, 5), ["-2 wk", "-1 wk", "review", "+1 wk", "+2 wk", "+3 wk", "+4 wk"])
    ax.set_xlim(-2.2, 9.5)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("weeks after the intermediate review", color=INK2)
    fig.suptitle("Plan of action: done (dark) and next milestones (light)", x=0.01, ha="left", fontsize=14,
                 fontweight="bold", y=1.0)
    fig.savefig(ROOT / "figures" / "1_original_runs" / "fig13_plan.png")


if __name__ == "__main__":
    main()
