"""
Generate all study figures from results/scores.csv.
Outputs PNGs to results/figures/.

Figures produced:
  01_scatter_substance_voice.png  — 2D scatter per condition with calibration band
  02_substance_by_condition.png   — bar chart substance aggregate by condition × mode
  03_voice_tier_gap.png           — Tier 1 vs Tier 2 per condition (mimicry gap)
  04_component_survival.png       — substance by component type per condition
  05_pipeline_delta.png           — qwen → qwen_pre_edit → qwen_haiku substance delta
  06_cost_vs_quality.png          — Haiku token cost vs substance score

Usage:
    python3 scripts/make_figures.py
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
SCORES_CSV = ROOT / "results" / "scores.csv"
CALIB_CSV  = ROOT / "results" / "calibration.csv"
FIG_DIR    = ROOT / "results" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Style ---
CONDITION_COLORS = {
    "qwen":          "#6B8EAD",
    "haiku":         "#E8956D",
    "qwen_pre_edit": "#A8C5A0",
    "qwen_haiku":    "#C17BB5",
}
CONDITION_LABELS = {
    "qwen":          "Qwen (standalone)",
    "haiku":         "Haiku (standalone)",
    "qwen_pre_edit": "Qwen pre-edit",
    "qwen_haiku":    "Qwen→Haiku",
}
MODE_MARKERS = {"blog": "o", "linkedin": "s"}

plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.size":     11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":    150,
})


def load_data():
    if not SCORES_CSV.exists():
        sys.exit(f"scores.csv not found at {SCORES_CSV} — run eval/run_all.py first")
    df = pd.read_csv(SCORES_CSV)
    # Exclude smoke test entries
    df = df[df["id"].str.startswith("sample_")].copy()
    return df


def load_calibration():
    if not CALIB_CSV.exists():
        return None, None
    calib = pd.read_csv(CALIB_CSV)
    gold = calib[calib["label"] == "gold_standard"]
    null = calib[calib["label"] == "null"]
    return (
        gold["substance_aggregate"].mean() if not gold.empty else None,
        null["substance_aggregate"].mean() if not null.empty else None,
        gold["voice_combined"].mean()      if not gold.empty else None,
        null["voice_combined"].mean()      if not null.empty else None,
    )


# --- Figure 1: 2D scatter substance × voice ---

def fig_scatter(df):
    calib = load_calibration()
    gold_sub, null_sub, gold_voice, null_voice = calib if calib else (None, None, None, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    modes = ["blog", "linkedin"]

    for ax, mode in zip(axes, modes):
        sub = df[df["mode"] == mode]
        for cond, grp in sub.groupby("condition"):
            color  = CONDITION_COLORS.get(cond, "#999")
            marker = MODE_MARKERS[mode]
            ax.scatter(
                grp["substance_aggregate"], grp["voice_combined"],
                color=color, marker=marker, alpha=0.7, s=60,
                label=CONDITION_LABELS.get(cond, cond),
                zorder=3,
            )

        # Calibration band
        if gold_sub is not None and null_sub is not None:
            ax.axhspan(null_voice, gold_voice, alpha=0.07, color="#2E8B57", label="Calibration band")
            ax.axvspan(null_sub,  gold_sub,  alpha=0.07, color="#2E8B57")
            ax.axhline(gold_voice, color="#2E8B57", lw=1, ls="--", alpha=0.5)
            ax.axhline(null_voice, color="#CC4444", lw=1, ls="--", alpha=0.5)
            ax.axvline(gold_sub,  color="#2E8B57", lw=1, ls="--", alpha=0.5)
            ax.axvline(null_sub,  color="#CC4444", lw=1, ls="--", alpha=0.5)

        ax.set_xlabel("Substance fidelity")
        ax.set_ylabel("Voice fidelity (combined)" if mode == "blog" else "")
        ax.set_title(f"{mode.title()} posts")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)

    # Single legend for all conditions
    handles = [
        mpatches.Patch(color=CONDITION_COLORS[c], label=CONDITION_LABELS[c])
        for c in CONDITION_COLORS if c in df["condition"].unique()
    ]
    if gold_sub is not None:
        handles += [
            plt.Line2D([0], [0], color="#2E8B57", ls="--", label="Gold ceiling"),
            plt.Line2D([0], [0], color="#CC4444", ls="--", label="Null floor"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Substance vs Voice Fidelity by Condition", fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "01_scatter_substance_voice.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# --- Figure 2: Bar chart substance by condition × mode ---

def fig_substance_bars(df):
    conditions = [c for c in CONDITION_COLORS if c in df["condition"].unique()]
    modes      = df["mode"].unique()
    x          = np.arange(len(conditions))
    width      = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mode in enumerate(modes):
        sub    = df[df["mode"] == mode]
        means  = [sub[sub["condition"] == c]["substance_aggregate"].mean() for c in conditions]
        sems   = [sub[sub["condition"] == c]["substance_aggregate"].sem()  for c in conditions]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=sems, capsize=4,
                      color=[CONDITION_COLORS[c] for c in conditions],
                      alpha=0.85 if mode == "blog" else 0.5,
                      label=mode.title(), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Substance fidelity (mean ± SE)")
    ax.set_ylim(0, 1)
    ax.set_title("Substance Fidelity by Condition and Mode")
    ax.legend(title="Mode")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "02_substance_by_condition.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# --- Figure 3: Tier 1 vs Tier 2 per condition (blog only) ---

def fig_voice_tier_gap(df):
    blog = df[(df["mode"] == "blog") & df["voice_tier1"].notna() & df["voice_tier2"].notna()]
    conditions = [c for c in CONDITION_COLORS if c in blog["condition"].unique()]

    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(len(conditions))
    width = 0.35

    t1_means = [blog[blog["condition"] == c]["voice_tier1"].mean() for c in conditions]
    t2_means = [blog[blog["condition"] == c]["voice_tier2"].mean() for c in conditions]

    ax.bar(x - width/2, t1_means, width, label="Tier 1 (surface)", color="#7BAFD4", edgecolor="white")
    ax.bar(x + width/2, t2_means, width, label="Tier 2 (structural)", color="#D4957B", edgecolor="white")

    # Annotate delta
    for i, (t1, t2) in enumerate(zip(t1_means, t2_means)):
        delta = t1 - t2
        ax.text(i, max(t1, t2) + 0.02, f"Δ{delta:+.2f}", ha="center", fontsize=9, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], rotation=15, ha="right")
    ax.set_ylabel("Voice score")
    ax.set_ylim(0, 1)
    ax.set_title("Surface vs Structural Voice — Blog Posts\n(high Δ = surface mimicry without structural voice)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "03_voice_tier_gap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# --- Figure 4: Component survival by condition ---

def fig_component_survival(df):
    components = ["substance_claim", "substance_evidence", "substance_logic", "substance_implication"]
    comp_labels = {"substance_claim": "Claim", "substance_evidence": "Evidence",
                   "substance_logic": "Logic", "substance_implication": "Implication"}
    conditions  = [c for c in CONDITION_COLORS if c in df["condition"].unique()]

    # Blog only — same pattern for linkedin, combine if needed
    sub = df[df["mode"] == "blog"]

    x     = np.arange(len(components))
    width = 0.8 / len(conditions)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cond in enumerate(conditions):
        grp    = sub[sub["condition"] == cond]
        means  = [grp[c].mean() for c in components]
        offset = (i - len(conditions)/2 + 0.5) * width
        ax.bar(x + offset, means, width, color=CONDITION_COLORS[cond],
               label=CONDITION_LABELS[cond], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([comp_labels[c] for c in components])
    ax.set_ylabel("Mean substance score")
    ax.set_ylim(0, 1)
    ax.set_title("Substance Score by Component Type — Blog Posts")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = FIG_DIR / "04_component_survival.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# --- Figure 5: Pipeline delta (qwen → pre_edit → qwen_haiku) ---

def fig_pipeline_delta(df):
    # Only samples present in all 3 pipeline stages
    pipeline_conds = ["qwen", "qwen_pre_edit", "qwen_haiku"]
    present = [c for c in pipeline_conds if c in df["condition"].unique()]
    if len(present) < 2:
        print("  Skipping pipeline delta — not enough pipeline conditions present")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode in zip(axes, ["blog", "linkedin"]):
        sub = df[df["mode"] == mode]

        # Only samples present in all pipeline conditions
        id_sets = [set(sub[sub["condition"] == c]["id"]) for c in present]
        common  = id_sets[0].intersection(*id_sets[1:])
        if not common:
            ax.set_title(f"{mode.title()} — no paired samples")
            continue

        for cond in present:
            grp     = sub[(sub["condition"] == cond) & (sub["id"].isin(common))]
            grp_sorted = grp.set_index("id").loc[sorted(common)]
            ax.plot(
                range(len(common)),
                grp_sorted["substance_aggregate"].values,
                marker="o", markersize=4,
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                alpha=0.8,
            )

        ax.set_xlabel("Sample")
        ax.set_ylabel("Substance fidelity")
        ax.set_title(f"{mode.title()} — Pipeline Delta")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Qwen → Qwen pre-edit → Qwen→Haiku: Substance Across Pipeline", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "05_pipeline_delta.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# --- Figure 6: Cost vs quality (Haiku conditions only) ---

def fig_cost_vs_quality(df):
    haiku_conds = [c for c in ["haiku", "qwen_haiku"] if c in df["condition"].unique()]
    if not haiku_conds:
        print("  Skipping cost vs quality — no Haiku conditions present")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode in zip(axes, ["blog", "linkedin"]):
        sub = df[(df["mode"] == mode) & df["haiku_cost_usd"].notna()]
        for cond in haiku_conds:
            grp = sub[sub["condition"] == cond]
            ax.scatter(
                grp["haiku_cost_usd"] * 1000,  # convert to millicents for readability
                grp["substance_aggregate"],
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                alpha=0.7, s=60,
            )
        ax.set_xlabel("Haiku cost (USD × 10⁻³)")
        ax.set_ylabel("Substance fidelity")
        ax.set_title(f"{mode.title()}")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle("Cost vs Substance Quality — Haiku Conditions", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "06_cost_vs_quality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# --- Main ---

def main():
    print("Loading scores.csv...")
    df = load_data()
    print(f"  {len(df)} rows  |  conditions: {sorted(df['condition'].unique())}  |  N samples: {df['id'].nunique()}")

    print("\nGenerating figures...")
    fig_scatter(df)
    fig_substance_bars(df)
    fig_voice_tier_gap(df)
    fig_component_survival(df)
    fig_pipeline_delta(df)
    fig_cost_vs_quality(df)

    print(f"\nDone. Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
