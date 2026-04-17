#!/usr/bin/env python3
"""Generate benchmark charts for llama-tq README.

Reads benchmarks.csv (own measurements) and other_projects.csv (reported
numbers from TheTom/buun for cross-project comparison), writes PNG charts
to ../../docs/img/.

Deps:  pip install pandas matplotlib

Usage: python3 gen_charts.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------

df = pd.read_csv(HERE / "benchmarks.csv")
other = pd.read_csv(HERE / "other_projects.csv")

# Derived columns
df["ppl_delta_pct"] = (df["ppl"] - df["ppl_baseline"]) / df["ppl_baseline"] * 100.0
df["config"] = df["k_cache"] + "/" + df["v_cache"]
df["model_label"] = df["model"] + " (" + df["model_quant"] + ")"


def _savefig(name: str, fig):
    path = OUT / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path.relative_to(HERE.parent.parent)}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Chart 1: PPL vs bpw (Pareto-style scatter)
# ----------------------------------------------------------------------------

def chart_ppl_vs_bpw():
    # Two-panel: left = bpw ≤ 7 (compressed region where points cluster),
    # right = bpw > 7 (f16-K territory, fewer points)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6),
                                   gridspec_kw={"width_ratios": [3, 1.2]},
                                   sharey=True)

    markers = {
        "Qwen3.5-35B-A3B (IQ2_XS)":   ("o", "#1f77b4"),
        "Qwen3.6-35B-A3B (IQ2_XXS)":  ("s", "#ff7f0e"),
        "Qwen3.5-35B-A3B (Q4_K_M)":   ("^", "#2ca02c"),
        "Qwen3.6-35B-A3B (Q4_K_M)":   ("D", "#d62728"),
        "Qwen3.5-27B Dense (Q4_K_M)": ("v", "#9467bd"),
    }

    have_ppl = df.dropna(subset=["ppl_delta_pct"]).copy()
    # Rename 'Qwen3.5-27B-Dense' → 'Qwen3.5-27B Dense' for consistency
    have_ppl["model_label"] = have_ppl["model_label"].str.replace(
        "Qwen3.5-27B-Dense", "Qwen3.5-27B Dense"
    )

    for label, (marker, color) in markers.items():
        sub = have_ppl[have_ppl["model_label"] == label]
        if sub.empty:
            continue
        for ax in (axL, axR):
            ax.scatter(
                sub["avg_bpw"], sub["ppl_delta_pct"],
                marker=marker, color=color, s=85, alpha=0.9,
                label=label if ax is axL else None,
                edgecolors="black", linewidth=0.5, zorder=3,
            )

    # Shade "near-lossless" band on left panel
    axL.axhspan(-0.5, 1.5, color="#e6f4ea", alpha=0.6, zorder=1,
                label="_nolegend_")
    axL.text(6.9, 0.5, "near-lossless (<1.5%)", fontsize=7, color="#2ca02c",
             ha="right", style="italic")

    # Annotate typical configs (one per cluster, placed outside the point)
    typical = {
        3.5:  ("q4_0 + vtq2_1", "3.5 bpw", (0.3, 0.8)),
        5.5:  ("q8_0 + vtq2_1", "5.5 bpw", (0.3, 0.6)),
        6.25: ("q8_0 + vtq3_1", "6.25 bpw", (0.3, -1.2)),
        4.5:  ("q4_0 / q4_0", "4.5 bpw", (0.3, -1.0)),
    }
    for bpw, (text, _, offset) in typical.items():
        axL.annotate(text, xy=(bpw, 10), xytext=(bpw + offset[0], 10.8),
                     fontsize=7.5, color="#333", ha="left",
                     arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.5))

    for ax in (axL, axR):
        ax.axhline(0, color="#666", lw=0.7, ls="--", zorder=2)
        ax.set_xlabel("Average KV-cache bpw (lower means more compression)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        ax.grid(True, ls=":", alpha=0.4, zorder=0)

    axL.set_ylabel("PPL delta vs f16 (lower is better)")
    axL.set_xlim(2.8, 7)
    axR.set_xlim(7.5, 16.8)
    axL.set_title("Compressed region (K or V below 8 bpw)")
    axR.set_title("f16 K region")

    # Hide duplicated y-axis on right panel
    axR.tick_params(labelleft=False)

    fig.suptitle("llama-tq: PPL delta vs KV-cache bpw (5 model/quant pairs, 2x RTX 2060)",
                 fontsize=12, y=0.98)
    axL.legend(loc="upper right", fontsize=8, framealpha=0.95,
               title="Model / weight quant")
    fig.tight_layout()

    _savefig("ppl_vs_bpw.png", fig)


# ----------------------------------------------------------------------------
# Chart 2: Decode throughput (TG128) bar chart
# ----------------------------------------------------------------------------

def chart_decode_throughput():
    # Pick one V-config per group, compare across models (Q4_K_M weights only,
    # fair cross-model comparison at same weight precision)
    configs = ["f16/f16", "q8_0/vtq3_1", "q8_0/vtq2_1", "q4_0/vtq2_1", "f16/q4_0", "q8_0/q4_0"]
    models = ["Qwen3.5-35B-A3B (Q4_K_M)", "Qwen3.6-35B-A3B (Q4_K_M)", "Qwen3.5-27B-Dense (Q4_K_M)"]

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.25
    x = range(len(configs))

    for i, model in enumerate(models):
        vals = []
        for cfg in configs:
            sub = df[(df["model_label"] == model) & (df["config"] == cfg)]
            vals.append(sub["tg128"].iloc[0] if not sub.empty and not pd.isna(sub["tg128"].iloc[0]) else 0)
        pos = [xi + i * width for xi in x]
        ax.bar(pos, vals, width, label=model, edgecolor="black", linewidth=0.4)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("TG128 tok/s (higher is better)")
    ax.set_xlabel("K-cache / V-cache configuration")
    ax.set_title("llama-tq: Decode throughput by KV cache config (Q4_K_M weights, 2x RTX 2060)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    _savefig("decode_throughput.png", fig)


# ----------------------------------------------------------------------------
# Chart 3: Cross-project Pareto frontier
# ----------------------------------------------------------------------------

def chart_cross_project():
    # Show only the representative config per model-quant for llama-tq so the
    # scatter isn't dominated by near-duplicate points.
    have_ppl = df.dropna(subset=["ppl_delta_pct"]).copy()
    have_ppl["model_label"] = have_ppl["model_label"].str.replace(
        "Qwen3.5-27B-Dense", "Qwen3.5-27B Dense"
    )

    # Use the two flagship configs only (vtq3_1 and vtq2_1 with q8_0 K)
    flagship = have_ppl[
        have_ppl["config"].isin(["q8_0/vtq3_1", "q8_0/vtq2_1"])
    ].copy()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Per-model colors for llama-tq
    model_colors = {
        "Qwen3.5-35B-A3B (IQ2_XS)":   "#1f77b4",
        "Qwen3.6-35B-A3B (IQ2_XXS)":  "#ff7f0e",
        "Qwen3.5-35B-A3B (Q4_K_M)":   "#2ca02c",
        "Qwen3.6-35B-A3B (Q4_K_M)":   "#d62728",
        "Qwen3.5-27B Dense (Q4_K_M)": "#9467bd",
    }

    # Plot llama-tq points, one marker per model, split by config
    for label, color in model_colors.items():
        sub = flagship[flagship["model_label"] == label]
        if sub.empty:
            continue
        # vtq3_1 = circle, vtq2_1 = square (so reader can distinguish configs)
        v3 = sub[sub["config"] == "q8_0/vtq3_1"]
        v2 = sub[sub["config"] == "q8_0/vtq2_1"]
        ax.scatter(v3["avg_bpw"], v3["ppl_delta_pct"],
                   marker="o", color=color, s=80, alpha=0.9,
                   edgecolors="black", linewidth=0.5, zorder=4)
        ax.scatter(v2["avg_bpw"], v2["ppl_delta_pct"],
                   marker="s", color=color, s=80, alpha=0.9,
                   edgecolors="black", linewidth=0.5, zorder=4)

    # TheTom (star)
    tt = other[other["project"] == "TheTom"]
    ax.scatter(tt["avg_bpw"], tt["ppl_delta_pct"],
               marker="*", color="#ff7f0e", s=240, alpha=0.9,
               edgecolors="black", linewidth=0.7, zorder=5,
               label="TheTom (M5 Max, Q4_K_M MoE)")
    for _, row in tt.iterrows():
        ax.annotate(
            row["config"],
            xy=(row["avg_bpw"], row["ppl_delta_pct"]),
            xytext=(9, 6), textcoords="offset points",
            fontsize=8, color="#cc5500", fontweight="bold",
        )

    # buun (diamond)
    bb = other[other["project"] == "buun"]
    ax.scatter(bb["avg_bpw"], bb["ppl_delta_pct"],
               marker="D", color="#2ca02c", s=160, alpha=0.9,
               edgecolors="black", linewidth=0.7, zorder=5,
               label="buun TCQ (RTX 3090, Q6_K)")
    for _, row in bb.iterrows():
        ax.annotate(
            row["config"],
            xy=(row["avg_bpw"], row["ppl_delta_pct"]),
            xytext=(9, -12), textcoords="offset points",
            fontsize=8, color="#206020", fontweight="bold",
        )

    # Custom legend: model colors + project markers (handmade so it's readable)
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markeredgecolor="black", markersize=9, label=m)
        for m, c in model_colors.items()
    ]
    config_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=9,
               label="llama-tq: q8_0 + vtq3_1"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=9,
               label="llama-tq: q8_0 + vtq2_1"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#ff7f0e",
               markeredgecolor="black", markersize=14,
               label="TheTom (M5 Max, Q4_K_M MoE)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=10,
               label="buun TCQ (RTX 3090, Q6_K)"),
    ]
    leg_models = ax.legend(handles=model_handles, loc="upper left",
                           fontsize=8, title="llama-tq: model",
                           title_fontsize=8, framealpha=0.95)
    ax.add_artist(leg_models)
    ax.legend(handles=config_handles, loc="lower left",
              fontsize=8, title="Config / project", title_fontsize=8,
              framealpha=0.95)

    ax.axhline(0, color="#666", lw=0.7, ls="--", zorder=2)
    ax.axhspan(-0.5, 1.5, color="#e6f4ea", alpha=0.4, zorder=1)

    ax.set_xlabel("Average KV-cache bpw (lower means more compression)")
    ax.set_ylabel("PPL delta vs f16 (lower is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("Cross-project PPL comparison (different HW/models, indicative only)")
    ax.grid(True, ls=":", alpha=0.4, zorder=0)
    ax.set_xlim(2, 7)

    # Caveat below the chart, outside the plotting area
    fig.text(
        0.5, -0.02,
        "Note: different hardware, model variants, and PPL methodologies. "
        "buun uses KL divergence on Q6_K; TheTom uses wikitext-2 on Q4_K_M MoE. "
        "Direct comparison is approximate.",
        ha="center", fontsize=8, color="#555", style="italic",
    )

    _savefig("cross_project.png", fig)


# ----------------------------------------------------------------------------
# Chart 4: VTQ2_1 PPL variance across models
# ----------------------------------------------------------------------------

def chart_vtq2_variance():
    vtq2_cfg = "q8_0/vtq2_1"
    vtq3_cfg = "q8_0/vtq3_1"
    have_ppl = df.dropna(subset=["ppl_delta_pct"])
    vtq2 = have_ppl[have_ppl["config"] == vtq2_cfg].sort_values("ppl_delta_pct")
    vtq3 = have_ppl[have_ppl["config"] == vtq3_cfg].sort_values("ppl_delta_pct")

    fig, ax = plt.subplots(figsize=(10, 5))
    y2 = range(len(vtq2))
    ax.barh(
        [yi - 0.2 for yi in y2], vtq2["ppl_delta_pct"], height=0.4,
        color="#d62728", edgecolor="black", linewidth=0.4,
        label=f"{vtq2_cfg} (5.5 bpw)",
    )
    # Plot vtq3 if available for same models
    vtq3_vals = []
    for label in vtq2["model_label"]:
        sub = vtq3[vtq3["model_label"] == label]
        vtq3_vals.append(sub["ppl_delta_pct"].iloc[0] if not sub.empty else 0)
    ax.barh(
        [yi + 0.2 for yi in y2], vtq3_vals, height=0.4,
        color="#2ca02c", edgecolor="black", linewidth=0.4,
        label=f"{vtq3_cfg} (6.25 bpw)",
    )

    ax.set_yticks(list(y2))
    ax.set_yticklabels(vtq2["model_label"])
    ax.set_xlabel("PPL delta vs f16 (lower is better)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("llama-tq: PPL delta varies by model architecture and weight quant")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    ax.axvline(0, color="#888", lw=0.6)

    _savefig("vtq2_variance.png", fig)


# ----------------------------------------------------------------------------

def main():
    print(f"Output dir: {OUT}")
    chart_ppl_vs_bpw()
    chart_decode_throughput()
    chart_cross_project()
    chart_vtq2_variance()
    print("done.")


if __name__ == "__main__":
    main()
