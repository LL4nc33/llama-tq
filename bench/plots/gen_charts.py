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
    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-model markers
    markers = {
        "Qwen3.5-35B-A3B (IQ2_XS)":   ("o", "#1f77b4"),
        "Qwen3.6-35B-A3B (IQ2_XXS)":  ("s", "#ff7f0e"),
        "Qwen3.5-35B-A3B (Q4_K_M)":   ("^", "#2ca02c"),
        "Qwen3.6-35B-A3B (Q4_K_M)":   ("D", "#d62728"),
        "Qwen3.5-27B-Dense (Q4_K_M)": ("v", "#9467bd"),
    }

    have_ppl = df.dropna(subset=["ppl_delta_pct"])
    for label, (marker, color) in markers.items():
        sub = have_ppl[have_ppl["model_label"] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["avg_bpw"], sub["ppl_delta_pct"],
            marker=marker, color=color, s=60, alpha=0.85,
            label=label, edgecolors="black", linewidth=0.4,
        )

    # Annotate our recommended configs
    highlights = [
        ("q8_0/vtq3_1", "q8_0+vtq3_1"),
        ("q8_0/vtq2_1", "q8_0+vtq2_1"),
    ]
    for cfg, text in highlights:
        sub = have_ppl[have_ppl["config"] == cfg]
        if sub.empty:
            continue
        # pick the median row for annotation
        row = sub.iloc[len(sub) // 2]
        ax.annotate(
            text,
            xy=(row["avg_bpw"], row["ppl_delta_pct"]),
            xytext=(10, 8), textcoords="offset points",
            fontsize=8, color="#444",
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5),
        )

    ax.axhline(0, color="#888", lw=0.6, ls="--")
    ax.set_xlabel("Average KV-cache bpw (lower = more compressed)")
    ax.set_ylabel("PPL delta vs f16 (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("llama-tq: PPL vs KV compression across 5 model/quant pairs")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(2, 17)

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
    ax.set_ylabel("TG128 tok/s (higher = faster)")
    ax.set_title("llama-tq: Decode throughput by KV cache config (Q4_K_M weights, 2x RTX 2060)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    _savefig("decode_throughput.png", fig)


# ----------------------------------------------------------------------------
# Chart 3: Cross-project Pareto frontier
# ----------------------------------------------------------------------------

def chart_cross_project():
    fig, ax = plt.subplots(figsize=(10, 6))

    # llama-tq points (just the non-broken configs on Qwen3.5-35B Q4_K_M for
    # apples-to-apples with TheTom)
    have_ppl = df.dropna(subset=["ppl_delta_pct"])
    our_q4km = have_ppl[have_ppl["model_quant"] == "Q4_K_M"]
    # only show valid VTQ / standard-type configs, not KTQ+VTQ broken combos
    our_q4km = our_q4km[~our_q4km["k_cache"].str.startswith("ktq")]

    ax.scatter(
        our_q4km["avg_bpw"], our_q4km["ppl_delta_pct"],
        marker="o", color="#1f77b4", s=70, alpha=0.85,
        label="llama-tq (2x RTX 2060, Q4_K_M)", edgecolors="black", linewidth=0.4,
    )

    # TheTom points
    tt = other[other["project"] == "TheTom"]
    ax.scatter(
        tt["avg_bpw"], tt["ppl_delta_pct"],
        marker="s", color="#ff7f0e", s=70, alpha=0.85,
        label="TheTom turboquant_plus (M5 Max, Q4_K_M)", edgecolors="black", linewidth=0.4,
    )

    # buun points
    bb = other[other["project"] == "buun"]
    ax.scatter(
        bb["avg_bpw"], bb["ppl_delta_pct"],
        marker="D", color="#2ca02c", s=70, alpha=0.85,
        label="buun TCQ (RTX 3090, Q6_K)", edgecolors="black", linewidth=0.4,
    )

    # Annotate specific points
    for _, row in tt.iterrows():
        ax.annotate(
            row["config"],
            xy=(row["avg_bpw"], row["ppl_delta_pct"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7, color="#cc5500",
        )
    for _, row in bb.iterrows():
        ax.annotate(
            row["config"],
            xy=(row["avg_bpw"], row["ppl_delta_pct"]),
            xytext=(6, -8), textcoords="offset points",
            fontsize=7, color="#2ca02c",
        )

    ax.axhline(0, color="#888", lw=0.6, ls="--")
    ax.set_xlabel("Average KV-cache bpw")
    ax.set_ylabel("PPL delta vs f16 (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("Cross-project comparison: PPL vs KV bpw (different HW/models, indicative only)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Caveat text
    ax.text(
        0.02, 0.02,
        "Note: different hardware, model variants, and PPL methodologies.\n"
        "buun uses KL-divergence on Q6_K; TheTom uses wikitext-2 on Q4_K_M MoE.\n"
        "Direct comparison is approximate.",
        transform=ax.transAxes, fontsize=7, color="#666",
        verticalalignment="bottom",
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
    ax.set_xlabel("PPL delta vs f16 (%)")
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
