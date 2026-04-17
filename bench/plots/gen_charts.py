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
    # Grouped bar chart: one group per KV config (sorted by typical avg bpw),
    # bars within a group = models. Immediately readable.
    have_ppl = df.dropna(subset=["ppl_delta_pct"]).copy()
    have_ppl["model_label"] = have_ppl["model_label"].str.replace(
        "Qwen3.5-27B-Dense", "Qwen3.5-27B Dense"
    )

    # Only show the representative asymmetric/clean configs (skip broken
    # KTQ+VTQ and skip redundant f16/f16 baseline which is always 0%)
    interesting = [
        ("q8_0/vtq3_1", "q8_0 + vtq3_1 (6.25 bpw)"),
        ("f16/vtq3_1",  "f16  + vtq3_1 (10.0 bpw)"),
        ("q4_0/q4_0",   "q4_0 + q4_0   (4.5 bpw)"),
        ("q8_0/vtq2_1", "q8_0 + vtq2_1 (5.5 bpw)"),
        ("q4_0/vtq2_1", "q4_0 + vtq2_1 (3.5 bpw)"),
    ]
    model_order = [
        "Qwen3.5-27B Dense (Q4_K_M)",
        "Qwen3.5-35B-A3B (IQ2_XS)",
        "Qwen3.6-35B-A3B (IQ2_XXS)",
        "Qwen3.6-35B-A3B (Q4_K_M)",
        "Qwen3.5-35B-A3B (Q4_K_M)",
    ]
    model_colors = {
        "Qwen3.5-27B Dense (Q4_K_M)": "#9467bd",
        "Qwen3.5-35B-A3B (IQ2_XS)":   "#1f77b4",
        "Qwen3.6-35B-A3B (IQ2_XXS)":  "#ff7f0e",
        "Qwen3.6-35B-A3B (Q4_K_M)":   "#d62728",
        "Qwen3.5-35B-A3B (Q4_K_M)":   "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    n_models = len(model_order)
    n_configs = len(interesting)
    width = 0.16
    x_base = range(n_configs)

    for i, model in enumerate(model_order):
        vals = []
        for cfg, _ in interesting:
            sub = have_ppl[(have_ppl["model_label"] == model)
                           & (have_ppl["config"] == cfg)]
            vals.append(sub["ppl_delta_pct"].iloc[0] if not sub.empty else 0)
        offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x_base]
        bars = ax.bar(offsets, vals, width, color=model_colors[model],
                      edgecolor="black", linewidth=0.4, label=model)
        # Put value on top of each bar
        for bar, v in zip(bars, vals):
            if v == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=7, color="#333")

    # Near-lossless band
    ax.axhspan(0, 1.5, color="#e6f4ea", alpha=0.5, zorder=0)
    ax.text(n_configs - 0.5, 0.75, "near-lossless zone (<1.5%)",
            fontsize=7.5, color="#2ca02c", ha="right",
            style="italic", va="center")

    ax.set_xticks(list(x_base))
    ax.set_xticklabels([label for _, label in interesting], fontsize=9)
    ax.set_ylabel("PPL delta vs f16 (lower is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("llama-tq: PPL delta by KV config and model "
                 "(5 model/quant pairs, 2x RTX 2060)")
    ax.legend(loc="upper left", fontsize=8, title="Model / weight quant",
              title_fontsize=8, framealpha=0.95)
    ax.grid(True, axis="y", ls=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(12, ax.get_ylim()[1]))

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

def chart_cross_project_2d():
    # Scatter: PPL delta (x) vs decode overhead (y). Each project gets
    # its own marker. Bottom-left = "best" (low PPL, low decode hit).
    # Only uses configs where both metrics are reported.
    fig, ax = plt.subplots(figsize=(12, 7.5))

    # llama-tq: compute TG overhead vs f16 baseline per model.
    # Use Q4_K_M MoE (Qwen3.5) as the reference for a fair cross-project
    # pairing with TheTom (who also uses Q4_K_M MoE).
    have = df.dropna(subset=["ppl_delta_pct", "tg128"]).copy()
    have["model_label"] = have["model_label"].str.replace(
        "Qwen3.5-27B-Dense", "Qwen3.5-27B Dense"
    )
    baselines = have[have["config"] == "f16/f16"].set_index("model_label")["tg128"]

    def tg_delta(row):
        b = baselines.get(row["model_label"])
        if b is None or pd.isna(b):
            return None
        return (row["tg128"] - b) / b * 100.0

    have["tg_delta_pct"] = have.apply(tg_delta, axis=1)
    ours = have[
        have["config"].isin(["q8_0/vtq3_1", "q8_0/vtq2_1", "q4_0/vtq2_1", "q4_0/q4_0"])
        & have["tg_delta_pct"].notna()
    ]

    # Plot llama-tq points grouped by config type
    cfg_style = {
        "q8_0/vtq3_1": ("o", "q8_0 + vtq3_1"),
        "q8_0/vtq2_1": ("s", "q8_0 + vtq2_1"),
        "q4_0/vtq2_1": ("^", "q4_0 + vtq2_1"),
        "q4_0/q4_0":   ("v", "q4_0 / q4_0 (baseline quant)"),
    }
    for cfg, (marker, label) in cfg_style.items():
        sub = ours[ours["config"] == cfg]
        ax.scatter(sub["ppl_delta_pct"], sub["tg_delta_pct"],
                   marker=marker, color="#1f77b4", s=120, alpha=0.85,
                   edgecolors="black", linewidth=0.7, zorder=4,
                   label=f"llama-tq {label}")

    # TheTom (PPL and decode delta already given)
    tt = other[other["project"] == "TheTom"]
    ax.scatter(tt["ppl_delta_pct"], tt["tg_delta_pct"],
               marker="*", color="#ff7f0e", s=300, alpha=0.9,
               edgecolors="black", linewidth=0.9, zorder=5,
               label="TheTom (turbo2/3/4)")
    for _, r in tt.iterrows():
        ax.annotate(r["config"],
                    xy=(r["ppl_delta_pct"], r["tg_delta_pct"]),
                    xytext=(10, 4), textcoords="offset points",
                    fontsize=9, color="#cc5500", fontweight="bold")

    # buun
    bb = other[other["project"] == "buun"]
    ax.scatter(bb["ppl_delta_pct"], bb["tg_delta_pct"],
               marker="D", color="#2ca02c", s=180, alpha=0.9,
               edgecolors="black", linewidth=0.9, zorder=5,
               label="buun TCQ")
    for _, r in bb.iterrows():
        ax.annotate(r["config"],
                    xy=(r["ppl_delta_pct"], r["tg_delta_pct"]),
                    xytext=(10, -10), textcoords="offset points",
                    fontsize=9, color="#206020", fontweight="bold")

    # "Better" direction indicator: bottom-left = low PPL loss + low decode cost
    ax.axhline(0, color="#888", lw=0.6, ls="--", zorder=1)
    ax.axvline(0, color="#888", lw=0.6, ls="--", zorder=1)

    # Shade the "best" quadrant (PPL <2%, decode >-5%)
    ax.add_patch(plt.Rectangle((-1, -5), 3, 5.5, alpha=0.15, color="green",
                                zorder=0, label="_nolegend_"))
    ax.text(1.9, -0.3, "best quadrant\n(low PPL hit,\nfast decode)",
            fontsize=9, color="#2ca02c", ha="right", va="top",
            style="italic")

    ax.set_xlabel("PPL delta vs f16 (lower is better →)")
    ax.set_ylabel("Decode throughput delta vs f16 (higher is better ↑)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("Cross-project: quality vs decode speed tradeoff\n"
                 "(each project on different hardware, see caveat)",
                 fontsize=12)
    ax.grid(True, ls=":", alpha=0.4, zorder=0)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-25, 3)

    # Caveat
    fig.text(
        0.5, 0.005,
        "Each project tested on different hardware (llama-tq: 2x RTX 2060, "
        "TheTom: M5 Max, buun: RTX 3090) and different model/metric combinations. "
        "Relative positions are indicative, not a strict ranking.",
        ha="center", fontsize=8.5, color="#555", style="italic",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    _savefig("cross_project.png", fig)


def _chart_cross_project_unused():
    # Two-panel small-multiples approach: one panel per bit-width tier,
    # comparing the three projects at approximately the same compression
    # level. This answers "at ~3.5 bpw, how do they compare?" without
    # forcing readers to parse a long list.
    have_ppl = df.dropna(subset=["ppl_delta_pct"]).copy()
    have_ppl["model_label"] = have_ppl["model_label"].str.replace(
        "Qwen3.5-27B-Dense", "Qwen3.5-27B Dense"
    )

    # Tier 1: ~3-3.5 bpw (aggressive compression)
    tier_low = []
    tier_low += [
        ("llama-tq", f"q4_0 + vtq2_1 ({r['avg_bpw']:.1f})",
         r["model_label"], r["ppl_delta_pct"])
        for _, r in have_ppl[have_ppl["config"] == "q4_0/vtq2_1"].iterrows()
    ]
    for _, r in other[(other["project"] == "TheTom")
                      & (other["config"].isin(["turbo2", "turbo3"]))].iterrows():
        tier_low.append(("TheTom", f"{r['config']} ({r['avg_bpw']:.1f})",
                         "Qwen3.5-35B MoE", r["ppl_delta_pct"]))
    for _, r in other[(other["project"] == "buun")
                      & (other["config"] == "turbo3_tcq")].iterrows():
        tier_low.append(("buun", f"{r['config']} ({r['avg_bpw']:.1f})",
                         "Qwen3.5-27B", r["ppl_delta_pct"]))

    # Tier 2: ~4-6 bpw (balanced)
    tier_mid = []
    tier_mid += [
        ("llama-tq", f"q8_0 + vtq3_1 ({r['avg_bpw']:.1f})",
         r["model_label"], r["ppl_delta_pct"])
        for _, r in have_ppl[have_ppl["config"] == "q8_0/vtq3_1"].iterrows()
    ]
    tier_mid += [
        ("llama-tq", f"q8_0 + vtq2_1 ({r['avg_bpw']:.1f})",
         r["model_label"], r["ppl_delta_pct"])
        for _, r in have_ppl[have_ppl["config"] == "q8_0/vtq2_1"].iterrows()
    ]
    for _, r in other[(other["project"] == "TheTom")
                      & (other["config"].isin(["turbo4", "q8_0+turbo3"]))].iterrows():
        tier_mid.append(("TheTom", f"{r['config']} ({r['avg_bpw']:.1f})",
                         "Qwen3.5-35B MoE", r["ppl_delta_pct"]))

    project_colors = {
        "llama-tq": "#1f77b4",
        "TheTom":   "#ff7f0e",
        "buun":     "#2ca02c",
    }

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 8), sharex=True)

    for ax, tier, title in [
        (axL, tier_low, "Aggressive compression (~2.25-3.5 bpw avg)"),
        (axR, tier_mid, "Balanced / conservative (~4.25-6.25 bpw avg)"),
    ]:
        # Sort by PPL delta ascending (best on top)
        tier = sorted(tier, key=lambda r: r[3])
        y_pos = list(range(len(tier)))
        labels = [f"{proj:9s}  {cfg}\n  on {model}" for proj, cfg, model, _ in tier]
        values = [v for *_, v in tier]
        colors = [project_colors[proj] for proj, *_ in tier]

        bars = ax.barh(y_pos, values, color=colors,
                       edgecolor="black", linewidth=0.4, height=0.65)

        # Value labels
        for bar, val in zip(bars, values):
            x_pos = bar.get_width()
            offset = 0.15 if val >= 0 else -0.15
            ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}%" if abs(val) < 1 else f"{val:+.1f}%",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=9, color="#222", fontweight="bold")

        # Near-lossless band
        ax.axvspan(-0.5, 1.5, color="#e6f4ea", alpha=0.5, zorder=0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8, family="monospace")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, pad=12)
        ax.grid(True, axis="x", ls=":", alpha=0.4, zorder=0)
        ax.axvline(0, color="#666", lw=0.7)
        ax.set_axisbelow(True)

    axL.set_xlabel("PPL delta vs f16 (lower is better)")
    axR.set_xlabel("PPL delta vs f16 (lower is better)")
    axL.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

    # Legend (once, above)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c, edgecolor="black", label=p)
        for p, c in project_colors.items()
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.94),
               frameon=False)

    fig.suptitle("Cross-project PPL comparison at matched bit-widths",
                 fontsize=14, y=0.99)

    # Caveat below
    fig.text(
        0.5, 0.02,
        "Different hardware, weight quants, and PPL methodologies. "
        "TheTom: M5 Max, Q4_K_M MoE. "
        "buun: RTX 3090, Q6_K, KL divergence at 2K ctx. "
        "llama-tq: 2x RTX 2060, wikitext-2 at 3 chunks. "
        "Comparison is approximate.",
        ha="center", fontsize=9, color="#555", style="italic",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.9])

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
    chart_cross_project_2d()
    chart_vtq2_variance()
    print("done.")


if __name__ == "__main__":
    main()
