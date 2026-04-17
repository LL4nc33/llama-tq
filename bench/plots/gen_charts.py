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

def chart_cross_project():
    # Horizontal bar chart: each row is one specific config, sorted by bpw
    # (most compressed at top). Color = project. No scatter, no overlap.
    have_ppl = df.dropna(subset=["ppl_delta_pct"]).copy()
    have_ppl["model_label"] = have_ppl["model_label"].str.replace(
        "Qwen3.5-27B-Dense", "Qwen3.5-27B Dense"
    )

    # Build comparison rows: each row = (label, bpw, ppl_delta, project)
    rows = []

    # Pick representative llama-tq configs: two flagship configs × models
    flagship_configs = ["q8_0/vtq3_1", "q8_0/vtq2_1"]
    for cfg in flagship_configs:
        sub = have_ppl[have_ppl["config"] == cfg]
        for _, r in sub.iterrows():
            cfg_short = cfg.replace("/", " + ")
            model_short = r["model_label"].replace("Qwen3.", "Q3.").replace(
                " Dense", " D").replace("-A3B", "")
            rows.append({
                "label": f"llama-tq  {cfg_short}  ({r['avg_bpw']:.2f} bpw)  on {model_short}",
                "bpw": r["avg_bpw"],
                "ppl_delta": r["ppl_delta_pct"],
                "project": "llama-tq",
            })

    # TheTom configs
    for _, r in other[other["project"] == "TheTom"].iterrows():
        rows.append({
            "label": f"TheTom    {r['config']}  ({r['avg_bpw']:.2f} bpw)  on Q3.5-35B MoE",
            "bpw": r["avg_bpw"],
            "ppl_delta": r["ppl_delta_pct"],
            "project": "TheTom",
        })

    # buun configs
    for _, r in other[other["project"] == "buun"].iterrows():
        rows.append({
            "label": f"buun      {r['config']}  ({r['avg_bpw']:.2f} bpw)  on Q3.5-27B",
            "bpw": r["avg_bpw"],
            "ppl_delta": r["ppl_delta_pct"],
            "project": "buun",
        })

    # Sort: by bpw ascending (most compressed at top), then by PPL delta
    rows.sort(key=lambda x: (x["bpw"], x["ppl_delta"]))

    project_colors = {
        "llama-tq": "#1f77b4",
        "TheTom":   "#ff7f0e",
        "buun":     "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(12, max(6, len(rows) * 0.38)))
    y_positions = list(range(len(rows)))
    labels = [r["label"] for r in rows]
    values = [r["ppl_delta"] for r in rows]
    colors = [project_colors[r["project"]] for r in rows]

    bars = ax.barh(y_positions, values, color=colors,
                   edgecolor="black", linewidth=0.4, height=0.7)

    # Put PPL delta value at the end of each bar
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        ax.text(x_pos + (0.2 if val >= 0 else -0.2),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}%" if abs(val) < 1 else f"{val:+.1f}%",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=8, color="#222")

    # Near-lossless band
    ax.axvspan(-0.5, 1.5, color="#e6f4ea", alpha=0.5, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.invert_yaxis()  # Top = most compressed
    ax.set_xlabel("PPL delta vs f16 (lower is better)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("Cross-project PPL comparison (sorted by KV bpw, most compressed at top)")
    ax.grid(True, axis="x", ls=":", alpha=0.4, zorder=0)
    ax.axvline(0, color="#666", lw=0.7)
    ax.set_axisbelow(True)

    # Legend: project colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c, edgecolor="black", label=p)
        for p, c in project_colors.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=9, framealpha=0.95)

    # Caveat below chart
    fig.text(
        0.5, -0.04 / max(1, len(rows) / 15),
        "Caveat: different hardware, weight quants, and PPL methodologies. "
        "TheTom uses Q4_K_M MoE on M5 Max. buun uses Q6_K with KL divergence on RTX 3090. "
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
