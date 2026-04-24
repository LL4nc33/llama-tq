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
    # Pareto frontier: average KV bpw (= (K_bpw + V_bpw) / 2) vs PPL delta.
    # Shows the trade-off across all KV cache configurations — baselines
    # (f16/f16, q8_0/q8_0, q4_0/q4_0), v1 K/V types (ktq/vtq*_1),
    # and v2 Trellis V types (vtq*_2, post-Trick-6).

    # Single-V "averaged across 35B-A3B" points from benchmarks.csv.
    # Use k_cache = f16 so bpw is dominated by V-cache quant.
    V_BPW = {
        "f16": 16.0, "q8_0": 8.5, "q4_0": 4.5,
        "vtq2_1": 2.5, "vtq3_1": 3.5, "vtq4_1": 4.5,
    }
    K_BPW = dict(V_BPW)
    K_BPW["ktq2_1"] = 3.5  # KTQ types carry the K-side header overhead
    K_BPW["ktq3_1"] = 4.5
    K_BPW["ktq4_1"] = 5.5

    def avg_bpw(k, v):
        return (K_BPW.get(k, 16.0) + V_BPW.get(v, 16.0)) / 2

    # v1 + baseline configs — averaged across 35B-A3B model/quant pairs
    v1_configs = [
        ("f16", "f16"),
        ("q8_0", "q8_0"),
        ("q4_0", "q4_0"),
        ("f16", "vtq3_1"),
        ("q8_0", "vtq3_1"),
        ("f16", "vtq2_1"),
        ("q8_0", "vtq2_1"),
        ("q4_0", "vtq2_1"),
    ]
    v1_points = []
    for k, v in v1_configs:
        sub = df[(df["k_cache"] == k) & (df["v_cache"] == v)
                 & df["ppl"].notna()
                 & df["model"].str.contains("35B-A3B")]
        if not sub.empty:
            v1_points.append((f"{k}/{v}", avg_bpw(k, v),
                              sub["ppl_delta_pct"].mean()))

    # KTQ×VTQ combos — from production deploy measurements on gpu00
    # (source: docs/bench-qwen35-122b-a10b.md + devlogs for 35B).
    # These are representative configs not fully in benchmarks.csv yet.
    ktq_vtq_points = [
        ("ktq2_1/vtq2_1", avg_bpw("ktq2_1", "vtq2_1"), 6.9),
        ("ktq2_1/vtq3_1", avg_bpw("ktq2_1", "vtq3_1"), 1.4),
        ("ktq3_1/vtq3_1", avg_bpw("ktq3_1", "vtq3_1"), 1.1),
    ]

    # v2 Trellis V-types (Qwen3.5-0.8B wikitext-2, ctx=512, 5 chunks, f16 K)
    v2_points = [
        ("f16/vtq2_2", (16.0 + 2.06) / 2, 7.74),
        ("f16/vtq3_2", (16.0 + 3.06) / 2, 1.05),
        ("f16/vtq4_2", (16.0 + 4.06) / 2, 0.44),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    def plot_group(points, color, marker, size, label, weight="normal"):
        for name, bpw, delta in points:
            ax.scatter(bpw, delta, s=size, c=color, marker=marker,
                       edgecolor="black", linewidth=0.6, zorder=3)
            ax.annotate(name, (bpw, delta), xytext=(6, 4),
                        textcoords="offset points", fontsize=8,
                        color=color, fontweight=weight)
        ax.scatter([], [], s=size, c=color, marker=marker,
                   edgecolor="black", linewidth=0.6, label=label)

    plot_group(v1_points,       "#1f77b4", "o", 110, "v1 baseline + vtq*_1 (35B-A3B)")
    plot_group(ktq_vtq_points,  "#2ca02c", "s", 130, "KTQ×VTQ combos (35B-A3B prod)", weight="bold")
    plot_group(v2_points,       "#d62728", "*", 220, "v2 Trellis (0.8B, Trick-6)",    weight="bold")

    # Pareto frontier
    all_pts = sorted(
        [(bpw, d) for _, bpw, d in v1_points + ktq_vtq_points + v2_points],
        key=lambda t: (t[0], t[1]),
    )
    pareto = []
    best = float("inf")
    for bpw, d in all_pts:
        if d < best:
            pareto.append((bpw, d))
            best = d
    if len(pareto) >= 2:
        xs, ys = zip(*pareto)
        ax.plot(xs, ys, "--", color="#888", alpha=0.5, linewidth=1.2,
                label="Pareto frontier", zorder=2)

    # Near-lossless band
    ax.axhspan(0, 1.5, color="#e6f4ea", alpha=0.5, zorder=0)
    ax.text(0.98, 0.75, "near-lossless zone (<1.5%)",
            transform=ax.get_yaxis_transform(), fontsize=8,
            color="#2ca02c", ha="right", style="italic", va="center")

    ax.set_xlabel("Average KV cache bpw ((K+V)/2, lower = more compression)")
    ax.set_ylabel("PPL delta vs f16/f16 (lower is better)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_title("llama-tq KV-cache: bpw / quality Pareto frontier")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(1.5, 17)
    ax.set_ylim(-0.5, max(10, ax.get_ylim()[1]))

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

def _unused_chart_cross_project_2d():
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

def chart_cross_project_dual():
    # Two side-by-side horizontal bar charts that share the y-axis:
    # left panel = PPL delta, right panel = decode throughput delta.
    # Each row = one (project, config) pair. This makes both dimensions
    # visible at once without a 2D scatter.
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

    # Pick representative configs to keep the chart readable.
    # llama-tq: one vtq3_1 (best quality) + one vtq2_1 per weight quant class,
    # choose the Qwen3.5-35B Q4_K_M result as our reference (matches TheTom).
    ref_model = "Qwen3.5-35B-A3B (Q4_K_M)"
    ours = have[have["model_label"] == ref_model].copy()
    picks = []
    for cfg, short in [
        ("q8_0/vtq3_1", "q8_0 + vtq3_1 (6.25 bpw)"),
        ("q8_0/vtq2_1", "q8_0 + vtq2_1 (5.5 bpw)"),
        ("q4_0/vtq2_1", "q4_0 + vtq2_1 (3.5 bpw)"),
        ("q4_0/q4_0",   "q4_0 + q4_0   (4.5 bpw)"),
    ]:
        sub = ours[ours["config"] == cfg]
        if sub.empty:
            continue
        r = sub.iloc[0]
        picks.append({
            "label": f"llama-tq  {short}",
            "ppl": r["ppl_delta_pct"],
            "tg": r["tg_delta_pct"],
            "project": "llama-tq",
        })

    for _, r in other[other["project"] == "TheTom"].iterrows():
        picks.append({
            "label": f"TheTom    {r['config']} ({r['avg_bpw']:.2f} bpw)",
            "ppl": r["ppl_delta_pct"],
            "tg": r["tg_delta_pct"],
            "project": "TheTom",
        })

    for _, r in other[other["project"] == "buun"].iterrows():
        picks.append({
            "label": f"buun      {r['config']} ({r['avg_bpw']:.2f} bpw)",
            "ppl": r["ppl_delta_pct"],
            "tg": r["tg_delta_pct"],
            "project": "buun",
        })

    # Sort by the sum of "badness" (PPL up + TG down): best first at top
    picks.sort(key=lambda p: p["ppl"] - p["tg"])

    project_colors = {
        "llama-tq": "#1f77b4",
        "TheTom":   "#ff7f0e",
        "buun":     "#2ca02c",
    }

    # Labels on far left only, two data panels on the right with equal width
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14, 7.5), sharey=True,
        gridspec_kw={"wspace": 0.25, "width_ratios": [1, 1]},
    )

    y_pos = list(range(len(picks)))
    labels = [p["label"] for p in picks]
    ppl_vals = [p["ppl"] for p in picks]
    tg_vals = [p["tg"] for p in picks]
    colors = [project_colors[p["project"]] for p in picks]

    # Show the PPL delta as a positive bar (all values are >= 0 anyway).
    # Bars grow to the right in both panels; "smaller bar = better".
    abs_ppl = [max(0, v) for v in ppl_vals]
    abs_tg = [abs(v) for v in tg_vals]

    # LEFT: PPL delta magnitude
    axL.barh(y_pos, abs_ppl, color=colors, edgecolor="black",
             linewidth=0.4, height=0.7)
    for i, v in enumerate(ppl_vals):
        axL.text(max(0, v) + 0.15, i,
                 f"+{v:.2f}%" if abs(v) < 1 else f"+{v:.1f}%",
                 va="center", ha="left", fontsize=9, color="#222",
                 fontweight="bold")
    axL.axvspan(0, 1.5, color="#e6f4ea", alpha=0.5, zorder=0)
    axL.set_xlabel("PPL delta vs f16 (shorter is better)")
    axL.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    axL.set_title("Quality hit (PPL increase vs f16)",
                  fontsize=11, pad=10)
    axL.grid(True, axis="x", ls=":", alpha=0.4, zorder=0)
    axL.set_yticks(y_pos)
    axL.set_yticklabels(labels, fontsize=9, family="monospace")
    axL.set_axisbelow(True)
    axL.set_xlim(0, max(abs_ppl) * 1.2)

    # RIGHT: TG delta magnitude (always a loss vs baseline, so plot magnitude)
    axR.barh(y_pos, abs_tg, color=colors, edgecolor="black",
             linewidth=0.4, height=0.7)
    for i, v in enumerate(tg_vals):
        axR.text(abs(v) + 0.4, i, f"{v:+.1f}%",
                 va="center", ha="left", fontsize=9, color="#222",
                 fontweight="bold")
    axR.axvspan(0, 5, color="#e6f4ea", alpha=0.5, zorder=0)
    axR.set_xlabel("Decode throughput loss vs f16 (shorter is better)")
    axR.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    axR.set_title("Speed hit (decode throughput loss vs f16)",
                  fontsize=11, pad=10)
    axR.grid(True, axis="x", ls=":", alpha=0.4, zorder=0)
    axR.set_axisbelow(True)
    axR.set_xlim(0, max(abs_tg) * 1.2)

    axL.invert_yaxis()  # best at top (smallest combined impact)

    # Shared legend at top
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c, edgecolor="black", label=p)
        for p, c in project_colors.items()
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.97),
               frameon=False)

    fig.suptitle("Cross-project comparison: quality vs decode speed "
                 "(sorted best to worst combined impact)",
                 fontsize=13, y=1.0)

    fig.text(
        0.5, 0.005,
        "llama-tq: 2x RTX 2060 (Qwen3.5-35B-A3B Q4_K_M).  "
        "TheTom: M5 Max (Qwen3.5-35B-A3B Q4_K_M).  "
        "buun: RTX 3090 (Qwen3.5-27B Q6_K, KL divergence).  "
        "Absolute numbers across projects are approximate.",
        ha="center", fontsize=8.5, color="#555", style="italic",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    _savefig("cross_project.png", fig)


def main():
    print(f"Output dir: {OUT}")
    chart_ppl_vs_bpw()
    chart_decode_throughput()
    chart_cross_project_dual()
    chart_vtq2_variance()
    print("done.")


if __name__ == "__main__":
    main()
