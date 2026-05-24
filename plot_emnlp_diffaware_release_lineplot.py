from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_JSON = Path("result/7-opensourced-llm.json")
GIANT_SUMMARY_CSV = Path("result/evaluate_summary_N.csv")
FIG_DIR = Path("figures")
OUTPUT_STEM = "emnlp_lineplot_diffaware_release_1x4"

# A4 / EMNLP double-column friendly size.
FIGSIZE = (7.15, 2.45)

BENCHMARKS = [
    (1, "Benchmark 1"),
    (2, "Benchmark 2"),
    (3, "Benchmark 3"),
    (4, "Benchmark 4"),
]

# Only plot the four requested prompts.
PROMPTS_TO_PLOT = [
    "base",
    "rawls-cott-nodefine",
    "rawls-cott-nodefine-gt",
    "rawls-cott-nodefine-reverse",
]

PROMPT_LABELS = {
    "base": "base",
    "rawls-cott-nodefine": "D-J",
    "rawls-cott-nodefine-gt": "D-J oracle",
    "rawls-cott-nodefine-reverse": "D-J reverse",
}

PROMPT_STYLES = {
    "base": {
        "color": "#2B2B2B",
        "marker": "o",
        "linestyle": "None",
    },
    "rawls-cott-nodefine": {
        "color": "#0072B2",
        "marker": "^",
        "linestyle": "-",
    },
    "rawls-cott-nodefine-gt": {
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-.",
    },
    "rawls-cott-nodefine-reverse": {
        "color": "#CC79A7",
        "marker": "v",
        "linestyle": ":",
    },
}

# Release dates are used only to order the x-axis.
# X tick labels show model names only, without year/month labels.
MODEL_INFO = [
    {
        "name": "Meta-Llama-3-8B-Instruct",
        "label": "L3-8B",
        "release": "2024-04-18",
        "group": "Small LMs",
        "tie_order": 0,
    },
    {
        "name": "gemma-2-9b-it",
        "label": "G2-9B",
        "release": "2024-06-27",
        "group": "Small LMs",
        "tie_order": 1,
    },
    {
        "name": "Meta-Llama-3.1-8B-Instruct",
        "label": "L3.1-8B",
        "release": "2024-07-23",
        "group": "Small LMs",
        "tie_order": 2,
    },
    {
        "name": "Qwen2.5-7B-Instruct",
        "label": "Q2.5-7B",
        "release": "2024-09-19",
        "group": "Small LMs",
        "tie_order": 3,
    },
    {
        "name": "Qwen2.5-14B-Instruct",
        "label": "Q2.5-14B",
        "release": "2024-09-19",
        "group": "Small LMs",
        "tie_order": 4,
    },
    {
        "name": "Qwen3-4B",
        "label": "Q3-4B",
        "release": "2025-04-29",
        "group": "Small LMs",
        "tie_order": 5,
    },
    {
        "name": "Qwen3-8B",
        "label": "Q3-8B",
        "release": "2025-04-29",
        "group": "Small LMs",
        "tie_order": 6,
    },
    {
        "name": "GLM-5.1",
        "label": "GLM-5.1",
        "release": "2026-04-07",
        "group": "Giant LMs",
        "tie_order": 7,
    },
    {
        "name": "Kimi-K2.6",
        "label": "Kimi-K2.6",
        "release": "2026-04-20",
        "group": "Giant LMs",
        "tie_order": 8,
    },
]


def set_paper_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.4,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.05,
            "legend.frameon": False,
        }
    )


def ordered_model_info():
    return sorted(MODEL_INFO, key=lambda item: (item["release"], item["tie_order"]))


def _model_lookup():
    models = ordered_model_info()
    return {item["name"]: {**item, "x": idx} for idx, item in enumerate(models)}


def load_small_lm_results(model_lookup):
    raw = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    records = []

    for model_name, meta in model_lookup.items():
        if meta["group"] != "Small LMs":
            continue
        if model_name not in raw:
            raise KeyError(f"Missing small LM in {RESULT_JSON}: {model_name}")

        for prompt in PROMPTS_TO_PLOT:
            if prompt not in raw[model_name]:
                continue
            scores = raw[model_name][prompt]["diffaware_dist"]
            for benchmark_idx, score in enumerate(scores, start=1):
                records.append(
                    {
                        **meta,
                        "prompt": prompt,
                        "prompt_label": PROMPT_LABELS.get(prompt, prompt),
                        "benchmark": benchmark_idx,
                        "score": float(score),
                    }
                )

    return pd.DataFrame.from_records(records)


def load_giant_lm_results(model_lookup):
    if not GIANT_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {GIANT_SUMMARY_CSV}")

    source = pd.read_csv(GIANT_SUMMARY_CSV)
    giant_names = {
        model_name
        for model_name, meta in model_lookup.items()
        if meta["group"] == "Giant LMs"
    }
    source = source[
        source["model"].isin(giant_names) & source["prompt"].isin(PROMPTS_TO_PLOT)
    ].copy()

    records = []
    for row in source.itertuples(index=False):
        meta = model_lookup[row.model]
        records.append(
            {
                **meta,
                "prompt": row.prompt,
                "prompt_label": PROMPT_LABELS.get(row.prompt, row.prompt),
                "benchmark": int(row.idx),
                "score": float(row.diffaware),
            }
        )

    return pd.DataFrame.from_records(records)


def load_plot_data():
    model_lookup = _model_lookup()
    data = pd.concat(
        [
            load_small_lm_results(model_lookup),
            load_giant_lm_results(model_lookup),
        ],
        ignore_index=True,
    )

    data["prompt"] = pd.Categorical(
        data["prompt"], categories=PROMPTS_TO_PLOT, ordered=True
    )
    return data.sort_values(["benchmark", "prompt", "x"])


def _rounded_ylim(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0, 100, 20, 10

    y_min = float(values.min())
    y_max = float(values.max())
    span = y_max - y_min
    padding = max(2.0, span * 0.12)
    lower = max(0.0, np.floor((y_min - padding) / 5.0) * 5.0)
    upper = min(100.0, np.ceil((y_max + padding) / 5.0) * 5.0)
    if upper - lower < 10:
        center = (upper + lower) / 2.0
        lower = max(0.0, center - 5.0)
        upper = min(100.0, center + 5.0)

    scale_span = upper - lower
    major_step = 5 if scale_span <= 25 else 10 if scale_span <= 55 else 20
    return lower, upper, major_step, major_step / 2


def _format_axis(ax, values):
    lower, upper, major_step, minor_step = _rounded_ylim(values)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(minor_step))
    ax.grid(axis="y", which="major", color="#D6D6D6", linewidth=0.48)
    ax.grid(axis="y", which="minor", color="#ECECEC", linewidth=0.32)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def _add_group_labels(ax):
    transform = ax.get_xaxis_transform()
    y_line = -0.42
    y_text = -0.48

    ax.plot(
        [-0.35, 6.35],
        [y_line, y_line],
        transform=transform,
        color="#7A7A7A",
        linewidth=0.55,
        clip_on=False,
    )
    ax.plot(
        [6.65, 8.35],
        [y_line, y_line],
        transform=transform,
        color="#7A7A7A",
        linewidth=0.55,
        clip_on=False,
    )
    ax.text(
        3,
        y_text,
        "Small LMs",
        transform=transform,
        ha="center",
        va="top",
        fontsize=5.4,
        fontweight="semibold",
        clip_on=False,
    )
    ax.text(
        7.5,
        y_text,
        "Giant LMs",
        transform=transform,
        ha="center",
        va="top",
        fontsize=5.4,
        fontweight="semibold",
        clip_on=False,
    )


def plot_diffaware_release_lineplot(data):
    models = ordered_model_info()
    x = np.arange(len(models))
    tick_labels = [item["label"] for item in models]

    fig, axes = plt.subplots(
        1,
        len(BENCHMARKS),
        figsize=FIGSIZE,
        sharex=True,
        sharey=False,
    )

    for ax, (benchmark_idx, benchmark_label) in zip(axes, BENCHMARKS):
        benchmark_df = data[data["benchmark"] == benchmark_idx].copy()

        for prompt in PROMPTS_TO_PLOT:
            prompt_df = benchmark_df[benchmark_df["prompt"] == prompt].copy()
            prompt_df = prompt_df.set_index("x").reindex(x)
            style = PROMPT_STYLES[prompt]
            scores = prompt_df["score"].to_numpy(dtype=float)
            if prompt == "base":
                ax.scatter(
                    x,
                    scores,
                    label=PROMPT_LABELS.get(prompt, prompt),
                    s=16,
                    facecolors="white",
                    edgecolors=style["color"],
                    linewidths=0.7,
                    alpha=0.86,
                    zorder=2,
                )
            else:
                ax.plot(
                    x,
                    scores,
                    label=PROMPT_LABELS.get(prompt, prompt),
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=3.0,
                    markerfacecolor=style["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    alpha=0.94,
                    zorder=3,
                )

        ax.set_title(benchmark_label, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            tick_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.tick_params(axis="x", pad=1.0)
        ax.set_xlim(-0.45, len(models) - 0.55)
        _format_axis(ax, benchmark_df["score"].to_numpy(dtype=float))
        _add_group_labels(ax)

    axes[0].set_ylabel("Diff-aware score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(PROMPTS_TO_PLOT),
        columnspacing=0.85,
        handlelength=1.55,
        handletextpad=0.35,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.835,
        bottom=0.34,
        wspace=0.18,
    )
    return fig, axes


def save_figure(fig, stem=OUTPUT_STEM):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(
            FIG_DIR / f"{stem}.{ext}",
            bbox_inches="tight",
            transparent=True,
            dpi=600,
        )


def main():
    set_paper_style()
    data = load_plot_data()
    fig, _axes = plot_diffaware_release_lineplot(data)
    save_figure(fig)
    print(f"Saved {FIG_DIR / (OUTPUT_STEM + '.pdf')}")
    print(f"Saved {FIG_DIR / (OUTPUT_STEM + '.svg')}")


if __name__ == "__main__":
    main()
