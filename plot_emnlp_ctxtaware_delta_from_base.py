from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_JSON = Path("result/7-opensourced-llm.json")
GIANT_SUMMARY_CSV = Path("result/evaluate_summary_N.csv")
FIG_DIR = Path("figures")
OUTPUT_STEM = "emnlp_ctxtaware_base_selfref_bar_1x4"

# EMNLP/ACL double-column friendly size.
FIGSIZE = (7.15, 3.25)

BENCHMARKS = [
    (1, "Benchmark 1"),
    (2, "Benchmark 2"),
    (3, "Benchmark 3"),
    (4, "Benchmark 4"),
]

# README: base is direct answering; debiasing/none/unbias-re are
# self-reflection baselines; rawls-cott-nodefine is the main D-J method.
PROMPTS_TO_PLOT = [
    "base",
    "debiasing",
    "none",
    "unbias-re",
    "rawls-cott-nodefine",
]

PROMPT_LABELS = {
    "base": "base",
    "debiasing": "debiasing",
    "none": "none",
    "unbias-re": "unbiased-re",
    "rawls-cott-nodefine": "D-J",
}

# Okabe-Ito inspired palette: readable in grayscale and colorblind-safe enough
# for a dense paper figure.
PROMPT_STYLES = {
    "base": {"color": "#2B2B2B", "edgecolor": "#2B2B2B"},
    "debiasing": {"color": "#AFAFAF", "edgecolor": "#8C8C8C"},
    "none": {"color": "#C7C7C7", "edgecolor": "#9E9E9E"},
    "unbias-re": {"color": "#969696", "edgecolor": "#777777"},
    "rawls-cott-nodefine": {"color": "#0072B2", "edgecolor": "#005A8F"},
}

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
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.3,
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
            "lines.linewidth": 1.0,
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
                raise KeyError(f"Missing prompt for {model_name}: {prompt}")
            scores = raw[model_name][prompt]["ctxtaware_dist"]
            for benchmark_idx, score in enumerate(scores, start=1):
                records.append(
                    {
                        **meta,
                        "prompt": prompt,
                        "prompt_label": PROMPT_LABELS[prompt],
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

    missing = sorted(
        {
            (model_name, prompt)
            for model_name in giant_names
            for prompt in PROMPTS_TO_PLOT
        }
        - set(zip(source["model"], source["prompt"]))
    )
    if missing:
        raise KeyError(f"Missing giant-LM rows in {GIANT_SUMMARY_CSV}: {missing}")

    records = []
    for row in source.itertuples(index=False):
        meta = model_lookup[row.model]
        records.append(
            {
                **meta,
                "prompt": row.prompt,
                "prompt_label": PROMPT_LABELS[row.prompt],
                "benchmark": int(row.idx),
                "score": float(row.ctxtaware),
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


def _format_axis(ax):
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
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


def plot_ctxtaware_base_selfref_bar(data):
    models = ordered_model_info()
    x = np.arange(len(models), dtype=float)
    tick_labels = [item["label"] for item in models]
    bar_width = 0.145
    offsets = dict(
        zip(
            PROMPTS_TO_PLOT,
            (np.arange(len(PROMPTS_TO_PLOT)) - (len(PROMPTS_TO_PLOT) - 1) / 2)
            * bar_width,
        )
    )

    fig, axes = plt.subplots(
        1,
        len(BENCHMARKS),
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )

    for ax, (benchmark_idx, benchmark_label) in zip(axes, BENCHMARKS):
        benchmark_df = data[data["benchmark"] == benchmark_idx].copy()

        for prompt in PROMPTS_TO_PLOT:
            prompt_df = benchmark_df[benchmark_df["prompt"] == prompt].copy()
            prompt_df = prompt_df.set_index("x").reindex(x)
            style = PROMPT_STYLES[prompt]
            ax.bar(
                x + offsets[prompt],
                prompt_df["score"].to_numpy(dtype=float),
                width=bar_width * 0.92,
                label=PROMPT_LABELS[prompt],
                color=style["color"],
                edgecolor=style["edgecolor"],
                linewidth=0.35,
                alpha=0.96,
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
        ax.set_xlim(-0.55, len(models) - 0.45)
        _format_axis(ax)
        _add_group_labels(ax)

    axes[0].set_ylabel("Ctxt-aware score")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(PROMPTS_TO_PLOT),
        columnspacing=0.7,
        handlelength=1.2,
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
    fig, _axes = plot_ctxtaware_base_selfref_bar(data)
    save_figure(fig)
    print(f"Saved {FIG_DIR / (OUTPUT_STEM + '.pdf')}")
    print(f"Saved {FIG_DIR / (OUTPUT_STEM + '.svg')}")


if __name__ == "__main__":
    main()
