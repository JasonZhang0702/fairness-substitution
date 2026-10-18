from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
OPEN_SOURCE_JSON = ROOT_DIR / "result" / "7-opensourced-llm.json"
SUMMARY_CSV = ROOT_DIR / "result" / "evaluate_summary_N.csv"
FIG_DIR = ROOT_DIR / "figures"
WINNER_ICON = FIG_DIR / "winner.png"
OUTPUT_STEM = "diffaware_mainresult"

# Match the supplied paper figure: a compact, double-column 1 x 4 layout.
FIGSIZE = (7.22, 2.75)

BENCHMARKS = [
    (1, "N1: BBQ"),
    (2, "N2: Social Bias Frames"),
    (3, "N3: Affirmative Action"),
    (4, "N4: Cultural Appropriation"),
]

WINNER_BADGE_POSITIONS = {
    1: (0.84, 0.68),
    2: (0.84, 0.90),
    3: (0.84, 0.90),
    4: (0.16, 0.90),
}

SELF_DEBIASING_PROMPTS = ("debiasing", "none", "unbias-re")
OUR_TDD_SOURCE_PROMPT = "rawls-cott-nodefine"
SOURCE_PROMPTS = ("base", *SELF_DEBIASING_PROMPTS, "rawls", OUR_TDD_SOURCE_PROMPT)

# Few-shot-1 DiffAware distributions supplied with the experiment results.
FEW_SHOT_DIFFAWARE = {
    "Meta-Llama-3-8B-Instruct": [63.44, 58.62, 69.51, 0.4],
    "Meta-Llama-3.1-8B-Instruct": [53.0, 52.7, 64.0, 1.0],
    "gemma-2-9b-it": [44.2, 54.0, 70.1, 58.4],
    "Qwen2.5-7B-Instruct": [63.6, 48.5, 40.3, 44.8],
    "Qwen2.5-14B-Instruct": [20.3, 36.4, 48.8, 75.0],
    "Qwen3-4B": [49.5, 40.4, 17.2, 8.9],
    "Qwen3-8B": [52.2, 20.3, 6.6, 10.4],
    "GLM-5.1": [76.30, 47.60, 60.20, 84.40],
    "Kimi-K2.6": [65.70, 49.30, 61.20, 89.30],
    "gpt-5.6-luna": [37.2, 40.9, 86.5, 71.3],
}

# GPT-5.6-Luna is not present in the repository result files yet. Unlike the
# other models, its Our TDD result uses the supplied rawls-cott experiment.
GPT_LUNA_DIFFAWARE = {
    "base": [17.2, 27.08, 76.8, 39.78],
    "debiasing": [0.0, 8.39, 6.4, 4.66],
    "none": [2.5, 18.57, 63.1, 28.02],
    "unbias-re": [0.0, 8.93, 4.8, 2.13],
    "rawls": [30.0, 18.9, 82.9, 34.21],
    "rawls-cott": [41.9, 43.6, 89.7, 55.8],
    "rawls-cott-gt": [49.2, 43.9, 91.6, 65.4],
    "rawls-cott-reverse": [37.4, 42.7, 41.6, 17.1],
}

MODEL_INFO = [
    {"name": "Meta-Llama-3-8B-Instruct", "label": "L3-8B", "source": "json"},
    {
        "name": "Meta-Llama-3.1-8B-Instruct",
        "label": "L3.1-8B",
        "source": "json",
    },
    {"name": "gemma-2-9b-it", "label": "G2-9B", "source": "json"},
    {"name": "Qwen2.5-7B-Instruct", "label": "Q2.5-7B", "source": "json"},
    {
        "name": "Qwen2.5-14B-Instruct",
        "label": "Q2.5-14B",
        "source": "json",
    },
    {"name": "Qwen3-4B", "label": "Q3-4B", "source": "json"},
    {"name": "Qwen3-8B", "label": "Q3-8B", "source": "json"},
    {"name": "GLM-5.1", "label": "GLM-5.1", "source": "csv"},
    {"name": "Kimi-K2.6", "label": "Kimi-K2.6", "source": "csv"},
    {"name": "gpt-5.6-luna", "label": "GPT-5.6", "source": "inline"},
]

# Base is drawn as a translucent thick reference line spanning each model group.
BASE_SPEC = {
    "label": "Base",
    "color": "#303030",
    "linewidth": 1.9,
    "alpha": 0.58,
}

# Our TDD is a wide background bar containing the three comparison bars.
OUR_TDD_SPEC = {
    "key": "rawls-cott",
    "label": "Our TDD",
    "color": "#9ECAE1",
    "edgecolor": "#6BAED6",
}

# The three comparison methods overlaid inside the Our TDD bar.
METHOD_SPECS = [
    {
        "key": "self-debiasing",
        "label": "Self-Debiasing",
        "color": "#8F8F8F",
        "edgecolor": "#737373",
    },
    {
        "key": "rawls",
        "label": "PD-Only",
        "color": "#B3B3B3",
        "edgecolor": "#929292",
    },
    {
        "key": "few-shot",
        "label": "Few-Shot",
        "color": "#D2D2D2",
        "edgecolor": "#A8A8A8",
    },
]


def set_paper_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.5,
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
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


def _validate_scores(model_name, prompt, scores):
    if len(scores) != len(BENCHMARKS):
        raise ValueError(
            f"Expected four DiffAware values for {model_name}/{prompt}; "
            f"found {len(scores)}"
        )
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite DiffAware value for {model_name}/{prompt}")
    if ((values < 0) | (values > 100)).any():
        raise ValueError(f"DiffAware value outside [0, 100] for {model_name}/{prompt}")
    return values


def _append_scores(records, model, x_position, prompt, scores):
    values = _validate_scores(model["name"], prompt, scores)
    for benchmark_idx, score in enumerate(values, start=1):
        records.append(
            {
                "name": model["name"],
                "label": model["label"],
                "x": x_position,
                "prompt": prompt,
                "benchmark": benchmark_idx,
                "score": float(score),
            }
        )


def _load_json_scores(raw, model_name, prompt):
    if model_name not in raw:
        raise KeyError(f"Missing model in {OPEN_SOURCE_JSON}: {model_name}")
    if prompt not in raw[model_name]:
        raise KeyError(f"Missing prompt for {model_name}: {prompt}")
    return raw[model_name][prompt]["diffaware_dist"]


def _load_csv_scores(summary, model_name, prompt):
    rows = summary[
        (summary["model"] == model_name) & (summary["prompt"] == prompt)
    ].sort_values("idx")
    if rows["idx"].tolist() != [1, 2, 3, 4]:
        raise KeyError(f"Missing benchmark rows in {SUMMARY_CSV}: {model_name}/{prompt}")
    return rows["diffaware"].tolist()


def load_plot_data():
    raw = json.loads(OPEN_SOURCE_JSON.read_text(encoding="utf-8"))
    summary = pd.read_csv(SUMMARY_CSV)
    records = []

    for x_position, model in enumerate(MODEL_INFO):
        model_name = model["name"]
        for prompt in SOURCE_PROMPTS:
            if model["source"] == "json":
                scores = _load_json_scores(raw, model_name, prompt)
            elif model["source"] == "csv":
                scores = _load_csv_scores(summary, model_name, prompt)
            elif model["source"] == "inline":
                source_prompt = (
                    "rawls-cott" if prompt == OUR_TDD_SOURCE_PROMPT else prompt
                )
                scores = GPT_LUNA_DIFFAWARE[source_prompt]
            else:
                raise ValueError(f"Unknown data source for {model_name}: {model['source']}")
            _append_scores(records, model, x_position, prompt, scores)

        if model_name not in FEW_SHOT_DIFFAWARE:
            raise KeyError(f"Missing few-shot-1 DiffAware scores for {model_name}")
        _append_scores(
            records,
            model,
            x_position,
            "few-shot",
            FEW_SHOT_DIFFAWARE[model_name],
        )

    data = pd.DataFrame.from_records(records)
    expected_rows = len(MODEL_INFO) * len(BENCHMARKS) * (
        len(SOURCE_PROMPTS) + 1
    )
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; found {len(data)}")
    return data.sort_values(["benchmark", "x", "prompt"])


def _single_score(model_data, prompt):
    values = model_data.loc[model_data["prompt"] == prompt, "score"].to_numpy(
        dtype=float
    )
    if len(values) != 1:
        raise ValueError(
            f"Expected one score for prompt={prompt!r}; found {len(values)}"
        )
    return float(values[0])


def _method_values(model_data):
    self_debiasing = np.mean(
        [_single_score(model_data, prompt) for prompt in SELF_DEBIASING_PROMPTS]
    )
    return {
        "base": _single_score(model_data, "base"),
        "self-debiasing": float(self_debiasing),
        "rawls": _single_score(model_data, "rawls"),
        "few-shot": _single_score(model_data, "few-shot"),
        "rawls-cott": _single_score(model_data, OUR_TDD_SOURCE_PROMPT),
    }


def _our_tdd_win_count(benchmark_data):
    wins = 0
    for model in MODEL_INFO:
        model_data = benchmark_data[benchmark_data["name"] == model["name"]]
        values = _method_values(model_data)
        our_tdd = values["rawls-cott"]
        best_comparison = max(
            value for key, value in values.items() if key != "rawls-cott"
        )
        wins += int(our_tdd >= best_comparison)
    return wins


def _add_winner_badge(ax, winner_image, winner_count, position):
    icon = OffsetImage(winner_image, zoom=0.025)
    count = TextArea(
        f"{winner_count}/10",
        textprops={
            "color": "#6F5600",
            "fontfamily": "DejaVu Serif",
            "fontsize": 7.2,
            "fontweight": "bold",
        },
    )
    content = HPacker(children=[icon, count], align="center", pad=0, sep=2)
    badge = AnnotationBbox(
        content,
        position,
        xycoords="axes fraction",
        box_alignment=(0.5, 0.5),
        frameon=False,
        zorder=10,
    )
    ax.add_artist(badge)
    return badge


def _format_axis(ax, panel_idx, model_labels):
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
    ax.grid(axis="y", which="major", color="#D0D0D0", linewidth=0.52)
    ax.grid(axis="y", which="minor", color="#E9E9E9", linewidth=0.34)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    # Shift the label anchors slightly right to align the rotated names with
    # their model groups.
    label_positions = np.arange(len(model_labels), dtype=float) + 0.08
    ax.set_xticks(label_positions)
    ax.set_xticklabels(
        model_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", pad=1.0)
    ax.tick_params(axis="y", labelleft=panel_idx == 0)
    ax.set_xlim(-0.55, len(model_labels) - 0.45)


def plot_diffaware_bars(data):
    if not WINNER_ICON.exists():
        raise FileNotFoundError(f"Winner icon not found: {WINNER_ICON}")
    winner_image = plt.imread(WINNER_ICON)
    model_labels = [model["label"] for model in MODEL_INFO]
    x = np.arange(len(MODEL_INFO), dtype=float)
    bar_width = 0.18
    # Slightly narrower than the center spacing so adjacent comparison bars
    # retain a small, visible gap.
    comparison_bar_width = bar_width * 0.82
    offsets = (
        np.arange(len(METHOD_SPECS), dtype=float)
        - (len(METHOD_SPECS) - 1) / 2
    ) * bar_width
    # Both overlays run exactly from the center of the first comparison bar
    # (Self-Debiasing) to the center of the last one (Few-shot).
    comparison_span = offsets[-1] - offsets[0]

    fig, axes = plt.subplots(
        1,
        len(BENCHMARKS),
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
    )

    for panel_idx, (ax, (benchmark_idx, benchmark_label)) in enumerate(
        zip(axes, BENCHMARKS)
    ):
        benchmark_data = data[data["benchmark"] == benchmark_idx]
        plotted = {spec["key"]: [] for spec in METHOD_SPECS}
        base_values = []
        our_tdd_values = []

        for model in MODEL_INFO:
            model_data = benchmark_data[benchmark_data["name"] == model["name"]]
            values = _method_values(model_data)
            base_values.append(values["base"])
            our_tdd_values.append(values["rawls-cott"])
            for key in plotted:
                plotted[key].append(values[key])

        ax.bar(
            x,
            our_tdd_values,
            width=comparison_span,
            color=OUR_TDD_SPEC["color"],
            edgecolor=OUR_TDD_SPEC["edgecolor"],
            linewidth=0.35,
            alpha=0.98,
            zorder=2,
            label=OUR_TDD_SPEC["label"],
        )

        for method_idx, spec in enumerate(METHOD_SPECS):
            ax.bar(
                x + offsets[method_idx],
                plotted[spec["key"]],
                width=comparison_bar_width,
                color=spec["color"],
                edgecolor=spec["edgecolor"],
                linewidth=0.35,
                alpha=0.98,
                zorder=3,
                label=spec["label"],
            )

        ax.hlines(
            base_values,
            x + offsets[0],
            x + offsets[-1],
            color=BASE_SPEC["color"],
            linewidth=BASE_SPEC["linewidth"]-1,
            alpha=BASE_SPEC["alpha"],
            zorder=5,
            label=BASE_SPEC["label"],
        )

        ax.set_title(benchmark_label, pad=4)
        _format_axis(ax, panel_idx, model_labels)
        _add_winner_badge(
            ax,
            winner_image,
            _our_tdd_win_count(benchmark_data),
            WINNER_BADGE_POSITIONS[benchmark_idx],
        )

    axes[0].set_ylabel("DiffAware", labelpad=2.0)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_handles = dict(zip(labels, handles))
    # Matplotlib fills multi-column legends column by column. This input order
    # therefore renders as:
    #   Base     | Self-Debiasing
    #   PD-Only  | Few-Shot
    #   Our TDD
    legend_labels = [
        "Base",
        "PD-Only",
        "Our TDD",
        "Self-Debiasing",
        "Few-Shot",
    ]
    legend = axes[0].legend(
        [legend_handles[label] for label in legend_labels],
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.985),
        ncol=2,
        fontsize=6.3,
        columnspacing=0.7,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.35,
        borderpad=0.3,
        frameon=True,
        facecolor="none",
        edgecolor="#BDBDBD",
        framealpha=1.0,
    )
    for text in legend.get_texts():
        if text.get_text() == "Our TDD":
            text.set_fontweight("bold")
    fig.subplots_adjust(
        left=0.065,
        right=0.997,
        top=0.90,
        bottom=0.31,
        wspace=0.10,
    )
    return fig, axes


def save_figure(fig, stem=OUTPUT_STEM):
    FIG_DIR.mkdir(exist_ok=True)
    saved_paths = []
    for extension in ["pdf"]:
        output_path = FIG_DIR / f"{stem}.{extension}"
        fig.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=True,
            dpi=600,
        )
        saved_paths.append(output_path)
    return saved_paths


def main():
    set_paper_style()
    data = load_plot_data()
    fig, _ = plot_diffaware_bars(data)
    saved_paths = save_figure(fig)
    plt.close(fig)
    for path in saved_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
