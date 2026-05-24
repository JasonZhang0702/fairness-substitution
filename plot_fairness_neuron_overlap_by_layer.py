from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAIRNESS_NEURON_DIR = Path("result/fairness-neuron")
FIG_DIR = Path("figures")
RESULT_DIR = Path("result")

OVERLAP_COUNTS_STEM = "fairness_neuron_pairwise_overlap_counts_by_layer_7models"
OVERLAP_JACCARD_STEM = "fairness_neuron_pairwise_overlap_jaccard_by_layer_7models"
SPECIFICITY_STEM = "fairness_neuron_specificity_percent_by_layer_7models"
SUMMARY_CSV = RESULT_DIR / "fairness_neuron_overlap_by_layer.csv"

FIGSIZE = (7.15, 4.55)

MODEL_INFO = [
    {
        "name": "Meta-Llama-3-8B-Instruct",
        "label": "L3-8B",
        "release": "2024-04-18",
        "tie_order": 0,
    },
    {
        "name": "gemma-2-9b-it",
        "label": "G2-9B",
        "release": "2024-06-27",
        "tie_order": 1,
    },
    {
        "name": "Meta-Llama-3.1-8B-Instruct",
        "label": "L3.1-8B",
        "release": "2024-07-23",
        "tie_order": 2,
    },
    {
        "name": "Qwen2.5-7B-Instruct",
        "label": "Q2.5-7B",
        "release": "2024-09-19",
        "tie_order": 3,
    },
    {
        "name": "Qwen2.5-14B-Instruct",
        "label": "Q2.5-14B",
        "release": "2024-09-19",
        "tie_order": 4,
    },
    {
        "name": "Qwen3-4B",
        "label": "Q3-4B",
        "release": "2025-04-29",
        "tie_order": 5,
    },
    {
        "name": "Qwen3-8B",
        "label": "Q3-8B",
        "release": "2025-04-29",
        "tie_order": 6,
    },
]

PAIRWISE_OVERLAPS = [
    {
        "key": "difference_base",
        "label": "Difference ∩ Base",
        "left": "difference",
        "right": "base",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "key": "liberty_base",
        "label": "Liberty ∩ Base",
        "left": "liberty",
        "right": "base",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
    },
    {
        "key": "difference_liberty",
        "label": "Difference ∩ Liberty",
        "left": "difference",
        "right": "liberty",
        "color": "#CC79A7",
        "marker": "s",
        "linestyle": ":",
    },
]

SPECIFICITY = [
    {
        "key": "difference_specific",
        "label": "Difference-specific",
        "source": "difference",
        "others": ("base", "liberty"),
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "key": "liberty_specific",
        "label": "Liberty-specific",
        "source": "liberty",
        "others": ("base", "difference"),
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
    },
]

JSON_KEYS = {
    "difference": "only_difference_neurons",
    "liberty": "only_liberty_neurons",
    "base": "base_neurons",
}


def set_paper_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.0,
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
            "lines.linewidth": 1.05,
            "legend.frameon": False,
        }
    )


def ordered_model_info():
    return sorted(MODEL_INFO, key=lambda item: (item["release"], item["tie_order"]))


def _parse_layer(neuron_id):
    layer_id, _neuron_id = str(neuron_id).split("_", 1)
    return int(layer_id)


def _model_file(model_name):
    path = FAIRNESS_NEURON_DIR / f"neuron_statistics_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing fairness-neuron file: {path}")
    return path


def _sets_by_layer(neuron_ids, max_layer):
    layer_sets = {layer: set() for layer in range(max_layer + 1)}
    for neuron_id in neuron_ids:
        layer_sets[_parse_layer(neuron_id)].add(str(neuron_id))
    return layer_sets


def _pct(numerator, denominator):
    return numerator / denominator * 100.0 if denominator else 0.0


def load_overlap_data():
    pairwise_records = []
    specificity_records = []
    summary_records = []

    for model_idx, model in enumerate(ordered_model_info()):
        raw = json.loads(_model_file(model["name"]).read_text(encoding="utf-8"))
        missing_keys = sorted(set(JSON_KEYS.values()) - set(raw))
        if missing_keys:
            raise KeyError(f"{model['name']} missing keys: {missing_keys}")

        category_sets = {
            category: set(str(item) for item in raw[json_key])
            for category, json_key in JSON_KEYS.items()
        }
        max_layer = max(
            _parse_layer(neuron_id)
            for category_set in category_sets.values()
            for neuron_id in category_set
        )
        layer_sets = {
            category: _sets_by_layer(category_set, max_layer)
            for category, category_set in category_sets.items()
        }

        for layer in range(max_layer + 1):
            layer_categories = {
                category: by_layer[layer]
                for category, by_layer in layer_sets.items()
            }
            layer_summary = {
                "model": model["name"],
                "model_label": model["label"],
                "model_idx": model_idx,
                "layer": layer,
                "difference_count": len(layer_categories["difference"]),
                "liberty_count": len(layer_categories["liberty"]),
                "base_count": len(layer_categories["base"]),
            }

            for overlap in PAIRWISE_OVERLAPS:
                left = layer_categories[overlap["left"]]
                right = layer_categories[overlap["right"]]
                intersection_count = len(left & right)
                union_count = len(left | right)
                left_count = len(left)
                right_count = len(right)
                jaccard_pct = _pct(intersection_count, union_count)

                pairwise_records.append(
                    {
                        "model": model["name"],
                        "model_label": model["label"],
                        "model_idx": model_idx,
                        "layer": layer,
                        "metric": overlap["key"],
                        "metric_label": overlap["label"],
                        "count": intersection_count,
                        "jaccard_pct": jaccard_pct,
                        "left_share_pct": _pct(intersection_count, left_count),
                        "right_share_pct": _pct(intersection_count, right_count),
                    }
                )
                layer_summary[f"{overlap['key']}_count"] = intersection_count
                layer_summary[f"{overlap['key']}_jaccard_pct"] = jaccard_pct

            for metric in SPECIFICITY:
                source = layer_categories[metric["source"]]
                other_union = set()
                for other in metric["others"]:
                    other_union |= layer_categories[other]
                specific_count = len(source - other_union)
                source_count = len(source)
                specificity_records.append(
                    {
                        "model": model["name"],
                        "model_label": model["label"],
                        "model_idx": model_idx,
                        "layer": layer,
                        "metric": metric["key"],
                        "metric_label": metric["label"],
                        "count": specific_count,
                        "specificity_pct": _pct(specific_count, source_count),
                    }
                )
                layer_summary[f"{metric['key']}_count"] = specific_count
                layer_summary[f"{metric['key']}_pct"] = _pct(
                    specific_count, source_count
                )

            summary_records.append(layer_summary)

    pairwise = pd.DataFrame.from_records(pairwise_records)
    specificity = pd.DataFrame.from_records(specificity_records)
    summary = pd.DataFrame.from_records(summary_records)
    return pairwise, specificity, summary


def _format_layer_axis(ax, max_layer):
    tick_step = 10 if max_layer >= 35 else 7
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_step))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.grid(axis="y", which="major", color="#D8D8D8", linewidth=0.45)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.28)
    ax.grid(axis="x", which="major", color="#ECECEC", linewidth=0.32)
    ax.set_axisbelow(True)


def _rounded_upper(values, step):
    max_value = float(np.nanmax(np.asarray(values, dtype=float)))
    if max_value <= 0:
        return step
    return np.ceil((max_value * 1.08) / step) * step


def _count_axis_limits(values):
    max_value = float(np.nanmax(np.asarray(values, dtype=float)))
    if max_value <= 0:
        return 0, 10, 5
    step = 10 if max_value <= 50 else 20 if max_value <= 120 else 50
    upper = np.ceil((max_value * 1.12) / step) * step
    return 0, upper, step


def _percent_axis_limits(values, floor_zero=True):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0, 100, 20

    y_min = float(values.min())
    y_max = float(values.max())
    span = y_max - y_min
    padding = max(1.0, span * 0.14)
    lower = np.floor((y_min - padding) / 5.0) * 5.0
    upper = np.ceil((y_max + padding) / 5.0) * 5.0
    if floor_zero:
        lower = max(0.0, lower)
    upper = min(100.0, max(upper, lower + 5.0))
    if upper - lower < 10:
        center = (upper + lower) / 2.0
        lower = max(0.0 if floor_zero else lower, center - 5.0)
        upper = min(100.0, center + 5.0)

    scale_span = upper - lower
    step = 2 if scale_span <= 14 else 5 if scale_span <= 35 else 10
    return lower, upper, step


def _plot_metric_grid(
    data,
    metric_styles,
    value_column,
    ylabel,
    output_stem,
    y_mode,
):
    models = ordered_model_info()
    style_by_key = {item["key"]: item for item in metric_styles}

    fig, axes = plt.subplots(
        2,
        4,
        figsize=FIGSIZE,
        sharey=False,
    )
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        model_data = data[data["model"] == model["name"]]
        max_layer = int(model_data["layer"].max())
        values = model_data[value_column].to_numpy(dtype=float)

        for metric in metric_styles:
            metric_data = model_data[model_data["metric"] == metric["key"]]
            ax.plot(
                metric_data["layer"].to_numpy(dtype=int),
                metric_data[value_column].to_numpy(dtype=float),
                label=metric["label"],
                color=style_by_key[metric["key"]]["color"],
                linestyle=style_by_key[metric["key"]]["linestyle"],
                marker=style_by_key[metric["key"]]["marker"],
                markersize=2.5,
                markevery=max(1, max_layer // 8),
                markerfacecolor=style_by_key[metric["key"]]["color"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                alpha=0.96,
                zorder=3,
            )

        ax.set_title(model["label"], pad=4)
        _format_layer_axis(ax, max_layer)

        if y_mode == "count":
            lower, upper, step = _count_axis_limits(values)
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step / 2))
        elif y_mode == "overlap_pct":
            lower, upper, step = _percent_axis_limits(values, floor_zero=True)
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step / 2))
        elif y_mode == "specificity_pct":
            lower, upper, step = _percent_axis_limits(values, floor_zero=False)
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(step))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(step / 2))
        else:
            raise ValueError(f"Unknown y_mode: {y_mode}")

    axes[-1].axis("off")

    for ax in axes[:7]:
        ax.set_xlabel("Layer")
    axes[0].set_ylabel(ylabel)
    axes[4].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(metric_styles),
        columnspacing=1.0,
        handlelength=1.6,
        handletextpad=0.38,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.86,
        bottom=0.105,
        hspace=0.38,
        wspace=0.18,
    )
    save_figure(fig, output_stem)
    return fig, axes


def plot_pairwise_overlap_counts(pairwise):
    return _plot_metric_grid(
        pairwise,
        PAIRWISE_OVERLAPS,
        value_column="count",
        ylabel="Overlap count",
        output_stem=OVERLAP_COUNTS_STEM,
        y_mode="count",
    )


def plot_pairwise_overlap_jaccard(pairwise):
    return _plot_metric_grid(
        pairwise,
        PAIRWISE_OVERLAPS,
        value_column="jaccard_pct",
        ylabel="Pairwise Jaccard (%)",
        output_stem=OVERLAP_JACCARD_STEM,
        y_mode="overlap_pct",
    )


def plot_specificity_percent(specificity):
    return _plot_metric_grid(
        specificity,
        SPECIFICITY,
        value_column="specificity_pct",
        ylabel="Specificity (%)",
        output_stem=SPECIFICITY_STEM,
        y_mode="specificity_pct",
    )


def save_figure(fig, stem):
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
    pairwise, specificity, summary = load_overlap_data()
    RESULT_DIR.mkdir(exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)

    plot_pairwise_overlap_counts(pairwise)
    plot_pairwise_overlap_jaccard(pairwise)
    plot_specificity_percent(specificity)

    for stem in (OVERLAP_COUNTS_STEM, OVERLAP_JACCARD_STEM, SPECIFICITY_STEM):
        print(f"Saved {FIG_DIR / (stem + '.pdf')}")
        print(f"Saved {FIG_DIR / (stem + '.svg')}")
    print(f"Saved {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
