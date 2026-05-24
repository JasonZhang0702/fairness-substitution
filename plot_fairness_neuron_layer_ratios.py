from collections import Counter
from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAIRNESS_NEURON_DIR = Path("result/fairness-neuron")
FIG_DIR = Path("figures")
OUTPUT_STEM = "fairness_neuron_layer_ratio_distribution_7models"
COMPOSITION_OUTPUT_STEM = "fairness_neuron_layer_category_composition_7models"
COMPOSITION_LINE_OUTPUT_STEM = "fairness_neuron_layer_category_composition_lines_7models"

# Double-column friendly, with enough room for seven model-specific panels.
FIGSIZE = (7.15, 4.55)

CATEGORIES = [
    {
        "key": "only_difference_neurons",
        "label": "Difference-only",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    {
        "key": "only_liberty_neurons",
        "label": "Liberty-only",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
    },
    {
        "key": "base_neurons",
        "label": "Base",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "-.",
    },
]

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


def set_paper_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.0,
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


def _parse_layer(neuron_id):
    layer_id, _neuron_id = str(neuron_id).split("_", 1)
    return int(layer_id)


def _model_file(model_name):
    path = FAIRNESS_NEURON_DIR / f"neuron_statistics_{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing fairness-neuron file: {path}")
    return path


def load_layer_ratio_data():
    records = []
    categories_by_key = {item["key"]: item for item in CATEGORIES}

    for model_idx, model in enumerate(ordered_model_info()):
        data = json.loads(_model_file(model["name"]).read_text(encoding="utf-8"))
        missing_keys = sorted(set(categories_by_key) - set(data))
        if missing_keys:
            raise KeyError(f"{model['name']} missing keys: {missing_keys}")

        max_layer = max(
            _parse_layer(neuron_id)
            for category in categories_by_key
            for neuron_id in data[category]
        )
        layers = np.arange(max_layer + 1)

        for category_idx, category in enumerate(CATEGORIES):
            key = category["key"]
            values = data[key]
            total = len(values)
            layer_counts = Counter(_parse_layer(neuron_id) for neuron_id in values)

            for layer in layers:
                count = layer_counts[layer]
                records.append(
                    {
                        "model": model["name"],
                        "model_label": model["label"],
                        "model_idx": model_idx,
                        "category": key,
                        "category_label": category["label"],
                        "category_idx": category_idx,
                        "layer": int(layer),
                        "count": int(count),
                        # Normalize within each category because base_neurons can
                        # overlap with the two "only" lists.
                        "ratio_pct": (count / total * 100.0) if total else 0.0,
                    }
                )

    data = pd.DataFrame.from_records(records)
    data["category"] = pd.Categorical(
        data["category"],
        categories=[item["key"] for item in CATEGORIES],
        ordered=True,
    )
    return data.sort_values(["model_idx", "category_idx", "layer"])


def _format_axis(ax, model_data):
    max_layer = int(model_data["layer"].max())
    tick_step = 10 if max_layer >= 35 else 7
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_step))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    ax.grid(axis="y", which="major", color="#D8D8D8", linewidth=0.45)
    ax.grid(axis="y", which="minor", color="#EEEEEE", linewidth=0.28)
    ax.grid(axis="x", which="major", color="#ECECEC", linewidth=0.32)
    ax.set_axisbelow(True)


def plot_layer_ratio_distribution(data):
    models = ordered_model_info()
    category_styles = {item["key"]: item for item in CATEGORIES}
    y_max = np.ceil((data["ratio_pct"].max() + 0.3) / 2.0) * 2.0

    fig, axes = plt.subplots(
        2,
        4,
        figsize=FIGSIZE,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        model_data = data[data["model"] == model["name"]]

        for category in CATEGORIES:
            category_data = model_data[model_data["category"] == category["key"]]
            style = category_styles[category["key"]]
            ax.plot(
                category_data["layer"].to_numpy(dtype=int),
                category_data["ratio_pct"].to_numpy(dtype=float),
                label=category["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=2.5,
                markevery=max(1, int(category_data["layer"].max() // 8)),
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                alpha=0.95,
                zorder=3,
            )

        ax.set_title(model["label"], pad=4)
        ax.set_ylim(0, y_max)
        _format_axis(ax, model_data)

    axes[-1].axis("off")

    for ax in axes[:4]:
        ax.tick_params(labelbottom=False)
    for ax in axes[4:7]:
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("Layer share (%)")
    axes[4].set_ylabel("Layer share (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(CATEGORIES),
        columnspacing=1.05,
        handlelength=1.65,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.88,
        bottom=0.105,
        hspace=0.28,
        wspace=0.16,
    )
    return fig, axes


def plot_layer_category_composition(data):
    models = ordered_model_info()
    category_labels = [item["label"] for item in CATEGORIES]
    category_colors = [item["color"] for item in CATEGORIES]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=FIGSIZE,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        model_data = data[data["model"] == model["name"]]
        composition = (
            model_data.pivot_table(
                index="layer",
                columns="category",
                values="count",
                aggfunc="sum",
                fill_value=0,
                observed=False,
            )
            .reindex(columns=[item["key"] for item in CATEGORIES], fill_value=0)
            .sort_index()
        )
        layer_totals = composition.sum(axis=1).replace(0, np.nan)
        composition_pct = composition.div(layer_totals, axis=0).fillna(0.0) * 100.0
        layers = composition_pct.index.to_numpy(dtype=int)
        values = [
            composition_pct[item["key"]].to_numpy(dtype=float)
            for item in CATEGORIES
        ]

        ax.stackplot(
            layers,
            values,
            labels=category_labels,
            colors=category_colors,
            alpha=0.82,
            linewidth=0.35,
            edgecolor="white",
        )
        ax.set_title(model["label"], pad=4)
        ax.set_ylim(0, 100)
        _format_axis(ax, model_data)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(12.5))

    axes[-1].axis("off")

    for ax in axes[:4]:
        ax.tick_params(labelbottom=False)
    for ax in axes[4:7]:
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("Within-layer share (%)")
    axes[4].set_ylabel("Within-layer share (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(CATEGORIES),
        columnspacing=1.05,
        handlelength=1.65,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.88,
        bottom=0.105,
        hspace=0.28,
        wspace=0.16,
    )
    return fig, axes


def _within_layer_composition(model_data):
    composition = (
        model_data.pivot_table(
            index="layer",
            columns="category",
            values="count",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=[item["key"] for item in CATEGORIES], fill_value=0)
        .sort_index()
    )
    layer_totals = composition.sum(axis=1).replace(0, np.nan)
    return composition.div(layer_totals, axis=0).fillna(0.0) * 100.0


def plot_layer_category_composition_lines(data):
    models = ordered_model_info()

    fig, axes = plt.subplots(
        2,
        4,
        figsize=FIGSIZE,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, model in zip(axes, models):
        model_data = data[data["model"] == model["name"]]
        composition_pct = _within_layer_composition(model_data)
        layers = composition_pct.index.to_numpy(dtype=int)

        for category in CATEGORIES:
            ax.plot(
                layers,
                composition_pct[category["key"]].to_numpy(dtype=float),
                label=category["label"],
                color=category["color"],
                linestyle=category["linestyle"],
                marker=category["marker"],
                markersize=2.7,
                markevery=max(1, int(layers.max() // 8)),
                markerfacecolor=category["color"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                alpha=0.96,
                zorder=3,
            )

        ax.set_title(model["label"], pad=4)
        ax.set_ylim(0, 100)
        _format_axis(ax, model_data)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(12.5))

    axes[-1].axis("off")

    for ax in axes[:4]:
        ax.tick_params(labelbottom=False)
    for ax in axes[4:7]:
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("Within-layer share (%)")
    axes[4].set_ylabel("Within-layer share (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=len(CATEGORIES),
        columnspacing=1.05,
        handlelength=1.65,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.88,
        bottom=0.105,
        hspace=0.28,
        wspace=0.16,
    )
    return fig, axes


def save_figure(fig, stem=OUTPUT_STEM):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=600)


def main():
    set_paper_style()
    data = load_layer_ratio_data()
    ratio_fig, _axes = plot_layer_ratio_distribution(data)
    save_figure(ratio_fig)
    composition_fig, _axes = plot_layer_category_composition(data)
    save_figure(composition_fig, COMPOSITION_OUTPUT_STEM)
    composition_line_fig, _axes = plot_layer_category_composition_lines(data)
    save_figure(composition_line_fig, COMPOSITION_LINE_OUTPUT_STEM)
    print(f"Saved {FIG_DIR / (OUTPUT_STEM + '.pdf')}")
    print(f"Saved {FIG_DIR / (OUTPUT_STEM + '.png')}")
    print(f"Saved {FIG_DIR / (COMPOSITION_OUTPUT_STEM + '.pdf')}")
    print(f"Saved {FIG_DIR / (COMPOSITION_OUTPUT_STEM + '.png')}")
    print(f"Saved {FIG_DIR / (COMPOSITION_LINE_OUTPUT_STEM + '.pdf')}")
    print(f"Saved {FIG_DIR / (COMPOSITION_LINE_OUTPUT_STEM + '.png')}")


if __name__ == "__main__":
    main()
