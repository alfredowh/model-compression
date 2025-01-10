import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Literal
import numpy as np
import argparse
import json
from pathlib import Path


def plot_sensitivity_analysis(data, acc_type: Literal["top1", "top5"]) -> None:
    if acc_type == "top1":
        acc = 71.87
        title = "Top-1 Accuracy"
    elif acc_type == "top5":
        acc = 90.31
        title = "Top-5 Accuracy"
    else:
        raise ValueError("Acc must be either top1 or top5")

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title(f"{title}: {acc}%")

    ax.plot(np.linspace(0, 1, 10), np.ones(10) * acc, color="red", linestyle="dashed", linewidth=3)

    lines = []
    bn_layers = list(data.keys())
    pruning_ratios = data["ratio"]

    for bn_layer in bn_layers:
        if bn_layer != 'ratio':
            line, = ax.plot(pruning_ratios,
                            np.array(data[bn_layer][acc_type]) * 100,
                            label=bn_layer,
                            marker='o',
                            markersize=5,
                            linewidth=2,
                            alpha=0.6)
            lines.append(line)

    legend = ax.legend(ncol=4)
    legend_texts = legend.get_texts()

    # Highlight function
    def on_pick(event):
        # Reset all lines and labels to default
        for line, text in zip(lines, legend_texts):
            line.set_linewidth(2)
            line.set_alpha(0.6)
            text.set_fontsize(10)
            text.set_color('black')

        # Highlight the selected line and label
        selected_line = event.artist
        selected_index = lines.index(selected_line)  # Get the index of the selected line
        selected_label = legend_texts[selected_index]

        # Modify line and label appearance
        selected_line.set_linewidth(4)
        selected_line.set_alpha(1.0)
        selected_label.set_fontsize(12)
        selected_label.set_color('red')

        # Redraw the figure
        fig.canvas.draw()

    # Enable picking for each line
    for line in lines:
        line.set_picker(True)

    # Connect the pick event
    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.xlabel("Pruning Ratio")
    plt.ylabel("Acc@1 (%)")
    plt.xlim([0, 1])
    plt.ylim([0, 120])
    plt.grid(axis="y")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruning-type', type=str, default='batchnorm', help='batchnorm or magnitude')
    parser.add_argument('--acc-type', type=str, default="top5", help='Acc Top-1 or Top-5')
    parser.add_argument('--root', type=str, default="./", help='Acc Top-1 or Top-5')

    opt = parser.parse_args()

    opt.root = Path(opt.root)

    if opt.pruning_type == "batchnorm":
        with open(opt.root / 'runs/scaling_pruning/sensivity_analysis.json', 'r') as f:
            data = json.load(f)
    elif opt.pruning_type == "magnitude":
        with open(opt.root / 'runs/magnitude_pruning/sensivity_analysis.json', 'r') as f:
            data = json.load(f)
    else:
        raise ValueError("Pruning type must be either batchnorm or magnitude")

    plot_sensitivity_analysis(data, opt.acc_type)
