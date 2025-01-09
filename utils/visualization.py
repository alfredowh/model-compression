import matplotlib.pyplot as plt
from typing import Literal
import numpy as np

def plot_sensitivity_analysis(data, acc_type: Literal["top1", "top5"]) -> None:
    if acc_type == "top1":
        acc = 71.87
        title = "Top-1 Accuracy"
    elif acc_type == "top5":
        acc = 90.31
        title = "Top-5 Accuracy"
    else:
        raise ValueError("Acc must be either top1 or top5")

    plt.figure(figsize=(15, 8))

    plt.title(f"{title}: {acc}%")
    plt.plot(np.linspace(0, 1, 10), np.ones(10) * acc, color="red", linestyle="dashed", linewidth=3)
    for bn_layer in data.keys():
        if bn_layer != 'ratio':
            plt.plot(data["ratio"], np.array(data[bn_layer][acc_type]) * 100, label=bn_layer, marker='o', markersize=3, linewidth=1)

    plt.xlabel("Pruning Ratio")
    plt.ylabel(f"{title} (%)")
    plt.xlim([0, 1])
    plt.ylim([0, 100])
    plt.grid(axis="y")
    plt.legend(ncol=4)
    plt.show()
