import matplotlib.pyplot as plt
from typing import Literal, List
import numpy as np

# Benchmark
acc_top1 = 71.87
acc_top5 = 90.31


def plot_sensitivity_analysis(data, acc_type: Literal["top1", "top5"]) -> None:
    if acc_type == "top1":
        acc = acc_top1
        title = "Top-1 Accuracy"
    elif acc_type == "top5":
        acc = acc_top5
        title = "Top-5 Accuracy"
    else:
        raise ValueError("Acc must be either top1 or top5")

    plt.figure(figsize=(15, 8))

    plt.title(f"{title}: {acc}%")
    plt.plot(np.linspace(0, 1, 10), np.ones(10) * acc, color="red", linestyle="dashed", linewidth=3)
    for bn_layer in data.keys():
        if bn_layer != 'ratio':
            plt.plot(data["ratio"], np.array(data[bn_layer][acc_type]) * 100, label=bn_layer, marker='o', markersize=3,
                     linewidth=1)

    plt.xlabel("Pruning Ratio")
    plt.ylabel(f"{title} (%)")
    plt.xlim([0, 1])
    plt.ylim([0, 100])
    plt.grid(axis="y")
    plt.legend(ncol=4)
    plt.show()


def plot_train_metrics(data) -> None:
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Training")

    plt.subplot(1, 2, 1)
    plt.title("Loss")

    for i, losses in enumerate(data["train_losses"]):
        plt.plot(losses, label=f"ratio {data["ratio"][i]}", marker='o', markersize=3, linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(axis="y")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    for i, acc in enumerate(data["train_accuracies"]):
        plt.plot(acc, label=f"ratio {data["ratio"][i]}", marker='o', markersize=3, linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplots_adjust(wspace=0.3)
    # plt.xlim([0, 1])
    # plt.ylim([0, 100])
    plt.grid(axis="y")
    plt.legend()
    plt.show()


def plot_test_metrics(data) -> None:
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"Evaluation")

    plt.subplot(1, 2, 1)
    plt.title("Acc@1")
    plt.plot(np.linspace(-1, 2, 10), np.ones(10) * acc_top1, color="red", linestyle="dashed", linewidth=1)
    plt.bar(data["ratio"], np.array(data["top1"]) * 100, edgecolor='black', width=.1)
    plt.xlabel('Pruning Ratio')
    plt.ylabel('Acc@1 in %')
    plt.xticks(data["ratio"])
    plt.ylim([0, 100])
    plt.xlim([-0.1, 1.1])
    plt.grid(ls="--", axis="y", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.title("Acc@5")
    plt.plot(np.linspace(-1, 2, 10), np.ones(10) * acc_top5, color="red", linestyle="dashed", linewidth=1)
    plt.bar(data["ratio"], np.array(data["top5"]) * 100, edgecolor='black', width=.1)
    plt.xlabel('Pruning Ratio')
    plt.ylabel('Acc@5 in %')
    plt.xticks(data["ratio"])
    plt.ylim([0, 100])
    plt.xlim([-0.1, 1.1])
    plt.grid(ls="--", axis="y", alpha=0.5)

    plt.subplots_adjust(wspace=0.3)
    plt.show()


def compare_test_metrics(datas, labels: List[str]) -> None:
    plt.figure(figsize=(15, 4))
    plt.suptitle("Evaluation")

    # Subplot 1: Acc@1
    plt.subplot(1, 2, 1)
    plt.title("Acc@1")
    x = np.arange(len(datas[0]["ratio"]))

    plt.plot(np.linspace(-1, len(x), 10), np.ones(10) * acc_top1, color="red", linestyle="dashed", linewidth=1)

    width = 0.5 / len(datas)

    for idx, data in enumerate(datas):
        plt.bar(x + idx * width, np.array(data["top1"]) * 100, width, label=labels[idx])

    plt.xticks(x + (width * (len(datas) - 1)) / 2, datas[0]["ratio"])
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Acc@1 in %")
    plt.ylim([0, 100])
    plt.xlim([-0.2, len(x) - 0.5])
    plt.grid(ls="--", axis="y", alpha=0.5)
    plt.legend()

    # Subplot 2: Acc@5
    plt.subplot(1, 2, 2)

    plt.title("Acc@5")
    plt.plot(np.linspace(-1, len(x), 10), np.ones(10) * acc_top5, color="red", linestyle="dashed", linewidth=1)

    for idx, data in enumerate(datas):
        plt.bar(x + idx * width, np.array(data["top5"]) * 100, width, label=labels[idx])

    plt.xticks(x + (width * (len(datas) - 1)) / 2, datas[0]["ratio"])
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Acc@5 in %")
    plt.ylim([0, 100])
    plt.xlim([-0.2, len(x) - 0.5])

    plt.grid(ls="--", axis="y", alpha=0.5)
    plt.legend()

    plt.show()
