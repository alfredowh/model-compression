import matplotlib.pyplot as plt
from typing import Literal, List, Tuple
import numpy as np
import torch
from torchvision import models
from models.pruning import Pruning
import time
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1').to(DEVICE).eval()

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

    for idx, data in enumerate(datas):
        plt.plot(datas[idx]["ratio"], np.array(data["top1"]) * 100, label=labels[idx], marker="o", linewidth=1)

    plt.xlabel("Pruning Ratio")
    plt.ylabel("Acc@1 in %")
    plt.ylim([0, 100])
    plt.xlim([0.0, 1.0])
    plt.grid(ls="--", axis="y", alpha=0.5)
    plt.legend()

    # Subplot 2: Acc@5
    plt.subplot(1, 2, 2)

    plt.title("Acc@5")
    plt.plot(np.linspace(-1, len(x), 10), np.ones(10) * acc_top5, color="red", linestyle="dashed", linewidth=1)

    for idx, data in enumerate(datas):
        plt.plot(datas[idx]["ratio"], np.array(data["top5"]) * 100, label=labels[idx], marker="o", linewidth=1)

    plt.xlabel("Pruning Ratio")
    plt.ylabel("Acc@5 in %")
    plt.ylim([0, 100])
    plt.xlim([0.0, 1.0])

    plt.grid(ls="--", axis="y", alpha=0.5)
    plt.legend()

    plt.show()


def plot_cp(data, label, original=False):
    if original:
        plt.plot(range(len(data.keys())), data.values(), label=label, c="red", linestyle="--", alpha=1.0, linewidth=0.5)
    else:
        plt.plot(range(len(data.keys())), data.values(), label=label)
    plt.xticks(range(len(data.keys())))


def iter_ratios(pruned_layers, pruning_ratios, pruning_type, level="global"):
    for r in pruning_ratios:
        model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1').to(DEVICE).eval()
        pruning = Pruning(model, DEVICE)
        if pruning_type == 'batchnorm':
            model = pruning.scaling_based_pruning(batch_norms=pruned_layers, pruning_ratio=r, level=level,
                                                  scale_threshold=False)
        elif pruning_type == 'magnitude':
            model = pruning.magnitude_based_pruning(conv_layers=pruned_layers, pruning_ratio=r, level=level,
                                                    scale_threshold=False)

        data, total = pruning.count_parameters()

        plot_cp(data, label=f"ratio: {r}, {level}, total: {total / 1e6:.3}M")

    plt.yscale("log")
    plt.ylabel("#Parameter")
    plt.xlabel("Conv Layer")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()


def measure_inference_speed(model, input_size: Tuple[int], device: torch.device) -> Tuple[float]:
    times = []
    with torch.no_grad():
        inputs = torch.randn(1, 3, *input_size).to(device)
        # Warm up the model (optional, especially for GPU)
        for _ in range(5):
            model(inputs)

        for _ in range(20):  # Run multiple iterations
            start_time = time.time()
            model(inputs)
            end_time = time.time()
            times.append(end_time - start_time)

    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


def plot_speed(pruning_ratios, title, device, level="global", input_size=(224, 224), width: float = 0.4) -> None:
    avg_times = []
    std_times = []

    for idx, ratio in enumerate(pruning_ratios):
        model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1').to(device).eval()

        if ratio == 0.0:
            avg_time, std_time = measure_inference_speed(model, input_size, device)

        elif ratio > 0.0 and ratio < 1.0:
            pruning = Pruning(model, device)
            pruned_layers = []
            for i in range(1, 18):
                if i == 1:
                    pruned_layers.append(f'features.{i}.conv.0.1')
                    continue
                pruned_layers.append(f'features.{i}.conv.1.1')
            model = pruning.scaling_based_pruning(batch_norms=pruned_layers, pruning_ratio=ratio, level=level,
                                                  scale_threshold=False)
            avg_time, std_time = measure_inference_speed(model, input_size, device)
        else:
            raise ValueError("Ratio must be < 1.0")

        avg_times.append(avg_time)
        std_times.append(std_time)

    x = np.arange(len(pruning_ratios))

    plt.figure(figsize=(8, 5))
    plt.title(title)

    plt.bar(x, avg_times, width, yerr=std_times, capsize=5)
    plt.xlabel("Pruning Ratio")
    plt.ylabel("Time (s)")
    plt.grid(ls="--", axis="y", alpha=0.5)
    plt.xticks(x, pruning_ratios)
    plt.show()

def plot_speed_acc(data, pruning_ratios, device, level="global", input_size=(224, 224)) -> None:
    avg_times = []

    for idx, ratio in enumerate(pruning_ratios):
        model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1').to(device).eval()

        if ratio == 0.0:
            avg_time, std_time = measure_inference_speed(model, input_size, device)

        elif ratio > 0.0 and ratio < 1.0:
            pruning = Pruning(model, device)
            pruned_layers = []
            for i in range(1, 18):
                if i == 1:
                    pruned_layers.append(f'features.{i}.conv.0.1')
                    continue
                pruned_layers.append(f'features.{i}.conv.1.1')
            model = pruning.scaling_based_pruning(batch_norms=pruned_layers, pruning_ratio=ratio, level=level,
                                                  scale_threshold=False)
            avg_time, _ = measure_inference_speed(model, input_size, device)
        else:
            raise ValueError("Ratio must be < 1.0")

        avg_times.append(avg_time)

    plt.figure(figsize=(5, 5))
    plt.title(f"Accuracy over Inference Time ({device})")

    accs = [acc_top5]
    accs.extend(data["top5"])

    for idx, r in enumerate(pruning_ratios):
        if r == 0.0:
            plt.plot(avg_times[idx], accs[idx], label="original", marker="o")
            continue
        plt.plot(avg_times[idx], accs[idx] * 100, label=f"ratio: {r}", marker="o")

    plt.ylim([0, 100])
    plt.xlabel("Inference Time (s)")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.show()
