from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from torch.utils.data import DataLoader
from utils.datasets import ImageNet, train_test_split
from models.pruning import Pruning
import json
import argparse
from typing import Tuple
import torch
from utils.general import increment_path
from pathlib import Path
import yaml


def test(model, device: torch.device, test_loader: DataLoader) -> Tuple[float, float, float]:
    print("Test starts ...")
    model.eval()

    losses = 0.0
    total_predictions = 0
    true_predictions_top1 = 0
    true_predictions_top5 = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets) / inputs.size(0)
            losses += loss.item()

            # Top-1 predictions
            _, predicted_top1 = torch.max(outputs, 1)
            batch_true_predictions_top1 = (predicted_top1 == targets).sum().item()
            true_predictions_top1 += batch_true_predictions_top1

            # Top-5 predictions
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            batch_true_predictions_top5 = sum(
                [targets[i].item() in predicted_top5[i].tolist() for i in range(targets.size(0))]
            )
            true_predictions_top5 += batch_true_predictions_top5

            # Update total predictions
            batch_total_predictions = outputs.size(0)
            total_predictions += batch_total_predictions

            # Print batch metrics
            # print(
            #     f'Batch {batch_idx}, Loss: {loss:.4f}, '
            #     f'Accuracy@1: {batch_true_predictions_top1 / batch_total_predictions * 100:.2f}%, '
            #     f'Accuracy@5: {batch_true_predictions_top5 / batch_total_predictions * 100:.2f}%'
            # )

    # Compute overall accuracies
    accuracy_top1 = true_predictions_top1 / total_predictions
    accuracy_top5 = true_predictions_top5 / total_predictions

    return accuracy_top1, accuracy_top5, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--ratios', nargs='+', type=int, default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help='Pruning ratio for eval iteration')
    parser.add_argument('--task', type=str, default='global_pruning', help='global_pruning or sensivity_analysis')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--scale-threshold', action="store_true",
                        help='Set scaling threshold for scaling-based pruning based on a heuristic')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')

    opt = parser.parse_args()

    if opt.test_size == 0.0:
        raise ValueError("Test set size should not be 0.0")
    if opt.task == 'sensivity_analysis' and opt.test_size != 1.0:
        raise UserWarning("Test set size is not 1.0")

    opt.save_dir = increment_path(Path(opt.project) / opt.name)

    with open(opt.save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Load dataset
    preprocess = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
    _, _, test_X, test_Y = train_test_split(test_size=opt.test_size, shuffle=False, num_imgs=50,
                                            root="./data/imagenet")
    test_dataset = ImageNet(test_X, test_Y, transform=preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=opt.workers, pin_memory=True)

    # Load full model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
    model.to(DEVICE).eval()

    # Eval full model
    accuracy_top1, accuracy_top5, losses = test(model, DEVICE, test_dataloader)
    print(f"Full Model - acc@1: {accuracy_top1 * 100}%, acc@5: {accuracy_top5 * 100}%, loss: {losses}")

    data = {}
    if opt.task == 'global_pruning':
        batch_norms = []
        for i in range(1, 18):
            if i == 1:
                batch_norms.append(f'features.{i}.conv.0.1')
                continue
            batch_norms.append(f'features.{i}.conv.1.1')

        for p in opt.ratios:
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
            model.eval().to(DEVICE)

            pruning = Pruning(model, DEVICE)
            model = pruning.scaling_based_pruning(batch_norms=batch_norms, pruning_ratio=p, level='global',
                                                  scale_threshold=opt.scale_threshold)

            accuracy_top1, accuracy_top5, losses = test(model, DEVICE, test_dataloader)

            print(
                f"ratio: {p} acc@1: {accuracy_top1 * 100}%, acc@5: {accuracy_top5 * 100}%, loss: {losses}")

            if data.get("ratio", -1) == -1:
                data["ratio"] = []
            if data.get("top1", -1) == -1:
                data["top1"] = []
            if data.get("top5", -1) == -1:
                data["top5"] = []
            if data.get("loss", -1) == -1:
                data["loss"] = []

            data["ratio"].append(p)
            data['top1'].append(accuracy_top1)
            data['top5'].append(accuracy_top5)
            data['loss'].append(losses)

    elif opt.task == 'sensivity_analysis':
        for p in opt.ratios:

            if data.get("ratio", -1) == -1:
                data["ratio"] = []
            data["ratio"].append(p)

            for i in range(1, 18):
                model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
                model.eval().to(DEVICE)

                if i == 1:
                    batch_norms = f'features.{i}.conv.0.1'
                else:
                    batch_norms = f'features.{i}.conv.1.1'

                pruning = Pruning(model, DEVICE)
                model = pruning.scaling_based_pruning(batch_norms=batch_norms, pruning_ratio=p, level='layerwise',
                                                      scale_threshold=False)

                accuracy_top1, accuracy_top5, losses = test(model, DEVICE, test_dataloader)

                print(
                    f"{batch_norms} ratio: {p} acc@1: {accuracy_top1 * 100}%, acc@5: {accuracy_top5 * 100}%, loss: {losses}")

                if data.get(batch_norms, -1) == -1:
                    data[batch_norms] = {}
                if data[batch_norms].get('top1', -1) == -1:
                    data[batch_norms]['top1'] = []
                if data[batch_norms].get('top5', -1) == -1:
                    data[batch_norms]['top5'] = []
                if data[batch_norms].get('loss', -1) == -1:
                    data[batch_norms]['loss'] = []

                data[batch_norms]['top1'].append(accuracy_top1)
                data[batch_norms]['top5'].append(accuracy_top5)
                data[batch_norms]['loss'].append(losses)
    else:
        raise ValueError("Task not supported")

    with open(opt.save_dir / f"{opt.task}.json", "w") as json_file:
        json.dump(data, json_file, indent=2)
