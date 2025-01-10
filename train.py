import torch
import torch.nn as nn
import argparse
import numpy as np
from torchvision.models import MobileNet_V2_Weights
from utils.datasets import ImageNet, train_test_split
from torch.utils.data import DataLoader
from models.pruning import Pruning
from torchvision import transforms, models
from test import test
import yaml
import json
from utils.general import increment_path, calc_total_ratio
from pathlib import Path


def train(model, hyp, opt, teacher=None):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    generator = torch.Generator()
    generator.manual_seed(opt.seed)  # randomness of the dataloader

    # Load dataset
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(float(hyp.get('fliph', 0.0))),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_X, train_Y, test_X, test_Y = train_test_split(test_size=opt.test_size, shuffle=False, num_imgs=50,
                                                        root=opt.root)
    train_dataset = ImageNet(train_X, train_Y, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, generator=generator,
                                  num_workers=opt.workers, pin_memory=True)

    test_dataset = ImageNet(test_X, test_Y, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                                 pin_memory=True)

    # Model training
    train_losses = []
    train_accuracies = []
    loss_fn = torch.nn.CrossEntropyLoss()

    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp.get('lr', 1e-3),
                                     betas=(float(hyp.get('momentum', 0.9)), 0.999),
                                     weight_decay=float(hyp.get('weight_decay', 0)))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr'], momentum=hyp['momentum'],
                                    weight_decay=float(hyp.get('weight_decay', 0)), nesterov=False)


    print("Training starts ...")
    for epoch in range(opt.epochs):
        model.train()

        train_total_imgs = 0
        train_correct_imgs = 0
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            batch_size, _, _, _ = inputs.size()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            train_loss = loss_fn(outputs, targets)

            if opt.kd:
                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                T = float(hyp.get("temperature", 2))    # temperature

                # Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(outputs / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[
                    0] * (T ** 2)

                # Weighted sum of the two losses
                kd_loss_weight = float(hyp.get("kd_loss_weight", 0.25))
                ce_loss_weight = float(hyp.get("ce_weight", 0.75))

                train_loss = kd_loss_weight * soft_targets_loss + ce_loss_weight * train_loss

            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

            _, predicted = torch.max(outputs, 1)
            train_correct_imgs += (predicted == targets).sum().item()
            train_total_imgs += batch_size

            idx = 200
            if batch_idx % idx == idx - 1:
                print(
                    f'Train Epoch: {epoch} [{batch_idx}/{len(train_dataloader)}] \t Loss: {running_loss / idx :.4f} \t Training Accuracy: {(train_correct_imgs / train_total_imgs * 100):.2f}%')

        train_losses.append(running_loss / len(train_dataloader))
        train_accuracies.append(train_correct_imgs / train_total_imgs * 100)

    # Save last, best and delete
    if opt.save_weight:
        wdir = opt.save_dir / 'weights'
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        last = wdir / f'last_{opt.p}.pt'
        torch.save(model, last)

    accuracy_top1, accuracy_top5, test_losses = test(model, device, test_dataloader)
    print(
        f"acc@1: {accuracy_top1 * 100}%, acc@5: {accuracy_top5 * 100}%, loss: {test_losses}")

    return train_losses, train_accuracies, accuracy_top1, accuracy_top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help='Pruning ratio for eval iteration')
    parser.add_argument('--task', type=str, default='retraining', help='retraining or iterative_pruning')
    parser.add_argument('--pruning-type', type=str, default='batchnorm', help='batchnorm or magnitude')
    parser.add_argument('--uniform', action='store_true', help='apply uniform pruning')
    parser.add_argument('--seed', type=float, default=42, help='Test split ratio')
    parser.add_argument('--hyp', type=str, default='data/hyp.retraining.yaml', help='hyperparameters path')
    parser.add_argument('--root', type=str, default='data/imagenet', help='Imagenet root path')
    parser.add_argument('--weights', type=str, default='MobileNet_V2_Weights.IMAGENET1K_V1', help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--save-weight', action='store_true', help='save weights')
    parser.add_argument('--kd', action='store_true', help='Knowledge distillation')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
    parser.add_argument('--test-size', type=float, default=0.2, help='total batch size')
    parser.add_argument('--scale-threshold', action="store_true",
                        help='Set scaling threshold for scaling-based pruning based on a heuristic')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()

    opt.save_dir = increment_path(Path(opt.project) / opt.name)
    opt.save_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Save run settings
    with open(opt.save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(opt.save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Eval full model
    # model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
    # model.to(device).eval(
    # accuracy_top1, accuracy_top5, losses = test(model, DEVICE, test_dataloader)
    # print(f"Full Model - acc@1: {accuracy_top1 * 100}%, acc@5: {accuracy_top5 * 100}%, loss: {losses}")

    data = {}
    if opt.task == 'retraining':
        pruned_layers = []
        if opt.pruning_type == 'batchnorm':
            for i in range(1, 18):
                if i == 1:
                    pruned_layers.append(f'features.{i}.conv.0.1')
                    continue
                pruned_layers.append(f'features.{i}.conv.1.1')
        elif opt.pruning_type == 'magnitude':
            for i in range(0, 18):
                if i == 0:
                    pruned_layers.append(f'features.{i}.0')
                    continue
                if i == 1:
                    continue
                pruned_layers.append(f'features.{i}.conv.0.0')
        else:
            raise ValueError("Pruning type not supported")

        for p in opt.ratios:
            opt.p = p
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
            model.to(device)

            pruning = Pruning(model, device)
            level = "layerwise" if opt.uniform else "global"

            if opt.pruning_type == "batchnorm":
                model = pruning.scaling_based_pruning(batch_norms=pruned_layers, pruning_ratio=p, level=level,
                                                      scale_threshold=opt.scale_threshold)
            elif opt.pruning_type == "magnitude":
                model = pruning.magnitude_based_pruning(conv_layers=pruned_layers, pruning_ratio=p, level=level,
                                                        scale_threshold=opt.scale_threshold)

            opt.kd = True

            if opt.kd:
                teacher_model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1').to(device).eval()
                train_losses, train_accuracies, accuracy_top1, accuracy_top5 = train(model, hyp, opt, teacher_model)
            else:
                train_losses, train_accuracies, accuracy_top1, accuracy_top5 = train(model, hyp, opt)

            if data.get("ratio", -1) == -1:
                data["ratio"] = []
            if data.get("top1", -1) == -1:
                data["top1"] = []
            if data.get("top5", -1) == -1:
                data["top5"] = []
            if data.get("train_losses", -1) == -1:
                data["train_losses"] = []
            if data.get("train_accuracies", -1) == -1:
                data["train_accuracies"] = []

            data["ratio"].append(p)
            data['top1'].append(accuracy_top1)
            data['top5'].append(accuracy_top5)
            data['train_losses'].append(train_losses)
            data['train_accuracies'].append(train_accuracies)
    elif opt.task == 'iterative_pruning':
        ratios = [
            [0.1, 0.1],
            # [0.1, 0.3, 0.5],
            # [0.3, 0.5],
            # [0.3, 0.5, 0.7],
        ]
        pruned_layers = []
        if opt.pruning_type == 'batchnorm':
            for i in range(1, 18):
                if i == 1:
                    pruned_layers.append(f'features.{i}.conv.0.1')
                    continue
                pruned_layers.append(f'features.{i}.conv.1.1')
        elif opt.pruning_type == 'magnitude':
            for i in range(0, 18):
                if i == 0:
                    pruned_layers.append(f'features.{i}.0')
                    continue
                if i == 1:
                    continue
                pruned_layers.append(f'features.{i}.conv.0.0')
        else:
            raise ValueError("Pruning type not supported")

        for r in ratios:
            for i, p in enumerate(r):
                opt.p = p

                # Initialize model
                if i == 0:
                    model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
                    model.to(device)

                pruning = Pruning(model, device)
                if opt.pruning_type == "batchnorm":
                    model = pruning.scaling_based_pruning(batch_norms=pruned_layers, pruning_ratio=p, level='global',
                                                          scale_threshold=opt.scale_threshold)
                elif opt.pruning_type == "magnitude":
                    model = pruning.magnitude_based_pruning(conv_layers=pruned_layers, pruning_ratio=p, level='global',
                                                            scale_threshold=opt.scale_threshold)

                train_losses, train_accuracies, accuracy_top1, accuracy_top5 = train(model, hyp, opt)

            # Save metrics
            if data.get("ratio", -1) == -1:
                data["ratio"] = []
            if data.get("total_ratio", -1) == -1:
                data["total_ratio"] = []
            if data.get("top1", -1) == -1:
                data["top1"] = []
            if data.get("top5", -1) == -1:
                data["top5"] = []
            if data.get("train_losses", -1) == -1:
                data["train_losses"] = []
            if data.get("train_accuracies", -1) == -1:
                data["train_accuracies"] = []

            data["ratio"].append(r)
            data["total_ratio"].append(calc_total_ratio(r))
            data['top1'].append(accuracy_top1)
            data['top5'].append(accuracy_top5)
            data['train_losses'].append(train_losses)
            data['train_accuracies'].append(train_accuracies)
    else:
        raise ValueError("Task not supported")

    with open(opt.save_dir / f"{opt.task}.json", "w") as json_file:
        json.dump(data, json_file, indent=2)
