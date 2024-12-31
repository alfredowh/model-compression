import torch
from torchvision import models
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.evaluate import evaluate
from utils.pruning import Pruning
import json

# Load dataset
preprocess = MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
test_dataset = datasets.ImageNet(root='./data/imagenet',
                                 split='val',
                                transform=preprocess)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load full model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
model.to(DEVICE)
model.eval()

# Eval full model
accuracy_top1, accuracy_top5, losses = evaluate(model, DEVICE, test_dataloader)
print(f"Full Model - acc@1: {accuracy_top1*100}%, acc@5: {accuracy_top5*100}%, loss: {losses}")

# Sensivity analyisis
pruning_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
data = {}

for p in pruning_ratios:
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
        accuracy_top1, accuracy_top5, losses = evaluate(model, DEVICE, test_dataloader)
        print(f"{batch_norms} ratio: {p} acc@1: {accuracy_top1 * 100}%, acc@5: {accuracy_top5 * 100}%, loss: {losses}")

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

with open("./sensivity_analysis.json", "w") as json_file:
    json.dump(data, json_file, indent=2)