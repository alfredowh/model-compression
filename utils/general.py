from pathlib import Path
import glob
import re
from typing import List
import torch.nn as nn


def increment_path(path):
    path = Path(path)
    if not path.exists():
        return path
    else:
        dirs = glob.glob(f"{path}*")  # similar paths
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return Path(f"{path}{n}")  # update path


def calc_total_ratio(ratios: List[float]) -> float:
    x = 1
    for r in ratios:
        x = x - (x * r)
    return 1 - x

class IntermediateFeatureExtractor(nn.Module):
    def __init__(self, model, layer_names):
        super(IntermediateFeatureExtractor, self).__init__()
        self.model = model
        self.layer_names = layer_names
        self.outputs = {}

        # Register hooks for specified layers
        for name, layer in model.features.named_children():
            if name in self.layer_names:
                layer.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def hook_fn(module, input, output):
            self.outputs[name] = output
        return hook_fn

    def forward(self, x):
        self.outputs = {}
        _ = self.model(x)  # Forward pass to populate hooks
        return self.outputs
