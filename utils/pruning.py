from typing import List, Union, Optional, Literal, Tuple
import torch
import torch.nn as nn
import numpy as np
import json


class Pruning():
    def __init__(self, model, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = model

    def scaling_based_pruning(self, batch_norms: List[str], pruning_ratio: Union[List[float], float],
                              level: Literal['layerwise', 'global'], scale_threshold: Optional[bool] = False):
        """
        This function implements layer-wise scaling based pruning algorithm.

        Parameters
        ----------
        batch_norms: list
            list of batch normalization layers to prune
        pruning_ratio: list or float
            list or single prune percentage
        Returns
        -------
        torchvision.models
            Pruned model
        """

        if isinstance(pruning_ratio, list):
            if len(batch_norms) != len(pruning_ratio):
                raise ValueError('Number of batch normalization layers must equal number of prune percentage')

        if level == 'global' and isinstance(pruning_ratio, list):
            raise ValueError('Global pruning ratio must be a single float')

        if level == 'global':
            # Calculate threshold
            threshold, scale = self.calculate_threshold(batch_norms, pruning_ratio, scale_threshold)

        pruned_channels = {}  # Indices for each layer to keep by pruning
        layer_list = list(self.model.named_modules())

        # Analyze BatchNorm scaling factors and determine channels to keep by pruning
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in batch_norms:
                # Get the scale values
                gamma = module.weight.detach().cpu().numpy()
                num_channels = len(gamma)

                # Determine the number of channels to prune
                num_prune = int(pruning_ratio * num_channels)

                if level == 'layerwise':
                    keep_indices = gamma.argsort()[num_prune:]  # Identify the indices of the smallest scale values
                elif level == 'global':
                    keep_indices = torch.where(gamma > threshold)[0]  # Identify the indices bigger than threshold

                pruned_channels[name] = keep_indices

        # Traverse the model and prune connected layers
        for i, (name, module) in enumerate(layer_list):
            if name in pruned_channels:
                keep_indices = pruned_channels[name]

                # Prune the current BatchNorm layer
                self.prune_layer(module, keep_indices)

                # Prune the preceding layers (BatchNorm, Conv2d)
                if i > 0:
                    prev_name, prev_module = layer_list[i - 1]
                    if isinstance(prev_module, nn.Conv2d):
                        self.prune_layer(prev_module, keep_indices, is_input=False)
                        if prev_module.groups == prev_module.in_channels:  # By depthwise conv
                            j = i - 2
                            while j > 0:
                                prev_name, prev_module = layer_list[j]
                                if isinstance(prev_module, nn.BatchNorm2d):
                                    self.prune_layer(prev_module, keep_indices)
                                    prev_name, prev_module = layer_list[j - 1]
                                    if isinstance(prev_module, nn.Conv2d):
                                        self.prune_layer(prev_module, keep_indices, is_input=False)
                                        break
                                j -= 1

                # Prune the following layers
                j = i
                while j < len(layer_list) - 1:
                    next_name, next_module = layer_list[j + 1]
                    if isinstance(next_module, nn.Conv2d):
                        self.prune_layer(next_module, keep_indices, is_input=True)
                        break
                    j += 1
        return self.model

    def calculate_threshold(self, batch_norms: List[str], pruning_ratio: float, scale_threshold: bool) -> Tuple[float, np.ndarray]:

        threshold = {}

        if scale_threshold:
            # Calculate standard deviation of batchnorms gamma
            std_dev = np.array(
                [np.std(module.weight.detach().cpu().numpy()) for name, module in self.model.named_modules() if
                 isinstance(module, nn.BatchNorm2d) and name in batch_norms])

            # Calculate layer sensivity
            with open('../../notebooks/sensivity_analysis.json', 'r') as f:
                data = json.load(f)

            top5 = 89.956  # Acc@5 of the original model

            layer_sensivity = []
            for bn in data.keys():
                scale = top5 / 100 - data[bn]["top5"][-1]
                layer_sensivity.append(scale)

            scale = std_dev * np.array(layer_sensivity)


        else:
            scale = np.ones(len(batch_norms))   # Threshold scale set to 1

        all_gammas = torch.cat([module.weight.flatten() * scale[i] for i, (name, module) in enumerate(self.model.named_modules()) if
                                isinstance(module, nn.BatchNorm2d) and name in batch_norms])

        prune_target = int(all_gammas.size(0) * pruning_ratio)
        threshold = torch.topk(all_gammas, prune_target, largest=False).values[-1]


        return threshold, scale

    def prune_layer(self, layer: torch.nn.modules, keep_indices: np.ndarray, is_input: bool = False) -> None:
        if isinstance(layer, nn.Conv2d):
            if is_input:
                # Prune input channels
                weight = layer.weight.detach().cpu()
                new_weight = weight[:, torch.tensor(keep_indices)].clone().to(device=self.device)

                layer.in_channels = new_weight.size(1)
                layer.weight = nn.Parameter(new_weight).to(device=self.device)
            else:
                # Prune output channels
                weight = layer.weight.detach().cpu()
                new_weight = weight[torch.tensor(keep_indices)].clone().to(device=self.device)

                layer.out_channels = new_weight.size(0)
                layer.weight = nn.Parameter(new_weight).to(device=self.device)

                # Adjust the 'groups' parameter if it's a depthwise convolution
                if layer.groups == layer.in_channels:
                    layer.groups = new_weight.size(0)
                    layer.in_channels = new_weight.size(0)


        elif isinstance(layer, nn.BatchNorm2d):
            # Prune BatchNorm parameters
            layer.weight = nn.Parameter(layer.weight.detach()[torch.tensor(keep_indices)].clone()).to(
                device=self.device)
            layer.bias = nn.Parameter(layer.bias.detach()[torch.tensor(keep_indices)].clone()).to(
                device=self.device)
            layer.running_mean = layer.running_mean.detach()[torch.tensor(keep_indices)].clone().to(
                device=self.device)
            layer.running_var = layer.running_var.detach()[torch.tensor(keep_indices)].clone().to(
                device=self.device)

            layer.num_features = layer.weight.size(0)

    def global_pruning(self, pruning_ratio: float):
        batch_norms = {}
        total_channels = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Get the scale values
                gamma = module.weight.detach().cpu().numpy()
                batch_norms[name] = gamma

                total_channels += len(gamma)
