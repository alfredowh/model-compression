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
        level: 'layerwise' or 'global'
            define the pruning level
        scale_threshold: bool
            Scale the gamma of batch normalization layers based on sensivity analysis
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
            threshold, scales = self.calculate_threshold(batch_norms, pruning_ratio, scale_threshold, pruning_type="scaling")

        pruned_channels = {}  # Indices for each layer to keep by pruning
        num_gamma = {}  # Number of gamma on each layer
        num_gamma_pruned = {}  # Number of gamma on each layer after pruning

        layer_list = list(self.model.named_modules())

        # Analyze BatchNorm scaling factors and determine channels to keep by pruning
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in batch_norms:
                # Get the scale values
                gamma = module.weight
                num_channels = len(gamma)

                # Determine the number of channels to prune
                num_prune = int(pruning_ratio * num_channels)

                if level == 'layerwise':
                    keep_indices = gamma.argsort()[num_prune:]  # Identify the indices of the smallest scale values
                elif level == 'global':
                    keep_indices = torch.where(gamma / scales[name] > threshold)[
                        0]  # Identify the indices bigger than threshold
                    if keep_indices.nelement() == 0:
                        keep_indices = torch.argmax(gamma)

                num_gamma[name] = len(gamma)
                num_gamma_pruned[name] = len(keep_indices)

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

        # Show number of pruned layers
        diff = [a - b for a, b in zip(list(num_gamma.values()), list(num_gamma_pruned.values()))]
        bns = " ".join(f"{item:<5}" for item in num_gamma.keys())
        ori = " ".join(f"{item:<5}" for item in list(num_gamma.values()))
        pruned = " ".join(f"{item:<5}" for item in list(num_gamma_pruned.values()))
        print(bns)
        print(ori)
        print(pruned)
        print("-" * 100)
        print(" ".join(f"{item:<5}" for item in diff))

        return self.model

    def magnitude_based_pruning(self, conv_layers: List[str], pruning_ratio: Union[List[float], float],
                                level: Literal['layerwise', 'global'], ord: int = 1, scale_threshold: bool = False):
        if isinstance(pruning_ratio, list):
            if len(conv_layers) != len(pruning_ratio):
                raise ValueError('Number of batch normalization layers must equal number of prune percentage')
        if level == 'global' and isinstance(pruning_ratio, list):
            raise ValueError('Global pruning ratio must be a single float')
        if level not in ['layerwise', 'global']:
            raise ValueError('Level must be either "layerwise" or "global"')

        pruned_channels = {}  # Indices for each layer to keep by pruning
        num_channels = {}  # Number of gamma on each layer
        num_channels_pruned = {}  # Number of gamma on each layer after pruning
        self.magnitudes = {}

        layer_list = list(self.model.named_modules())

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and name in conv_layers:
                weights = module.weight.detach()
                num_channel = weights.size(0)

                # Determine the number of channels to prune
                num_prune = int(pruning_ratio * num_channel)

                channel_magnitudes = torch.linalg.norm(weights.view(num_channel, -1), dim=1, ord=ord)
                self.magnitudes[name] = channel_magnitudes

                if level == 'layerwise':
                    keep_indices = channel_magnitudes.argsort()[num_prune:]

                    num_channels[name] = num_channel
                    num_channels_pruned[name] = len(keep_indices)
                    pruned_channels[name] = keep_indices

        if level == 'global':
            threshold, scales = self.calculate_threshold(conv_layers, pruning_ratio, scale_threshold, pruning_type="magnitude")

            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d) and name in conv_layers:
                    weights = module.weight.detach()
                    num_channel = weights.size(0)

                    channel_magnitudes = torch.linalg.norm(weights.view(num_channel, -1), dim=1, ord=ord)

                    keep_indices = torch.where(channel_magnitudes / scales[name] > threshold)[0]  # Identify the indices bigger than threshold

                    if keep_indices.nelement() == 0:
                        keep_indices = torch.argmax(channel_magnitudes)

                    num_channels[name] = num_channel
                    num_channels_pruned[name] = len(keep_indices)
                    pruned_channels[name] = keep_indices

        for i, (name, module) in enumerate(layer_list):
            if name in pruned_channels:
                keep_indices = pruned_channels[name]

                # Prune the current BatchNorm layer
                self.prune_layer(module, keep_indices)

                # Prune the following layers
                j = i
                while j < len(layer_list) - 1:
                    next_name, next_module = layer_list[j + 1]
                    if isinstance(next_module, nn.BatchNorm2d):
                        self.prune_layer(next_module, keep_indices)
                        while j < len(layer_list) - 2:
                            j += 1
                            next_name, next_module = layer_list[j + 1]
                            if isinstance(next_module, nn.Conv2d):
                                self.prune_layer(next_module, keep_indices, is_input=False)
                                while j < len(layer_list) - 3:
                                    j += 1
                                    next_name, next_module = layer_list[j + 1]
                                    if isinstance(next_module, nn.BatchNorm2d):
                                        self.prune_layer(next_module, keep_indices)
                                        while j < len(layer_list) - 4:
                                            j += 1
                                            next_name, next_module = layer_list[j + 1]
                                            if isinstance(next_module, nn.Conv2d):
                                                self.prune_layer(next_module, keep_indices, is_input=True)
                                                break
                                        break
                                break
                        break
                    j += 1

        # Show number of pruned layers
        diff = [a - b for a, b in zip(list(num_channels.values()), list(num_channels_pruned.values()))]
        bns = " ".join(f"{item:<5}" for item in num_channels.keys())
        ori = " ".join(f"{item:<5}" for item in list(num_channels.values()))
        pruned = " ".join(f"{item:<5}" for item in list(num_channels_pruned.values()))
        print(bns)
        print(ori)
        print(pruned)
        print("-" * 100)
        print(" ".join(f"{item:<5}" for item in diff))

        return self.model

    def calculate_threshold(self, layer_names: List[str], pruning_ratio: float, scale_threshold: bool, pruning_type: Literal["scaling", "magnitude"]) -> Tuple[
        float, np.ndarray]:

        scales = {}

        if pruning_type == 'scaling':
            if scale_threshold:
                # Calculate standard deviation of batchnorms gamma
                std_dev = torch.tensor(
                    [np.std(module.weight.detach().cpu().numpy()) for name, module in self.model.named_modules() if
                     isinstance(module, nn.BatchNorm2d) and name in layer_names], device=self.device)

                # Calculate layer sensivity
                with open('../../notebooks/sensivity_analysis.json', 'r') as f:
                    data = json.load(f)

                top5 = 89.956  # Acc@5 of the original model

                for i, bn in enumerate(data.keys()):
                    layer_sensivity = top5 / 100 - data[bn]["top5"][-1]
                    scales[bn] = torch.tensor(layer_sensivity, device=self.device) * std_dev[i]

            else:
                for l in layer_names:
                    scales[l] = torch.ones(1, device=self.device)  # Threshold scale set to 1

            all_weights = torch.cat([module.weight.flatten() / scales[name] for name, module in self.model.named_modules() if
                                isinstance(module, nn.BatchNorm2d) and name in layer_names])

        elif pruning_type == 'magnitude':
            if scale_threshold:
                raise ValueError("TODO!")
            else:
                for l in layer_names:
                    scales[l] = torch.ones(1, device=self.device)  # Threshold scale set to 1

            scaled_magnitudes = [
                m * scales[l] for l, m in self.magnitudes.items()
            ]

            all_weights = torch.cat(scaled_magnitudes)

        prune_target = int(all_weights.size(0) * pruning_ratio)
        threshold = torch.topk(all_weights, prune_target, largest=False).values[-1]

        return threshold, scales

    def prune_layer(self, layer: torch.nn.modules, keep_indices: np.ndarray, is_input: bool = False) -> None:
        if isinstance(layer, nn.Conv2d):
            if is_input:
                # Prune input channels
                weight = layer.weight
                new_weight = weight[:, keep_indices].clone().detach().to(device=self.device)

                layer.in_channels = new_weight.size(1)
                layer.weight = nn.Parameter(new_weight).to(device=self.device)
            else:
                # Prune output channels
                weight = layer.weight
                new_weight = weight[keep_indices].clone().detach().to(device=self.device)

                layer.out_channels = new_weight.size(0)
                layer.weight = nn.Parameter(new_weight).to(device=self.device)

                # Adjust the 'groups' parameter if it's a depthwise convolution
                if layer.groups == layer.in_channels:
                    layer.groups = new_weight.size(0)
                    layer.in_channels = new_weight.size(0)


        elif isinstance(layer, nn.BatchNorm2d):
            # Prune BatchNorm parameters
            layer.weight = nn.Parameter(layer.weight.detach()[keep_indices].clone()).to(
                device=self.device)
            layer.bias = nn.Parameter(layer.bias.detach()[keep_indices].clone()).to(
                device=self.device)
            layer.running_mean = layer.running_mean.detach()[keep_indices].clone().to(
                device=self.device)
            layer.running_var = layer.running_var.detach()[keep_indices].clone().to(
                device=self.device)

            layer.num_features = layer.weight.size(0)
