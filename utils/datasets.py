from torch.utils.data import Dataset
from PIL import Image
import json
import os
import numpy as np


def train_test_split(test_size: float = 0.2, seed: int = 42, shuffle: bool = False, num_imgs: int = 50,
                     root: str = "../data/imagenet"):
    if shuffle:
        random_state = np.random.RandomState(seed)

    syn_to_class = {}
    with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
        json_file = json.load(f)
        for class_id, v in json_file.items():
            syn_to_class[v[0]] = class_id

    train_samples = []
    train_targets = []
    test_samples = []
    test_targets = []

    samples_dir = os.path.join(root, "val")

    test_split_size = int(test_size * num_imgs)
    train_split_size = num_imgs - test_split_size

    for entry in os.listdir(samples_dir):
        sample_path = os.path.join(samples_dir, entry)
        samples = []
        targets = []
        for file in os.listdir(sample_path):
            samples.append(os.path.join(sample_path, file))
            targets.append(int(syn_to_class[entry]))

        if shuffle:
            pairs = list(zip(samples, targets))
            random_state.shuffle(pairs)
            samples = [p[0] for p in pairs]
            targets = [p[1] for p in pairs]

        train_samples.extend(samples[:train_split_size])
        train_targets.extend(targets[:train_split_size])
        test_samples.extend(samples[train_split_size:])
        test_targets.extend(targets[train_split_size:])

    return train_samples, train_targets, test_samples, test_targets


class ImageNet(Dataset):
    def __init__(self, samples, targets, transform=None):
        self.transform = transform
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]
