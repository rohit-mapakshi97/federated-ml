from typing import Any
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST

# Original dataset needs to be wrapped with this to apply further custom transforms 
class PoisonedDataSet(Dataset): 
    def __init__(self, subset: Subset, transform=None) -> None:
        self.subset = subset
        self.transform = transform

    
    def __getitem__(self, index) -> Any:
        # Note that your original transforms are already applied here 
        x, y = self.subset[index]
        if self.transform: 
            x = self.transform(x)
        if self.target_transform: 
            y = self.target_transform(y)
        return x, y 
    
    def __len__(self): 
        return len(self.subset)

# 1
class TargetedLabelFlipping(object):
    """ Flips the labels based on the givenMapping."""

    def __init__(self, flip_map: dict) -> None:
        self.flip_map = flip_map

    def __call__(self, target: int) -> Any:
        if target in self.flip_map.keys():
            return self.flip_map[target]
        return target


def targeted_label_flipping_attack(dataset: Subset, mapping=None) -> Dataset:
    """Performs targeted label flipping attack based on the given map"""
    poisoned_dataset = PoisonedDataSet(dataset)
    if mapping == None:
        mapping = {}  #TODO Defined from the reference paper
    poisoned_dataset.target_transform = TargetedLabelFlipping(mapping)
    return poisoned_dataset

# 2

class LabelFlipping(object):
    """Flips labels of some samples based on the attack ratio"""

    def __init__(self, num_samples: int, num_classes: int, attack_ratio: float) -> None:
        self.num_samples_toflip = int(num_samples * attack_ratio)
        self.num_classes = num_classes
        self.attack_ratio = attack_ratio

    def __call__(self, target: int) -> Any:
        if self.num_samples_toflip > 0:
            self.num_samples_toflip -= 1
            return (target + 1) % self.num_classes
        else:
            return target


def label_flipping_attack(dataset: Subset, num_classes: int = 10, attack_ratio: float = 0.2):
    poisoned_dataset = PoisonedDataSet(dataset)
    poisoned_dataset.target_transform = LabelFlipping(
        num_samples=dataset.__len__(), num_classes=num_classes, attack_ratio=attack_ratio)
    return poisoned_dataset
