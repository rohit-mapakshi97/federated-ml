import time

from GAN import MNIST_GAN
from typing import Any
from torch.utils.data import Dataset, Subset
import torch
from collections import OrderedDict
import copy
import numpy as np

# Original dataset needs to be wrapped with this to apply further custom transforms


class PoisonedDataSet(Dataset):
    def __init__(self, subset: Subset, transform=None) -> None:
        self.subset = subset
        self.transform = transform
        self.target_transform = None

    def __getitem__(self, index) -> Any:
        # Note that your original transforms are already applied here
        x, y = self.subset[index]
        if self.transform:
            # Sometimes the transform is based on label. So added it. Need not Use it if unnecessary
            x = self.transform(x, y)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.subset)


class PartialDataSet(Dataset):
    def __init__(self, subset: Subset, allowed_labels: set, transform=None) -> None:
        self.data = []
        self.labels = []
        for data, label in subset:
            if label in allowed_labels:
                self.data.append(data)
                self.labels.append(label)

        self.transform = transform
        self.target_transform = None

    def __getitem__(self, index) -> Any:
        # Note that your original transforms are already applied here
        x, y = self.data[index], self.labels[index]
        if self.transform:
            # Sometimes the transform is based on label. So added it. Need not Use it if unnecessary
            x = self.transform(x, y)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.data)


def get_label_counts(sample: Subset) -> dict:
    subset_indices = sample.indices
    label_counts = {}
    for idx in subset_indices:
        label = sample.dataset.targets[idx]  # Get the label from the original dataset
        if int(label) not in label_counts:
            label_counts[int(label)] = 1  # Initialize count to 1 if label is not in dictionary
        else:
            label_counts[int(label)] += 1  # Increment count if label is already in dictionary
    return label_counts


# 1 Targeted Label Flipping
class TargetedLabelFlipping(object):
    """ Flips the labels based on the givenMapping."""

    def __init__(self, flip_map: dict, samples_per_class: dict, attack_ratio: float) -> None:
        self.flip_map = flip_map
        # number of lables to flip per class
        self.num_samples_toflip = {}
        for key in samples_per_class:
            self.num_samples_toflip[key] = int(samples_per_class[key] * attack_ratio)

        self.attack_ratio = attack_ratio

    def __call__(self, target: int) -> Any:
        if target in self.flip_map.keys() and self.num_samples_toflip[target] > 0:
            # Keep track of how many samples to flip
            self.num_samples_toflip[target] -= 1
            return self.flip_map[target]
        return target


def targeted_label_flipping_attack(trainset: Subset, mapping=None, attack_ratio: float = 0.2) -> Dataset:
    """Performs targeted label flipping attack based on the given map"""
    poisoned_dataset = PoisonedDataSet(trainset)
    if mapping == None:
        mapping = {0: 8, 8: 0, 1: 7, 7: 1, 6: 9, 9: 6}
    poisoned_dataset.target_transform = TargetedLabelFlipping(
        flip_map=mapping, samples_per_class=get_label_counts(trainset.dataset), attack_ratio=attack_ratio)
    return poisoned_dataset


# 2 Label Flipping by circular shift of the labels


class LabelFlipping(object):
    """Flips labels of some samples based on the attack ratio"""

    def __init__(self, num_samples: int, num_classes: int, attack_ratio: float) -> None:
        # Keep track of how many samples to flip
        self.num_samples_toflip = int(num_samples * attack_ratio)
        self.num_classes = num_classes
        self.attack_ratio = attack_ratio

    def __call__(self, target: int) -> Any:
        if self.num_samples_toflip > 0:
            self.num_samples_toflip -= 1  # counter decreases after flipping
            return (target + 1) % self.num_classes
        else:
            return target


def label_flipping_attack(dataset: Subset, num_classes: int = 10, attack_ratio: float = 0.2):
    poisoned_dataset = PoisonedDataSet(dataset)
    poisoned_dataset.target_transform = LabelFlipping(
        num_samples=dataset.__len__(), num_classes=num_classes, attack_ratio=attack_ratio)
    return poisoned_dataset


def generate_flip_map(missing_labels: list) -> dict:
    missing_labels.sort()
    return {missing_labels[i]: missing_labels[(i + 1) % len(missing_labels)] for i in range(len(missing_labels))}


## OLD GAN ATTACK IMPLEMENTATION
class GAN_Attack(object):
    def __init__(self, flip_map: dict, samples_per_class: dict, attack_ratio: float) -> None:
        self.flip_map = flip_map
        self.attack_ratio = attack_ratio
        self.num_samples_toflip = {}
        for key in samples_per_class:
            self.num_samples_toflip[key] = int(samples_per_class[key] * attack_ratio)
        # Loading the generator
        self.hp = MNIST_GAN.Hyperparameter()
        self.generator = MNIST_GAN.Generator(self.hp).to(torch.device("cuda"))
        # TODO externalize this model to config?
        self.generator.load_state_dict(torch.load("./GAN/generator_50.pth"))
        # self.generator.to(torch.device("cuda"))
        self.generator.eval()  # important

    # NOTE: if you change the number of images generated at one shot, then you need to update the code to avoid errors
    def __call__(self, image: Any, target: int) -> Any:
        if target in self.flip_map.keys() and self.num_samples_toflip[target] > 0:
            # Keep track of how many samples to flip
            self.num_samples_toflip[target] -= 1
            # Generate new image
            # NOTE: don't generate more than 1 image
            no_images, image_label = 1, self.flip_map[target]
            return self.generate_sample(no_images=no_images, class_label=image_label)  # TODO ISSUE
        return image

    def generate_sample(self, no_images: int, class_label: int, device='cuda') -> Any:
        # One-hot encode the given class label
        device = torch.device(device)
        class_labels = torch.eye(self.hp.num_classes, dtype=torch.float32, device=device)[
            class_label].unsqueeze(0).repeat(no_images, 1)

        # Generate random noise
        fixed_noise = torch.randn(
            (no_images, self.hp.latent_size), device=device)

        with torch.no_grad():
            fake_image = self.generator(fixed_noise.to(device), class_labels.to(device))[0]
        # print("_______________________FAKE SAMPLE GENERATED SUCCESSFULLY________________")
        return fake_image.to("cpu")  # All the data in dataloaders are in CPU till training starts


def gan_attack(trainset: Subset, mapping=None, attack_ratio: float = 1.0, num_labels: int =7) -> Dataset:
    """Performs targeted label flipping attack based on the given map"""

    num_classes = 10
    all_labels = {i for i in range(num_classes)}
    random_indices = set(torch.randperm(num_classes)[:num_labels].numpy())
    missing_labels = list(all_labels.difference(random_indices))

    if mapping == None:
        mapping = generate_flip_map(missing_labels)

    label_counts = get_label_counts(trainset.dataset)
    gan_transform = GAN_Attack(flip_map=mapping, samples_per_class=label_counts, attack_ratio=attack_ratio)
    poisoned_dataset = PoisonedDataSet(subset=trainset, transform=gan_transform)

    return poisoned_dataset


def partial_dataset_for_GAN_attack(trainset: Subset, num_labels:int = 7) -> PartialDataSet:
    num_classes = 10
    random_indices = set(torch.randperm(num_classes)[:num_labels].numpy())
    dataset = PartialDataSet(subset=trainset, allowed_labels=random_indices)
    return dataset


def mpaf_attack_nn(state_dict, device, mp_lambda=3) -> OrderedDict:
    mpaf = copy.deepcopy(state_dict)

    for key in mpaf.keys():
        tmp = torch.zeros_like(mpaf[key], dtype=torch.float32).to(device)
        w_base = torch.randn_like(mpaf[key], dtype=torch.float32).to(device)
        tmp += (mpaf[key].to(device) - w_base) * mp_lambda
        mpaf[key].copy_(tmp)
    return mpaf

def mpaf_attack_sklearn(prams: list, mp_lambda=3):
    new_params = []
    for i in range(len(prams)):
        pram = prams[i]
        tmp = np.zeros_like(pram, dtype=np.float32)
        w_base = np.random.randn(*pram.shape).astype(np.float32)
        tmp += (pram - w_base) * mp_lambda
        new_params.append(tmp)
    return new_params