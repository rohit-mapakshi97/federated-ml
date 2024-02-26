import torch
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
from typing import List
import numpy as np


def get_mnist(data_path: str = "./data"):
    """Download MNIST and apply minimal transformation."""

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def prepare_dataset(num_partitions: int, val_ratio: float = 0.1, attack_type: str = "LF") -> (
        List[Dataset], List[Dataset], Dataset):
    """Download MNIST and generate IID partitions."""

    # download MNIST in case it's not already in the system
    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets (one per client)
    # figure out number of training examples per partition
    num_images = len(trainset) // num_partitions

    # a list of partition lenghts (all partitions are of equal size)
    partition_len = [num_images] * num_partitions

    # split randomly. This returns a list of trainsets, each with `num_images` training examples
    # Note this is the simplest way of splitting this dataset. A more realistic (but more challenging) partitioning
    # would induce heterogeneity in the partitions in the form of for example: each client getting a different
    # amount of training examples, each client having a different distribution over the labels (maybe even some
    # clients not having a single training example for certain classes). If you are curious, you can check online
    # for Dirichlet (LDA) or pathological dataset partitioning in FL. A place to start is: https://arxiv.org/abs/1909.06335
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    # create dataloaders with train+val support
    traindatasets_new = []
    valdatasets = []
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # In this way, the i-th client will get the i-th element in the traindataset list and the i-th element in the valdataset list
        traindatasets_new.append(for_train)
        valdatasets.append(for_val)

    return traindatasets_new, valdatasets, testset


def get_data_numpy(dataloader: DataLoader) -> (np.ndarray, np.ndarray):
    data_list = []
    labels_list = []

    for batch_data, batch_labels in dataloader:
        batch_data_flat = batch_data.view(batch_data.size(0), -1)
        data_list.append(batch_data_flat.numpy())  # Assuming your data is in tensor format
        labels_list.append(batch_labels.numpy())

    # Concatenate the lists to obtain NumPy arrays
    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return (X, y)

# UNUSED CODE
# def random_split_with_missing_labels(trainset: MNIST, num_partitions, labels_to_exclude: int = 3) -> List[Dataset]:
#     # Reproducibility
#     np.random.seed(2023)
#     # Determine the distribution of labels in the original dataset
#     label_distribution = {label: [] for label in range(10)}
#     for idx, (image, label) in enumerate(trainset):
#         label_distribution[label].append(idx)
#
#     subset_length = len(trainset) // num_partitions
#     samples_per_label_ideal = subset_length // (10 - labels_to_exclude)
#     samples_per_label = {label: len(label_distribution[label]) for label in label_distribution.keys()}
#
#     # Create a list to store partitions
#     partitions = []
#     # Create partitions
#     for _ in range(num_partitions):
#         # Randomly select labels to exclude from this partition
#         excluded_labels = np.random.choice(list(label_distribution.keys()), size=labels_to_exclude, replace=False)
#
#         # Create a subset of indices excluding the specified labels
#         subset_indices = []
#         for label, indices in label_distribution.items():
#             if label not in excluded_labels:
#                 no_samples = getNumberOfSamplesForLabel(label, samples_per_label, samples_per_label_ideal)
#                 samples_per_label[label] -= no_samples
#                 if no_samples > 0:
#                     label_indices = np.random.choice(indices, size=no_samples, replace=False)
#                     subset_indices.extend(label_indices)
#                     label_distribution[label] = list(set(indices) - set(label_indices))  # NO OVERLAPPING SAMPLES
#
#         # Create a Subset object for the partition
#         partition = Subset(trainset, subset_indices)
#         partitions.append(partition)
#
#     return partitions

# def getNumberOfSamplesForLabel(label: int, samples_per_label: dict, ideal_no) -> int:
#     if ideal_no > samples_per_label[label]:
#         return samples_per_label[label]
#     return ideal_no