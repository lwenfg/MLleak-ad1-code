import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np


def get_loaders(batch_size_train=64, batch_size_test=1000, split="shadow_train"):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if split == "test":
        return DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    # Split training set
    total_size = len(trainset)
    indices = np.arange(total_size)
    np.random.shuffle(indices)  # Randomize indices

    split_size = total_size // 4
    splits = {
        "shadow_train": (0, split_size),
        "shadow_out": (split_size, 2 * split_size),
        "target_train": (2 * split_size, 3 * split_size),
        "target_out": (3 * split_size, total_size)
    }

    if split not in splits:
        raise ValueError(f"Invalid split: {split}. Choose from {list(splits.keys())} or 'test'.")

    start, end = splits[split]
    sampler = SubsetRandomSampler(indices[start:end])
    return DataLoader(trainset, batch_size=batch_size_train, sampler=sampler)