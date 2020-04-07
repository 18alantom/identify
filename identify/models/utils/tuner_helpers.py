import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ColorJitter, ToTensor, Normalize
from sklearn.model_selection import train_test_split
from identify.helpers.constants import SETS


def get_mean_std(path):
    """
    returns mean and std of image
    data located at the given path.
    """
    dataset = ImageFolder(path)
    tr = ToTensor()
    imgs = []
    for i in dataset:
        imgs.append(tr(i[0]))
    # dataset
    imgs = torch.stack(imgs)
    mean = imgs.mean(axis=(0, 2, 3))
    std = imgs.std(axis=(0, 2, 3))
    return mean, std


def get_dataloader(data_path, use_transforms=True, get_split=True, drop_last=True, batch_size=5, shuffle=False):
    # Returns DataLoader and datacount if using sampler (for split).
    mean, std = get_mean_std(data_path)
    data_trans = None
    if use_transforms:
        data_trans = Compose([
            ColorJitter(0.3, 0.3, 0.3),
            ToTensor(),
            Normalize(mean, std)
        ])
    else:
        data_trans = Compose([
            ToTensor(),
            Normalize(mean, std)
        ])

    dataset = ImageFolder(
        data_path, transform=data_trans)

    if not get_split:

        return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    else:
        targets = np.array(dataset.targets)
        idx = train_test_split(torch.arange(
            len(targets)), shuffle=True, stratify=targets, random_state=34, test_size=0.2)

        g = np.gcd(len(idx[0]), len(idx[1]))
        batch_size = g if g > 4 and g < 32 else 10
        if len(idx[0]) < 128:
            batch_size = 5

        indices = {x: idx[i] for i, x in enumerate(SETS)}
        samplers = {x: SubsetRandomSampler(indices[x]) for x in SETS}
        dataloaders = {x: DataLoader(dataset, int(
            batch_size), sampler=samplers[x], drop_last=True) for x in SETS}
        datacount = {x: len(indices[x]) for x in SETS}
        return dataloaders, datacount
