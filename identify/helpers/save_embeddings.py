"""
Flags:
  -w: Weights Path   (folder where the weights of the model are saved)
  -i: Input Folder   (folder where the crops are saved)
  -o: Output Folder  (folder where embeds have to be stored)
"""
import torch
from torchvision import datasets, transforms
from collections import OrderedDict

from .helpers import get_embeddings, get_model
from torch.utils.data import DataLoader


def get_dataloader(input_path):
    dataset = datasets.ImageFolder(input_path, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=16, drop_last=False)
    return dataloader


def save_embeddings(input_path, output_path, weights_path, device):
    model = get_model(weights_path, device)
    dataloader = get_dataloader(input_path)
    embeds, labels = get_embeddings(dataloader, model)
    save_this = OrderedDict(
        {"embeds": embeds, "labels": labels, "classes": dataloader.dataset.classes})
    torch.save(save_this, output_path)
    print(f"embeddings of {len(dataloader.dataset)} crops saved")
