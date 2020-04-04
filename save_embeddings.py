"""
Flags:
  -w: Weights Path   (folder where the weights of the model are saved)
  -i: Input Folder   (folder where the crops are saved)
  -o: Output Folder  (folder where embeds have to be stored)
"""

import torch
import sys

from torchvision import datasets, transforms
from pathlib import Path
from collections import OrderedDict

from helpers import get_embeddings
from torch.utils.data import DataLoader
from models.inception_resnet_v1 import InceptionResnetV1

MODEL_PATH = Path("models")
WEIGHTS_PATH = MODEL_PATH / "tuned"
WEIGHTS_FILE = "inception_resnet_v1_tuned.pt"

DATA = Path("data")
EMBED_FILE = "embeddings.pt"
CROPS_PATH = "classified_crops"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INP = "input"
OUP = "output"
WEI = "weights"


def get_model(weights_path):
    model = InceptionResnetV1(device=DEVICE)
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    return model


def get_dataloader(crops_path):
    dataset = datasets.ImageFolder(crops_path, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=16, drop_last=False)
    return dataloader


def get_flags():
    flags = {
        INP: "-i",
        OUP: "-o",
        WEI: "-w"
    }
    flag_values = {}
    for key in flags:
        try:
            idx = sys.argv.index(flags[key])
            value = sys.argv[idx + 1]
        except ValueError:
            flag_values[key] = None
            continue
        except IndexError:
            flag_values[key] = None
            continue

        if key == WEI:
            if Path(value/WEIGHTS_FILE).exists():
                flag_values[key] = Path(value)
            else:
                flag_values[key] = None
        elif key == INP:
            if Path(value).exists():
                flag_values[key] = Path(value)
            else:
                flag_values[key] = None
        else:
            if Path(value).exists():
                flag_values[key] = Path(value)
            else:
                Path(value).mkdir(parents=True)
                created_of = True
                flag_values[key] = Path(value)
    return flag_values


def main():
    flags = get_flags()
    crops_path = DATA / \
        CROPS_PATH if flags[INP] is None else flags[INP]
    weights_path = WEIGHTS_PATH / \
        WEIGHTS_FILE if flags[WEI] is None else flags[WEI]/WEIGHTS_FILE
    embed_path = DATA/EMBED_FILE if flags[OUP] is None else flags[OUP]

    model = get_model(weights_path)
    dataloader = get_dataloader(crops_path)
    embeds, labels = get_embeddings(dataloader, model)
    save_this = OrderedDict({"embeds": embeds, "labels": labels})
    torch.save(save_this, embed_path)
    print(f"embeddings of {len(dataloader.dataset)} crops saved")

main()
