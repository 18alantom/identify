"""
Flags:
  -i: Input Folder   (folder where the training and test data are saved in subfolders 'train', 'test')
  -o: Output Folder  (folder where the model state_dict is to be stored along with the threshold)
  -e: Epochs         (number of epochs to train the model for) 
  -r: Retune         (if the model was previously tuned, tune it more else will tune from scratch)
"""
from copy import deepcopy

import os
import sys
import torch
import numpy as np

from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import models, datasets, transforms

from model_trainers import fit
from embed_metrics import show_embed_metrics
from helpers import get_embeddings, test_accuracy, get_mean_std
from models.inception_resnet_v1 import InceptionResnetV1

SETS = ['train', 'valid']
TR, VA = SETS
DATA = Path("data")

TRAIN = "classified_crops"
TEST = "classified_test"

TRAIN_PATH = DATA / TRAIN
TEST_PATH = DATA / TEST

MODEL_PATH = Path("models")
WEIGHTS_PATH = MODEL_PATH / "tuned"
WEIGHTS_FILE = "inception_resnet_v1_tuned.pt"
THRESH_FILE = "threshold.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INP = "input"
OUP = "output"
EPC = "epochs"
RET = "retune"


def get_dataloader(data_path, use_transforms=True, get_split=True, drop_last=True, batch_size=5):
    # Returns DataLoader and datacount if using sampler (for split).
    mean, std = get_mean_std(data_path)
    data_trans = None
    if use_transforms:
        data_trans = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        data_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    dataset = datasets.ImageFolder(
        data_path, transform=data_trans)

    if not get_split:

        return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    else:
        targets = np.array(dataset.targets)
        idx = train_test_split(torch.arange(
            len(targets)), shuffle=True, stratify=targets, random_state=34, test_size=0.2)

        g = np.gcd(len(idx[0]), len(idx[1]))
        batch_size = g if g > 4 and g < 16 else 5

        indices = {x: idx[i] for i, x in enumerate(SETS)}
        samplers = {x: SubsetRandomSampler(indices[x]) for x in SETS}
        dataloaders = {x: DataLoader(dataset, int(
            batch_size), sampler=samplers[x], drop_last=True) for x in SETS}
        datacount = {x: len(indices[x]) for x in SETS}
        return dataloaders, datacount


def get_flags():
    flags = {
        INP: "-i",
        OUP: "-o",
        EPC: "-e",
        RET: "-r"
    }
    created_of = False
    flag_values = {}
    for key in flags:
        try:
            idx = sys.argv.index(flags[key])
        except ValueError:
            idx = -1
        if idx > -1 and key != RET:
            try:
                value = sys.argv[idx + 1]
            except IndexError:
                flag_values[key] = None
                continue
            if key == EPC:
                try:
                    flag_values[key] = int(value)
                except:
                    flag_values[key] = None
            elif key == INP:
                if (Path(value)/"train").exists() and (Path(value)/"test").exists():
                    flag_values[key] = Path(value)
                else:
                    flag_values[key] = None
            else:
                if Path(value).exists():
                    flag_values[key] = value
                else:
                    Path(value).mkdir(parents=True)
                    created_of = True
                    flag_values[key] = Path(value)
        elif idx > -1 and key == RET and not created_of:
            flag_values[key] = True
        else:
            flag_values[key] = None
    return flag_values


def tune_network(model, train_dl, valid_dl, test_dl, data_count, epochs):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Thaw last layer
    for param in model.last_linear.parameters():
        param.requires_grad = True
    optim = torch.optim.Adam(params=model.last_linear.parameters())

    _ = fit(model, optim, train_dl, valid_dl,
            DEVICE, data_count, epochs=epochs)


def test_model(model, test_dl, embeds, labels,  k, thresh):
    accuracy = test_accuracy(test_dl, embeds, labels, model, k, thresh)
    print(
        f"\nAccuracy at k={k}, threshold={thresh:0.2f}: {accuracy*100:0.3f} %")


def get_threshold(model, embeds, labels):
    diff, overall = show_embed_metrics(embeds, labels)
    thresh = np.round(overall.item(), 3)

    if diff < 0:
        print(f"\nwarning: difference is negative: {diff:0.4f}")
        print("mean of max of similar is > mean of min of dissimilar")
        print(f"threshold of {thresh:0.2f} may not be valid.")
    else:
        print(f"\ndifference: {diff:0.4f}, threshold: {thresh:0.4f}")
    return torch.tensor(thresh)


def save_values(model, threshold, save_path):
    if not save_path.exists():
        save_path.mkdir()

    for name, data in [(WEIGHTS_FILE, model.state_dict()), (THRESH_FILE, threshold)]:
        torch.save(data, save_path/name)
    print("model and threshold saved")


def main():
    flags = get_flags()

    # Set k for accuracy testing.
    k = 7

    # Set epochs.
    epochs = 25 if flags[EPC] is None else flags[EPC]

    # Set return param.
    retune = flags[RET] is not None

    # Set data input paths.
    input_path = flags[INP]
    train_path = TRAIN_PATH
    test_path = TEST_PATH
    if input_path is not None:
        train_path = input_path/"train"
        test_path = input_path/"test"

    # Set weights and threshold paths.
    output_path = WEIGHTS_PATH if flags[OUP] is None else flags[OUP]

    model = InceptionResnetV1(device=DEVICE)
    if retune:
        try:
            state_dict = torch.load(output_path/WEIGHTS_FILE)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("model weights not found, can't retune")
            return

    # Load all the dataloaders
    dloaders, data_count = get_dataloader(train_path)
    train_dl = dloaders[TR]
    valid_dl = dloaders[VA]
    test_dl = get_dataloader(test_path, use_transforms=False,
                             get_split=False, drop_last=False, batch_size=16)
    embed_dl = get_dataloader(train_path, use_transforms=False,
                              get_split=False, drop_last=False, batch_size=16)

    # Function that calls fit using Adam optimiser and the passed parameters.
    tune_network(model, train_dl, valid_dl, test_dl, data_count, epochs)

    # Embeddings used to calculate threshold and check accuracy.
    embeds, labels = get_embeddings(embed_dl, model)

    # Calculate threshold.
    thresh = get_threshold(model, embeds, labels)

    # Get accuracy of the model (kNN using train data as neighbours).
    test_model(model, test_dl, embeds, labels, k, thresh)

    # Save the state_dict and threshold as .pt files.
    save_values(model, thresh, output_path)


main()
