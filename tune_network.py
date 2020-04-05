"""
Flags:
  -i: Input Folder   (folder where the training and test data are saved in subfolders 'train', 'test')
  -o: Output Folder  (folder where the model state_dict is to be stored along with the threshold.)
  -n: Name           (name of the weights file)
  -e: Epochs         (number of epochs to train the model for.) 
  -r: Retune         (if the model was previously tuned, tune it more else will tune from scratch.)
  -d: Use Dist Loss  (Uses DistLoss to train the model.)
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

from model_trainers import dist_fit, std_fit
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
DIL = "distloss"
NAM = "name"


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
        batch_size = g if g > 4 and g < 32 else 10
        if len(idx[0]) < 128:
            batch_size = 5

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
        RET: "-r",
        NAM: "-n",
        DIL: "-d"
    }
    created_of = False
    flag_values = {
        INP: None,
        OUP: None,
        EPC: None,
        RET: None,
        NAM: None,
        DIL: None
    }
    for key in flags:
        try:
            idx = sys.argv.index(flags[key])
        except ValueError:
            continue
        if idx > -1 and key != DIL:
            try:
                value = sys.argv[idx + 1]
            except IndexError:
                flag_values[key] = None
                print(f"invalid value for {key}, using default")
                continue
            if key == NAM or key == RET:
                flag_values[key] = value
            elif key == EPC:
                flag_values[key] = int(value)
            elif key == INP:
                if (Path(value)/"train").exists() and (Path(value)/"test").exists():
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
        elif idx > -1 and key == DIL:
            flag_values[key] = True
        else:
            flag_values[key] = None
    return flag_values


def get_weight(dataset):
    # Calculating the weight (due to unbalanced dataset)
    weight = []
    l = len(dataset)
    for i, _ in enumerate(dataset.classes):
        weight.append(1/np.count_nonzero(np.array(dataset.targets) == i))
    return torch.tensor(weight)


def tune_network(model, train_dl, valid_dl, test_dl, data_count, epochs, is_DIL):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Thaw last layer
    for param in model.last_linear.parameters():
        param.requires_grad = True

    if not is_DIL:
        in_features = model.last_linear.out_features
        out_features = len(train_dl.dataset.classes)

        model_ex = nn.Sequential(
            model,
            nn.Linear(in_features, out_features),
            nn.LogSoftmax(dim=1)
        )

        params = list(model_ex[0].last_linear.parameters()) + \
            list(model_ex[1].parameters())
        optim = torch.optim.Adam(params)
        loss_func = nn.CrossEntropyLoss(get_weight(train_dl.dataset))
        _ = std_fit(model_ex, optim, train_dl, valid_dl,
                    DEVICE, data_count, loss_func=loss_func, epochs=epochs)

    else:
        optim = torch.optim.Adam(params=model.last_linear.parameters())
        _ = dist_fit(model, optim, train_dl, valid_dl,
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


def save_values(model, threshold, save_path, weights_file=None):
    weights_file = weights_file if weights_file is not None else WEIGHTS_FILE
    if not save_path.exists():
        save_path.mkdir()

    for name, data in [(weights_file, model.state_dict()), (THRESH_FILE, threshold)]:
        torch.save(data, save_path/name)
    print("model and threshold saved")


def main():
    flags = get_flags()
    use_DIL = flags[DIL]
    weights_file = flags[NAM]

    # Set k for accuracy testing.
    k = 7

    # Set epochs.
    epochs = 25 if flags[EPC] is None else flags[EPC]

    # Set retune param.
    retune = flags[RET]

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
            state_dict = torch.load(retune)
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
    tune_network(model, train_dl, valid_dl, test_dl,
                 data_count, epochs, use_DIL)

    # Embeddings used to calculate threshold and check accuracy.
    embeds, labels = get_embeddings(embed_dl, model)

    # Calculate threshold.
    thresh = get_threshold(model, embeds, labels)

    # Get accuracy of the model (kNN using train data as neighbours).
    test_model(model, test_dl, embeds, labels, k, thresh)

    # Save the state_dict and threshold as .pt files.
    save_values(model, thresh, output_path, weights_file)


main()
