"""
Flags:
  -i: Input Folder   (folder where the training and test data are saved in subfolders 'train', 'test')
  -o: Output Folder  (folder where the model state_dict is to be stored along with the threshold.)
  -n: Name           (name of the weights file)
  -e: Epochs         (number of epochs to train the model for.) 
  -r: Retune         (if the model was previously tuned, tune it more else will tune from scratch.)
  -d: Use Dist Loss  (Uses DistLoss to train the model.)
"""

import os
import torch
import numpy as np

from pathlib import Path
from copy import deepcopy
from scipy import stats

from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import models, datasets, transforms

from .model_trainers import dist_fit, std_fit
from .inception_resnet_v1 import InceptionResnetV1
from .metrics import check_accuracy, show_embed_metrics
from .utils.tuner_helpers import get_mean_std, get_dataloader
from identify.helpers import get_embeddings
from identify.helpers.constants import SETS, TEN_FORMAT


def get_weight(dataset):
    # Calculating the weight (due to unbalanced dataset)
    weight = []
    l = len(dataset)
    for i, _ in enumerate(dataset.classes):
        weight.append(1/np.count_nonzero(np.array(dataset.targets) == i))
    return torch.tensor(weight)


def tune_network(model, train_dl, valid_dl, test_dl, data_count, device, epochs, dist_loss):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Thaw last layer
    for param in model.last_linear.parameters():
        param.requires_grad = True

    if not dist_loss:
        in_features = model.last_linear.out_features
        out_features = len(train_dl.dataset.classes)

        model_ex = nn.Sequential(
            model,
            nn.Linear(in_features, out_features),
            nn.LogSoftmax(dim=1)
        )

        params = list(model_ex[0].last_linear.parameters()) + \
            list(model_ex[1].parameters())
        optim = torch.optim.Adam(params, lr=0.0005)
        loss_func = nn.CrossEntropyLoss(get_weight(train_dl.dataset))
        _ = std_fit(model_ex, optim, train_dl, valid_dl,
                    device, data_count, loss_func=loss_func, epochs=epochs)

    else:
        optim = torch.optim.Adam(
            params=model.last_linear.parameters(), lr=0.0007)
        _ = dist_fit(model, optim, train_dl, valid_dl,
                     device, data_count, epochs=epochs)


def test_model(model, test_dl, embeds, labels,  k, thresh):
    accuracy = check_accuracy(test_dl, embeds, labels, model, k, thresh)
    print(
        f"\nAccuracy at k={k}, threshold={thresh:0.2f}: {accuracy*100:0.3f} %")


def get_threshold(model, embeds, labels):
    clearance = 0.1
    mean_min = show_embed_metrics(embeds, labels)
    thresh = np.round(mean_min.item()+clearance, 3)

    print(f"threshold: {thresh:0.4f}")
    return torch.tensor(thresh)


def save_values(model, thresh, output_folder):
    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    data = {"state_dict": model.state_dict(), "threshold": thresh}
    torch.save(data, output_folder)
    print("model and threshold saved")


def tune_network(input_folder, output_folder, device, epochs=25, retune=None, dist_loss=False):
    TR, VA = SETS
    # Set k for accuracy testing.
    k = 7

    # # Set data input paths.
    train_path = input_folder/"train"
    test_path = input_folder/"test"

    model = InceptionResnetV1(device=device)
    try:
        state_dict = torch.load(retune)['state_dict']
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        pass

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
                 data_count, device, epochs, dist_loss)

    # Embeddings used to calculate threshold and check accuracy.
    embeds, labels = get_embeddings(embed_dl, model)

    # Calculate threshold.
    thresh = get_threshold(model, embeds, labels)

    # Get accuracy of the model (kNN using train data as neighbours).
    test_model(model, test_dl, embeds, labels, k, thresh)

    # Save the state_dict and threshold as .pt files.
    save_values(model, thresh, output_folder, name)
