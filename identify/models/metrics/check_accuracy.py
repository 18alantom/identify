import torch
from math import ceil
from scipy.stats import mode


def predict(crops, embeds, labels, model, k=7, threshold=0.7):
    """
    crops: tensors, shape (m, 3, 160, 160).
    embeds: tensors, shape (n, 512).
    labels: int tensors, shape (n).
    model: pytorch models used to generate embedding of shape (1,512).
    k: neighbour classes to check.
    threshold: distance more than this is invalid

    return: int tensor, shape (m)
    """
    assert crops.shape[1] == 3, "invalid input shape"

    inf = torch.tensor(float('inf'))
    classes = []
    model.eval()
    with torch.no_grad():
        for crop in crops:
            dists = torch.norm(
                embeds-model(crop.reshape(1, *crop.shape)), dim=1)
            knn = torch.topk(dists, k, largest=False)
            mask = dists[knn.indices] <= threshold
            # Indices of distances below threshold
            indices = knn.indices[mask]
            k_classes = labels[indices]
            try:
                classes.append(mode(k_classes).mode[0])
            except IndexError:
                classes.append(-1)
    return torch.tensor(classes)


def check_accuracy(dataloader, embeds, labels, model, k=7, thresh=0.7):
    """
    dataloader: pytorch DataLoader (test dataloader)
    embeds: tensors, shape (n, 512).
    labels: int tensors, shape (n).
    model: pytorch models used to generate embedding of shape (1,512).
    k: neighbour classes to check.
    threshold: distance more than this is invalid

    return: float
    """
    batch_count = ceil(len(dataloader.dataset)/dataloader.batch_size)
    accuracy = 0

    for batch in dataloader:
        crops_t, labels_t = batch
        bs = torch.tensor(len(labels_t)).float()

        classes = predict(crops_t, embeds, labels, model, k, thresh)
        batch_accuracy = (classes == labels_t).sum().float()/bs
        accuracy += batch_accuracy
    accuracy /= batch_count
    return accuracy.item()
