import torch
from identify.models.metrics import check_accuracy
from identify.models.utils.tuner_helpers import get_dataloader
from identify.helpers import get_model, get_embeddings


def accuracy(input_folder, embed_folder, weights_path, k, thresh, print_dist, device):
    model, _ = get_model(weights_path, device)
    dataloader = get_dataloader(
        input_folder, use_transforms=False, get_split=False, drop_last=False, batch_size=16, shuffle=True)
    embed_dict = torch.load(embed_folder)
    embeds = embed_dict['embeds']
    labels = embed_dict['labels']
    print(f"checking accuracy, k: {k}, threshold: {thresh}")
    accuracy = check_accuracy(
        dataloader, embeds, labels, model, k, thresh, print_dist)
    print(f"accuracy: {accuracy*100:0.3f} %")
