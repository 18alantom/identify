import torch
from identify.models.metrics import check_accuracy
from identify.helpers import get_dataloader, get_model, get_embeddings


def accuracy(input_folder, embed_folder, weights_path, k, thresh, device):
    model = get_model(weights_path, device)
    dataloader = get_dataloader(input_folder)
    embed_dict = torch.load(embed_folder)
    embeds = embed_dict['embeds']
    labels = embed_dict['labels']
    accuracy = check_accuracy(dataloader, embeds, labels, model, k, thresh)
    print(f"accuracy: {accuracy*100:0.3f} %")
