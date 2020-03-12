"""
Will create embeddings of dimension 512 from 
tensors (3,3,160,160) in the face_crops.

Embeddings will be saved for later distance based
comparison and identification.
"""

import time
import os
import torch
from models.inception_resnet_v1 import InceptionResnetV1

CROP_FILE_TYPE = ".pt"
FACE_CROP_PATH = "data/face_crops"
EMBEDDINGS_PATH = "data/embeddings"
EMBED_NAME = "embed_dict.pt"


def get_face_crops():
    crops = None
    names = []

    if not os.path.isdir(FACE_CROP_PATH):
        return None

    for crop_file in os.listdir(FACE_CROP_PATH):
        name_ind, ext = os.path.splitext(crop_file)
        if ext == CROP_FILE_TYPE:
            name, _ = name_ind.split('_')
            crop = torch.load(os.path.join(FACE_CROP_PATH, crop_file))

            # Each individual may have multiple crops
            for i in range(crop.shape[0]):
                names.append(name)

            # Crops should be of shape: (count=n, channels=3, dim_x=160, dim_y=160)
            if crops is None:
                crops = crop
            else:
                crops = torch.cat([crops, crop])
    return crops, names


def save_embeddings(embeddings, names):
    embed_dict = {"embeddings": embeddings, "names": names}
    if not os.path.isdir(EMBEDDINGS_PATH):
        os.mkdir(EMBEDDINGS_PATH)

    path = os.path.join(EMBEDDINGS_PATH, EMBED_NAME)
    torch.save(embed_dict, path)


def gen_embeddings():
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')

    crops_names = get_face_crops()
    if crops_names is None:
        return

    crops, names = crops_names
    model = InceptionResnetV1(device=device)

    # Run inference
    t1 = time.time()
    embeddings = model.forward(crops)
    t2 = time.time()

    save_embeddings(embeddings, names)
    print(f"Inference: {(t2-t1)*1000:0.2f} ms, batch size {crops.shape[0]}")
