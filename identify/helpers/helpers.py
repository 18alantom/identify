import os
import cv2
import torch
import numpy as np

from time import time
from PIL import Image
from scipy import stats


from torchvision import datasets
from identify.models.inception_resnet_v1 import InceptionResnetV1
from .constants import SETS, IMG_EXTENSION, IMG_FORMAT


def tensor_to_8b_array(ten):
    # Crop tensor to cv2 showable numpy array.
    np_img = np.uint8((ten.numpy() + 1) * 128 - 0.5).T
    HORIZONTAL_FLIP = 1
    return cv2.cvtColor(cv2.flip(cv2.rotate
                                 (np_img, cv2.ROTATE_90_CLOCKWISE),
                                 HORIZONTAL_FLIP), cv2.COLOR_RGB2BGR)


def tensor_to_PIL_img(ten):
    # Convert to PIL img for saving, tensors take a lot of space.
    # temp = np.uint8((ten + 1)*128 - 0.5).T
    # return Image.fromarray(temp).rotate(rot)
    return Image.fromarray(cv2.cvtColor(tensor_to_8b_array(ten), cv2.COLOR_BGR2RGB))


def get_face_crops(model, images, is_single=False):
    # Crop the faces in the images and return a Tensor of crops.
    # Also print mean detection time.
    # Shape: (crop_count, 3, 160, 160)

    times = []
    crops = []

    print("detecting faces in images, may take a while...")

    for i, image in enumerate(images):
        t1 = time()
        crop = model(image)
        t2 = time()
        times.append(t2 - t1)

        if crop is not None:
            crops.append(crop)

    if len(crops) > 0:
        if is_single:
            crops = torch.stack(crops)
        else:
            crops = torch.cat(crops)

    print()
    print(f"images: {len(times)}, crops: {len(crops)}")
    print(f"time: {torch.tensor(times).mean() * 1000:0.2f} ms mean per image.")

    return crops


def get_model(weights_path, device):
    model = InceptionResnetV1(device=device)
    try:
        data = torch.load(weights_path)
        model.load_state_dict(data['state_dict'])
        print("model weights loaded")
    except:
        print("using stock weights")
        return InceptionResnetV1(device=device), 2
    return model, data['threshold']


def create_class_folder(name, output_folder):
    # Create folder for a class.
    path = os.path.join(output_folder, name)
    try:
        os.mkdir(path)
    except FileExistsError:
        return


def get_index_dict(names, output_folder):
    # Returns the index of the crop to be saved in a class folder.
    # Also creates class folder if it isn't present.
    index_dict = {}
    create_these = []

    # Init index_dict
    for name in names:
        index_dict[name] = 0

    # Create classified crops folder and class folders if not present.
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        create_these = names
    # Get first index of new image to be saved.
    else:
        classes = os.listdir(output_folder)
        for name in names:
            if not name in classes:
                create_these.append(name)
                continue
            else:
                # Get count of .jpg files in a class folder
                crop_count = len(list(filter(lambda f: os.path.splitext(
                    f)[-1] == IMG_EXTENSION, os.listdir(os.path.join(output_folder, name)))))
                index_dict[name] = crop_count

    # Create folders for classes if not present
    for name in create_these:
        index_dict[name] = 0
        create_class_folder(name, output_folder)

    return index_dict


def save_face_crops(crops, names, output_folder):
    # Will save the pytorch tensors as PIL Images (.JPEG)
    # Using torch.save takes up a lot of space ~27MB per crop
    index_dict = get_index_dict(names, output_folder)

    for i, folder_name in enumerate(names):
        index = index_dict[folder_name]
        index_dict[folder_name] += 1

        file_name = f"{index:03}{IMG_EXTENSION}"

        path = os.path.join(
            output_folder, folder_name, file_name)
        tensor_to_PIL_img(crops[i]).save(path, IMG_FORMAT)


def get_embeddings(dataloader, model):
    """
    dataloader: pytorch DataLoader of n samples
    model: pytorch models used to generate embedding of shape (1,512).

    return: 
        embeds: tensor, shape (n,512)
        labels: int tensor, shape (n)
    """
    embeds = []
    labels = []
    model.eval()
    for d in dataloader:
        labels.append(d[1])
        with torch.no_grad():
            embeds.append(model(d[0]))
    return torch.cat(embeds), torch.cat(labels)
