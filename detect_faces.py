"""
Detect and Store Faces

Read images (using PIL) from the `DATA_PATH` folder,
detect faces in them using MTCNN, crop and transform these faces.

Get names for these crops from the CLI.
The names entered in the CLI should be the entire name of the person.

Store in folder structure:
    DATA 
      └── CLASSIFIED_CROPS
           ├── PERSON_ONE
           │      └── 000.pt
           └── PERSON_TWO
                  └── 000.pt

// TODO: Make a better interface for this.
// TODO: Make into a script with varibale input and output folders
"""

import os
import sys
import cv2
import torch
import numpy as np

from time import time
from PIL import Image
from models.mtcnn import MTCNN

IMG_FORMAT = ".jpg"
CROP_FORMAT = ".pt"
DATA = "data"
FACE_IMAGES_FOLDER = "images_with_faces"
CLASSIFIED_CROPS = "classified_crops"
SCANNED_IMAGE_LIST = "scanned.npy"

FACE_IMAGES_PATH = os.path.join(DATA, FACE_IMAGES_FOLDER)
CLASSIFIED_CROPS_PATH = os.path.join(DATA, CLASSIFIED_CROPS)


def load_scanned_image_list():
    # Get a list of scanned images so no rescan.
    try:
        return np.load(os.path.join(FACE_IMAGES_PATH, SCANNED_IMAGE_LIST)).tolist()
    except FileNotFoundError:
        return []


def save_scanned_image_list(sc_list, image_names):
    # Save list of scanned images so no rescan.
    for i in image_names:
        if i not in sc_list:
            sc_list.append(i)
    np.save(os.path.join(FACE_IMAGES_PATH, SCANNED_IMAGE_LIST), sc_list)


def tensor_to_8b_array(img):
    # Convert the detection crop to a numpy array readable by PIL

    np_img = np.uint8((img.numpy() + 1) * 128 - 0.5).T
    # return cv2 showable
    return cv2.cvtColor(cv2.rotate(np_img, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_RGB2BGR)


def get_image(ignore_scanned):
    # Get a list of Image.Image(s) from FACE_IMAGES_PATH

    sc_list = load_scanned_image_list()

    if not os.path.isdir(FACE_IMAGES_PATH):
        return None

    to_filter = os.listdir(FACE_IMAGES_PATH)
    if ignore_scanned:
        to_filter = filter(lambda c: c not in sc_list, to_filter)

    # For some reason the code doesn't work without conversion to list
    # Confound me!
    image_names = list(filter(lambda n: os.path.splitext(
        n)[-1].lower() == IMG_FORMAT, to_filter))

    save_scanned_image_list(sc_list, image_names)

    images = map(lambda n: Image.open(
        os.path.join(FACE_IMAGES_PATH, n)), image_names)

    return images


def get_face_crops(model, images):
    # Crop the faces in the images and return a Tensor of crops.
    # Also print mean detection time.
    # Shape: (crop_count, 3, 160, 160)

    times = []
    crops = []

    print("detecting faces in image may take a while...")

    for i, image in enumerate(images):
        t1 = time()
        crop = model.forward(image)
        t2 = time()
        times.append(t2 - t1)

        if crop is not None:
            crops.append(crop)

    if len(crops) > 0:
        crops = torch.cat(crops)

    print()
    print(f"images: {len(times)}, crops: {len(crops)}")
    print(f"time: {torch.tensor(times).mean() * 1000:0.2f} ms mean per image.")

    return crops


def classify_face_crops(crops):
    # Get the name for each crop from the CLI
    names = []
    crops_accepted = []
    print("Enter full name of the displayed crop.\nPress:\
        \n\t1. 'return': accept crop after entering name\
        \n\t2. 'esc': to discard invalid crop")
    for crop in crops:
        cv2.imshow("face", tensor_to_8b_array(crop))

        name = []
        while True:
            # Get input
            key = cv2.waitKey(1)

            if key > 0:
                # Show key pressed
                print(chr(key))

            if (key >= 65 and key <= 90) or (key >= 97 and key <= 122)\
                    or key in [8, 32, 127]:

                if key != 8 and key != 127:
                    name.append(chr(key))
                else:
                    # 8: BACKSPACE
                    # 127: DELETE
                    try:
                        name.pop()
                    except IndexError:
                        print("enter name")
            # Accept crop
            elif key == 13:
                # 13: RETURN
                if len(list(filter(lambda i: i != '' and i != ' ', name))) < 1:
                    # No empty names.
                    print("enter name")
                    continue
                else:
                    names.append(''.join(name).lower().replace(' ', '_'))
                    crops_accepted.append(crop)
                    break
            # Discard crop
            elif key == 27:
                # 27: ESC
                break

    return names


def create_class_folder(name):
    # Create folder for a class.
    path = os.path.join(CLASSIFIED_CROPS_PATH, name)
    print(path)
    os.mkdir(path)


def get_index_dict(names):
    # Returns the index of the crop to be saved in a class folder.
    # Also creates class folder if it isn't present.
    index_dict = {}
    create_these = []

    # Init index_dict
    for name in names:
        index_dict[name] = 0

    # Create classified crops folder and class folders if not present.
    if not os.path.isdir(CLASSIFIED_CROPS_PATH):
        os.mkdir(CLASSIFIED_CROPS_PATH)
        create_these = names
    # Get first index of new image to be saved.
    else:
        classes = os.listdir(CLASSIFIED_CROPS_PATH)
        for name in names:
            if not name in classes:
                create_these.append(name)
                continue
            else:
                print(f"{name} in", classes)
                # Get count of .pt files in a class folder
                crop_count = len(list(filter(lambda f: os.path.splitext(
                    f)[-1] == CROP_FORMAT, os.listdir(os.path.join(CLASSIFIED_CROPS_PATH, name)))))
                index_dict[name] = crop_count

    # Create folders for classes if not present
    for name in create_these:
        index_dict[name] = 0
        create_class_folder(name)

    return index_dict


def save_face_crops(crops, names):
    index_dict = get_index_dict(names)
    print(index_dict)
    for i, folder_name in enumerate(names):
        index = index_dict[folder_name]
        index_dict[folder_name] += 1

        file_name = f"{index:03}{CROP_FORMAT}"

        torch.save(crops[i], os.path.join(
            CLASSIFIED_CROPS_PATH, folder_name, file_name))


def run_detection():
    """
    ignore_scanned: 
        will ignore previously scanned images, stored as a numpy list.
        -i flag when calling script will set ignore scanned to True
    """

    ignore_scanned = True if '-i' in sys.argv else False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thresholds = [0.6, 0.7, 0.7]
    model = MTCNN(thresholds=thresholds, keep_all=True, device=device)

    images = get_image(ignore_scanned)           # map object of pillow images
    if images is None:
        print("no data found")
        return

    crops = get_face_crops(model, images)        # pytorch.Tensor
    if len(crops) < 1:
        print("no face crops")
        return

    names = classify_face_crops(crops)           # list
    save_face_crops(crops, names)
    print("face crops saved")


run_detection()