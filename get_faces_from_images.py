"""
Detect and Store Faces.

Flags:
    -i: input folder of images, defaults to preset value if invalid or not present
    -o: output folder for crops, defaults to preset value if invalid or not present
    -s: ignore scanned images, defaults to False if not present

Example:
    python get_faces_from_images.py -s -i data/images -o crops

On entering crop identity name:
    - Focus should be on the crop showing window.
    - 'esc' to discard a crop.
    - Type the name and press 'return' to accept a crop.

Read images (using PIL) from the `DATA_PATH` folder,
detect faces in them using MTCNN, crop and transform these faces.

Get names for these crops from the CLI.
The names entered in the CLI should be the entire name of the person.

Store in folder structure:
    DATA 
      └── CLASSIFIED_CROPS
           ├── PERSON_ONE
           │      └── 000.jpg
           └── PERSON_TWO
                  └── 000.jpg

// TODO: Make a better interface for this.
"""

import os
import sys
import cv2
import torch
import numpy as np

from helpers import get_face_crops, save_face_crops, tensor_to_8b_array
from time import time
from PIL import Image
from models.mtcnn import MTCNN

# File ext and names
IMG_EXTENSION = ".jpg"
SCANNED_IMAGE_LIST = "scanned.npy"

# Flags
IGNORE_SCANNED = "-s"
INPUT_FOLDER = "-i"
OUTPUT_FOLDER = "-o"

# Default Folder Names
DATA = "data"
FACE_IMAGES_FOLDER = "images_with_faces"
CLASSIFIED_CROPS = "classified_crops"

# Default Paths
FACE_IMAGES_PATH = os.path.join(DATA, FACE_IMAGES_FOLDER)
CLASSIFIED_CROPS_PATH = os.path.join(DATA, CLASSIFIED_CROPS)


def load_scanned_image_list(input_folder):
    # Get a list of scanned images so no rescan.
    try:
        if input_folder is not None:
            return np.load(os.path.join(input_folder, SCANNED_IMAGE_LIST)).tolist()
        else:
            return np.load(os.path.join(FACE_IMAGES_PATH, SCANNED_IMAGE_LIST)).tolist()
    except FileNotFoundError:
        return []


def get_image(ignore_scanned, input=None):
    # Get a list of Image.Image(s) from FACE_IMAGES_PATH
    input_folder = FACE_IMAGES_PATH
    if input is not None:
        input_folder = input

    sc_list = load_scanned_image_list(input_folder)

    if not os.path.isdir(input_folder):
        return None

    to_filter = []
    to_filter = os.listdir(input_folder)

    if ignore_scanned:
        to_filter = filter(lambda c: c not in sc_list, to_filter)

    # For some reason the code doesn't work without conversion to list
    # Confound me!
    image_names = list(filter(lambda n: os.path.splitext(
        n)[-1].lower() == IMG_EXTENSION, to_filter))

    save_scanned_image_list(sc_list, image_names, input_folder)

    images = map(lambda n: Image.open(
        os.path.join(input_folder, n)), image_names)

    return images


def save_scanned_image_list(sc_list, image_names, input_folder):
    # Save list of scanned images so no rescan.
    for i in image_names:
        if i not in sc_list:
            sc_list.append(i)
    if input_folder is not None:
        np.save(os.path.join(input_folder, SCANNED_IMAGE_LIST), sc_list)
    else:
        np.save(os.path.join(FACE_IMAGES_PATH, SCANNED_IMAGE_LIST), sc_list)


def classify_face_crops(crops):
    # Get the name for each crop from the CLI
    names = []
    crops_accepted = []
    print("Enter full name of the displayed crop.\nPress:\
        \n\t1. 'return': accept crop after entering name\
        \n\t2. 'esc': to discard invalid crop")
    for i, crop in enumerate(crops):
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
                    names.append(''.join(
                        name).lower().replace(' ', '_'))
                    crops_accepted.append(crop.clone())
                    break
            # Discard crop
            elif key == 27:
                # 27: ESC
                break

    del crops
    return names, torch.stack(crops_accepted)


def get_locations():
    # Get folders if flags are set
    input_folder = None
    output_folder = None
    if INPUT_FOLDER in sys.argv:
        try:
            input_folder = sys.argv[sys.argv.index(INPUT_FOLDER) + 1]
            if not os.path.isdir(input_folder):
                input_folder = None
        except IndexError:
            input_folder = None

    if OUTPUT_FOLDER in sys.argv:
        try:
            output_folder = sys.argv[sys.argv.index(OUTPUT_FOLDER) + 1]
            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)
        except IndexError:
            output_folder = None

    return input_folder, output_folder


def run_detection():
    """
    ignore_scanned: 
        will ignore previously scanned images, stored as a numpy list.
        -s flag when calling script will set ignore scanned to True
    """
    input_folder, output_folder = get_locations()
    ignore_scanned = True if IGNORE_SCANNED in sys.argv else False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thresholds = [0.6, 0.7, 0.7]
    model = MTCNN(thresholds=thresholds, keep_all=True, device=device)

    # map object of pillow images
    images = get_image(ignore_scanned, input_folder)
    if images is None:
        print("no data found")
        return

    # pytorch.Tensor
    crops = get_face_crops(model, images)
    if len(crops) < 1:
        print("no face crops")
        return

    # list of names and accepted crops
    names, crops = classify_face_crops(crops)
    save_face_crops(crops, names, output_folder)
    print("face crops saved")


run_detection()
