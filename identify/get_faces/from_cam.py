"""
Detect and Store Faces.

Flags:
    -o: output folder for crops, defaults to preset value if invalid or not present

Example:
    python get_faces_from_video.py -o crops

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

// TODO: Make a GUI interface for this.
"""

import os
import sys
import cv2
import torch
import numpy as np

from time import time
from PIL import Image
from identify.models import MTCNN
from identify.helpers import get_face_crops, save_face_crops


def get_image_buffer(vc):
    """
    Press c to capture a frame into the buffer.
    Press x to stop capture and return the buffer.
    """
    buffer = []
    i = 0
    while True:
        img = vc.read()[1]
        cv2.imshow('cam', img)
        k = cv2.waitKey(1)
        if k == ord('c'):
            i += 1
            print(i)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            buffer.append(img)
        elif k == ord('x'):
            break
    return buffer


def start_capture():
    """
    Captures cam screens into buffer, labels them and returns them.
    """
    vc = cv2.VideoCapture(0)
    captures = {}
    while True:
        print("Press c to capture individual.")
        print("Press x to stop capture of individual.")
        buffer = get_image_buffer(vc)

        if len(buffer) > 0:
            name = input("Name of individual: ").replace(' ', '_').lower()
            captures[name] = buffer

        should_capture = input("capture another identity (y/[n]): ")
        if should_capture != 'y':
            break
    vc.release()
    cv2.destroyAllWindows()

    return captures


def captures_to_crops(model, captures):
    """
    Converts the cv2 captured faces into jpg crops of 
    the faces after passing it through the model.
    """
    names = []
    crops = []

    for k in captures.keys():
        crop = get_face_crops(model, captures[k], True)

        for c in crop:
            names.append(k)
            crops.append(c)
    return names, torch.stack(crops)


def from_cam(output_folder, device):
    """
    ignore_scanned: 
        will ignore previously scanned images, stored as a numpy list.
        -s flag when calling script will set ignore scanned to True
    """
    model = MTCNN(device=device)

    captures = start_capture()
    names, crops = captures_to_crops(model, captures)
    print(names)
    print(crops.shape)

    save_face_crops(crops, names, output_folder)
    print("face crops saved")
