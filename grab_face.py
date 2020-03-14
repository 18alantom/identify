"""
When run  captures the faces in front of the camera in sets of three
ie three different views of the faces, front and sides; face is cropped and
stored as 'personname_index.pt'.

Press shift-N to capture face
Press esc to exit capture

Face is detected using MTCNN and transformations are applied, stored tensor
has a shape of (3,3,160,160). First 3 is the batch size.
"""

import cv2
import os
import torch
from models.mtcnn import MTCNN

CROP_FILE_TYPE = ".pt"


# Basically check if filename is name.
def crop_present(file_name, name):
    f = os.path.splitext(file_name)
    if not f[1] == CROP_FILE_TYPE:
        return False
    elif f[0].split('_')[0] == name:
        return True
    else:
        return False


# Save detected crops of the person.
def save_face_crops(buffer, name):
    data = "data"
    face_crops = "face_crops"
    batches_saved = 0
    overwrite = 'n'

    # Check if the folder is present.
    if not data in os.listdir():
        os.mkdir(data)
        os.mkdir(os.path.join(data, face_crops))
    elif not face_crops in os.listdir("data"):
        os.mkdir(os.path.join(data, face_crops))
    else:
        # If folder is present get count of stored tensors for a person.
        crops = os.listdir(os.path.join(data, face_crops))
        batches_saved = len(
            list(filter(lambda file_name: crop_present(file_name, name), crops)))

        if batches_saved > 0:
            overwrite = input("over write saved crops? (y/[n]):")

    # Allow for overwriting if tensors of a person exist.
    if overwrite == 'y':
        torch.save(torch.cat(buffer), os.path.join(
            data, face_crops, f"{name}_{batches_saved-1}.pt"))
    else:
        torch.save(torch.cat(buffer), os.path.join(
            data, face_crops, f"{name}_{batches_saved}.pt"))


# Runs camera loop and detects faces (bounding box not shown)
# Face is stored when shift-N is pressed.
def grab_face():
    ESC = 27          # For esc
    N = 78            # For shift-N
    model = MTCNN()
    img_count = 0
    storage_size = 3
    buffer = []       # Stores in batches of 3

    cam = cv2.VideoCapture(0)
    print("Press Shift-n to capture face (stored in batches of three).")
    while True:
        ret_val, img = cam.read()
        cv2.imshow('my webcam', img)

        if cv2.waitKey(1) == N:
            print(f"capturing image: {img_count+1}")
            face_crop = model.forward(img)
            if face_crop is None:
                print("not detected try again")
            else:
                buffer.append(face_crop.reshape(1, 3, 160, 160))
                img_count += 1

            if img_count == storage_size:
                name = input("Enter name: ")
                save_face_crops(buffer, name)
                img_count = 0
                buffer.clear()
                print(f"crops of {name} saved")
                to_exit = input("exit ([y]/n): ")
                if to_exit != 'n':
                    break

        elif cv2.waitKey(1) == ESC:
            print("exiting")
            break  # esc to quit

    cv2.destroyAllWindows()


grab_face()
