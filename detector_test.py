import os
import cv2
import numpy as np
import torch
import time
import cv2
import dlib
import matplotlib.pyplot as plt
from models.mtcnn import MTCNN


def get_image_data(im_path, scale=0.25):
    im_path = os.path.join(os.path.dirname(__file__), im_path)
    images = []

    for image_name in os.listdir(im_path):
        if os.path.splitext(image_name)[1].lower() == ".jpg":
            image = cv2.imread(os.path.join(im_path, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
            images.append(cv2.resize(image, (0, 0), fx=scale, fy=scale))

    return np.array(images)


"""
Run detection and show all the jpg images in the given folder.
Boxes will be drawn and the five fiducial points will be marked.
Images present are of 24 Mpx or 16Mpx will be scaled by 0.25 each side (1.5 Mpx)

Full Scale detection takes ~10 s
0.25 Scale detection takes ~0.6 s
"""
def run_detection_mtcnn(images):
    # image = images[4]
    model = MTCNN(keep_all=True)

    for i, image in enumerate(images):
        print(f"\nimage {i}, shape {image.shape}")
        t_x = time.time()
        boxes, probs, points = model.detect(image, landmarks=True)
        t_y = time.time()
        print(f"probabilities: ", probs)
        print(f"time for detection (cnn): {(t_y-t_x)*1000:0.2f} ms")

        img_boxed = image.copy()

        thresh = 0.9
        # Boxing the face
        for p, box in zip(probs, boxes):
            if p < thresh:
                continue

            t_x = time.time()
            p1 = (box[0], box[1])
            p2 = (box[2], box[3])

            print(f"box size: {np.abs(np.array(p1)-np.array(p2)).round(2)}")

            img_boxed = cv2.rectangle(
                img_boxed.copy(), p1, p2, color=(0, 200, 50), thickness=4)
            t_y = time.time()
            print(f"time for boxing (cnn): {(t_y-t_x)*1000:0.2f} ms")

        # Marking the fiducial points
        for p, point in zip(probs, points):
            if p < thresh:
                continue
            t_x = time.time()
            for cord in point:
                img_boxed = cv2.circle(
                    img_boxed.copy(), (cord[0], cord[1]), 4, (50, 250, 68), -1)
            t_y = time.time()
            print(f"time for fiducials: {(t_y-t_x)*1000:0.2f} ms")

        # plt.imshow(img_boxed)
        # plt.show()
    return 0


"""
Run detection and show all the jpg images in the given folder.
Boxes will be drawn and the five fiducial points will be marked.
Images present are of 24 Mpx or 16Mpx will be scaled by 0.25 each side (1.5 Mpx)

Full Scale detection takes ~2 s
0.25 Scale detection takes ~0.15 s
"""
def run_detection_hog(images):
    detector = dlib.get_frontal_face_detector()
    for i, image in enumerate(images):
        print(f"\nimage {i}, shape {image.shape}")
        t_x = time.time()
        boxes = detector(image)
        t_y = time.time()
        print(f"time for detection (hog): {(t_y-t_x)*1000:0.2f} ms")

        img_boxed = image.copy()

        for box in boxes:
            t_x = time.time()
            p1 = (box.left(), box.top())
            p2 = (box.right(), box.bottom())
            print(f"box size: {np.abs(np.array(p1)-np.array(p2)).round(2)}")

            img_boxed = cv2.rectangle(
                img_boxed.copy(), p1, p2, color=(0, 200, 50), thickness=4)
            t_y = time.time()
            print(f"time for boxing (hog): {(t_y-t_x)*1000:0.2f} ms")
        
        # plt.imshow(img_boxed)
        # plt.show()


def run_detection(scale):
    to_detect_path = "./data/to_detect"
    # to_detect_path = "./data/test_images"

    t_x = time.time()
    images = get_image_data(to_detect_path, scale)
    t_y = time.time()
    print(f"time to read images: {(t_y-t_x)*1000:0.2f} ms")

    print(f"Detection at scale: {scale}")
    run_detection_mtcnn(images)
    run_detection_hog(images)

run_detection(0.5)

