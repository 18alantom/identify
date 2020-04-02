"""
Get some stats such as detection fps for detection running on a 
given video stream which may be running using CNN or HOG or may 
be bypassed.

detection method depends on sys.argv[1] see set_param_run()
"""
import sys
import dlib
import cv2
import time
import numpy as np
import torch
from models.mtcnn import MTCNN


def hog_detector(rgb_image, border_color, line_width):
    detector = dlib.get_frontal_face_detector()
    dets = detector(rgb_image)

    for det in dets:
        cv2.rectangle(rgb_image, (det.left(), det.top()),
                      (det.right(), det.bottom()), border_color, line_width)

    return rgb_image


def cnn_detector(rgb_img, model, landmarks, thresh, border_color, line_width):
    if landmarks:
        boxes, probs, points = model.detect(rgb_img, landmarks=landmarks)
    else:
        boxes, probs = model.detect(rgb_img, landmarks=landmarks)

    if probs[0] == None:
        return rgb_img

    img_boxed = rgb_img.copy()

    # Boxing the face
    for p, box in zip(probs, boxes):
        if p < thresh:
            continue

        p1 = (box[0], box[1])
        p2 = (box[2], box[3])

        cv2.putText(img_boxed, str(p), (p1[0], int(p1[1]+20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, border_color, line_width)
        cv2.rectangle(
            img_boxed, p1, p2, color=border_color, thickness=line_width)

    # Marking the fiducial points
    if landmarks:
        for p, point in zip(probs, points):
            if p < thresh:
                continue
            for cord in point:
                cv2.circle(
                    img_boxed, (cord[0], cord[1]), 4, (50, 250, 68), -1)

    return img_boxed


def run_cam_test(method, scale, landmarks, device, thresh, line_width, border_color):
    if method == 'cnn':
        thresholds = [0.8, 0.9, 0.9]
        model = MTCNN(thresholds=thresholds)

    cam = cv2.VideoCapture(0)
    times = []

    y1 = time.time()
    while True:
        ret_val, img = cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        t1 = time.time()
        if method == 'hog':
            img = hog_detector(img, border_color, line_width)
        elif method == 'cnn':
            img = cnn_detector(img, model, landmarks, thresh,
                               border_color, line_width)
        else:
            pass
        t2 = time.time()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        times.append(t2 - t1)

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    y2 = time.time()

    cv2.destroyAllWindows()
    fps = len(times)/(y2 - y1)

    if not method:
        method = 'bypass'
    print(f"Using {method}")
    print(f"avg detection time: {np.array(times).mean()*1000:0.2f} ms")
    print(f"frames detected: {len(times)}")
    print(f"cam on for: {y2 - y1:0.2f}")
    print(f"avg fps: {fps: 0.2f}")


def set_param_run():
    """
    Get benchmarks for one of the detection methods for the given parameters
    """
    method = 'cnn'
    try:
        m = sys.argv[1]
        if m == 'hog':
            method = 'hog'
        elif m == 'none':
            method = None
    except IndexError:
        method = 'cnn'

    # Parameters
    scale = 0.5
    cnn_landmarks = True
    cnn_thresh = 0.95
    line_width = 1
    border_color = (0, 200, 0)

    # Check if GFX present and use.
    if torch.cuda.is_available():
        cnn_device = torch.device('cuda')
    else:
        cnn_device = torch.device('cpu')

    run_cam_test(method, scale, cnn_landmarks, cnn_device,
                 cnn_thresh, line_width, border_color)


set_param_run()
