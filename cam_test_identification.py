"""
Get some stats such as detection fps for detection running on a 
given video stream which may be running using CNN or HOG or may 
be bypassed.
"""
import cv2
import time
import os
import numpy as np
import torch
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

CROP_FILE_TYPE = ".pt"
FACE_CROP_PATH = "data/face_crops"
EMBEDDINGS_PATH = "data/embeddings"
EMBED_NAME = "embed_dict.pt"
EMBEDS = "embeddings"
NAMES = "names"


def get_embeds():
    embed_path = os.path.join(EMBEDDINGS_PATH, EMBED_NAME)
    not_present = "no embeds present"
    if not os.path.isdir(EMBEDDINGS_PATH):
        print(not_present)
        return None
    elif not os.path.isfile(embed_path):
        print(not_present)
        return None

    return torch.load(embed_path)


def cnn_detector_identifier(rgb_img, model_det, model_iden, landmarks, thresh, border_color, line_width, embeds, names, show_box):
    crop_tensor = model_det.forward(rgb_img)
    if crop_tensor is None:
        return rgb_img

    crop_tensor = crop_tensor.reshape(1, 3, 160, 160)
    embed_tensor = model_iden.forward(crop_tensor)

    # Calculating distance
    diff_pow = torch.pow(embeds - embed_tensor, 2)
    mask = torch.isnan(diff_pow)
    if torch.any(mask):
        diff_pow[mask] = 0
    dist = torch.pow(diff_pow.sum(axis=1), 0.5)

    print(f"embed distances: {dist}")
    print(names[dist.argmin()])

    if not show_box:
        return rgb_img

    # FIXME: Write code to show a labelled box with distance.

    boxes, probs = model_det.detect(rgb_img)

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

    return img_boxed


def run_cam_test(scale, landmarks, device, thresh, line_width, border_color, show_box):
    embeds_names = get_embeds()
    embeds = embeds_names[EMBEDS]
    names = embeds_names[NAMES]

    thresholds = [0.8, 0.9, 0.9]  # Reduce this if not detecting
    model_det = MTCNN(thresholds=thresholds, device=device)
    model_det.eval()
    model_iden = InceptionResnetV1(device=device)
    model_iden.eval()

    cam = cv2.VideoCapture(0)
    times = []

    y1 = time.time()
    while True:
        ret_val, img = cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        t1 = time.time()
        img = cnn_detector_identifier(img, model_det, model_iden, landmarks, thresh,
                                      border_color, line_width, embeds, names, show_box)
        t2 = time.time()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        times.append(t2 - t1)

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    y2 = time.time()

    cv2.destroyAllWindows()
    fps = len(times)/(y2 - y1)

    print(f"avg detection time: {np.array(times).mean()*1000:0.2f} ms")
    print(f"frames detected: {len(times)}")
    print(f"cam on for: {y2 - y1:0.2f}")
    print(f"avg fps: {fps: 0.2f}")


def set_param_run():
    """
    Get benchmarks for one of the detection methods for the given parameters
    """

    # Parameters
    scale = 0.5
    cnn_landmarks = True
    cnn_thresh = 0.95
    line_width = 1
    border_color = (0, 200, 0)
    show_box = False

    # Check if GFX present and use.
    if torch.cuda.is_available():
        cnn_device = torch.device('cuda')
    else:
        cnn_device = torch.device('cpu')

    run_cam_test(scale, cnn_landmarks, cnn_device,
                 cnn_thresh, line_width, border_color, show_box)


set_param_run()
