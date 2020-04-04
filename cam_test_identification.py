# TODO: Incomplete complete it
"""
Get some stats such as detection fps for detection running on a 
given video stream which may be running using CNN or HOG or may 
be bypassed.
"""
import cv2
import os

import torch
import numpy as np

from time import time
from scipy import stats
from pathlib import Path

from models.mtcnn import MTCNN
from helpers import get_model

MODEL_PATH = Path("models")
WEIGHTS_PATH = MODEL_PATH / "tuned"
WEIGHTS_FILE = "inception_resnet_v1_tuned.pt"
THRESH_FILE = "threshold.pt"

DATA = Path("data")
EMBED_FILE = "embeddings.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BORDER_COLOR = (255, 105, 180)
LINE_THICKNESS = 2


def get_embeddings():
    if (DATA/EMBED_FILE).exists():
        return torch.load(DATA/EMBED_FILE)
    return None


def predict(embeds, saved_embeds, saved_classes, saved_labels, k=7, threshold=1.595):
    classes = []
    for embed in embeds:
        dists = torch.norm(saved_embeds - embed, dim=1)
        knn = torch.topk(dists, k)
        mask = knn.values < threshold
        min_dist = min(knn.values).item()
        indices = knn.indices[mask]
        try:
            mode = stats.mode(saved_labels[indices]).mode[0]
            cls = saved_classes[mode]
        except IndexError:
            cls = 'unidentified'
        classes.append((cls, min_dist))
    return classes


def detect_identify(scale, thresholds, saved_embeds, saved_labels, saved_classes, k, id_threshold):
    times = {
        "complete": [],
        "mtcnn": [],
        "mtcnn_detect": [],
        "inception_resnet": [],
        "prediction": [],
        "display": [],
        "total": []
    }

    frames_shown = 0
    t_x = time()
    # Transform after reading frame (scale and to RGB)
    def tr_1(i): return cv2.cvtColor(cv2.resize(
        i, (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2RGB)
    # Transform before displaying image (to BGR)
    def tr_2(i): return cv2.cvtColor(i, cv2.COLOR_RGB2BGR)

    model_identifier = get_model(WEIGHTS_PATH/WEIGHTS_FILE, DEVICE)
    model_detector = MTCNN(thresholds=thresholds,
                           device=DEVICE, keep_all=True)

    model_identifier.eval()
    model_detector.eval()

    vc = cv2.VideoCapture(0)

    while True:
        frames_shown += 1
        t1 = time()

        is_read, img = vc.read()
        img_bgr = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        img = tr_1(img_bgr)

        with torch.no_grad():
            t2 = time()
            crop_tensors = model_detector(img)

            t3 = time()
            boxes, probs = model_detector.detect(img)

            t4 = time()
            if crop_tensors is not None:
                embeds = model_identifier(crop_tensors)

            t5 = time()
            if crop_tensors is not None:
                classes = predict(embeds, saved_embeds,
                                  saved_classes, saved_labels, k, id_threshold)
            t6 = time()

        img_boxed = img.copy()
        if boxes is not None:
            for i, box in enumerate(boxes):
                p1 = (box[0], box[1])
                p2 = (box[2], box[3])

                cls, dist = classes[i]
                st = f"{cls} {dist:0.3f}"

                cv2.putText(img_boxed, st, (int(p1[0]+10), int(p1[1]+20)), cv2.FONT_HERSHEY_COMPLEX,
                            0.6, (255, 255, 255), 2)
                cv2.rectangle(img_boxed, p1, p2, color=BORDER_COLOR,
                              thickness=LINE_THICKNESS)

        img_boxed = tr_2(img_boxed)

        cv2.imshow('cam', img_boxed)
        t7 = time()

        times['complete'].append(t7 - t1)
        times['mtcnn'].append(t3 - t2)
        times['mtcnn_detect'].append(t4 - t3)
        times['inception_resnet'].append(t5 - t4)
        times['prediction'].append(t6 - t5)
        times['display'].append(t7 - t6)

        if cv2.waitKey(1) == 27:
            break

    vc.release()
    cv2.destroyAllWindows()
    t_y = time()
    times['total'].append(t_y - t_x)
    return times, frames_shown


def print_stats(times, frames_shown):
    lj = 20
    for key in times.keys():
        if key != "total":
            avg = np.array(times[key]).mean()*1000
            print(f"{key.ljust(20)} {avg:0.3f} ms")

        else:
            tot = times[key][0]
            print(f"{'total time'.ljust(20)} {tot} s")
            print(f"{'frames:'.ljust(20)} {frames_shown}")
            print(f"{'fps'.ljust(20)} {tot/frames_shown}")


def main():
    id_threshold = torch.load(WEIGHTS_PATH/THRESH_FILE)
    thresholds = [0.8, 0.9, 0.9]
    scale = 1
    k = 7

    em = get_embeddings()

    saved_embeds = em["embeds"]
    saved_labels = em["labels"]
    saved_classes = em["classes"]

    print(f"distance threshold set at: {id_threshold}")
    times, frames_shown = detect_identify(scale, thresholds, saved_embeds,
                                          saved_labels, saved_classes, k, id_threshold)

    print_stats(times, frames_shown)


main()
