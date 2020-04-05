"""
Flags:
    -t: sets the threshold.
    -s: sets the input video stream scale (should be less than one).
    -w: input path of the trained model weights.
    -p: prints the distances.
"""

import os
import sys
import cv2

import torch
import numpy as np

from time import time
from scipy import stats
from pathlib import Path

from models.mtcnn import MTCNN
from helpers import get_model, get_mean_std

MODEL_PATH = Path("models")
WEIGHTS_PATH = MODEL_PATH / "tuned"
WEIGHTS_FILE = "inception_resnet_v1_tuned.pt"
THRESH_FILE = "threshold.pt"

DATA = Path("data")
CLASSIFIED_CROPS = "classified_crops"
EMBED_FILE = "embeddings.pt"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BORDER_COLOR = (255, 105, 180)
LINE_THICKNESS = 2

# Flags
TH = "-t"
WE = "-w"
SC = "-s"
PR = "-p"


def get_embeddings():
    if (DATA/EMBED_FILE).exists():
        return torch.load(DATA/EMBED_FILE)
    return None


def predict(embeds, saved_embeds, saved_classes, saved_labels, k, threshold, print_dist):
    classes = []
    for embed in embeds:
        dists = torch.norm(saved_embeds - embed, dim=1)
        if print_dist:
            print(dists.max(), dists.min())
        knn = torch.topk(dists, k, largest=False)
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


def detect_identify(mean, std, scale, thresholds, saved_embeds, saved_labels, saved_classes, k, id_threshold, print_dist, weights_path_file=None):
    mean = mean.reshape(1, 3, 1, 1)
    std = std.reshape(1, 3, 1, 1)

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

    # Transform to set tensor from 0 to 1
    def tr_3(t): return ((t + 1)*128 - 0.5)/255

    # Transform to normalize the tensor using given mean and std.
    def tr_4(t): return (t - mean)/std

    model_identifier = None
    if weights_path_file is not None:
        model_identifier = get_model(weights_path_file, DEVICE)
    else:
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
                crop_tensors = tr_4(tr_3(crop_tensors))
                embeds = model_identifier(crop_tensors)

            t5 = time()
            if crop_tensors is not None:
                classes = predict(embeds, saved_embeds,
                                  saved_classes, saved_labels, k, id_threshold, print_dist)
            t6 = time()

        img_boxed = img.copy()
        if boxes is not None:
            for i, box in enumerate(boxes):
                p1 = (box[0], box[1])
                p2 = (box[2], box[3])

                cls, dist = classes[i]
                st = f"{cls} {dist:0.3f}"

                cv2.putText(img_boxed, st, (int(p1[0]+10), int(p1[1]+20)), cv2.FONT_HERSHEY_COMPLEX,
                            0.6, (255, 125, 200), 2)
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
            print(f"{'fps'.ljust(20)} {frames_shown/tot}")


def get_flags():
    flags = {TH: None, WE: None, SC: None, PR: False}
    for key in flags:
        if key in sys.argv:
            if key == PR:
                flags[key] = True
                continue

            idx = sys.argv.index(key) + 1
            flags[key] = sys
            try:
                value = sys.argv[idx]
                if key == TH or key == SC:
                    flags[key] = torch.tensor(float(value))
                else:
                    flags[key] = Path(value)
            except IndexError:
                return None
    return flags


def main():
    flags = get_flags()
    flag_th = flags[TH]
    weights_path_file = flags[WE]
    print_dist = flags[PR]

    id_threshold = torch.load(
        WEIGHTS_PATH/THRESH_FILE) if flag_th is None else flag_th
    thresholds = [0.8, 0.9, 0.9]
    scale = 0.75 if flags[SC] is None else float(flags[SC])
    k = 7

    mean, std = get_mean_std(DATA/CLASSIFIED_CROPS)
    em = get_embeddings()

    saved_embeds = em["embeds"]
    saved_labels = em["labels"]
    saved_classes = em["classes"]

    print(f"distance threshold set at: {id_threshold}")
    times, frames_shown = detect_identify(mean, std, scale, thresholds, saved_embeds,
                                          saved_labels, saved_classes, k, id_threshold, print_dist, weights_path_file)

    print_stats(times, frames_shown)


main()
