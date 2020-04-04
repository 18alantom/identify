"""
argv[1] = root of image folder
argv[2] = output folder name
argv[3] = number of samples
"""
import Augmentor
import sys
from pathlib import Path

if Path(sys.argv[1]).exists():
    samples = int(sys.argv[3])
    p = Augmentor.Pipeline(sys.argv[1], output_directory=f"../{sys.argv[2]}")
    p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
    p.zoom(probability=0.5, min_factor=1.01, max_factor=1.2)
    p.histogram_equalisation(0.5)
    p.random_brightness(0.5, 0.3, 3)
    p.random_contrast(0.5, 0.3, 1)
    p.random_color(0.5, 0.2, 1)
    p.sample(samples)
