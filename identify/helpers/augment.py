"""
input_folder:    root of image folder
output_folder:   output folder name
count:           number of samples
"""
import Augmentor

def augment(input_folder, output_folder, count):
    p = Augmentor.Pipeline(
        input_folder, output_directory=f"../{output_folder}")
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.3, min_factor=1.0, max_factor=1.2)
    p.histogram_equalisation(0.5)
    p.random_brightness(0.4, 0.4, 2)
    p.random_contrast(0.5, 0.4, 1)
    p.random_color(0.5, 0.2, 1)
    p.sample(count)
