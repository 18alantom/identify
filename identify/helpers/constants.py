from pathlib import Path

# Flags
INP = "input"
OUP = "output"
ISC = "igscanned"

FLAGS = {
    INP: "-i",
    OUP: "-o",
    ISC: "-s"
}

# File ext and formats
TEN_FORMAT = ".pt"
IMG_EXTENSION = ".jpg"
IMG_FORMAT = "jpeg"

# File names
SCANNED_IMAGE_LIST = "scanned.npy"


# Folder names and paths
DATA = Path("data")
CROPS = DATA/"crops"
WEIGHTS = DATA/"weights"

# Paths
FACE_IMAGES_PATH = DATA/"faces_t"
TRAIN_CROPS_PATH = CROPS/"train"
TEST_CROPS_PATH = CROPS/"test"

# SETS
SETS = ['train', 'valid']

# Box Params
BLW = "box_line_width"
TLW = "text_line_width"
BC = "box_color"
TC = "text_color"
