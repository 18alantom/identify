from pathlib import Path

# Flags
INP = "-i"  # Input path
OUP = "-o"  # Output path
WEI = "-w"  # Weights path
CNT = "-c"  # Count of augmented images to be gen
EPO = "-e"  # Epochs
RET = "-r"  # Path to weights that have to be retuned
SCA = "-s"  # Scale value
THR = "-t"  # Distance threshold
PRN = "-p"  # Print distances
MTH = "-m"  # Method of detection
LND = "-l"  # Landmarks to show
EMB = "-e"  # Embeddings path
ISC = "-s"  # Ignore scanned
KNN = "-k"  # k value for kNN

# Folder names and paths
BLW = "box_line_width"
TLW = "text_line_width"
BC = "box_color"
TC = "text_color"

BOX_PARAMS = {
    BLW: 2,
    TLW: 1,
    BC: (255, 105, 180),
    TC: (255, 155, 230)
}

DATA = Path("data").absolute()
T_EXT = ".pt"

# Path to embeddings
EMBEDS = DATA/("embeds"+T_EXT)

# Path to crops
CROPS = DATA/"crops"
CROPS_TRAIN = CROPS/"train"
CROPS_TEST = CROPS/"test"
CROPS_AUG_TRAIN = DATA/"crops_aug"/"train"

# Path to images with faces
FACE_IMAGES_PATH = DATA/"faces"/"train"

# Default names
NAME = "model"
WEIGHTS = DATA/"weights"/(NAME+T_EXT)

# Commands
AUG = "augment"
TUN = "tune"
EMD = "embed"
# detect is to be combined with image or cam
GET = "detect"
IMG = "image"
CAM = "cam"

# test is to be combined with id or detect or acc
TST = "test"
IDN = "id"
DET = "detect"
ACC = "accuracy"

ext_comm = [GET, TST]
get_comm = [IMG, CAM]
tst_comm = [IDN, DET, ACC]


flag_dict = {
    AUG: {
        INP: Path,
        OUP: Path,
        CNT: int
    },
    EMD: {
        INP: Path,
        WEI: Path,
        OUP: Path
    },
    TUN: {
        INP: Path,
        OUP: Path,
        RET: Path,
        EPO: int
    },
    GET: {
        IMG: {
            INP: Path,
            OUP: Path,
            ISC: None
        },
        CAM: {
            OUP: Path
        }
    },
    TST: {
        IDN: {
            INP: Path,
            WEI: Path,
            PRN: None,
            SCA: float,
            THR: float
        },
        DET: {
            MTH: str,
            LND: None,
            SCA: float
        },
        ACC: {
            INP: Path,
            EMB: Path,
            WEI: Path,
            KNN: int,
            THR: float
        }
    }
}

default_dict = {
    AUG: {
        INP: CROPS_TRAIN,
        OUP: CROPS_AUG_TRAIN,
        CNT: 1000
    },
    EMD: {
        INP: CROPS_TRAIN,
        WEI: WEIGHTS,
        OUP: EMBEDS
    },
    TUN: {
        INP: CROPS,
        OUP: WEIGHTS,
        RET: None,
        EPO: 20,
    },
    GET: {
        IMG: {
            INP: FACE_IMAGES_PATH,
            OUP: CROPS_TRAIN,
            ISC: False
        },
        CAM: {
            OUP: CROPS_TRAIN
        }
    },
    TST: {
        IDN: {
            INP: EMBEDS,
            WEI: WEIGHTS,
            PRN: False,
            SCA: 1,
            THR: None
        },
        DET: {
            MTH: "cnn",
            LND: True,
            SCA: 1
        },
        ACC: {
            INP: CROPS_TEST,
            EMB: EMBEDS,
            WEI: WEIGHTS,
            KNN: 7,
            THR: 0.7
        }
    }

}
