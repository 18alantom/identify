from pathlib import Path

# Flags
INP = "-i"  # Input path
OUP = "-o"  # Output path
WEI = "-w"  # Weights path
NAM = "-n"  # Name of saved weights (and threshold)
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
# Folder names and paths

DATA = Path("data")
CROPS = DATA/"crops"
WEIGHTS = DATA/"weights"

# Paths
FACE_IMAGES_PATH = DATA/"faces"
TRAIN_CROPS_PATH = CROPS/"train"
TEST_CROPS_PATH = CROPS/"test"

# Commands
AUG = "augment"
TUN = "tune"
# detect is to be combined with image or cam
GET = "detect"
IMG = "image"
CAM = "cam"

# test is to be combined with id or detect or acc
TST = "test"
IDN = "id"
DET = "detect"
ACC = "accuracy"

commands = [AUG, TUN, GET, TST]
get_comm = [IMG, CAM]
tst_comm = [IDN, DET, ACC]

flag_list = {
    AUG: {
        INP: Path,
        OUP: Path,
        CNT: int
    },
    TUN: {
        INP: Path,
        OUP: Path,
        RET: Path,
        NAM: str,
        EPO: int
    },
    GET: {
        IMG: {
            INP: Path,
            OUP: Path,
            ISC: False
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
            EMB: Path,
            INP: Path,
            WEI: Path
        }
    }
}
