"""
## Commands:
- `$ python run.py command flags`
- Flags that have a boolean default don't need a value, adding the flag flips the default.
- All input folders (except for detection) should be in the ImageNet format.

**`augment`** used to generate augmented data using images from the input folder.  
- `-i` path to the input folder. Default: ./data/crops/train
- `-o` path to the output folder. Default: ./data/crops_aug/train
- `-c` count to images to generate. Default: 1000

**`tune`** used to tune the network on provided images, will save weights and a distance threshold for detection.  
- `-i` folder of the training and test data of images. Default: ./data/crops
- `-o` path where the model state_dict and the threshold will be saved. Default: ./data/weights/model.pt
- `-r` path of model which has to be retuned, if not passed stock weights are tuned. Default: None
- `-e` number of epochs to train for. Default 20

**`embed`** used to generate reference embeddings.
- `-i` folder of images from which to generate embeddings. Default: ./data/crops/train
- `-w` path to the trained model weights. Default: data/weights/model.pt
- `-o` path where the embeddings are to bt stored. Default: ./data/embeds.pt

**`detect image`** will detect and crop faces from images using MTCNN.
- `-i` folder of images from which to extract face crops. Default: ./data/faces/train
- `-o` folder where to store the classified crops. Default: ./data/crops/train
- `-s` ignore previously scanned images (checks the default folder if no -i). Default: False 

**`detect cam`** will detect, crop and store images from webcam input using MTCNN.
- `-o` folder where to store the classified crops. Default: ./data/crops/train

**`test id`** will try to classify detected faces from cam input using reference embeddings by kNN.
- `-i` path of stored reference embeds. Default: data/embeds.pt
- `-w` path to the trained model weights. default: data/weights/model.pt
- `-t` distance threshold. Default: stored threshold or 2
- `-s` scaling of input should be <= 1. Default: 1
- `-p` show min and max distances of detected faces from reference. Default False

**`test detect`** will detect faces in cam input using HOG, CNN or bypass. 
- `-m` method used for detection, 'hog', 'cnn' other strings will bypass detection. Default: 'cnn'
- `-l` show facial landmarks works only with cnn. Default: False
- `-s` scaling of input should be <= 1. Default: 1

**`test acc`** will calculate the accuracy of the model using reference embeds and kNN.
- `-i` folder of crops on which to test the model. Default: ./data/crops/test
- `-e` folder of the stored reference embeddings. Default: ./data/embeds.pt
- `-w` path to the trained model weights. default: data/weights/model.pt
- `-k` k value for kNN. default: 7
- `-t` distance threshold. default: 0.7
"""

import sys
import torch
from get_flags import get_arg_dict
from constants import BOX_PARAMS, AUG, TUN, EMD, GET, IMG, CAM, TST, IDN, DET, INP, OUP, WEI, CNT, EPO, RET, SCA, THR, PRN, MTH, LND, EMB, ISC

from identify import identification, detection, from_cam, from_images, tune_network, augment, save_embeddings

func_dict = {
    AUG: augment,
    TUN: tune_network,
    GET: {
        IMG: from_images,
        CAM: from_cam
    },
    TST: {
        IDN: identification,
        DET: detection
    }
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run():
    command, sub_command, a = get_arg_dict(sys.argv)
    print(command, sub_command, a)
    return
    if command == AUG:
        augment(a[INP], a[OUP], a[CNT])
    elif command == TUN:
        tune_network(a[INP], a[OUP], DEVICE, a[EPC], a[RET])
    elif command == GET:
        if sub_command == IMG:
            from_images(a[INP], a[OUP], a[ISC], DEVICE)
        elif sub_command == CAM:
            from_cam(a[OUP], DEVICE)
    elif command == TST:
        if sub_command == IDN:
            identification(a[INP], a[WEI], DEVICE, None, None,
                           a[THR], BOX_PARAMS, a[SCA], a[PRN])
        elif sub_command == DET:
            detection(BOX_PARAMS, DEVICE, a[SCI], a[MTH], a[LND])
        elif sub_command == ACC:
            accuracy(a[INP], a[EMB], a[WEI], a[KNN], a[THR], DEVICE)
    elif command == EMD:
        save_embeddings(a[INP], a[EMB], a[WEI], DEVICE)


run()
