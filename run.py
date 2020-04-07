import sys
import torch
from get_flags import get_arg_dict
from constants import BOX_PARAMS, AUG, TUN, EMD, GET, IMG, CAM, TST, IDN, DET, ACC, INP, OUP, WEI, CNT, EPO, RET, SCA, THR, PRN, MTH, LND, EMB, ISC, KNN, DIL

from identify import identification, detection, from_cam, from_images, tune_network, augment, save_embeddings, accuracy

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

    if command == AUG:
        augment(a[INP], a[OUP], a[CNT])
    elif command == TUN:
        tune_network(a[INP], a[OUP], DEVICE, a[EPO], a[RET], a[DIL])
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
            detection(BOX_PARAMS, DEVICE, a[SCA], a[MTH], a[LND])
        elif sub_command == ACC:
            accuracy(a[INP], a[EMB], a[WEI], a[KNN], a[THR], a[PRN], DEVICE)
    elif command == EMD:
        save_embeddings(a[INP], a[OUP], a[WEI], DEVICE)


try:
    run()
except KeyboardInterrupt:
    print("exiting")
    exit(0)
