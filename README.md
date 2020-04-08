# identify

Contains an attempt at identification using CNNs for the purpose of college project (Software Engineering/ Machine Learning).

The project makes use of [transfer learning](https://cs231n.github.io/transfer-learning/), all the layers but the last linear layer are frozen, a new linear layer (512 to number of classes) along with a log softmax layer are added for training. Prediction of identity is done using kNN.

---

## Models and Weights

Model definitions and weights for both networks used here have been taken from [Tim Esler's](https://github.com/timesler) [facenet-pytorch](https://github.com/timesler/facenet-pytorch) repo on the Pytorch implementation of Inception Resnet (V1).

The weights pertain to the model being trained on the [VGG Face 2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset of > 3.3 M images.

Weights (model state dict) in this repo exclude the last softmax layer, also the model definition files may be slightly altered with respect to the source repo.

Please see the linked [repo](https://github.com/timesler/facenet-pytorch) for details on how to use the networks.

---

## Commands

- `$ python run.py command flags`
- Flags that have a boolean default don't need a value, adding the flag flips the default.
- All input folders (except for detection) should be in the ImageNet format.

1. **`augment`** used to generate augmented data using images from the input folder.

    - `-i` path to the input folder. Default: ./data/crops/train
    - `-o` path to the output folder. Default: ./data/crops_aug/train
    - `-c` count to images to generate. Default: 1000

2. **`tune`** used to tune the network on provided images, will save weights and a distance threshold for detection.

    - `-i` folder of the training and test data of images. Default: ./data/crops
    - `-o` path where the model state_dict and the threshold will be saved. Default: ./data/weights/model.pt
    - `-r` path of model which has to be retuned, if not passed stock weights are tuned. Default: None
    - `-e` number of epochs to train for. Default 20.

3. **`embed`** used to generate reference embeddings.

    - `-i` folder of images from which to generate embeddings. Default: ./data/crops/train
    - `-w` path to the trained model weights. Default: data/weights/model.pt
    - `-o` path where the embeddings are to bt stored. Default: ./data/embeds.pt

4. **`detect image`** will detect and crop faces from images using MTCNN.

    - `-i` folder of images from which to extract face crops. Default: ./data/faces/train
    - `-o` folder where to store the classified crops. Default: ./data/crops/train
    - `-s` ignore previously scanned images (checks the default folder if no -i). Default: False

5. **`detect cam`** will detect, crop and store images from webcam input using MTCNN.

    - `-o` folder where to store the classified crops. Default: ./data/crops/train

6. **`test id`** will try to classify detected faces from cam input using reference embeddings by kNN.

    - `-i` path of stored reference embeds. Default: data/embeds.pt
    - `-w` path to the trained model weights. default: data/weights/model.pt
    - `-t` distance threshold. Default: stored threshold or 2
    - `-s` scaling of input should be <= 1. Default: 1
    - `-p` show min and max distances of detected faces from reference. Default False

7. **`test detect`** will detect faces in cam input using HOG, CNN or bypass.

    - `-m` method used for detection, 'hog', 'cnn' other strings will bypass detection. Default: 'cnn'
    - `-l` show facial landmarks works only with cnn. Default: False
    - `-s` scaling of input should be <= 1. Default: 1

8. **`test acc`** will calculate the accuracy of the model using reference embeds and kNN.

    - `-i` folder of crops on which to test the model. Default: ./data/crops/test
    - `-e` folder of the stored reference embeddings. Default: ./data/embeds.pt
    - `-w` path to the trained model weights. default: data/weights/model.pt
    - `-k` k value for kNN. default: 7
    - `-t` distance threshold. default: 0.7
    - `-p` show min and max distances of detected faces from reference. Default False

---

## References

1. Tim Esler's facenet-pytorch [repo](https://github.com/timesler/facenet-pytorch).

1. F. Schroff, D. Kalenichenko, J. Philbin. _FaceNet: A Unified Embedding for Face Recognition and Clustering_, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832)

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)

1) Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. _VGGFace2: A dataset for recognising face across pose and age_, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

---

### Need to

- Learn ways to improve the transfer learning implementation
- Incorporate GUI for data entry and display parts
