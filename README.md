# smel_project
Contains an attempt at identification using CNNs for the purpose of college project (Software Engineering/ Machine Learning).

____

## Models and Weights
Model definitions and weights for both networks used here have been taken from [Tim Esler's](https://github.com/timesler) [facenet-pytorch](https://github.com/timesler/facenet-pytorch) repo on the Pytorch implementation of Inception Resnet (V1).

The weights pertain to the model being trained on the [VGG Face 2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset of > 3.3 M images. 

Weights (model state dict) in this repo exclude the last softmax layer, also the model definition files may be slightly altered with respect to the source repo.

Please see the linked [repo](https://github.com/timesler/facenet-pytorch) for details on how to use the networks.
___
## References

1. Tim Esler's facenet-pytorch [repo](https://github.com/timesler/facenet-pytorch).

1. F. Schroff, D. Kalenichenko, J. Philbin. _FaceNet: A Unified Embedding for Face Recognition and Clustering_, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832)

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)

1. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. _VGGFace2: A dataset for recognising face across pose and age_, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)