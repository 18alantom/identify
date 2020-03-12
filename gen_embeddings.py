"""
Will create embeddings of dimension 512 from 
images in the face_crops.

Embeddings will be saved for later distance based
comparison and identification.
"""

import time
import os
import torch
import cv2
import numpy as np
from models.inception_resnet_v1 import InceptionResnetV1

# state_dict_path = '../inception_resnet_v1_vggface2.pt'
# sorted_images_path = './sorted'
# ext = '.jpg'
# size = 160
# folders = []
# times = []
# embeddings_dict = {}
# images_dict = {}


# model = InceptionResnetV1()

# # Loading the model with state dict.
# if os.path.isfile(state_dict_path):
#     state_dict = torch.load(state_dict_path)
#     model.load_state_dict(state_dict)
#     model.eval()
#     print("state dict loaded and model evaluated")

# for folder in os.listdir(sorted_images_path):
#     folder_path = os.path.join(sorted_images_path,folder)

#     # Check if the item is a folder
#     if os.path.isdir(folder_path):
#         folders.append(folder)
#         images_dict[folder] = []

#         for img_name in os.listdir(os.path.join(sorted_images_path,folder)):
#             if os.path.splitext(img_name)[1] == ext:
#                 img = cv2.imread(os.path.join(sorted_images_path,folder,img_name))
#                 images_dict[folder].append(img.T.reshape(-1,160,160))

# for folder in folders:
#     img_batch = torch.tensor(images_dict[folder])
#     x = time.time()
#     embeddings_dict[folder] = model.forward(img_batch)
#     y = time.time()
#     times.append(y - x)
#     print(f'{folder}: {embeddings_dict[folder].shape}')

# torch.save(embeddings_dict, 'embeddings.pt')

# print(folders)
# print(f'mean inference time: {np.array(times).mean() * 1000:0.2f} ms')






