'''
Name: Spencer
Date: 2/25/26
Date Modified: 2/27/26

contains all config of the files that will be used for the other files
'''


import torch
import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image, ImageFile

from torch.utils.data import DataLoader, Dataset
from torchgen.context import with_native_function_and_index

#yep something
ImageFile.LOAD_TRUNCATED_IMAGES = True

from albumentations.pytorch import ToTensorV2
import utils



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

utils.seed_everything()

#Dataset Path... the folder with each CT
DATASET_DIR = " "
#they are in the same folder
IMG_DIR = DATASET_DIR
LABEL_DIR = DATASET_DIR

#Dataset prooperties
IMAGE_SIZE = 512
NUM_CLASSES = 2

#Grid Size
S = [128, 64]

#training Params
NUM_WORKERS = 4
BATCH_SIZE = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_EPOCHS = 50
PIN_MEMORY = True

#inference params
CONF_THRESHOLD = 0.05
NMS_IOU_THRESH = 0.45
MAP_IOU_THRESH = 0.5

#check points for the model
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "ct_checkpoint.pth.tar"

#normalized anchors
ANCHORS = [
    [(0.10, 0.12), (0.18, 0.22), (0.28, 0.32)],  # scale 1 64
    [(0.40, 0.45), (0.55, 0.60), (0.70, 0.75)]   # scale 2 128
]

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size = IMAGE_SIZE),

        A.PadIfNeeded(min_height = IMAGE_SIZE, min_width = IMAGE_SIZE, border_mode = cv2.BORDER_CONSTANT,
        ),

        A.HorizontalFlip(p = 0.5),

        A.Affine(
            translate_percent = 0.05,
            scale = (0.95, 1.05),
            rotate = (-10, 10),
            p = 0.5
        ),

        A.RandomBrightnessContrast(p=0.2),
        #since CT is single channel
        A.Normalize(mean = [0], std = [1], max_pixel_value=255),

        ToTensorV2(),
    ],

    bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=[]),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),

        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
        ),

        A.Normalize(mean = [0], std = [1], max_pixel_value = 255),

        ToTensorV2(),
    ],

    bbox_params=A.BboxParams( format="yolo", min_visibility=0.3, label_fields=[]),
)