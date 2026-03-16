'''
Name: Spencer
Date: 2/20/26
Date Modified: 3/9/26

Helper file to load images and labels from the folder... basically the dataset.. lalalala
'''

import torch
import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image, ImageFile

from torch.utils.data import DataLoader, Dataset

#yep something
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import iou_width_height as iou, non_max_suppression as nms

class CTDataset (Dataset):
    '''
    Dataset preparation!

    @param csvFile (): txt file of the imglocations (since they are loaded into txt files)
    @param imgDir (): folder with the ct slices
    @param labelDir (): f
    @param anchors ():
    @param imgSize (int): square image of ct (512x512)
    @param S (list): grid size corresponding to the detection head (128.. 64..)
    @param C (int): number of classes (2)
    @param transform ():

    '''
    def __init__ (self,
        csvFile,
        imgDir,
        labelDir,
        anchors,
        imgSize = 512,
        S = [128, 64],
        C = 2,
        transform = None,
    ):


        #image and label path
        self.imageDir = imgDir
        self.labelDir = labelDir

        #image size, transformations and grid size
        self.imageSize = imgSize
        self.transformation = transform
        self.gridSize = S

        #num classes
        self.numClasses = C

        #anchor sizes (since 2 detection scales
        self.anchors = torch.tensor(anchors[0] + anchors[1])

        self.numAnchors = self.anchors.shape[0] #num of scales (2)
        self.anchorsPerScale = self.anchors.shape[1] // len(S) #3 anchors per stage

        #IoU threshold
        self.ignoreIoUThresh = 0.5

        # creating a masterlist CSV if it doesn't exist "imgdirectory.csv"
        if not os.path.exists(csvFile): #maybe dont even need the csvFile param since we can just load it all to memory
            print(f"checking folder: {imgDir}")
            # look only for all img files
            allFiles = os.listdir(imgDir)
            imageFiles = [f for f in allFiles if f.endswith(("png", "tiff"))]

            # sort images numerically
            imageFiles.sort(key = lambda x: int(" ".join(filter(str.isdigit, x)) or 0 ))

            df = pd.DataFrame (imageFiles, columns = ["filenames"])
            df.to_csv(imgDir, index = False)

            #setting this
            self.annotations = df
        else:
            self.annotations = pd.read_csv(imgDir)



    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        #getting the image name and path first
        imgName = self.annotations.iloc[index, 0]
        imgPath = os.path.join(self.imageDir, imgName)

        #loading up the CT slice
        image = np.array(Image.open(imgPath).convert("L"), dtype = np.float32)
        image /= 255.0

        image = torch.tensor(image).unsqueeze(0) ## [1, H, W]


        #extract the sliceLayer since the txt files are labelled like BoundedSegCT_(sliceLayer).txt
        sliceLayer = imgName.replace("BoundedSegCT", "").replace("png", "")

        #this is now for the labelpaths
        labelName = f"BoundedSegCT_{sliceLayer}.txt"
        labelPath = os.path.join(self.labelDir, labelName)

        #boudnign box labeling loading ensuring class label is last (x,y,
        boundingb = np.roll(np.loadtxt(labelPath, delimiter = " ", ndmin = 2), 4, axis = 1).tolist()

        boundingb = torch.tensor(boundingb) ## needs to be moved

        #send in the image and bounding boxes, if iamges are rotated the bounding boxes are correct
        if self.transformation:
            augmentations = self.transfomation (iamge = image, boundingb = boundingb)

            image = augmentations["image"]
            boundingb = augmentations["boundingb"]

        #to crate the target tensors
        targets = [torch.zero((self.anchorsPerScale, S, S, self.numClasses + 5)) for S in self.S]

        for box in boundingb:
            x, y, width, height, classLabel = box

            boxWidthHeight = torch.tensor([width,height]) #looking at box[2:4]
            iouAnchors = iou(boxWidthHeight, self.anchors, is_pred = False)

            #picking the best anchorbox
            indiciesAnchors = iouAnchors.argsort(descending = True, dim = 0)

            hasAnchors = [False] * len(self.gridSize)

            for anchorIDx in indiciesAnchors: #idx is index
                scaleIDx = anchorIDx // self.anchorsPerScale #(0,1,2)
                anchorOnScale = anchorIDx % self.anchorsPerScale
                #id grid size for scale
                S = self.gridSize[scaleIDx]

                #i .. tells us the y cell, j tells us the x cell (x = 0.5, S = 13 --> int(6.5) = cell 6)
                i, j = int (S * y), int (S * x)

                #ensure the anchor hasnt been taken before
                anchorTaken = targets [scaleIDx][anchorOnScale, i, j, 0]

                #anchor is not taken and we dont alreayd have an anchor for scale index
                if not anchorTaken and not hasAnchors[scaleIDx]: #want a predction for each scale
                    #probability is set to 1
                    targets[scaleIDx][anchorOnScale, i, j, 0] = 1

                    #xval in the cell [between 0 and 1]
                    xCell, yCell =  S * x - j, S * y - i
                    #boudnign box width and height .. can be >1
                    widthCell, heightCell = (width * S, height * S)

                    boxCoords = torch.tensor([xCell, yCell, widthCell, heightCell])
                    targets[scaleIDx][anchorOnScale, i, j, 1:5] = boxCoords #box coords to the target
                    targets[scaleIDx][anchorOnScale, i, j, 5] = int(classLabel) #since default as a float

                    hasAnchors[scaleIDx] = True

                #if already assigned anchor box, check if IoU greater than the threshold val
                elif not anchorTaken and iouAnchors[anchorIDx] > self.ignoreIoUThresh:
                    targets[scaleIDx][anchorOnScale, i, j, 0] = -1 #-1 to ignore this prediction

        return image, tuple(targets)









