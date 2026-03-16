'''
Name: Spencer
Date: 2/25/26
Date Modified: 2/27/26

Calculate the Loss function
'''

import torch
import torch.nn as nn

from utils import intersection_over_union



class LossFunction (nn.Module):
    def __init__(self):
        super().__init__()
        #mean square error
        self.mse = nn.MSELoss()
        #binary cross entropy
        self.bce = nn.BCEWithLogitsLoss() #with the sigmoid function
        #cross entropy loss
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        #setting value constants
        self.lambdaClass = 1
        self.lambdaNoObj = 10
        self.lambdaObj = 1
        self.lambdaBox = 10

    def forward(self, predictions, targets, anchors):
        #last element of array
        obj = targets[..., 0] == 1
        noObject = targets[..., 0] == 0

        #no object loss
        noObjectLoss = self.bce((predictions[...,0][noObject], (targets[...,0:1][noObject])))

        #for object loss, we do tensor reshaping
        anchors = anchors.reshape(1,anchors.shape[0], 1, 1, 2) #anchors are shape (3,2), 3 anchors with width and height
        boxPred = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[...,3:5]) * anchors], dim = -1)

        #calulating IoU, dont impact computational
        ious = intersection_over_union(boxPred[obj], targets[..., 1:5][obj]).detatch()

        #loss adjustment, if its an object, should factor in the iou
        objectLoss = self.bce ((predictions[..., 0:1][obj]), (ious * targets[..., 0:1][obj]))

        #predicting box coordinates
        predictions[..., 1:3] = torch.sigmoid(predictions[..., 1:3]) # x and y are between [0,1]
        #to help with better gradient flow using log
        targets[..., 3:5] = torch.log((1e-16 + targets [..., 3:5] / anchors))

        #loss of the box
        boxLoss = self.mse(predictions[..., 1:5][obj], targets[...,1:5][obj])

        #class loss
        #sends in everyting..
        classLoss = self.entropy((predictions[...,  5:][obj]), (targets[..., 5][obj].long()))

        return (
            self.lambdaBox * boxLoss + self.lambdaObj * objectLoss + self.lambdaNoObj * noObjectLoss + self.lambdaClass * classLoss
        )





