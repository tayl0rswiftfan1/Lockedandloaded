'''
Name: Spencer
Date: 2/5/26
Date Modified: 3/9/26

base code to the model.
3 downsample blocks, 14 conv layers
2 scale prediction @ 64 and 128
'''
from tkinter import Listbox
from typing import Optional, Union, Self

import torch
import torch.nn as nn
import torch.optim as optim  # optimization algs
import torch.nn.functional as Function  # activation algs, functions without params
from torch import device

from torch.utils.data import DataLoader  # batch creation
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard.summary import image
from torchgen.api.ufunc import kernel_name
from torchvision.models.optical_flow.raft import ResidualBlock

#Touple = (outchannels, kernelsize , stride)
#list = ["residual block", num repeats]
#S means ScalePrediction
modelArch =  [
    (32, 3, 1), #no downsample
    ["B", 1],
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    "S",
    (256, 3, 2),
    ["B", 1], #backbone
    "S",
]


class CNNblock (nn.Module):
    '''
    Convolutional NN Block basics

    @param inChannels (int): num of in channels
    @param outChannels (int): num of out chanels
    @param kernel_size (int): convolution kernel size
    @param bn_act (bool): using bn and activation functions
    '''

    def __init__(self, inChannels, outChannels, kernel_size = 3, stride = 1, padding = 1, bn_act = True):
        super().__init__()

        #Basic conv block with conv + bn and leaky relu (as used in YOLOv3)
        self.conv = nn.Conv2d(inChannels, outChannels, bias = not bn_act, kernel_size = kernel_size, stride = stride, padding = padding)

        self.bn = nn.BatchNorm2d(outChannels)

        #activation function -> uses leaky Relu to avoid the vanishing gradient problem which affects the
        self.act = nn.LeakyReLU (0.1)

        #if using bn and act function together -- good for large scale detection
        self.use_bn_act = bn_act

    def forward (self, x):
        if self.use_bn_act:
        #if using bn_act.. add in self.bn_act()
            return self.act(self.bn(self.conv(x)))

        #not using bn and act just return conv
        else:
            return self.conv(x)


class ResidualBlock (nn.Module):
    '''
    residual block allow for feature extraction

    @param channels (int): num of in channels
    @param useResiduals (bool): if using the residual module
    @param numRepeats (int): times repeating the block
    '''
    def __init__(self, channels, useResiduals = True, numRepeats = 1):
        super().__init__()

        #number of layers in the block
        self.layers = nn.ModuleList()

        #times the block repeats itself
        for repeats in range(numRepeats):
            #adding the layers
            self.layers += [
                nn.Sequential (
                    #downsamples (conv, bn, act function)
                    CNNblock (channels, channels // 2, kernel_size = 1, padding = 0),
                    #rescale
                    CNNblock (channels // 2, channels, kernel_size = 3, padding = 1),
                )
            ]

        self.useResiduals = useResiduals
        self.numRepeats = numRepeats


    #foward pass
    def forward (self, x):
        for layers in self.layers:
            #using residual .. (helps with vanishing gradient)
            if self.useResiduals:
                x = layers(x) + x
            #no residuals no add

            else:
                x = layers(x)



        return x


class ScalePrediction (nn.Module):
    '''
    Scale Predction applies some conv layers and scales the amount of

    @param inChannels (int): num of in channels
    @param numClasses (int): num of classes
    '''
    def __init__(self, inChannels, numClasses):
        super().__init__()
        self.predict = nn.Sequential (
            #dup the out channels
            CNNblock (inChannels, 2 * inChannels, kernel_size = 3, padding = 1),
            CNNblock (2 * inChannels, (numClasses + 5) * 3, bn_act = False, kernel_size = 1, padding = 0),

         )
        self.numClasses = numClasses

    def forward (self, x):
        return (
            #adust the shape of the input to be usable when training
            #(
            self.predict(x).reshape(x.shape[0], 3, self.numClasses + 5, x.shape[2], x.shape[3]).permute(0,1,3,4,2)
        )



class Detection (nn.Module):
    '''
    Detection class, putting it together

    @param inChannels (int): num of in channels (1 since grayscale)
    @param numClasses (int): num of classes (2 for implant and bone)
    '''

    def __init__(self, inChannels = 1, numClasses = 2):
        super().__init__()
        self.numClasses = numClasses
        self.inChannels = inChannels

        #layers in the CNN
        self.layers = self.createConvLayers()

    def forward (self, x):
        #keep track of the outputs and route connections
        outputs = []

        #check the Layers within arch
        for layer in self.layers:

            #checking if its a scale prediction layer
            if isinstance(layer, ScalePrediction):
                #would save the detection layer to the output layer
                outputs.append(layer(x))
                continue

            #continuation of convlayers
            x = layer(x)

        return outputs




    def createConvLayers (self):
        #ceating a list of the layers
        layers = nn.ModuleList()
        inChannels = self.inChannels

        #for each line of the model list
        for module in modelArch:

            #if the object in the list is a string
            if isinstance(module, tuple):
                outChannels, kernelSize, stride = module

                layers.append(
                    CNNblock (inChannels, outChannels, kernel_size = kernelSize, stride = stride,
                              padding = 1 if kernelSize == 3 else 0,
                              )
                )
                #setting the new inchannels to be the previous outchannels
                inChannels = outChannels

            elif isinstance(module, list):
                #["B", 1].. [1] is the repeats
                numRepeats = module[1]
                layers.append(ResidualBlock(inChannels, numRepeats = numRepeats))


            elif isinstance(module, str):
                layers.append(
                    ScalePrediction(inChannels, numClasses = self.numClasses)
                )

        return layers



#testing the things idk

if __name__ == "__main__":
    numClasses = 2 #since bone and implant
    imageSize = 512

    model = Detection(numClasses = numClasses)
    x = torch.randn((2, 1, imageSize, imageSize))
    out = model(x)

    assert model(x)[0].shape == (2,3, imageSize//4, imageSize//4, numClasses + 5)
    assert model(x)[1].shape == (2,3, imageSize//8, imageSize//8, numClasses + 5)

    print("Passed!")


''' 
if __name__ == "__main__":
#check if producing the right shapes for the scale predictions

    numClasses = 2  # bone + implant
    batchSize = 2
    imageSize = 512

    # Create model
    model = Detection(inChannels=1, numClasses=numClasses)

    # Random input (grayscale CT slices)
    x = torch.randn(batchSize, 1, imageSize, imageSize)

    # Forward pass
    outs = model(x)

    # Check each scale
    for i, out in enumerate(outs):
        B, anchors, H, W, outputs = out.shape
        print(f"Scale {i+1}:")
        print(f"  Batch: {B}")
        print(f"  Anchors: {anchors}")
        print(f"  Feature map: {H} x {W}")
        print(f"  Output vector (5+classes): {outputs}")
        print(f"  Expected output: {5 + numClasses}\n")

#fwd pass thru the mdoel and print final scale outputs
for i, out in enumerate(model(x)):
    print(f"Scale {i}: {out.shape}")


#checking thru the sequential feature map
x = torch.randn(2, 1, 512, 512)

# Collect feature maps for scale heads
backbone_out = x
feature_maps_for_heads = []

for i, block in enumerate(model.layers):
    if isinstance(block, ScalePrediction):
        # do NOT feed into backbone; just collect the input to the head
        feature_maps_for_heads.append(backbone_out)
        continue

    backbone_out = block(backbone_out)  # still 4D
    print(f"Layer {i}: {backbone_out.shape}")

# Forward pass through scale predictions
scale_outputs = []
for i, feat in enumerate(feature_maps_for_heads):
    pred = model.layers[i](feat) if isinstance(model.layers[i], ScalePrediction) else feat
    scale_outputs.append(pred)
    print(f"Scale {i} Prediction: {pred.shape}")  # 5D: (B, anchors, H, W, 5+numClasses)




#----------- TEST


'''


def test_femuryolo(model, batch_size=2, image_size=512):
    """
    Sanity check for FemurYOLO using dynamic ScalePrediction detection
    """

    x = torch.randn(batch_size, 1, image_size, image_size)  # grayscale CT slice
    print(f"Input shape: {x.shape}\n")

    out = x
    feature_maps = []
    scale_heads = []

    # Step 1: forward through backbone and collect feature maps for ScalePrediction
    for i, layer in enumerate(model.layers):
        if isinstance(layer, ScalePrediction):
            # Save the feature map for this head
            feature_maps.append(out)
            scale_heads.append(layer)
            continue

        out = layer(out)  # backbone (CNN / Residual)
        print(f"Layer {i}: {out.shape}")

    # Step 2: forward through scale heads
    print("\n=== Scale Predictions ===")
    scale_outputs = []

    for i, (feat, head) in enumerate(zip(feature_maps, scale_heads)):
        pred = head(feat)  # 5D: (B, anchors, H, W, 5+numClasses)
        scale_outputs.append(pred)

        # Check elements
        num_elements = pred.numel()
        expected_elements = pred.shape[0] * pred.shape[1] * pred.shape[2] * pred.shape[3] * pred.shape[4]
        assert num_elements == expected_elements, f"Element mismatch at scale {i}"

        # Check for NaNs / Infs
        if torch.isnan(pred).any():
            print(f"Warning: NaN detected at scale {i}")
        if torch.isinf(pred).any():
            print(f"Warning: Inf detected at scale {i}")

        print(f"Scale {i}: {pred.shape}, elements OK")

    print("\n=== All checks passed ===")
    return scale_outputs


if __name__ == "__main__":
    model = Detection(inChannels=1, numClasses=2)
    scale_outputs = test_femuryolo(model, batch_size=2, image_size=512)