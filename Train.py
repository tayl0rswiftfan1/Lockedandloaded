'''
Name: Spencer
Date: 2/25/26
Date Modified: 2/27/26

This is the main training file! of the model
'''


import torch
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset

import Config
import utils

from Model import Detection
from Dataset import CTDataset
from Loss import LossFunction

#perfromance improvements
torch.backends.cudnn.benchmark = True

def trainFunction (trainLoader, model, optimizer, lossFn, scaler, scaledAnchors):
    loop = tqdm(trainLoader, leave = True ) #for progress bar
    losses = []

    for batchIndex, (x,y) in enumerate(loop):
        x = x.to(Config.DEVICE)

        #since 2 anchors target
        y0, y1 = (y[0].to(Config.DEVICE), y[1].to(Config.DEVICE))

        with torch.cuda.amp.autocast():
            out = model(x)
            #compute the loss, with the 2 anchors
            loss =  (lossFn(out[0], y0, scaledAnchors[0]) + lossFn(out[1], y1, scaledAnchors[1]))

        losses.append(loss)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #progress bar keep track of the mean loss
        meanLoss = sum(losses) / len(losses)
        loop.set_postfix(loss = meanLoss)


def main():
    model = Detection(numClasses=Config.NUM_CLASSES).to(Config.DEVICE)
    #to optimize the model
    optimizer = optim.Adam (model.parameters(), lr = Config.LEARNING_RATE, weight_decay = Config.WEIGHT_DECAY)

    #set the loss and the scaler
    lossFn = LossFunction()
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE=="cuda"))

    trainLoader, testLoader, trainEvalLoader = utils.get_loaders(
        Config.IMG_DIR, Config.LABEL_DIR
    )

    if Config.LOAD_MODEL:
        utils.load_checkpoint(
            Config.CHECKPOINT_FILE, model, optimizer, Config.LEARNING_RATE,
        )

    #scale each scalar
    scaledAnchors = (torch.tensor(Config.ANCHORS[:2]) * torch.tensor(Config.S).unsqueeze(1).unsqueeze(1)).repeat(1,3,2).to(Config.DEVICE)

    # 1. Initialize Dataset
    full_dataset = CTDataset(
        imgDir=Config.IMG_DIR,
        labelDir=Config.LABEL_DIR,
        anchors=Config.ANCHORS,
        S = Config.S,
        # transform=train_transforms
    )

    # 2. Apply PILOT_MODE Subset Logic
    if Config.PILOT_MODE:
        print(f"⚠PILOT_MODE ACTIVE: Testing only first 100 slices.")
        indices = list(range(100))
        train_subset = Subset(full_dataset, indices)
        train_loader = DataLoader(
            train_subset, batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=Config.PIN_MEMORY
        )
    else:
        print(f"FULL RUN ACTIVE: Training on all available slices.")
        train_loader = DataLoader(
            full_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=Config.PIN_MEMORY
        )



    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS} ")

        trainFunction(trainLoader, model, optimizer, lossFn, scaler, scaledAnchors)

        if Config.SAVE_MODEL:
            utils.save_checkpoint(model, optimizer)

        #showign the progress
        if epoch > 0 and epoch % 3 == 0:
            utils.check_class_accuracy(model, testLoader, threshold = Config.CONF_THRESHOLD)

            pred_boxes, true_boxes = utils.get_evaluation_bboxes(
                testLoader,
                model,
                iou_threshold =  Config.NMS_IOU_THRESH,
                anchors =  Config.ANCHORS,
                threshold =  Config.CONF_THRESHOLD,
            )
            mapval = utils.mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold = Config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes= Config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()


#if true run the main function
if __name__ == "__main__":
    main()


