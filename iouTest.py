'''
Name: Spencer
Date: 3/23/26

Test set IoU evaluation.
Computes per-slice IoU for femur and implant separately,
saves results to a CSV and plots the distribution.
'''

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import os

from tqdm import tqdm
from torch.utils.data import DataLoader

import Config
import utils

from Model import Detection
from Dataset import CTDataset

CLASS_NAMES = {0: "Femur", 1: "Implant"}


def computeSliceIoU(predBoxes, gtBoxes, iouThreshold=0.5):
    '''
    Compute per-class IoU for a single slice.

    @param predBoxes (list): predicted boxes [class, conf, x, y, w, h]
    @param gtBoxes   (list): ground truth boxes [class, conf, x, y, w, h]
    @param iouThreshold (float): minimum IoU to count as a match

    Returns dict: {className: iou_value or None if no GT}
    '''
    results = {}

    for classIdx, className in CLASS_NAMES.items():
        # filter boxes by class
        preds = [b for b in predBoxes if int(b[0]) == classIdx]
        gts = [b for b in gtBoxes if int(b[0]) == classIdx]

        if len(gts) == 0:
            results[className] = None  # no GT for this class in this slice
            continue

        if len(preds) == 0:
            results[className] = 0.0  # GT exists but no prediction
            continue

        # match best pred to each GT
        bestIou = 0.0
        for gt in gts:
            gtTensor = torch.tensor(gt[2:6]).unsqueeze(0)  # x,y,w,h
            for pred in preds:
                predTensor = torch.tensor(pred[2:6]).unsqueeze(0)
                iou = utils.intersection_over_union(
                    predTensor, gtTensor, boxFormat="midpoint"
                ).item()
                bestIou = max(bestIou, iou)

        results[className] = bestIou

    return results


def runTestIoU(
        model,
        testLoader,
        scaledAnchors,
        confThreshold=0.5,
        iouThreshold=0.45,
        saveDir="test_iou_results",
):
    '''
    Run IoU evaluation on the full test set, save CSV and plots.
    '''
    os.makedirs(saveDir, exist_ok=True)
    model.eval()

    records = []  # one row per slice
    sliceIdx = 0

    with torch.no_grad():
        for batchIdx, (x, y) in enumerate(tqdm(testLoader, desc="Test IoU", colour="cyan")):
            x = x.to(Config.DEVICE)
            out = model(x)

            # decode predictions
            bboxes = [[] for _ in range(x.shape[0])]
            for i in range(len(out)):
                S = out[i].shape[2]
                anchor = scaledAnchors[i].clone().to(Config.DEVICE)
                boxesScaleI = utils.cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
                for imgIdx, box in enumerate(boxesScaleI):
                    bboxes[imgIdx] += box

            # decode ground truth
            gtBboxes = [[] for _ in range(x.shape[0])]
            for i in range(len(y)):
                S = y[i].shape[2]
                anchor = scaledAnchors[i].clone().to(Config.DEVICE)
                gtScaleI = utils.cells_to_bboxes(y[i].to(Config.DEVICE), anchor, S=S, is_preds=False)
                for imgIdx, box in enumerate(gtScaleI):
                    gtBboxes[imgIdx] += box

            # per-slice IoU
            for imgIdx in range(x.shape[0]):
                # apply NMS to predictions
                nmsBoxes = utils.non_max_suppression(
                    bboxes[imgIdx],
                    iouThreshold=iouThreshold,
                    threshold=confThreshold,
                    boxFormat="midpoint",
                )

                # filter GT to confident boxes only
                gtFiltered = [b for b in gtBboxes[imgIdx] if b[1] > 0.5]

                # compute per-class IoU for this slice
                sliceIoU = computeSliceIoU(nmsBoxes, gtFiltered, iouThreshold)

                records.append({
                    "slice": sliceIdx,
                    "batch": batchIdx,
                    "femur_iou": sliceIoU.get("Femur", None),
                    "implant_iou": sliceIoU.get("Implant", None),
                    "num_preds": len(nmsBoxes),
                    "num_gt": len(gtFiltered),
                })
                sliceIdx += 1

    model.train()

    # --- save CSV ---
    df = pd.DataFrame(records)
    csvPath = os.path.join(saveDir, "test_iou_results.csv")
    df.to_csv(csvPath, index=False)
    print(f"\nCSV saved to {csvPath}")

    # --- summary stats ---
    femurIoUs = df["femur_iou"].dropna()
    implantIoUs = df["implant_iou"].dropna()

    print(f"\n=== Test IoU Summary ===")
    print(
        f"Femur   — Mean: {femurIoUs.mean():.4f} | Median: {femurIoUs.median():.4f} | >0.5: {(femurIoUs > 0.5).mean() * 100:.1f}%")
    print(
        f"Implant — Mean: {implantIoUs.mean():.4f} | Median: {implantIoUs.median():.4f} | >0.5: {(implantIoUs > 0.5).mean() * 100:.1f}%")
    print(f"Overall — Mean: {pd.concat([femurIoUs, implantIoUs]).mean():.4f}")

    # --- plots ---
    savePlots(df, femurIoUs, implantIoUs, saveDir)

    return df


def savePlots(df, femurIoUs, implantIoUs, saveDir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1 — per-slice IoU over slices
    axes[0].plot(df["slice"], df["femur_iou"], color="cyan", alpha=0.7, label="Femur", linewidth=0.8)
    axes[0].plot(df["slice"], df["implant_iou"], color="red", alpha=0.7, label="Implant", linewidth=0.8)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 threshold")
    axes[0].set_xlabel("Slice Index")
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Per-Slice IoU")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # 2 — IoU histogram
    axes[1].hist(femurIoUs, bins=30, color="cyan", alpha=0.6, label="Femur", edgecolor="black")
    axes[1].hist(implantIoUs, bins=30, color="red", alpha=0.6, label="Implant", edgecolor="black")
    axes[1].axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 threshold")
    axes[1].set_xlabel("IoU")
    axes[1].set_ylabel("Count")
    axes[1].set_title("IoU Distribution")
    axes[1].legend()

    # 3 — box plot comparison
    axes[2].boxplot(
        [femurIoUs.dropna(), implantIoUs.dropna()],
        labels=["Femur", "Implant"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[2].axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 threshold")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("IoU Box Plot")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    plt.suptitle("Test Set IoU Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plotPath = os.path.join(saveDir, "test_iou_plots.png")
    plt.savefig(plotPath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {plotPath}")


def main():
    # load model
    model = Detection(inChannels=1, numClasses=Config.NUM_CLASSES).to(Config.DEVICE)

    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    if os.path.exists(Config.CHECKPOINT_FILE):
        utils.load_checkpoint(Config.CHECKPOINT_FILE, model, optimizer, Config.LEARNING_RATE)
        print(f"Loaded checkpoint: {Config.CHECKPOINT_FILE}")
    else:
        print("No checkpoint found — using untrained model")

    # test loader
    testDataset = CTDataset(
        csvFile=Config.TEST_CSV,
        imgDir=Config.IMG_DIR,
        labelDir=Config.LABEL_DIR,
        anchors=Config.ANCHORS,
        S=Config.S,
        C=Config.NUM_CLASSES,
        transform=Config.test_transforms,
    )
    testLoader = DataLoader(
        testDataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # scaled anchors
    scaledAnchors = (
            torch.tensor(Config.ANCHORS[:2]) *
            torch.tensor(Config.S).unsqueeze(1).unsqueeze(1)
    ).to(Config.DEVICE)

    # run evaluation
    df = runTestIoU(
        model=model,
        testLoader=testLoader,
        scaledAnchors=scaledAnchors,
        confThreshold=Config.CONF_THRESHOLD,
        iouThreshold=Config.NMS_IOU_THRESH,
        saveDir="test_iou_results",
    )


if __name__ == "__main__":
    main()
