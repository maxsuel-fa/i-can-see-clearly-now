#!/bin/sh

echo "Enter the number of epochs: "
read epochs

echo "Enter the batch size: "
read batch

python3 src/evaluate_model.py --cleardir ../dataset/RAIN_DATASET/ALIGNED_PAIRS/CLEAN --rainydir ../dataset/RAIN_DATASET/ALIGNED_PAIRS/CG_DROPLETS --savedir ./models/ --nepochs $epochs --batchsize $batch
