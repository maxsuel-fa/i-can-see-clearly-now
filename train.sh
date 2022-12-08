#!/bin/sh

echo "Enter the number of epochs: "
read epochs

echo "Enter the batch size: "
read batch

python3 src/train_model.py --cleardir ../dataset/RAIN_DATASET/ALIGNED_PAIRS/CLEAN --rainydir ../dataset/RAIN_DATASET/ALIGNED_PAIRS/REAL_DROPLETS --savedir ./models/ --nepochs $epochs --batchsize $batch
