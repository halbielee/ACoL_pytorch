#!/bin/bash

# ACoL - VGG16 script

gpu=0
arch=vgg16_acol
name=acol_train1
dataset=CUB
data_root="../CUB_200_2011/images"
epoch=150
decay=40
batch=32
wd=1e-4
lr=0.001

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
--multiprocessing-distributed \
--world-size 1 \
--workers 4 \
--arch ${arch} \
--name ${name} \
--dataset ${dataset} \
--data-root ${data_root} \
--pretrained True \
--batch-size ${batch} \
--epochs ${epoch} \
--lr ${lr} \
--LR-decay ${decay} \
--wd ${wd} \
--nest True \
--erase-thr 0.7 \
--acol-cls False \
--VAL-CROP False \
--loc True
