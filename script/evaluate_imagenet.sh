#!/bin/bash

# ACoL - VGG16 script

gpu=0
arch=vgg16_acol
name=YOUT_TRAIN_NAME
dataset=ILSVRC
data_root="/srv/ILSVRC2012/"
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
--VAL-CROP True \
--evaluate True \
--cam-thr 0.15 \
--resume train_log/acol_train1/acol_train1/model_best.pth.tar
