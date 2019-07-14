gpu=0
name1=vgg_13
epoch=90
decay=30
model=vgg16
server=tcp://127.0.0.1:11
batch=64
wd=1e-4
lr=0.001



CUDA_VISIBLE_DEVICES=${gpu} python evaluate.py -a ${model} --dist-url ${server} \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --pretrained \
--workers=6 \
--erase-thr 0.7 \
--cam-thr 0.15 \
--num-classes=200 \
--img_dir=../CUB_200_2011/CUB_200_2011/ \
--resume checkpoints/train_11/model_best.pth.tar \
--batch-size ${batch} --epochs ${epoch} --LR-decay ${decay} --wd ${wd} --lr ${lr} --nest
