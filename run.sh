gpu=0
name=train_6
epoch=250
decay=70
model=vgg16
server=tcp://127.0.0.1:06
batch=32
wd=1e-4
lr=0.001



CUDA_VISIBLE_DEVICES=${gpu} python train.py -a ${model} --dist-url ${server} \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --pretrained \
--workers=6 \
--erase-thr 0.7 \
--validation eval \
--num-classes=200 \
--img_dir=../CUB_200_2011/ \
--batch-size ${batch} --epochs ${epoch} --LR-decay ${decay} --wd ${wd} --lr ${lr} --nest --name ${name}
