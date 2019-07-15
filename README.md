# [Adversarial Complementary Learning for Weakly Supervised Object Localization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.pdf)

This is the reproducing version of [Adversarial Complementary Learning for Weakly Supervised Object Localization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Adversarial_Complementary_Learning_CVPR_2018_paper.pdf)



## Prerequisites

- Python 3.6+
- Pytorch ( >= 1.1)
- Python bindings for OpenCV
- tensorboardX



## Data Preparation

### 	CUB-200-2011

- Download the dataset from [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)




#### Train

```
git clone https://github.com/halbielee/ACoL_reproducing.git
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz
cd ACoL_reproducing
bash run.sh
```



#### Evaluate

```
bash evaluate.sh
```



#### Performance

| Name     | Acc1 | Acc5 | Top1_LOC | GT-known | condition                                    |
| -------- | ---- | ---- | -------- | ------- | -------------------------------------------- |
| train_1  | 75.337 | 92.492 | 48.155   | 63.616   | batch 32, lr 0.001, wd 1e-4, 40/150, thr 0.7 |
| train_2  | 75.768 | 92.717 | 48.887   | 64.617   | batch 32, lr 0.001, wd 1e-4, 40/150, thr 0.7 |
| train_3  | 75.923 | 92.544 | 49.212 | 64.076 | batch 32, lr 0.001, wd 1e-4, 60/200, thr 0.7 |
| train_4  | 77.097 | 92.475 | 50.935 | 64.863 | batch 32, lr 0.001, wd 1e-4, 60/200, thr 0.7 |
| train_5  | 75.233 | 91.905 | 46.284 | 61.794 | batch 32, lr 0.001, wd 1e-4, 70/250, thr 0.7 |
| train_6  | 75.820 | 91.992 | 49.236 | 64.708 | batch 32, lr 0.001, wd 1e-4, 70/250, thr 0.7 |
| train_7  | 73.990 | 91.163 | 42.404 | 56.433 | batch 32, lr 0.001, wd 1e-4, 70/250, thr 0.7 / (crop_size 224, resize_size 224) |
| train_8  | 74.336 | 91.405 | 41.585 | 54.971 | batch 32, lr 0.001, wd 1e-4, 70/250, thr 0.7 / (crop_size 224, resize_size 224) |
