import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

import numpy as np

from network import vgg
from tensorboardX import SummaryWriter
from utils.util_args import get_args
from utils.util_loss import Loss
from utils.util_loader import data_loader
from utils.util_cam import *
from utils.util_acc import calculate_IOU
from utils.dataset_cub import get_image_name
from utils.util_bbox import load_bbox_size
import torch.nn.functional as F
best_acc1 = 0
best_loc1 = 0
loc1_at_best_acc1 = 0
acc1_at_best_loc1 = 0
gtknown_at_best_acc1 = 0
gtknown_at_best_loc1 = 0

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch != 0 and epoch % args.LR_decay == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print('LR is adjusted at {}/{}'.format(
            epoch, args.epochs
        ))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_loc1, \
        loc1_at_best_acc1, acc1_at_best_loc1, \
        gtknown_at_best_acc1, gtknown_at_best_loc1

    args.gpu = gpu

    if args.gpu == 0:
        writer = SummaryWriter(logdir=os.path.join('runs', args.name))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    models = None

    if args.arch.startswith('vgg'):
        models = getattr(vgg, args.arch)
    else:
        raise Exception("Fail to recognize the architecture")

    if args.pretrained:
        model = models(pretrained=True, writer=writer, args=args)
    else:
        model = models(pretrained=False, writer=writer, args=args)

    # define loss function (criterion) and optimizer
    criterion = Loss(args.gpu)
    param_features = []
    param_classifier = []

    for name, param in model.named_parameters():
        if 'features.' in name:
            param_features.append(param)
        else:
            param_classifier.append(param)

    optimizer = torch.optim.SGD([
        {'params': param_features, 'lr': args.lr},
        {'params': param_classifier, 'lr': args.lr*args.lr_ratio}],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nest)

    # Change the last fully connected layers.
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    train_loader, val_loader, train_sampler = data_loader(args, test_path=True)





    val_acc1, val_acc5, val_loss, \
    val_gtloc, val_loc = evaluate_loc(val_loader, model, criterion, 0, args, writer)


    if args.gpu == 0:
        print("Best Acc@1 %.3f\tloc@1 %.3f\tgt-known %.3f" %
              (val_acc1, val_loc, val_gtloc))




def evaluate_loc(val_loader, model, criterion, epoch, args, writer):
    batch_time = AverageMeter('Time')
    losses = AverageMeter('Loss')
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    GT_loc = AverageMeter('GT-Known')
    top1_loc = AverageMeter('LOC@1')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top1_loc, GT_loc],
        prefix='EVAL: ')

    # image 개별 저장할 때 필요
    image_names = get_image_name(args.img_dir, file='test.txt')
    gt_bbox = load_bbox_size(resize_size = args.resize_size,
                             crop_size = args.crop_size)

    cnt = 0
    cnt_false = 0
    hit_known = 0
    hit_top1 = 0

    means = [0.485, .456, .406]
    stds = [.229, .224, .225]
    means = torch.reshape(torch.tensor(means), (1, 3, 1, 1)).cuda(args.gpu)
    stds = torch.reshape(torch.tensor(stds), (1, 3, 1, 1)).cuda(args.gpu)

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, image_ids) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.long().cuda(args.gpu, non_blocking=True)
            image_ids = image_ids.data.cpu().numpy()
            logits, feature_maps = model(images, target)
            loss = criterion.get_loss(logits[0], logits[1], target)

            acc1, acc5 = accuracy(logits[0], target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))


            _, pred = logits[0].topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            wrongs = [c==0 for c in correct.cpu().numpy()][0]

            feature_map_A = get_attention_map(feature_maps[0], pred.squeeze())
            feature_map_B = get_attention_map(feature_maps[1], pred.squeeze())

            loc_map = torch.max(feature_map_A, feature_map_B)

            loc_map = F.interpolate(loc_map.unsqueeze(dim=1),
                                    (images.size(2), images.size(3)),
                                    mode='bilinear')

            image_ = images * stds + means
            image_ = image_.cpu().detach().numpy()
            loc_map = loc_map.cpu().detach().numpy()

            image_ = np.transpose(image_, (0, 2, 3, 1))
            loc_map = np.transpose(loc_map, (0, 2, 3, 1))

            saving_image = torch.zeros(images.shape)
            for j in range(images.size(0)):

                _, cammed = cammed_image(image_[j], loc_map[j])
                heatmap = intensity_to_rgb(np.squeeze(loc_map[j]), normalize=True).astype('uint8')
                img_bbox = image_[j].copy()

                gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)

                thr_val = args.cam_thr * np.max(gray_heatmap)

                _, th_gray_heatmap = cv2.threshold(gray_heatmap, int(thr_val), 255, cv2.THRESH_BINARY)

                try:
                    _, contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                bbox = gt_bbox[image_ids[j].item()]
                _img_bbox = (img_bbox.copy() * 255).astype('uint8')
                gxa = max(int(bbox[0]), 0)
                gya = max(int(bbox[1]), 0)
                gxb = min(int(bbox[2]), args.crop_size - 1)
                gyb = min(int(bbox[3]), args.crop_size - 1)
                cammed = cv2.rectangle(cammed, (gxa, gya), (gxb, gyb), (0, 255, 0), 2)

                adjusted_bbox = [gxa, gya, gxb, gyb]
                if len(contours) != 0:
                    c = max(contours, key=cv2.contourArea)

                    x, y, w, h = cv2.boundingRect(c)
                    estimated_box = [x, y, x+w, y+h]
                    IOU = calculate_IOU(adjusted_bbox, estimated_box)

                    cnt += 1
                    if IOU >= 0.5:
                        hit_known +=1
                        if not wrongs[j]:
                            hit_top1 += 1
                    if wrongs[j]:
                        cnt_false += 1

                    cammed = cv2.rectangle(cammed,
                                           (max(0, estimated_box[0]), max(0, estimated_box[1])),
                                           (min(args.crop_size, estimated_box[2]), min(args.crop_size, estimated_box[3])),
                                           (0,0,255),
                                           2)
                saving_image[j] = torch.tensor(cammed.transpose(2, 0, 1))


            if args.gpu == 0:
                saving_folder = os.path.join('image_path', args.name)
                if not os.path.isdir(saving_folder):
                    os.makedirs(saving_folder)
                file_name = 'HEATMAP_{}_{}.jpg'.format(epoch, i)
                saving_path = os.path.join(saving_folder, file_name)
                save_image(saving_image, saving_path, normalize=True)

            loc_gt = hit_known / cnt * 100
            loc_top1 = hit_top1 / cnt * 100

            GT_loc.update(loc_gt, images.size(0))
            top1_loc.update(loc_top1, images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if args.gpu == 0 and i % args.print_freq == 0:
                progress.display(i)
        if args.gpu == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} \
            Loc@1 {loc1.avg:.3f} gt-known {GT_loc.avg:.3f}'.
                  format(top1=top1, top5=top5, loc1=top1_loc, GT_loc=GT_loc))

    torch.cuda.empty_cache()

    return top1.avg, top5.avg, losses.avg, GT_loc.avg, top1_loc.avg



if __name__ == '__main__':
    main()