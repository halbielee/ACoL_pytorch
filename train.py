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
        model = models(pretrained=True)
    else:
        model = models(pretrained=False)

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
        {'params': param_classifier, 'lr': args.lr*10}],
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


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)


        # train for one epoch
        # train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        # val_acc1, val_loss = validate(val_loader, model, criterion, epoch, args)

        val_acc1, val_acc5, val_loss, \
        val_gtloc, val_loc = evaluate_loc(val_loader, model, criterion, epoch, args)

        if args.gpu == 0:
            writer.add_scalar(args.name + '/train_acc', train_acc, epoch)
            writer.add_scalar(args.name + '/train_loss', train_loss, epoch)
            writer.add_scalar(args.name + '/val_cls_acc', val_acc1, epoch)
            writer.add_scalar(args.name + '/val_loss', val_loss, epoch)

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if args.gpu == 0:
            print("Until %d epochs, Best Acc@1 %.3f" % (epoch+1, best_acc1))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            saving_dir = os.path.join(args.save_dir, args.name)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, saving_dir)



def train(train_loader, model, criterion, optimizer, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    length = len(train_loader)

    for i, (images, target, image_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.long().cuda(args.gpu, non_blocking=True)

        # compute output
        logits_A, logits_B = model(images, target)
        loss = criterion.get_loss(logits_A, logits_B, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits_A, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target_in, image_ids) in enumerate(val_loader):
            if args.tencrop:
                b, n_crop, c, h, w = images.size()
                images = images.view(-1, c, h, w)
                target_input = target_in.repeat(10, 1)
                target = target_input.view(-1)
            else:
                target = target_in
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.long().cuda(args.gpu, non_blocking=True)

            # compute output
            logits_A, logits_B = model(images, target)
            loss = criterion.get_loss(logits_A, logits_B, target)

            if args.tencrop:
                logits_A = logits_A.view(b, n_crop, -1).mean(1)

            if args.gpu == 0 and i < 3:
                loc_map = model.module.generate_localization_map(images, args.erase_thr)
                denormed_image = get_denorm_tensor(images)
                heatmaps = get_heatmap_tensor(denormed_image, loc_map)
                file_name = time.strftime('%c', time.localtime(time.time())) + '.jpg'
                folder_path = os.path.join('save_image', args.name)
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                saving_path = os.path.join(folder_path, str(epoch)+'_'+file_name)
                save_image(heatmaps, saving_path, normalize=True)


            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits_A, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.gpu == 0 and i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if args.gpu == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

def evaluate_loc(val_loader, model, criterion, epoch, args):
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

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, image_ids) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.long().cuda(args.gpu, non_blocking=True)
            image_ids = image_ids.data.cpu().numpy()
            logits_A, logits_B = model(images, target)
            loss = criterion.get_loss(logits_A, target)

            acc1, acc5 = accuracy(logits_A, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))


            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            wrongs = [c==0 for c in correct.cpu().numpy()][0]

            loc_map = model.module.generate_localization_map(images, args.erase_thr)
            loc_map = loc_map.cpu().numpy()
            denorm_image = ((images * 0.22 + 0.45) * 255.0).cpu().\
                               detach().numpy().transpose([0, 2, 3, 1])[..., ::-1]
            denorm_image = denorm_image - np.min(denorm_image)
            denorm_image = denorm_image / np.max(denorm_image) * 255.0

            saving_image = torch.zeros_like(images)
            for j in range(images.size(0)):

                estimated_bbox, adjusted_gt_bbox, blend_box = \
                    get_bbox(denorm_image[j],
                             loc_map[j],
                             args.cam_thr,
                             gt_bbox[image_ids[j]],
                             image_names[image_ids[j]],
                             args.name,
                             args.image_save)
                if i == 0:

                    saving_image[j] = torch.tensor(blend_box.transpose(2,0,1))

                # print(estimated_bbox)
                IOU_ = calculate_IOU(estimated_bbox, adjusted_gt_bbox)
                if IOU_ > 0.5 or IOU_ == 0.5:
                    hit_known = hit_known + 1
                if (IOU_ > 0.5 or IOU_ == 0.5) and not wrongs[j]:
                    hit_top1 = hit_top1 + 1
                if wrongs[j]:
                    cnt_false += 1
                cnt += 1

            if i == 0 and args.gpu == 0:
                saving_folder = os.path.join('image_path', args.name)
                if not os.path.isdir(saving_folder):
                    os.makedirs(saving_folder)
                file_name = time.strftime('%c', time.localtime(time.time())) + '.jpg'
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
