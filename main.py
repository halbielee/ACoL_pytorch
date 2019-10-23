import os
import random
import warnings
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import network as models
from utils.util_args import get_args
from utils.util_cam import load_bbox, get_cam, resize_cam, get_bboxes, blend_cam
from utils.util_loader import data_loader
from utils.util import \
    accuracy, adjust_learning_rate, \
    save_checkpoint, load_model, AverageMeter, IMAGE_MEAN_VALUE, IMAGE_STD_VALUE, calculate_IOU, draw_bbox, save_images

best_acc1 = 0
def main():
    args = get_args()

    args.log_folder = os.path.join('train_log', args.name)
    if not os.path.join(args.log_folder):
        os.makedirs(args.log_folder)

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
    global best_acc1
    args.gpu = gpu

    if args.gpu == 0 and not args.evaluate:
        writer = SummaryWriter(logdir=args.log_folder)

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

    if args.dataset == "CUB":
        num_classes = 200
    elif args.dataset == "ILSVRC":
        num_classes = 1000
    else:
        raise Exception("No dataset named {}".format(args.dataset))

    # Define Model
    model = models.__dict__[args.arch](pretrained=args.pretrained,
                                       num_classes=num_classes,
                                       drop_thr=args.erase_thr,
                                       turnoff=args.acol_cls)

    # Define loss function (criterion) and optimizer
    criterion = None
    param_features = []
    param_classifier = []

    # Give different learning rate
    for name, param in model.named_parameters():
        if 'features.' in name:
            param_features.append(param)
        else:
            param_classifier.append(param)

    optimizer = torch.optim.SGD([
        {'params': param_features, 'lr': args.lr},
        {'params': param_classifier, 'lr': args.lr*args.lr_ratio}],
        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nest)

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
        model, optimizer = load_model(model, optimizer, args)

    # Loading training/validation dataset
    cudnn.benchmark = True
    train_loader, val_loader, train_sampler = data_loader(args)

    if args.evaluate:
        evaluate_loc(val_loader, model, criterion, 0, args)
        return

    # Training Phase
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train_acc1, train_loss = \
            train(train_loader, model, criterion, optimizer, epoch, args)

        if not args.loc:
            val_acc1, val_loss = \
                validate(val_loader, model, criterion, epoch, args)

        else:
            val_acc1, val_acc5, top1_loc, top5_loc, gt_loc, val_loss = \
                evaluate_loc(val_loader, model, criterion, epoch, args)


        if args.gpu == 0 and not args.loc:
            writer.add_scalars(args.name, {'t_acc': train_acc1,
                                           't_loss': train_loss,
                                           'v_acc': val_acc1,
                                           'v_loss':val_loss,}, epoch)

        elif args.gpu == 0 and args.loc:
            writer.add_scalars(args.name, {'t_acc': train_acc1,
                                           't_loss': train_loss,
                                           'v_acc': val_acc1,
                                           'v_loss': val_loss,
                                           'top1-loc': top1_loc,
                                           'top5-loc': top5_loc,
                                           'gt-loc': gt_loc}, epoch)

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        if args.gpu == 0:
            print("Until %d epochs, Best Acc@1 %.3f" % (epoch+1, best_acc1))

        # remember best acc@1 and save checkpoint
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.log_folder)

    top1_cls, top5_cls, top1_loc, top5_loc, gt_loc, val_loss = \
        evaluate_loc(val_loader, model, criterion, 0, args)

    with open(os.path.join(args.log_folder, 'result.txt'), 'w') as f:
        f.write("Training Result"
                "Top1-cls: {0}\n"
                "Top5-cls: {1}\n"
                "Top1-loc: {2}\n"
                "Top5-loc: {3}\n"
                "  GT-loc: {4}\n".format(top1_cls, top5_cls, top1_loc, top5_loc, gt_loc))

def train(train_loader, model, criterion, optimizer, epoch, args):

    # AverageMeter for Performance
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # Switch to train mode
    model.train()

    train_t = tqdm(train_loader)
    for i, (images, target) in enumerate(train_t):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # Compute outputs
        output = model(images, target)
        if args.distributed:
            loss = model.module.get_loss(output, target)
        else:
            loss = model.get_loss(output, target)
        output = output[0]

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        description="[T:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ".\
            format(epoch, args.epochs, top1.avg, top5.avg, losses.avg)
        train_t.set_description(desc=description)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        val_t = tqdm(val_loader)
        for i, (images, target, image_ids) in enumerate(val_t):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.long().cuda(args.gpu, non_blocking=True)

            # Compute output
            output = model(images, target)
            if args.distributed:
                loss = model.module.get_loss(output, target)
            else:
                loss = model.get_loss(output, target)
            output = output[0]

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))


            description="[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ".\
                format(epoch, args.epochs, top1.avg, top5.avg, losses.avg)
            val_t.set_description(desc=description)

    return top1.avg, losses.avg


def evaluate_loc(val_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss')
    top1_cls = AverageMeter('Acc@1')
    top5_cls = AverageMeter('Acc@5')

    # image 개별 저장할 때 필요
    gt_bbox = load_bbox(args)

    cnt = 0
    cnt_false_top1 = 0
    cnt_false_top5 = 0
    hit_known = 0
    hit_top1 = 0
    hit_top5 = 0

    image_mean = torch.reshape(torch.tensor(IMAGE_MEAN_VALUE), (1, 3, 1, 1))
    image_std = torch.reshape(torch.tensor(IMAGE_STD_VALUE), (1, 3, 1, 1))

    model.eval()
    with torch.no_grad():
        val_t = tqdm(val_loader)
        for i, (images, target, image_ids) in enumerate(val_t):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.long().cuda(args.gpu, non_blocking=True)
            if args.dataset == 'CUB':
                image_ids = image_ids.data.cpu().numpy()

            # Compute output
            output = model(images, target)
            if args.distributed:
                loss = model.module.get_loss(output, target)
            else:
                loss = model.get_loss(output, target)
            output = output[0]

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1_cls.update(acc1[0], images.size(0))
            top5_cls.update(acc5[0], images.size(0))

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_1 = correct[:1].flatten(1).float().sum(dim=0)
            correct_5 = correct[:5].flatten(1).float().sum(dim=0)


            # Get cam
            cam = get_cam(model=model, target=target, args=args)
            cam = cam.cpu().numpy().transpose(0, 2, 3, 1)

            image_ = images.clone().detach().cpu() * image_mean + image_std
            blend_tensor = torch.empty_like(image_)
            image_ = image_.numpy().transpose(0, 2, 3, 1)
            image_ = image_[:, :, :, ::-1] * 255

            for j in range(images.size(0)):
                # Resize and Normalize CAM
                cam_ = resize_cam(cam[j], size=(224, 224))

                # Estimate BBOX
                estimated_bbox = get_bboxes(cam_, cam_thr=args.cam_thr)

                # Calculate IoU
                iou = calculate_IOU(gt_bbox[image_ids[j]][0], estimated_bbox)

                # Get blended image
                blend, heatmap = blend_cam(image_[j], cam_)

                if iou >= 0.5 and correct_1[j] > 0:
                    boxed_image = draw_bbox(blend, iou, gt_bbox[image_ids[j]][0], estimated_bbox, True)
                else:
                    boxed_image = draw_bbox(blend, iou, gt_bbox[image_ids[j]][0], estimated_bbox, False)

                # reverse the color representation(RGB -> BGR) and reshape
                boxed_image = boxed_image[:, :, ::-1] / 255.
                boxed_image = boxed_image.transpose(2, 0, 1)
                blend_tensor[j] = torch.tensor(boxed_image)

                cnt += 1
                if iou >= 0.5:
                    hit_known += 1
                    if correct_5[j] > 0:
                        hit_top5 += 1
                        if correct_1[j] > 0:
                            hit_top1 += 1
                        elif correct_1[j] == 0:
                            cnt_false_top1 += 1
                    elif correct_5[j] == 0:
                        cnt_false_top1 += 1
                        cnt_false_top5 += 1
                else:
                    if correct_5[j] > 0:
                        if correct_1[j] == 0:
                            cnt_false_top1 += 1
                    elif correct_5[j] == 0:
                        cnt_false_top1 += 1
                        cnt_false_top5 += 1

            # Save tensors
            if args.gpu == 0:
                save_images('results_best', 0, i, blend_tensor, args)

            description="[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ".\
                format(epoch, args.epochs, top1_cls.avg, top5_cls.avg, losses.avg)
            val_t.set_description(desc=description)

        loc_gt = hit_known / cnt * 100
        loc_top1 = hit_top1 / cnt * 100
        loc_top5 = hit_top5 / cnt * 100
        cls_top1 = (1 - cnt_false_top1 / cnt) * 100
        cls_top5 = (1 - cnt_false_top5 / cnt) * 100

        if args.gpu == 0:
            print("Evaluation Result:\n"
                  "LOC GT:{0:6.2f} Top1: {1:6.2f} Top5: {2:6.2f}\n"
                  "CLS TOP1: {3:6.3f} Top5: {4:6.3f}".
                  format(loc_gt, loc_top1, loc_top5, cls_top1, cls_top5))

    return cls_top1, cls_top5, loc_top1, loc_top5, loc_gt, losses.avg


if __name__ == '__main__':
    main()
