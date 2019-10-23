import os
import cv2
import torch
import shutil
import torchvision.utils as vutils

IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
IMAGE_STD_VALUE = [0.229, 0.224, 0.225]


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


def calculate_IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch != 0 and epoch % args.LR_decay == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print('LR is adjusted at {}/{}'.format(
            epoch, args.epochs
        ))


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))


def load_model(model, optimizer, args):
    """ Loading pretrained / trained model. """
    if os.path.isfile(args.resume):
        if args.gpu == 0:
            print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        try:
            args.start_epoch = checkpoint['epoch']
        except (TypeError, KeyError) as e:
            print("=> No 'epoch' keyword in checkpoint.")
        try:
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
                pass
        except (TypeError, KeyError) as e:
            print("=> No 'best_acc1' keyword in checkpoint.")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except (TypeError, KeyError, ValueError) as e:
            print("=> Fail to load 'optimizer' in checkpoint.")
        try:
            if args.gpu == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        except (TypeError, KeyError) as e:
            if args.gpu == 0:
                print("=> No 'epoch' in checkpoint.")

        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        if args.gpu == 0:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model, optimizer


def draw_bbox(image, iou, gt_box, pred_box, is_top1=False):

    def draw_bbox(img, box1, box2, color1=(0, 0, 255), color2=(0, 255, 0)):
        cv2.rectangle(img, (box1[0], box1[1]), (box1[2], box1[3]), color1, 2)
        cv2.rectangle(img, (box2[0], box2[1]), (box2[2], box2[3]), color2, 2)
        return img
    def mark_target(img, text='target', pos=(25, 25), size=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), size)
        return img

    boxed_image = image.copy()

    # draw bbox on image
    boxed_image = draw_bbox(boxed_image, gt_box, pred_box)

    # mark the iou
    mark_target(boxed_image, '%.1f' % (iou * 100), (150, 30), 2)
    if is_top1:
        mark_target(boxed_image, 'TOP1', pos=(15, 30))

    return boxed_image


def save_images(folder_name, epoch, i, blend_tensor, args):
    """ Save Tensor image in the folder. """
    saving_folder = os.path.join(args.log_folder, folder_name)
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    file_name = 'HEAT_TEST_{}_{}.jpg'.format(epoch+1, i)
    saving_path = os.path.join(saving_folder, file_name)
    if args.gpu == 0:
        vutils.save_image(blend_tensor, saving_path)