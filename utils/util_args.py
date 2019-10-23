import socket
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dist_url_generator():
    address = '127.0.0.1'
    for port in range(50000, 50100):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        is_used = sock.connect_ex((address, port))
        if is_used != 0:
            sock.close()
            return 'tcp://' + address + ':' + str(port)
        sock.close()
    raise Exception("Cannot find available port.")


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                        help='model architecture: default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=30, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=str2bool,
                        nargs='?', const=True, default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=False, help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    # Training
    parser.add_argument('--name', type=str, default='test_case')
    parser.add_argument('--LR-decay', type=int, default=30)
    parser.add_argument('--lr-ratio', type=float, default=10)
    parser.add_argument('--nest', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--loc', type=str2bool, nargs='?', const=True, default=False)

    # Dataset
    parser.add_argument('--dataset', type=str, default='CUB')
    parser.add_argument("--data-list", type=str, help="data list path")
    parser.add_argument("--data-root", type=str, default='', help='Directory of training images')
    parser.add_argument("--label-folder", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--resize-size', type=int, default=256, help='validation resize size')
    parser.add_argument('--crop-size', type=int, default=224, help='validation crop size')
    parser.add_argument('--VAL-CROP', type=str2bool, nargs='?', const=True, default=False,
                        help='Evaluation method'
                             'If True, Evaluate on 256x256 resized and center cropped 224x224 map'
                             'If False, Evaluate on directly 224x224 resized map')

    # ACoL
    parser.add_argument('--erase-thr', type=float, default=0.7, help='ACoL erasing threshold')
    parser.add_argument('--acol-cls', type=str2bool, nargs='?', const=True, default=False,
                        help='For no erasing training in ACoL')

    # CAM
    parser.add_argument('--cam-thr', type=float, default=0.2, help='cam threshold value')

    args = parser.parse_args()
    args.dist_url = dist_url_generator()
    if args.dataset == 'CUB':
        args.data_list = 'datalist/CUB'
    elif args.dataset == 'ILSVRC':
        args.data_list = 'datalist/ILSVRC'

    return args
