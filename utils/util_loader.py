from torchvision import transforms
from torch.utils.data import DataLoader
from .dataset_cub import dataset

import torchvision
import torch
import numpy as np

def data_loader(args, test_path=False):

    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.resize_size)
    crop_size = int(args.crop_size)

    tsfm_train = transforms.Compose([transforms.Resize(input_size),
                                     transforms.RandomCrop(crop_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = []
        # print input_size, crop_size
        if input_size == 0 or crop_size == 0:
            pass
        else:
            func_transforms.append(transforms.Resize(input_size))
            func_transforms.append(transforms.CenterCrop(crop_size))

        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    img_train = dataset(args.train_list, root_dir=args.img_dir,
                           transform=tsfm_train, with_path=True)

    img_test = dataset(args.test_list, root_dir=args.img_dir,
                          transform=tsfm_test, with_path=test_path)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(img_train)
    else:
        train_sampler = None

    train_loader = DataLoader(img_train,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=args.workers)

    val_loader = DataLoader(img_test,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers)

    return train_loader, val_loader, train_sampler
