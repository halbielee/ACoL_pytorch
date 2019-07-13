from torch.utils.data import Dataset
import numpy as np
import os
import torch
from PIL import Image
import random
# from .transforms import functional
# random.seed(1234)
# from .transforms import functional
import cv2
import math

class dataset(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir, self.image_list[idx])
        img_name =  self.image_list[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.with_path:
            return img_name, image, self.label_list[idx]
        else:
            return image, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)

def get_image_name(dataset_path, file):
    '''
    return dict{image_id: image name}
    '''
    image_names = {}
    with open(os.path.join(dataset_path, file)) as f:
        for line in f:
            image_id, image_name, image_label = line.strip().split()
            image_names[int(image_id)] = image_name
    return image_names


class dataset_with_mask(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, datalist_file, root_dir, mask_dir, transform=None, with_path=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.with_path = with_path
        self.datalist_file =  datalist_file
        self.image_list, self.label_list = \
            self.read_labeled_image_list(self.root_dir, self.datalist_file)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')

        mask_name = os.path.join(self.mask_dir, get_name_id(self.image_list[idx])+'.png')
        mask = cv2.imread(mask_name)
        mask[mask==0] = 255
        mask = mask - 1
        mask[mask==254] = 255

        if self.transform is not None:
            image = self.transform(image)

        if self.with_path:
            return img_name, image, mask, self.label_list[idx]
        else:
            return image, mask, self.label_list[idx]

    def read_labeled_image_list(self, data_dir, data_list):
        """
        Reads txt file containing paths to images and ground truth masks.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

        Returns:
          Two lists with all file names for images and masks, respectively.
        """
        f = open(data_list, 'r')
        img_name_list = []
        img_labels = []
        for line in f:
            if ';' in line:
                image, labels = line.strip("\n").split(';')
            else:
                if len(line.strip().split()) == 2:
                    image, labels = line.strip().split()
                    if '.' not in image:
                        image += '.jpg'
                    labels = int(labels)
                else:
                    line = line.strip().split()
                    image = line[0]
                    labels = map(int, line[1:])
            img_name_list.append(os.path.join(data_dir, image))
            img_labels.append(labels)
        return img_name_list, np.array(img_labels, dtype=np.float32)
