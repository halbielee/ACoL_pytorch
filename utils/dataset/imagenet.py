import os
from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, root=None, datalist=None, transform=None, is_train=True):

        self.root = root
        self.datalist = datalist
        self.transform = transform
        self.is_train = is_train
        image_ids = []
        image_labels = []
        with open(self.datalist) as f:
            for line in f:
                info = line.strip().split()
                image_ids.append(info[0][:-5])
                image_labels.append(int(info[1]))
        self.image_ids = image_ids
        self.image_labels = image_labels

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[idx]
        image = Image.open(os.path.join(self.root, image_id+'.JPEG')).convert('RGB')
        if '/' in image_id:
            image_id = image_id.split('/')[1]
        if self.transform is not None:
            image = self.transform(image)
        if self.is_train:
            return image, image_label
        else:
            return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)
