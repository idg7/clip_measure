import torch.utils.data
import glob
import os
from PIL import Image
import pandas as pd
from torchvision import datasets


class ImgTxtLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str, transforms: torch.nn.Module, labels_path: str):
        self.class_to_idx = datasets.ImageFolder(dir_path).class_to_idx
        self.dir_path = dir_path
        self.transforms = transforms
        self.classes = glob.glob(os.path.join(dir_path, '*'))
        self.labels_df = pd.read_csv(labels_path, index_col='class')
        self.labels = []
        self.images = []
        self.cls_to_imgs = {}
        self.cls_to_txt = {}
        self.cls_filter = None
        for cl_path in self.classes:
            cl_id = os.path.basename(cl_path)
            cl_label = self.labels_df.loc[cl_id, 'name']
            cl_images = glob.glob(os.path.join(cl_path, '*'))
            self.cls_to_imgs[self.class_to_idx[cl_id]] = cl_images
            self.cls_to_txt[self.class_to_idx[cl_id]] = cl_id
            self.labels = self.labels + [cl_label for _ in cl_images]
            self.images = self.images + cl_images

    def set_cls_filter(self, i: int):
        self.cls_filter = i

    def __getitem__(self, idx):
        if self.cls_filter is not None:
            txt = self.cls_to_txt[self.cls_filter]
            return self.transforms(Image.open(self.cls_to_imgs[self.cls_filter][idx])), txt, self.cls_filter
        txt = self.labels[idx]
        cls = self.class_to_idx[txt]
        return self.transforms(Image.open(self.images[idx])), txt, cls

    def __len__(self):
        if self.cls_filter is not None:
            return len(self.cls_to_imgs[self.cls_filter])
        return len(self.images)
