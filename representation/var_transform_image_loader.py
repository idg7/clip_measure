import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from data_prep.Datasets.activations_dataset import ActivationsDatasets
from PIL import Image
import torch


class ImageLoader(object):
    def __init__(self, source_transforms, target_transform=None):
        self.target_transform = target_transform
        self.test_transforms = source_transforms
        self.train_transforms = source_transforms

    def load_dataset(self, dir_path, test=True, with_path=False):
        """Loads a dataset based on a specific structure (dataset/classes/images)"""
        tt = self.train_transforms
        if test:
            tt = self.test_transforms
        # if with_path:
        #     return ActivationsDatasets(dir_path, tt, self.target_transform)
        return datasets.ImageFolder(dir_path, tt, target_transform=self.target_transform)

    def load_image(self, image_path, test=True):
        if test:
            tt = self.test_transforms
        else:
            tt = self.train_transforms

        im1 = Image.open(image_path)
        # fix bug from grayscale images
        # duplicate to make 3 channels
        if im1.mode != 'RGB':
            im1 = im1.convert('RGB')

        im1t = tt(im1)
        im1t = im1t.unsqueeze(0)

        if torch.cuda.is_available():
            im1t = im1t.cuda()
        return im1t.half()
