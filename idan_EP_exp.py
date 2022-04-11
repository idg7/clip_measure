from typing import Tuple

import clip
import torch.utils.data
import glob
import os
import torch

# import plotly.express as px
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tqdm import tqdm
from numpy import random
from torch.utils.data import DataLoader
from PIL import Image
from util import rdm, rdm_corr
from argparse import ArgumentParser


class ComparisonDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, image_transforms, class_naming: pd.DataFrame):
        self.dir_path = dir_path
        self.image_transforms = image_transforms
        self.classes = glob.glob(os.path.join(dir_path, '*'))
        self.class_naming = class_naming
        self.images = []
        # print(self.classes)
        for cl in self.classes:
            class_img = random.choice(glob.glob(os.path.join(cl, '*')), replace=False, size=1)
            self.images.append(class_img[0])

    def __getitem__(self, idx):
        class_id = os.path.dirname(os.path.relpath(self.images[idx], self.dir_path))
        return self.image_transforms(Image.open(self.images[idx])), self.class_naming.loc[class_id, 'Name'], class_id

    def __len__(self):
        return len(self.images)


def create_RDMs(model: torch.nn.Module, dataset: DataLoader, mode: str='cos') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.requires_grad_(False)

    with torch.set_grad_enabled(False):
        for imgs, names, class_id in tqdm(dataset):
            imgs = imgs.cuda(non_blocking=True)
            text = clip.tokenize(names).cuda(non_blocking=True)

            image_features = model.encode_image(imgs).float()
            text_features = model.encode_text(text).float()

            img2img = rdm(image_features, image_features, names, mode)
            img2txt = rdm(image_features, text_features, names, mode)
            txt2txt = rdm(text_features, text_features, names, mode)

        return img2img, img2txt, txt2txt


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--image2txt_map',
                        default='/home/ssd_storage/datasets/Cognitive_exp/adva_images_and_more/adva_second_images/map.csv', type=str,
                        help='mapping between image cls dir to the used name')
    parser.add_argument('--arch', default='RN50x16', type=str, help='CLIP architecture to use')
    parser.add_argument('--dataset_path', default='/home/ssd_storage/datasets/Cognitive_exp/adva_images_and_more/adva_second_images/images_used', type=str, help='Where to find the classes dir')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model, preprocess = clip.load(args.arch)

    naming = pd.read_csv(args.image2txt_map, index_col='Class_Id', sep=',')

    dataset = DataLoader(ComparisonDataset(args.dataset_path, preprocess, naming),
                         batch_size=26, num_workers=4, pin_memory=True, shuffle=True)

    cos_img2img, cos_img2txt, cos_txt2txt = create_RDMs(model, dataset, mode='cos')
    img2img, img2txt, txt2txt  = create_RDMs(model, dataset, mode='l2')
    actors = [
        'angelina_jolie',
        'jennifer_aniston',
        'Judi_Dench',
         'Kate-Winslet',
        'keira_knightley',
        'sandra_bullock',
        'anthony_hopkins',
        'Hugh_Grant',
        'liam_neeson',
        'Martin_Freeman',
        'michael_caine',
        'nicolas_cage',
        'robert_de_niro',
        'tom_hanks',
    ]
    politicians = [
        'angela_rayner',
        'esther_mcvey',
        'hillary_clinton',
        'nicola_sturgeon',
        'PRITI_PATEL',
        'theresa may',
        'bill_clinton',
        'boris johnson',
        'david_cameron',
        'donald_trump',
        'george_Wbush',
        'tony_blair',
    ]

    order = actors + politicians
    img2img = img2img.loc[order, order]
    img2txt = img2txt.loc[order, order]
    txt2txt = txt2txt.loc[order, order]

    cos_img2img = cos_img2img.loc[order, order]
    cos_img2txt = cos_img2txt.loc[order, order]
    cos_txt2txt = cos_txt2txt.loc[order, order]

    fig = make_subplots(2,3, column_titles = ['img2img', 'img2txt', 'txt2txt'], row_titles=['cos', 'l2'])

    fig.add_trace(go.Heatmap(z=img2img, x=img2img.columns, y=img2img.index), row=2, col=1)
    fig.add_trace(go.Heatmap(z=img2txt, x=img2txt.columns, y=img2txt.index), row=2, col=2)
    fig.add_trace(go.Heatmap(z=txt2txt, x=txt2txt.columns, y=txt2txt.index), row=2, col=3)

    fig.add_trace(go.Heatmap(z=cos_img2img, x=cos_img2img.columns, y=cos_img2img.index), row=1, col=1)
    fig.add_trace(go.Heatmap(z=cos_img2txt, x=cos_img2txt.columns, y=cos_img2txt.index), row=1, col=2)
    fig.add_trace(go.Heatmap(z=cos_txt2txt, x=cos_txt2txt.columns, y=cos_txt2txt.index), row=1, col=3)
    fig.update_layout(height=2 * 600, width= 3*600)
    fig.show()

    print(rdm_corr(img2img, cos_img2img))
    print(rdm_corr(txt2txt, cos_txt2txt))