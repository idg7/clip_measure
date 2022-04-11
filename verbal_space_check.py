from typing import List, Dict

import clip
import torch.utils.data
from glob import glob
import os
import torch

import plotly.express as px
import pandas as pd
from util import acc_at_1, rdm
from tqdm import tqdm
from numpy import random
from torch.utils.data import DataLoader


def create_RDMs(model: torch.nn.Module, text: List[str]) -> pd.DataFrame:
    model.requires_grad_(False)

    with torch.set_grad_enabled(False):
        tokenized = clip.tokenize(text).cuda(non_blocking=True)
        text_features = model.encode_text(tokenized)
        return rdm(text_features, text, mode='cos')


def name2text_classification(model: torch.nn.Module, names: List[str], cls: List[str]) -> pd.DataFrame:
    model.requires_grad_(False)

    with torch.set_grad_enabled(False):
        names_tokens = clip.tokenize(names).cuda(non_blocking=True)
        cls_tokens = clip.tokenize(cls).cuda(non_blocking=True)
        names_features = model.encode_text(names_tokens)
        cls_features = model.encode_text(cls_tokens)
        return acc_at_1(names_features, cls_features, cls)


def multi_class_name2text(model: torch.nn.Module, names: List[str], multi_cls: Dict[str, List[str]]) -> pd.DataFrame:
    classification = {}
    for category in tqdm(multi_cls):
        top_cls, top_score = name2text_classification(model, names, multi_cls[category])
        classification[category] = top_cls
        classification[f'{category} score'] = top_score
    return pd.DataFrame(classification, index=names)


if __name__ == '__main__':
    dir = '/home/ssd_storage/datasets/celebA_crops'
    names = glob(os.path.join(dir, '*'))
    names = [os.path.basename(name) for name in names]
    categories = {
        'ethnicity': ['caucasian', 'american', 'black', 'white', 'african american', 'asian', 'european', 'african', 'indian', 'aborigini', 'jewish'],
        'hair': ['bald', 'balding', 'receding hairline', 'buzz cut', 'long hair', 'short hair'],
        'facial_hair': ['beard', 'side brows', 'moustache', 'shave'],
        'hair_color': ['redhead', 'ginger', 'blonde', 'brunette', 'brown hair', 'red hair', 'blonde hair', 'brown hair', 'black hair', 'jet black hair', 'bald'],
        'facial_features': ['wide', 'long', 'round', 'symmetrical', 'smooth', 'nose', 'long nose', 'thick eyebrows', 'thin eyebrows', 'ears', 'big ears', 'small ears', 'chin', 'big chin', 'small chin', 'hairline'],
        'eye_color': ['blue eyes', 'green eyes', 'brown eyes'],
        'subjective': ['pretty', 'ugly', 'beautiful', 'hideous'],
        'personality': ['friendly', 'reliable', 'dominant', 'aggressive', 'mean', 'evil', 'good', 'intelligent', 'kind', 'dumb', 'stupid']
    }

    feats = names
    for category in categories:
        feats = feats + categories[category]
    architecture = 'ViT-B/32'
    model, _ = clip.load(architecture, device='cuda')
    curr_rdm = create_RDMs(model, feats)
    px.imshow(curr_rdm).show()
    # curr_rdm.to_csv('/home/ssd_storage/experiments/clip_decoder/verbal_features_rdm.csv')
    # multi_class_name2text(model, names, categories).to_csv('/home/ssd_storage/experiments/clip_decoder/verbal_features_classification.csv')
