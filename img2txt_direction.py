from typing import List

import clip
from datasets import ImgTxtLabelDataset
from torch.utils import data
from torch import nn, tensor
from util import cartesian_to_spherical, tensor_rdm
import torch
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm


def calc_im2txt_direction(model: nn.Module, dl: data.DataLoader, chosen_cls: int = None):
    with torch.set_grad_enabled(False):
        imgs = tensor([]).cuda(non_blocking=True)
        euclidian_diffs = tensor([]).cuda(non_blocking=True)
        angular_diffs = tensor([]).cuda(non_blocking=True)
        for img, txt, cls in dl:
            if (chosen_cls is None) or (cls == chosen_cls).all():
                img = img.cuda(non_blocking=True)
                txt = clip.tokenize(txt).cuda(non_blocking=True)
                txt_encoding = model.encode_text(txt)
                img_encoding = model.encode_image(img)
                if ANGULAR:
                    angle_txt_encoding = cartesian_to_spherical(txt_encoding)
                    angle_img_encoding = cartesian_to_spherical(img_encoding)
                    angular_diffs = torch.cat((angular_diffs, angle_txt_encoding - angle_img_encoding), dim=0)
                diff = txt_encoding - img_encoding
                euclidian_diffs = torch.cat((euclidian_diffs, diff), dim=0)
                imgs = torch.cat((imgs, img_encoding), dim=0)
        mean = torch.mean(euclidian_diffs, dim=0)
        std = torch.std(euclidian_diffs, dim=0)
        U, S, Vh = torch.linalg.svd(imgs, full_matrices=True)
        sort_order = torch.argsort(-S)
        S = S[sort_order][0].unsqueeze(0)
        U = Vh.T[sort_order, :][0, :].unsqueeze(0)
        return mean, std, S, U


def cl_imgs_embeddings(model: nn.Module, dl: data.DataLoader, chosen_cls: int = None):
    imgs = []
    classes = []
    name = None
    all_names = []
    with torch.set_grad_enabled(False):
        for img, txt, cls in dl:
            if (chosen_cls is None) or (cls == chosen_cls).all():
                txt = clip.tokenize(txt).cuda(non_blocking=True)
                txt_encoding = model.encode_text(txt)
                img = img.cuda(non_blocking=True)
                cls = cls.cuda(non_blocking=True)
                img_encoding = model.encode_image(img)
                imgs.append(img_encoding.float())
                all_names.append(txt_encoding.float())
                classes.append(cls)
                name = txt_encoding.float()[0].unsqueeze(0)
        imgs = torch.cat(imgs, dim=0)
        classes = torch.cat(classes, dim=0)
        return imgs, classes, name#, (torch.cat(all_names, dim=0) - imgs)

def names2imgs_histogram(model: nn.Module, dl: data.DataLoader):
    names = [None] * len(dl.dataset.class_to_idx)
    imgs = []
    cls = []
    corr_dists = [None] * len(dl.dataset.class_to_idx)
    non_corr_dists = [None] * len(dl.dataset.class_to_idx)
    for i in tqdm(range(len(dl.dataset.class_to_idx))):
        dl.dataset.set_cls_filter(i)
        curr_imgs, curr_cls, curr_names = cl_imgs_embeddings(model, dl, i)
        imgs.append(curr_imgs)
        cls.append(curr_cls)
        names[i] = curr_names
        dists = torch.cdist(curr_names, curr_imgs, p=2)
        corr_dists[i] = torch.flatten(dists)
    imgs = torch.cat(imgs, dim=0)
    cls = torch.cat(cls, dim=0)
    names = torch.cat(names, dim=0).unsqueeze(1)

    same_name_deltas = [None] * len(dl.dataset.class_to_idx)
    for i in tqdm(range(len(dl.dataset.class_to_idx))):
        non_corr_dists[i] = torch.flatten(torch.cdist(names[i], imgs[cls != i], p=2))
        # same_name_deltas[i] = torch.flatten(torch.tile(non_corr_dists[i], (1, corr_dists[i].shape[0])) - corr_dists[i][:, None])

    # corr_dists = torch.cat(corr_dists).flatten().detach().cpu().numpy()

    # non_corr_dists = torch.cat(non_corr_dists).flatten().detach().cpu().numpy()

    # same_name_deltas = torch.cat(same_name_deltas).flatten().detach().cpu().numpy()

    fig_cos = go.Figure()
    for i in tqdm(range(len(dl.dataset.class_to_idx))):
        # fig_cos.add_trace(go.Histogram(x=same_name_deltas, name=f'non-corr - corr'))
        fig_cos.add_trace(go.Histogram(x=corr_dists[i].detach().cpu().numpy(), name=f'{i} corresponding'))
        fig_cos.add_trace(go.Histogram(x=non_corr_dists[i].detach().cpu().numpy(), name=f'{i} non corresponding'))
    # Overlay both histograms
    fig_cos.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig_cos.update_traces(opacity=0.75)

    fig_cos.show()



def svd_direction(Us: List[torch.Tensor], Ss: List[torch.Tensor], i: int):
    Us = torch.cat(Us, dim=0)
    print(Us.shape)
    # Us = Us[:, i, :]
    dists = 1 - torch.abs(tensor_rdm(Us))
    dists = torch.flatten(dists).cpu().detach().numpy()

    fig_cos = go.Figure()
    fig_cos.add_trace(go.Histogram(x=dists, name=f'{i} Principal Component Cosine Distance'))
    fig_cos.show()


if __name__ == '__main__':
    ANGULAR = False
    model, preprocess = clip.load('ViT-B/32')
    model = model.cuda()
    model.eval()
    dl = data.DataLoader(
        ImgTxtLabelDataset('/home/ssd_storage/datasets/celebA_crops', preprocess, '/home/ssd_storage/experiments/clip_decoder/celebA_cls_names.csv'),
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        shuffle=False)

    mean_direction = []
    var_direction = []
    Us = []
    Ss = []
    # euc_mean, euc_var, S, U = calc_im2txt_direction(model, dl, None)
    # mean_direction.append(euc_mean.unsqueeze(0))
    # var_direction.append(euc_var.unsqueeze(0))
    # print('Euclidean')
    # print(euc_mean.norm(p=2), euc_var.norm(p=2))

    names2imgs_histogram(model,dl)
    #
    # for i in tqdm(range(len(dl.dataset.class_to_idx))):
    #     dl.dataset.set_cls_filter(i)
    #     # print(i)
    #     euc_mean, euc_var, S, U = calc_im2txt_direction(model, dl, i)
    #     mean_direction.append(euc_mean.unsqueeze(0))
    #     var_direction.append(euc_var.unsqueeze(0))
    #     Ss.append(S)
    #     Us.append(U)
    # #
    # svd_direction(Us, Ss, 0)
    #
    # mean_direction = torch.cat(mean_direction)
    # var_direction = torch.cat(var_direction)
    #
    # mean_step_size = mean_direction.norm(dim=1, p=2).cpu().detach().numpy()
    # step_size_var = var_direction.norm(dim=1, p=2).cpu().detach().numpy()
    #
    # fig_l2 = go.Figure()
    # fig_l2.add_trace(go.Histogram(x=mean_step_size, name='Mean txt2img distance'))
    # fig_l2.add_trace(go.Histogram(x=step_size_var, name='Txt2img distance SD'))
    #
    # # Overlay both histograms
    # fig_l2.update_layout(barmode='overlay')
    # # Reduce opacity to see both histograms
    # fig_l2.update_traces(opacity=0.75)
    # fig_l2.show()
    #
    # mean_direction = mean_direction / mean_direction.norm(dim=1, p=2)[:, None]
    # mean_direction = 1 - torch.mm(mean_direction, mean_direction.transpose(0, 1))
    # mean_direction = torch.flatten(mean_direction).cpu().detach().numpy()
    #
    # var_direction = var_direction / var_direction.norm(dim=1, p=2)[:, None]
    # var_direction = 1 - torch.mm(var_direction, var_direction.transpose(0, 1))
    # var_direction = torch.flatten(var_direction).cpu().detach().numpy()
    #
    # fig_cos = go.Figure()
    # fig_cos.add_trace(go.Histogram(x=mean_direction, name='Mean txt2img direction corr'))
    # fig_cos.add_trace(go.Histogram(x=var_direction, name='Txt2img direction corr SD'))
    #
    # # Overlay both histograms
    # fig_cos.update_layout(barmode='overlay')
    # # Reduce opacity to see both histograms
    # fig_cos.update_traces(opacity=0.75)
    # fig_cos.show()

