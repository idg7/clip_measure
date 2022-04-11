from typing import List

import pandas as pd
import torch


def l2_rdm(features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
    return - torch.cdist(features1, features2, p=2)


def tensor_rdm(features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
    features1 = features1 / features1.norm(dim=1, p=2)[:, None]
    features2 = features2 / features2.norm(dim=1, p=2)[:, None]
    sim = torch.mm(features1, features2.transpose(0, 1))
    return sim


def rdm(features1: torch.Tensor, features2: torch.Tensor, labels: List[str], mode='cos') -> pd.DataFrame:
    if mode == 'cos':
        sim = tensor_rdm(features1, features2)
    elif mode == 'l2':
        sim = l2_rdm(features1, features2)

    rdm_data_frame = sim.cpu().detach().numpy()
    return pd.DataFrame(rdm_data_frame, columns=labels, index=labels)
