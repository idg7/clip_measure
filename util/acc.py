from typing import List, Tuple
import torch


def acc_at_1(input_feats: torch.Tensor, class_feats: torch.Tensor, class_names: List[str]) -> Tuple[List[str], List[str]]:
    input_feats = input_feats / input_feats.norm(dim=1, p=2)[:, None]
    class_feats = class_feats / class_feats.norm(dim=1, p=2)[:, None]
    sim = torch.mm(input_feats, class_feats.transpose(0, 1))
    top_cls = torch.argmax(sim, dim=1).cpu().detach().numpy()
    top_score = torch.max(sim, dim=1).values.cpu().detach().numpy()
    classifications = [class_names[int(cls)] for cls in top_cls]
    scores = [score for score in top_score]
    return classifications, scores
