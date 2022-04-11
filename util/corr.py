import numpy as np
from scipy import stats
import pandas as pd


def rdm_to_np(rdm: pd.DataFrame) -> np.ndarray:
    mat = rdm.to_numpy()
    # np.fill_diagonal(mat, np.NaN)
    vec = mat.reshape(-1)
    return vec[~np.isnan(vec)]


def rdm_corr(rdm1: pd.DataFrame, rdm2: pd.DataFrame):
    vec1 = rdm_to_np(rdm1)
    vec2 = rdm_to_np(rdm2)
    return stats.pearsonr(vec1, vec2), stats.kendalltau(vec1, vec2)