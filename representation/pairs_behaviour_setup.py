from typing import Dict

import clip
import json
import os
import torch.utils.data as data
import torch

from representation.analysis.MultiDatasetCompare import MultiDatasetComparer
from representation.analysis.metrics.euclidian_distance_compare import EuclidianDistanceCompare
from representation.analysis.metrics.cosine_distance_compare import CosineDistanceCompare
from representation.analysis.pairs_list_compare import PairsListComparer
from representation.analysis.multi_list_comparer import MultiListComparer
from representation.analysis.rep_dist_mat import DistMatrixComparer
from representation.analysis.efficient_rdm import EfficientRDM
from representation.analysis.metrics.correlated_firing import CountCorrelatedFiring
from representation.analysis.metrics.normalized_correlated_firing import NormalizedCountCorrelatedFiring
from representation.analysis.metrics.count_firing import CountFiring
from representation.activations.activation_acquisition import ActivationAcquisition
from representation.activations.deep_layers_activations import DeepLayersActivations
from representation.activations.multi_list_activations_acquisition import MultiListAcquisition
from representation.acquisition.model_layer_dicts import *
from representation.var_transform_image_loader import ImageLoader


def get_pairs_comparison(
        pairs_image_dirs: Dict[str, str],
        pairs_paths: Dict[str, str],
        reps_cache_path: str,
        metric: str,
        model: clip.model.CLIP,
        preprocess: torch.nn.Module):
    image_loader = ImageLoader(preprocess)

    if type(model.visual) == clip.model.ModifiedResNet:
        model_layers_dict = ResNetBottleNeckLayersDict()
    else:
        model_layers_dict = ViTLayersDict()

    if metric.lower() == 'l2':
        comparison_calc = EuclidianDistanceCompare()
    if metric.lower() == 'cos':
        comparison_calc = CosineDistanceCompare()

    pairs_types_to_lists = {}
    for pairs_type in pairs_paths:
        print(pairs_type)
        pairs_types_to_lists[pairs_type] = []
        with open(pairs_paths[pairs_type], 'r') as f:
            for line in f:
                labeled_pair = line.split(' ')
                labeled_pair[1] = labeled_pair[1].replace(os.linesep, '')
                pairs_types_to_lists[pairs_type].append(labeled_pair)
    pairs_list_comparison = PairsListComparer(reps_cache_path, image_loader, comparison_calc,
                                              model_layers_dict)
    return MultiListComparer(pairs_types_to_lists, pairs_image_dirs, pairs_list_comparison)



# def get_dist_mat_extractor(model: clip.model.CLIP):
#         datasets = config['REP_BEHAVIOUR']['datasets']
#
#         # return MultiDatasetComparer(json.loads(datasets),
#         #                             DistMatrixComparer(reps_cache_path, image_loader, comparison_calc, ReflectionFactory().get_dict_extractor(config['REP_BEHAVIOUR']['reps_layers'])),
#         #                             config['REP_BEHAVIOUR']['reps_results_path'])
#         return MultiDatasetComparer(json.loads(datasets),
#                                     EfficientRDM(reps_cache_path, image_loader,
#                                                        ReflectionFactory().get_dict_extractor(
#                                                            config['REP_BEHAVIOUR']['reps_layers'])),
#                                     config['REP_BEHAVIOUR']['reps_results_path'])



# def setup_pairs_reps_behaviour(model: clip.model.CLIP, preprocess, opts):
#
#     # if 'activations' in config['REP_BEHAVIOUR'] and config['REP_BEHAVIOUR']['activations'] == 'True':
#     #     ds_path = config['REP_BEHAVIOUR']['activations_dataset']
#     #     whitelist = json.loads(config['REP_BEHAVIOUR']['whitelist'])
#     #     activations_dataset = data.DataLoader(
#     #         image_loader.load_dataset(ds_path),
#     #         batch_size=int(config['REP_BEHAVIOUR']['batch_size']),
#     #         num_workers=4,
#     #         shuffle=False,
#     #         pin_memory=True,
#     #         drop_last=False)
#     #     return ActivationAcquisition(activations_dataset, whitelist, int(config['MODELLING']['num_classes']))
#
#     reps_cache_path = config['REP_BEHAVIOUR']['reps_cache_path']
#     rep_dict_factory = ReflectionFactory()
#     get_model_layers_dict = rep_dict_factory.get_dict_extractor(config['REP_BEHAVIOUR']['reps_layers'])
#
#     if config['REP_BEHAVIOUR']['comparison_metric'] == 'l2' or config['REP_BEHAVIOUR']['comparison_metric'] == 'euclidian':
#         comparison_calc = EuclidianDistanceCompare()
#     if config['REP_BEHAVIOUR']['comparison_metric'] == 'cos' or config['REP_BEHAVIOUR']['comparison_metric'] == 'CosineSimilarity':
#         comparison_calc = CosineDistanceCompare()
#     if config['REP_BEHAVIOUR']['comparison_metric'] == 'correlated_firing_count':
#         comparison_calc = CountCorrelatedFiring()
#     if config['REP_BEHAVIOUR']['comparison_metric'] == 'normalized_correlated_firing_count':
#         comparison_calc = NormalizedCountCorrelatedFiring()
#     if config['REP_BEHAVIOUR']['comparison_metric'] == 'firing_count':
#         comparison_calc = CountFiring()
#
#     else:
#         pairs_image_dirs = json.loads(config['REP_BEHAVIOUR']['pairs_image_dirs'])
#         pairs_paths = json.loads(config['REP_BEHAVIOUR']['pairs_paths'])
#         pairs_types_to_lists = {}
#         for pairs_type in pairs_paths:
#             print(pairs_type)
#             pairs_types_to_lists[pairs_type] = []
#             with open(pairs_paths[pairs_type], 'r') as f:
#                 for line in f:
#                     labeled_pair = line.split(' ')
#                     print(labeled_pair)
#                     labeled_pair[1] = labeled_pair[1].replace(os.linesep, '')
#                     pairs_types_to_lists[pairs_type].append(labeled_pair)
#         pairs_list_comparison = PairsListComparer(reps_cache_path, image_loader, comparison_calc,
#                                                   get_model_layers_dict)
#         return MultiListComparer(pairs_types_to_lists, pairs_image_dirs, pairs_list_comparison)
#
