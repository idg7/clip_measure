import clip
import torch


class ViTLayersDict(object):
    def __call__(self, vit: clip.model.VisionTransformer):
        layers = {vit: f'ViT output'}
        index = 1
        for mod in vit.modules():
            if type(mod) == clip.model.ResidualAttentionBlock:
                layers[mod] = f'Residual attention block {index}'
                index += 1
        return layers
