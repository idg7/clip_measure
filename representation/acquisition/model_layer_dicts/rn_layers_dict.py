import clip
import torch


class ResNetLayersDict(object):
    def __call__(self, rn: clip.model.ModifiedResNet):
        layers = {rn: f'ResNet output',
                  rn.layer1: f'ResNet layer 1',
                  rn.layer2: f'ResNet layer 2',
                  rn.layer3: f'ResNet layer 3',
                  rn.layer4: f'ResNet layer 4'}

        return layers
