from typing import Dict

import clip
import torch


class ResNetBottleNeckLayersDict(object):
    def __layer_bottlenecks(self, layers: Dict[torch.nn.Module, str], layer: torch.nn.Sequential, layer_idx: int) -> Dict[torch.nn.Module, str]:
        i = 1
        for mod in layer.modules():
            if type(mod) == clip.model.Bottleneck:
                layers[mod] = f'Layer {layer_idx}, Bottleneck {i}'
                i += 1
        return layers

    def __call__(self, rn: clip.model.ModifiedResNet):
        layers = {rn: f'ResNet output'}
        layers = self.__layer_bottlenecks(layers, rn.layer1, 1)
        layers = self.__layer_bottlenecks(layers, rn.layer2, 2)
        layers = self.__layer_bottlenecks(layers, rn.layer3, 3)
        layers = self.__layer_bottlenecks(layers, rn.layer4, 4)

        return layers
