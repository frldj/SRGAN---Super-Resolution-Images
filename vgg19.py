from collections import namedtuple
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class VGG19(nn.Module):
    VGG_LAYERS = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
    ]

    def __init__(self, pretrained=True, require_grad=False):
        super().__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.layers = nn.ModuleList(models.vgg19(weights=weights).features)

        if not require_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = {}
        for name, layer in zip(self.VGG_LAYERS, self.layers):
            x = layer(x)
            outputs[name] = x

        VGGOutput = namedtuple('VGGOutput', self.VGG_LAYERS)
        return VGGOutput(**outputs)
