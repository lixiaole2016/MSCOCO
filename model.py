import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn as nn
from torchvision import models

alexnet_model = models.alexnet(pretrained=True)

class ImgNet(nn.Module):
    def __init__(self):
        super(ImgNet, self).__init__()
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        features = self.remain(x)
        return features




