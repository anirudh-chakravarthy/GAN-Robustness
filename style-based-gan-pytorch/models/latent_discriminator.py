"""
Implements discriminator on latent vectors (z_real vs z_fake, w_real vs w_fake)
Supports both z and w spaces
"""
import torch
import torch.nn as nn
import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator


class ResBlockDown(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlockDown, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features, out_features, bias=True),
            nn.ReLU(),
        )
        self.residual = nn.Linear(out_features, out_features, bias=True)
        self.shortcut = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        out = self.layers(x)
        out = self.residual(out)
        out = out + self.shortcut(x)
        return out


class ResBlock(nn.Module):

    def __init__(self, in_features=512, out_features=512):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features, out_features, bias=True),
            nn.ReLU(),
            nn.Linear(out_features, out_features, bias=True)
        )

    def forward(self, x):
        out = self.layers(x)
        out = out + x
        return out


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=512, mode='z', device='cuda'):
        """
        Args:
            mode: 'z' to run discriminator on z-space, 'w' for w-space
        Returns:
        """
        super(LatentDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            ResBlockDown(in_features=512, out_features=256),
            ResBlockDown(in_features=256, out_features=128),
            ResBlock(in_features=128, out_features=128),
            ResBlock(in_features=128, out_features=128),
            nn.ReLU()
        )
        self.dense = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, latent):
        out = self.layers(latent)
        out = self.dense(out)
        return out
        

# ----------- UNIT TESTS ------------- #

def test_resblock_down():
    x = torch.randn(size=(10, 256))
    resblockdown = ResBlockDown(256, 128)
    print(resblockdown(x).shape)

def test_resblock():
    x = torch.randn(size=(10, 512))
    resblock = ResBlock(in_features=512, out_features=512)
    print(resblock(x).shape)

def test_discriminator():
    x = torch.randn(size=(10, 512))
    disc = LatentDiscriminator()
    print(disc(x).shape)

if __name__ == "__main__":

    test_resblock_down()
    test_resblock()
    test_discriminator()
        




        
        

