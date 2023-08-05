"""
Visualize features in Chain-of-Look
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
import os
# %matplotlib inline


def draw_fea(fea: list, names, save_path=None):
    r"""
    fea: [_, _, _, _]
    feature shape: torch.Size([8, 100, 1024]) or torch.Size([8, 10, 1024])
    """
    # fig = plt.figure(figsize=(30, 50))
    # num_frames, num_fea = fea.shape[0], fea.shape[1]
    num_fea, num_frames = len(fea), fea[0].shape[0]  # 4, 8
    # print(num_fea, num_frames)
    # print(names)

    # fig = plt.figure(figsize=(30, 50))
    # a = fig.add_subplot(1,4)
    # _fea = fea[]

    for i in range(num_frames):
        fig = plt.figure(figsize=(30, 50))
        a = fig.add_subplot(1, 8, i+1)
        # print(fea[i].shape)  # torch.Size([8, 100, 1024])
        # print(fea[i][0].shape)  # torch.Size([100, 1024])
        # _fea = torch.sum(fea[i][0], dim=0) / fea[i][0].shape[0]
        _fea = fea[0][i]
        _fea = torch.sum(_fea, dim=0) / _fea.shape[0]
        # print(_fea.shape)  # torch.Size([100, 1024])
        plt.plot(_fea.data.cpu().numpy())
        a.axis("off")
        a.set_title(names[i][0], fontsize=30)
        print(names[i])  # VID70/000136.png
        plt.savefig('{}'.format(names[i][0]), bbox_inches='tight')


if __name__ == '__main__':
    draw_fea()
