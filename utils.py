import numpy as np
from torch import nn
from torch import  autograd
import torch
# from visualize import VisdomPlotter
import os
# import pdb

class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=projected_embed_dim),
            nn.BatchNorm1d(num_features=projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)

        return hidden_concat


def save_loss(x, file_name, mode):
    """
    @ param: list: x
    @ param: string: name
    @ param: string: mode
    return: None
    """
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            for subitem in x:
                f.write("%.4f" % subitem)
                f.write(" ")
            f.write("\n")
    else:
        with open(file_name, 'a') as f:
            for subitem in x:
                f.write("%.4f" % subitem)
                f.write(" ")
            f.write("\n")


def save_log(x, file_name, mode):
    """
    @ param: list: x
    @ param: string: name
    @ param: string: mode
    return: None
    """
    with open(file_name, mode) as f:
        # for item in x:
        f.writelines(x)
        f.writelines("\n")
    f.close()






