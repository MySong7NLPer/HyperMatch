#-*- coding:utf-8 -*-

import torch
import torch.nn.init as init
import math

class MobiusLinear(torch.nn.Module):
    """
        Mobius linear layer.
    """
    def __init__(self, manifold, in_features, out_features, c, use_bias=True):
        super(MobiusLinear, self).__init__()
        self.use_bias = use_bias
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.manifold = manifold
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0.0)

    def forward(self, x):
        mv = self.manifold.mobius_matvec(self.weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1))
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features_size={}, out_features_size={}, curvalture={}'.format(
            self.in_features, self.out_features, self.c
        )



