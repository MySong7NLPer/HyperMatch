#-*- coding:utf-8 -*-

import torch

eps = 1e-15

def artanh(x):
    return Artanh.apply(x)

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + eps, 1 - eps)
        ctx.save_for_backward(x)
        out = (torch.log(1 + x.double()).sub(torch.log(1 - x.double()))).mul(0.5)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 - x ** 2)

