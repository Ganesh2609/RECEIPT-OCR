import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class NoBiasLayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(NoBiasLayerNorm, self).__init__()
        normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(WithBiasLayerNorm, self).__init__()
        normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape:int, bias:bool):
        super(LayerNorm, self).__init__()
        if bias:
            self.body = WithBiasLayerNorm(normalized_shape)
        else:
            self.body = NoBiasLayerNorm(normalized_shape)

    def forward(self, x):
        H, W = x.shape[-2:]
        return self.to_4d(self.body(self.to_3d(x)), H, W)

    def to_3d(self, x):
        return rearrange(x, 'b c h w -> b (h w) c')

    def to_4d(self, x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
