import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from layernorm import LayerNorm



class GatedDConvFeedForwardNetwork(nn.Module):

    def __init__(self, channels, expansion_ratio, bias):
        
        super(GatedDConvFeedForwardNetwork, self).__init__()
        hidden = int(channels*expansion_ratio)

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=hidden*2, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=hidden*2, out_channels=hidden*2, kernel_size=3, stride=1, padding=1, groups=hidden, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=hidden, out_channels=channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = self.conv2(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.conv3(x)
        return x
    


class MultiDConvHeadTransposedAttention(nn.Module):
    
    def __init__(self, channels, nheads, bias):
        super(MultiDConvHeadTransposedAttention, self).__init__()
        self.nheads = nheads
        self.temperature = nn.Parameter(torch.ones(nheads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3, stride=1, padding=1, groups=channels*3, bias=bias)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        

    def forward(self, x):
        
        N, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'N (head C) H W -> N head C (H W)', head=self.nheads)
        k = rearrange(k, 'N (head C) H W -> N head C (H W)', head=self.nheads)
        v = rearrange(v, 'N (head C) H W -> N head C (H W)', head=self.nheads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        out = rearrange(out, 'N head C (H W) -> N (head C) H W', head=self.nheads, H=H, W=W)
        out = self.final_conv(out)

        return out



class TransformerBlock(nn.Module):

    def __init__(self, channels, nheads, expansion_ratio, bias, layernorm_bias):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(channels, layernorm_bias)
        self.attn = MultiDConvHeadTransposedAttention(channels, nheads, bias)
        self.norm2 = LayerNorm(channels, layernorm_bias)
        self.ffn = GatedDConvFeedForwardNetwork(channels, expansion_ratio, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    


class Downsample(nn.Module):
    
    def __init__(self, num_features):
        super(Downsample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_features, num_features//2, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.layers(x)



class Upsample(nn.Module):
    
    def __init__(self, num_features):
        super(Upsample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.layers(x)