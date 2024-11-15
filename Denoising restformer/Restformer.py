import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from layernorm import LayerNorm
from RestformerBlocks import Downsample, Upsample, TransformerBlock



class Restformer(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_ratio=2.66, bias=False, layernorm_bias=False):

        super(Restformer, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.encoder1 = nn.Sequential( *[TransformerBlock(channels=dim, nheads=heads[0], expansion_ratio=ffn_expansion_ratio, bias=bias, layernorm_bias=layernorm_bias) for _ in range(num_blocks[0])] )
        self.encoder2 = nn.Sequential( *[TransformerBlock(channels=int(dim*2), nheads=heads[1], expansion_ratio=ffn_expansion_ratio, bias=bias, layernorm_bias=layernorm_bias) for _ in range(num_blocks[1])] )
        self.encoder3 = nn.Sequential( *[TransformerBlock(channels=int(dim*4), nheads=heads[2], expansion_ratio=ffn_expansion_ratio, bias=bias, layernorm_bias=layernorm_bias) for _ in range(num_blocks[2])] )
        self.latent = nn.Sequential( *[TransformerBlock(channels=int(dim*8), nheads=heads[3], expansion_ratio=ffn_expansion_ratio, bias=bias, layernorm_bias=layernorm_bias) for _ in range(num_blocks[3])] )

        self.down1 = Downsample(dim)
        self.down2 = Downsample(int(dim*2))
        self.down3 = Downsample(int(dim*4))

        self.up3 = Upsample(int(dim*8))
        self.up2 = Upsample(int(dim*4))
        self.up1 = Upsample(int(dim*2))

        self.upsample_conv_3 = nn.Conv2d(int(dim*8), int(dim*4), kernel_size=1, bias=bias)
        self.upsample_conv_2 = nn.Conv2d(int(dim*4), int(dim*2), kernel_size=1, bias=bias)

        self.decoder3 = nn.Sequential( *[TransformerBlock(channels=int(dim*4), nheads=heads[2], expansion_ratio=ffn_expansion_ratio, bias=bias, layernorm_bias=layernorm_bias) for _ in range(num_blocks[2])] )
        self.decoder2 = nn.Sequential( *[TransformerBlock(channels=int(dim*2), nheads=heads[1], expansion_ratio=ffn_expansion_ratio, bias=bias, layernorm_bias=layernorm_bias) for _ in range(num_blocks[2])] )
        self.decoder1 = nn.Sequential( *[TransformerBlock(channels=int(dim*2), nheads=heads[0], expansion_ratio=ffn_expansion_ratio, bias=bias, layernorm_bias=layernorm_bias) for _ in range(num_blocks[2])] )
        self.refinement = nn.Sequential( *[TransformerBlock(channels=int(dim*2), nheads=heads[0], expansion_ratio=ffn_expansion_ratio, bias=bias, layernorm_bias=layernorm_bias) for _ in range(num_refinement_blocks)] )

        self.final_conv = nn.Conv2d(in_channels=int(dim*2), out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    

    def forward(self, x):

        x_conv = self.initial_conv(x)

        e1 = self.encoder1(x_conv)
        e2 = self.encoder2(self.down1(e1))
        e3 = self.encoder3(self.down2(e2))
        lat = self.latent(self.down3(e3))

        d3_inp = self.upsample_conv_3(torch.cat([e3, self.up3(lat)], dim=1))
        d3_out = self.decoder3(d3_inp)

        d2_inp = self.upsample_conv_2(torch.cat([e2, self.up2(d3_out)], dim=1))
        d2_out = self.decoder2(d2_inp)

        d1_inp = torch.cat([e1, self.up1(d2_out)], dim=1)
        d1_out = self.decoder1(d1_inp)

        refined = self.refinement(d1_out)
        out = self.final_conv(refined) + x

        return out

