import torch 
from torch import nn

# Encoder
class EncoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, channel_ratio=1, kernel_size=3, padding=1):
        super(EncoderBlock, self).__init__()
        self.conv1 = self.conv_block(in_channels, out_channels, channel_ratio, kernel_size, padding)
        self.conv2 = self.conv_block(out_channels, out_channels, channel_ratio, kernel_size, padding)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=out_channels)
        )
        
    def conv_block(self, in_channels, out_channels, channel_ratio, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*channel_ratio, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels),
            nn.BatchNorm2d(num_features=in_channels*channel_ratio),
            nn.Conv2d(in_channels=in_channels*channel_ratio, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout()
        )
        
    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x +  residual 
        return x                
     
     
# Decoder 
class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, channel_ratio=1, kernel_size=3, padding=1):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*channel_ratio, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=out_channels*channel_ratio),
            nn.Conv2d(in_channels=out_channels*channel_ratio, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*channel_ratio, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels),
            nn.BatchNorm2d(num_features=in_channels*channel_ratio),
            nn.Conv2d(in_channels=in_channels*channel_ratio, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout()
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        
    def forward(self, x, attention):
        residual = self.residual_conv(x)
        x = self.conv1(x)
        x = torch.cat((attention, x), dim=1)
        x = self.conv2(x)
        x = x + residual 
        return x
       
          
# Bottleneck
class BottleNeckBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, channel_ratio=1, kernel_size=3, padding=1):
        super(BottleNeckBlock, self).__init__()
        self.conv1 = self.conv_block(in_channels, in_channels, channel_ratio, kernel_size, padding)
        self.conv2 = self.conv_block(in_channels, out_channels, channel_ratio, kernel_size, padding)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
    
    def conv_block(self, in_channels, out_channels, channel_ratio, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*channel_ratio, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels),
            nn.BatchNorm2d(num_features=in_channels*channel_ratio),
            nn.Conv2d(in_channels=in_channels*channel_ratio, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout()
        )
    
    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual 
        return x


# Attention
class AttentionBlock(nn.Module):
    
    def __init__(self, in_channels, g_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels=g_channels, out_channels=g_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=g_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=g_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=g_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=g_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1)
        )
        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.final_conv = nn.Conv2d(in_channels=in_channels, out_channels=g_channels, kernel_size=1, stride=1, padding=0)
            
    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        x1 = self.upsampler(x1)
        out = self.relu(g1+x1)
        out = self.psi(out)
        out = self.sigmoid(out)
        out = self.upsampler(x) * out 
        out = self.final_conv(out)
        return out
    
    
# Final UNet
class UNet(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1):
        
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder4 = EncoderBlock(in_channels=256, out_channels=512)
        
        self.bottleneck = BottleNeckBlock(in_channels=512, out_channels=1024)

        self.att1 = AttentionBlock(in_channels=1024, g_channels=512)
        self.att2 = AttentionBlock(in_channels=512, g_channels=256)
        self.att3 = AttentionBlock(in_channels=256, g_channels=128)
        self.att4 = AttentionBlock(in_channels=128, g_channels=64)
        
        self.decoder1 = DecoderBlock(in_channels=1024, out_channels=512)
        self.decoder2 = DecoderBlock(in_channels=512, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=128)
        self.decoder4 = DecoderBlock(in_channels=128, out_channels=64)
        
        self.output_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    
    def forward(self, x):
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))

        d1 = self.decoder1(b, self.att1(b, e4))
        d2 = self.decoder2(d1, self.att2(d1, e3))
        d3 = self.decoder3(d2, self.att3(d2, e2))
        d4 = self.decoder4(d3, self.att4(d3, e1))
        
        out = self.output_conv(d4)
            
        return out
