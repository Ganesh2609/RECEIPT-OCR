import torch
from torch import nn
import torch.nn.functional as F
from unet import UNet
import math


class RoI(nn.Module):

    def __init__(self, out_size):
        super(RoI, self).__init__()
        self.out_size = out_size

    def forward(self, x, rois):
        pooled_features = []
        batch_size, seq_len, _ = rois.shape
        for i in range(batch_size):
            batch_pooled = []
            for j in range(seq_len):
                _, x1, y1, x2, y2 = rois[i, j].int()
                roi_region = x[i:i+1, :, y1:y2, x1:x2]
                pooled_feature = F.adaptive_avg_pool2d(roi_region, self.out_size)
                batch_pooled.append(pooled_feature)
            pooled_features.append(torch.cat(batch_pooled, dim=1))
        
        return torch.cat(pooled_features, dim=0)


class VisionModel(nn.Module):

    def __init__(self, img_size=256, out_size=(7, 7), out_features=512):
        super(VisionModel, self).__init__()
        self.img_size = img_size
        self.out_size = out_size
        self.unet = UNet()
        self.roi = RoI(self.out_size)
        self.fc_out = nn.Linear(in_features=out_size[0] * out_size[1], out_features=out_features)

    def forward(self, x, rois):
        m, n = x.shape[2], x.shape[3]
        x_scale, y_scale = self.img_size/m, self.img_size/n
        
        rois[:, :, 1] = (rois[:, :, 1] * x_scale).int()
        rois[:, :, 2] = (rois[:, :, 2] * y_scale).int()
        rois[:, :, 3] = (rois[:, :, 3] * x_scale).int()
        rois[:, :, 4] = (rois[:, :, 4] * y_scale).int()

        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x1 = self.unet(x)
        x2 = self.roi(x1, rois)
        print(x2.shape)
        
        out = torch.cat([F.adaptive_avg_pool2d(x1, self.out_size), x2], dim=1)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.fc_out(out)
        return out
