import torch
from torch import nn
import torch.nn.functional as F
from unet import UNet


class RoI(nn.Module):

    def __init__(self, out_size):
        super(RoI, self).__init__()
        self.out_size = out_size

    def forward(self, x, rois):
        pooled_features = []
        for roi in rois:
            x1, y1, x2, y2 = roi
            roi_region = x[:, :, y1:y2, x1:x2]
            pooled_feature = F.adaptive_avg_pool2d(roi_region, self.out_size)
            pooled_features.append(pooled_feature)
        return torch.cat(pooled_features, dim=1)
    

class VisionModel(nn.Module):

    def __init__(self, img_size=256, out_size=(7,7), out_features=512):
        super(VisionModel, self).__init__()
        self.img_size = img_size
        self.out_size = out_size
        self.unet = UNet()
        self.roi = RoI(self.out_size)
        self.fc_out = nn.Linear(in_features=out_size[0]*out_size[1], out_features=out_features)

    def forward(self, x, rois):
        m, n = x.shape[2], x.shape[3]
        x_scale, y_scale = self.img_size//m, self.img_size//n
        for i in range(len(rois)):
            x1, y1, x2, y2 = rois[i]
            rois[i] = [x1//x_scale, y1//y_scale, x2//x_scale, y2//y_scale]
        x1 = self.unet(x)
        x2 = self.roi(x1, rois)
        print(x1.shape, x2.shape)
        out = torch.cat([F.adaptive_avg_pool2d(x1, self.out_size), x2], dim=1)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.fc_out(out)
        return out
