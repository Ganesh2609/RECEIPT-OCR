import torch 
from torch import nn
from VisionModel import VisionModel
from transformer import TransformerEncoder


class DualityProcessor(nn.Module):

    def __init__(self):
        super(DualityProcessor, self).__init__()
        self.vision = VisionModel()
        self.text = TransformerEncoder()
    
    def forward(self, img, bbox):
        vision_embeddings = self.vision(img, bbox)
        text_embeddings = self.text(bbox)
        out = torch.cat([vision_embeddings, text_embeddings], dim=1)
        return out