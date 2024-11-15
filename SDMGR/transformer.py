import torch 
from torch import nn


class TransformerEncoder(nn.Module):
    
    def __init__(self, img_size=4096, num_tokens=8192, embedding_dim=512, num_layers=4, max_seq_len=128):
        
        super(TransformerEncoder, self).__init__()
        self.img_size = img_size

        self.bbox_embedder = nn.Embedding(num_embeddings=2*img_size, embedding_dim=embedding_dim)
        self.word_embedder = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim)
        self.positional_embedder = nn.Parameter(torch.rand(1, max_seq_len, embedding_dim))

        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, activation='relu', dim_feedforward=256, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(p=0.1)
    
    
    def forward(self, x):        
        
        x_text = self.word_embedder(x[:, :, 0])
        x1 = self.bbox_embedder(x[:, :, 1])
        y1 = self.bbox_embedder(x[:, :, 2] + self.img_size)
        x2 = self.bbox_embedder(x[:, :, 3])
        y2 = self.bbox_embedder(x[:, :, 4] + self.img_size)

        combined_x = torch.mean(torch.stack([x_text, x1, y1, x2, y2]), dim=0)
        embeddings = combined_x + self.positional_embedder[:, :combined_x.shape[1], :].repeat(combined_x.shape[0], 1, 1)

        out = self.encoder(embeddings)

        return out
    