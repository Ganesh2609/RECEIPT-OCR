import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv

class LayoutGNN(torch.nn.Module):
    """
    Graph Neural Network for receipt node classification using LayoutLM-inspired approach.
    
    Features:
    - Uses pre-computed BERT embeddings from the graph construction stage
    - LayoutLM-style 2D positional embeddings with concatenation
    - Heterogeneous Graph Attention Networks
    - Supports three relationship types: spatial, textual, and directed
    """
    def __init__(self, hidden_channels, out_channels, 
                 num_layers=2, heads=8, dropout=0.3, use_gat=True, use_edge_features=True):
        super(LayoutGNN, self).__init__()
        
        self.use_gat = use_gat
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        self.num_layers = num_layers
        
        # BERT embedding size
        self.bert_dim = 768
        
        # Projection for pre-computed BERT embeddings
        self.projection = nn.Linear(self.bert_dim, hidden_channels // 2)
        
        # LayoutLM-style 2D position embeddings
        self.x_embedding = nn.Embedding(1001, hidden_channels // 4)  # 0-1000
        self.y_embedding = nn.Embedding(1001, hidden_channels // 4)  # 0-1000
        
        # Projection for concatenated features
        total_concat_dim = hidden_channels // 2 + hidden_channels  # Projected text + 4 spatial embeddings
        self.concat_projection = nn.Linear(total_concat_dim, hidden_channels)
        
        # Initialize heterogeneous graph layers
        self._init_hetero_layers(hidden_channels, num_layers, heads, dropout, use_gat)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])
        
        # Final classification layer
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def _init_hetero_layers(self, hidden_channels, num_layers, heads, dropout, use_gat):
        """Initialize layers for heterogeneous graph processing."""
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            if use_gat:
                # Use GAT for heterogeneous graphs with three edge types
                conv = HeteroConv({
                    ('node', 'spatial', 'node'): GATConv(
                        hidden_channels, hidden_channels // heads, 
                        heads=heads, dropout=dropout, edge_dim=1),
                    # ('node', 'textual', 'node'): GATConv(
                    #     hidden_channels, hidden_channels // heads, 
                    #     heads=heads, dropout=dropout, edge_dim=1),
                    ('node', 'directed', 'node'): GATConv(  # Add directed edge type
                        hidden_channels, hidden_channels // heads, 
                        heads=heads, dropout=dropout, edge_dim=1)
                })
            else:
                # Use standard GCN/SAGE for heterogeneous graphs
                from torch_geometric.nn import GCNConv, SAGEConv
                conv = HeteroConv({
                    ('node', 'spatial', 'node'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                    ('node', 'textual', 'node'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                    ('node', 'directed', 'node'): SAGEConv((hidden_channels, hidden_channels), hidden_channels)
                })
            self.convs.append(conv)
    
    def _get_spatial_embeddings(self, bbox):
        """
        Get LayoutLM-style spatial embeddings from bounding boxes.
        
        Args:
            bbox: Tensor of shape [batch_size, 4] with normalized coordinates
                 in LayoutLM format (x0, y0, x1, y1)
        
        Returns:
            Concatenated embeddings for spatial features
        """
        # Extract coordinates
        x0, y0, x1, y1 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        
        # Get embeddings for each coordinate (shared embedding tables)
        x0_emb = self.x_embedding(x0)
        y0_emb = self.y_embedding(y0)
        x1_emb = self.x_embedding(x1)
        y1_emb = self.y_embedding(y1)
        
        # Concatenate embeddings
        return torch.cat([x0_emb, y0_emb, x1_emb, y1_emb], dim=1)
    
    def forward(self, data):
        """Forward pass for the heterogeneous graph."""
        # Get pre-computed BERT embeddings
        text_emb = data['node'].x
        
        if text_emb.dim() == 3:
            text_emb = text_emb.squeeze(1)  # Remove the middle dimension
        
        # Project text embeddings
        text_emb = self.projection(text_emb)
        
        # Get spatial embeddings
        spatial_emb = self._get_spatial_embeddings(data['node'].bbox)
        
        # Concatenate text and spatial embeddings
        x = torch.cat([text_emb, spatial_emb], dim=1)
        
        # Project concatenated features to hidden dimension
        x = self.concat_projection(x)
        x = F.relu(x)
        
        # Apply dropout to initial embeddings
        x = self.dropout_layer(x)
        
        # Initialize node features dict
        x_dict = {'node': x}
        
        # Store original features for residual connection
        original_x = x_dict['node']
        
        # Edge indices dictionary
        edge_index_dict = {
            ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_index,
            # ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_index,
            ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_index
        }
        
        # Edge attributes dictionary (if using edge features)
        edge_attr_dict = None
        if self.use_edge_features:
            edge_attr_dict = {
                ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_attr,
                # ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_attr,
                ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_attr
            }
        
        # Process through graph convolution layers
        for i, conv in enumerate(self.convs):
            # Apply convolution
            if self.use_edge_features and edge_attr_dict is not None:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation and normalization
            x_dict = {key: self.layer_norms[i](F.relu(x)) for key, x in x_dict.items()}
            
            # Apply dropout
            x_dict = {key: self.dropout_layer(x) for key, x in x_dict.items()}
            
            # Add residual connection for last layer
            if i == self.num_layers - 1:
                x_dict['node'] = x_dict['node'] + original_x
        
        # Final prediction
        out = self.lin_out(x_dict['node'])
        
        return out