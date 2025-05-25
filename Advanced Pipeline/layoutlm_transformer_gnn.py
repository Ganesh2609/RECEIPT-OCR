import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from transformers import BertConfig

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing document text elements.
    Uses a simpler implementation of transformer blocks compared to the full BERT model.
    """
    def __init__(self, hidden_size, num_attention_heads=8, num_hidden_layers=2, 
                 intermediate_size=1024, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Create a BERT-style configuration
        self.config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )
        
        # Initialize layers
        self.embeddings_layer_norm = nn.LayerNorm(hidden_size)
        self.embeddings_dropout = nn.Dropout(hidden_dropout_prob)
        
        # Create transformer blocks
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(
                hidden_size, 
                num_attention_heads, 
                intermediate_size, 
                hidden_dropout_prob, 
                attention_probs_dropout_prob
            ) for _ in range(num_hidden_layers)
        ])
    
    def forward(self, embeddings, attention_mask=None):
        """
        Process embeddings through transformer layers.
        
        Args:
            embeddings: Input embeddings tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask (1 for tokens to attend to, 0 for masked tokens)
            
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Apply layer norm and dropout to input embeddings
        hidden_states = self.embeddings_layer_norm(embeddings)
        hidden_states = self.embeddings_dropout(hidden_states)
        
        # Process through transformer layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states


class TransformerLayer(nn.Module):
    """
    Single transformer layer with self-attention and feed-forward network.
    """
    def __init__(self, hidden_size, num_attention_heads, 
                 intermediate_size, hidden_dropout_prob, attention_probs_dropout_prob):
        super(TransformerLayer, self).__init__()
        
        # Self-attention block
        self.attention = MultiHeadAttention(
            hidden_size, 
            num_attention_heads, 
            attention_probs_dropout_prob
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.attention_dropout = nn.Dropout(hidden_dropout_prob)
        
        # Feed-forward block
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        """Process input through self-attention and feed-forward blocks."""
        # Self-attention block with residual connection and layer norm
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(hidden_states + attention_output)
        
        # Feed-forward block with residual connection and layer norm
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(attention_output + layer_output)
        
        return layer_output


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    """
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(MultiHeadAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention computation."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        """Compute multi-head self-attention."""
        batch_size, seq_length = hidden_states.size(0), hidden_states.size(1)
        
        # Project inputs to queries, keys, and values
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for attention computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between query and key to get attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure attention_mask is correctly shaped: [batch_size, 1, seq_length, seq_length]
            if attention_mask.dim() == 3:
                # If mask is [batch_size, seq_length, seq_length]
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 2:
                # If mask is [batch_size, seq_length]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
            
            # Add large negative value to masked positions (1.0 for positions to attend to, 0.0 for masked)
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention probabilities to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Transpose the result back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape back to original dimensions
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Apply output projection
        output = self.output(context_layer)
        
        return output


class LayoutTransformerGNN(torch.nn.Module):
    """
    Combined model that integrates LayoutLM-inspired features, Graph Neural Networks,
    and a transformer encoder for receipt information extraction.
    
    Architecture:
    1. Text embeddings (BERT) combined with spatial embeddings (LayoutLM-style)
    2. Graph embeddings from GNN layers on the document graph
    3. Combination of all embeddings passed to transformer encoder
    4. Final classification based on the transformer output
    """
    def __init__(self, hidden_channels, out_channels, 
                 num_gnn_layers=2, num_transformer_layers=2,
                 gnn_heads=8, transformer_heads=8,
                 dropout=0.3, use_gat=True, use_edge_features=True):
        super(LayoutTransformerGNN, self).__init__()
        
        self.use_gat = use_gat
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.num_transformer_layers = num_transformer_layers
        
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
        self._init_hetero_layers(hidden_channels, num_gnn_layers, gnn_heads, dropout, use_gat)
        
        # Layer normalization for GNN outputs
        self.gnn_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_gnn_layers)
        ])
        
        # Projection layer to combine initial features with GNN outputs
        self.combination_projection = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Transformer encoder for global attention
        self.transformer_encoder = TransformerEncoder(
            hidden_size=hidden_channels,
            num_attention_heads=transformer_heads,
            num_hidden_layers=num_transformer_layers,
            intermediate_size=hidden_channels * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
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
                    ('node', 'textual', 'node'): GATConv(
                        hidden_channels, hidden_channels // heads, 
                        heads=heads, dropout=dropout, edge_dim=1),
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
    
    def _create_attention_mask(self, num_nodes, batch_size, device):
        """
        Create an attention mask for the transformer based on the document structure.
        Each node can attend to all other nodes in the same document.
        
        Args:
            num_nodes: List of number of nodes in each batch item
            batch_size: Number of documents in the batch
            device: Computation device
            
        Returns:
            A binary attention mask of shape [batch_size, max_nodes, max_nodes]
        """
        max_nodes = max(num_nodes)
        attention_mask = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
        
        for i in range(batch_size):
            # Allow attention between all nodes in the same document
            attention_mask[i, :num_nodes[i], :num_nodes[i]] = 1.0
        
        return attention_mask
    
    def forward(self, data):
        """
        Forward pass for the combined GNN and transformer model.
        
        Args:
            data: Heterogeneous graph data batch
            
        Returns:
            Classification predictions for each node
        """
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
        
        # Store original features for later combination
        original_x = x_dict['node']
        
        # Edge indices dictionary
        edge_index_dict = {
            ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_index,
            ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_index,
            ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_index
        }
        
        # Edge attributes dictionary (if using edge features)
        edge_attr_dict = None
        if self.use_edge_features:
            edge_attr_dict = {
                ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_attr,
                ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_attr,
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
            x_dict = {key: self.gnn_layer_norms[i](F.relu(x)) for key, x in x_dict.items()}
            
            # Apply dropout
            x_dict = {key: self.dropout_layer(x) for key, x in x_dict.items()}
        
        # Get the GNN output
        gnn_output = x_dict['node']
        
        # Combine original features and GNN output
        # This creates an embedding that has: text + bbox + graph information
        combined_features = torch.cat([original_x, gnn_output], dim=1)
        combined_features = self.combination_projection(combined_features)
        combined_features = F.relu(combined_features)
        combined_features = self.dropout_layer(combined_features)
        
        # Prepare data for transformer processing
        # First, get the batch assignment for each node
        batch = data['node'].batch if hasattr(data['node'], 'batch') else None
        
        if batch is not None:
            # Count nodes per document in the batch
            unique_batch, counts = torch.unique(batch, return_counts=True)
            batch_size = len(unique_batch)
            nodes_per_doc = [counts[i].item() for i in range(batch_size)]
            max_nodes = max(nodes_per_doc)
            
            # Create a padded tensor for transformer input
            # Shape: [batch_size, max_nodes, hidden_dim]
            padded_features = torch.zeros(batch_size, max_nodes, combined_features.size(1), 
                                         device=combined_features.device)
            
            # Fill in the actual features
            node_idx = 0
            for i, count in enumerate(nodes_per_doc):
                padded_features[i, :count, :] = combined_features[node_idx:node_idx+count, :]
                node_idx += count
            
            # Create attention mask for transformer
            attention_mask = self._create_attention_mask(nodes_per_doc, batch_size, combined_features.device)
            
            # Process through transformer encoder
            # Make sure attention_mask has proper shape for transformer
            transformer_output = self.transformer_encoder(padded_features, attention_mask)
            
            # Unpack the output back to the original node-level format
            unpacked_output = []
            for i, count in enumerate(nodes_per_doc):
                unpacked_output.append(transformer_output[i, :count, :])
            
            transformer_output = torch.cat(unpacked_output, dim=0)
        else:
            # If we only have a single document, we can process it directly
            # First, reshape to add batch dimension
            batched_features = combined_features.unsqueeze(0)
            
            # Process through transformer
            transformer_output = self.transformer_encoder(batched_features)
            
            # Remove batch dimension
            transformer_output = transformer_output.squeeze(0)
        
        # Final classification layer
        out = self.lin_out(transformer_output)
        
        return out
    
    def get_embeddings(self, data):
        """
        Extract different types of embeddings for visualization or analysis.
        
        Args:
            data: Heterogeneous graph data batch
            
        Returns:
            Dictionary containing different types of embeddings:
            - text_embeddings: Original BERT embeddings
            - spatial_embeddings: LayoutLM-style spatial embeddings
            - combined_initial: Initial combined embeddings (text + spatial)
            - graph_embeddings: Output from the GNN layers
            - final_embeddings: Final embeddings after transformer processing
        """
        # Get pre-computed BERT embeddings
        text_emb = data['node'].x
        
        if text_emb.dim() == 3:
            text_emb = text_emb.squeeze(1)
        
        # Get spatial embeddings
        spatial_emb = self._get_spatial_embeddings(data['node'].bbox)
        
        # Project text embeddings
        projected_text = self.projection(text_emb)
        
        # Concatenate and project
        initial_combined = torch.cat([projected_text, spatial_emb], dim=1)
        initial_combined = self.concat_projection(initial_combined)
        initial_combined = F.relu(initial_combined)
        
        # Process through GNN (simplified from the forward method)
        x_dict = {'node': initial_combined}
        
        edge_index_dict = {
            ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_index,
            ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_index,
            ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_index
        }
        
        edge_attr_dict = None
        if self.use_edge_features:
            edge_attr_dict = {
                ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_attr,
                ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_attr,
                ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_attr
            }
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            if self.use_edge_features and edge_attr_dict is not None:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            
            x_dict = {key: self.gnn_layer_norms[i](F.relu(x)) for key, x in x_dict.items()}
        
        graph_embeddings = x_dict['node']
        
        # Combine embeddings
        combined_features = torch.cat([initial_combined, graph_embeddings], dim=1)
        combined_features = self.combination_projection(combined_features)
        combined_features = F.relu(combined_features)
        
        # We can't easily extract the transformer embeddings without running the full forward pass
        # So we'll just use the combined features here
        
        return {
            'text_embeddings': text_emb,
            'spatial_embeddings': spatial_emb,
            'combined_initial': initial_combined,
            'graph_embeddings': graph_embeddings,
            'final_embeddings': combined_features
        }
        
class LayoutImageTransformerGNN(torch.nn.Module):
    """
    Combined model that integrates LayoutLM-inspired features, image features,
    Graph Neural Networks, and a transformer encoder for receipt information extraction.
    
    Architecture:
    1. Text embeddings (BERT) combined with spatial embeddings (LayoutLM-style)
    2. Image embeddings from regions corresponding to bounding boxes
    3. Graph embeddings from GNN layers on the document graph
    4. Combination of all embeddings passed to transformer encoder
    5. Final classification based on the transformer output
    """
    def __init__(self, hidden_channels, out_channels, 
                 image_embedding_dim=512,
                 num_gnn_layers=2, num_transformer_layers=2,
                 gnn_heads=8, transformer_heads=8,
                 dropout=0.3, use_gat=True, use_edge_features=True,
                 use_image_features=True):
        super(LayoutImageTransformerGNN, self).__init__()
        
        self.use_gat = use_gat
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.num_transformer_layers = num_transformer_layers
        self.use_image_features = use_image_features
        
        # BERT embedding size
        self.bert_dim = 768
        self.image_embedding_dim = image_embedding_dim
        
        # Projection for pre-computed BERT embeddings
        self.projection = nn.Linear(self.bert_dim, hidden_channels // 4)
        
        # Projection for pre-computed image embeddings
        if self.use_image_features:
            self.image_projection = nn.Linear(self.image_embedding_dim, hidden_channels // 4)
        
        # LayoutLM-style 2D position embeddings
        self.x_embedding = nn.Embedding(1001, hidden_channels // 8)  # 0-1000
        self.y_embedding = nn.Embedding(1001, hidden_channels // 8)  # 0-1000
        
        # Projection for concatenated features
        if use_image_features:
            total_concat_dim = hidden_channels // 4 + hidden_channels // 4 + hidden_channels // 2  # Projected text + image + 4 spatial embeddings
        else:
            total_concat_dim = hidden_channels // 4 + hidden_channels // 2  # Projected text + 4 spatial embeddings
            
        self.concat_projection = nn.Linear(total_concat_dim, hidden_channels)
        
        # Initialize heterogeneous graph layers
        self._init_hetero_layers(hidden_channels, num_gnn_layers, gnn_heads, dropout, use_gat)
        
        # Layer normalization for GNN outputs
        self.gnn_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_gnn_layers)
        ])
        
        # Projection layer to combine initial features with GNN outputs
        self.combination_projection = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Transformer encoder for global attention
        self.transformer_encoder = TransformerEncoder(
            hidden_size=hidden_channels,
            num_attention_heads=transformer_heads,
            num_hidden_layers=num_transformer_layers,
            intermediate_size=hidden_channels * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
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
                    ('node', 'textual', 'node'): GATConv(
                        hidden_channels, hidden_channels // heads, 
                        heads=heads, dropout=dropout, edge_dim=1),
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
    
    def _create_attention_mask(self, num_nodes, batch_size, device):
        """
        Create an attention mask for the transformer based on the document structure.
        Each node can attend to all other nodes in the same document.
        
        Args:
            num_nodes: List of number of nodes in each batch item
            batch_size: Number of documents in the batch
            device: Computation device
            
        Returns:
            A binary attention mask of shape [batch_size, max_nodes, max_nodes]
        """
        max_nodes = max(num_nodes)
        attention_mask = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
        
        for i in range(batch_size):
            # Allow attention between all nodes in the same document
            attention_mask[i, :num_nodes[i], :num_nodes[i]] = 1.0
        
        return attention_mask
    
    def forward(self, data):
        """
        Forward pass for the combined GNN and transformer model.
        
        Args:
            data: Heterogeneous graph data batch
            
        Returns:
            Classification predictions for each node
        """
        # Get pre-computed BERT embeddings
        text_emb = data['node'].x
        
        if text_emb.dim() == 3:
            text_emb = text_emb.squeeze(1)  # Remove the middle dimension
        
        # Project text embeddings
        text_emb = self.projection(text_emb)
        
        # Get image embeddings if available
        if self.use_image_features and hasattr(data['node'], 'img_x'):
            img_emb = data['node'].img_x
            if img_emb.dim() == 3:
                img_emb = img_emb.squeeze(1)
            # Project image embeddings
            img_emb = self.image_projection(img_emb)
        
        # Get spatial embeddings
        spatial_emb = self._get_spatial_embeddings(data['node'].bbox)
        
        # Concatenate all embeddings
        if self.use_image_features and hasattr(data['node'], 'img_x'):
            x = torch.cat([text_emb, img_emb, spatial_emb], dim=1)
        else:
            x = torch.cat([text_emb, spatial_emb], dim=1)
        
        # Project concatenated features to hidden dimension
        x = self.concat_projection(x)
        x = F.relu(x)
        
        # Apply dropout to initial embeddings
        x = self.dropout_layer(x)
        
        # Initialize node features dict
        x_dict = {'node': x}
        
        # Store original features for later combination
        original_x = x_dict['node']
        
        # Edge indices dictionary
        edge_index_dict = {
            ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_index,
            ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_index,
            ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_index
        }
        
        # Edge attributes dictionary (if using edge features)
        edge_attr_dict = None
        if self.use_edge_features:
            edge_attr_dict = {
                ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_attr,
                ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_attr,
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
            x_dict = {key: self.gnn_layer_norms[i](F.relu(x)) for key, x in x_dict.items()}
            
            # Apply dropout
            x_dict = {key: self.dropout_layer(x) for key, x in x_dict.items()}
        
        # Get the GNN output
        gnn_output = x_dict['node']
        
        # Combine original features and GNN output
        # This creates an embedding that has: text + bbox + graph information
        combined_features = torch.cat([original_x, gnn_output], dim=1)
        combined_features = self.combination_projection(combined_features)
        combined_features = F.relu(combined_features)
        combined_features = self.dropout_layer(combined_features)
        
        # Prepare data for transformer processing
        # First, get the batch assignment for each node
        batch = data['node'].batch if hasattr(data['node'], 'batch') else None
        
        if batch is not None:
            # Count nodes per document in the batch
            unique_batch, counts = torch.unique(batch, return_counts=True)
            batch_size = len(unique_batch)
            nodes_per_doc = [counts[i].item() for i in range(batch_size)]
            max_nodes = max(nodes_per_doc)
            
            # Create a padded tensor for transformer input
            # Shape: [batch_size, max_nodes, hidden_dim]
            padded_features = torch.zeros(batch_size, max_nodes, combined_features.size(1), 
                                         device=combined_features.device)
            
            # Fill in the actual features
            node_idx = 0
            for i, count in enumerate(nodes_per_doc):
                padded_features[i, :count, :] = combined_features[node_idx:node_idx+count, :]
                node_idx += count
            
            # Create attention mask for transformer
            attention_mask = self._create_attention_mask(nodes_per_doc, batch_size, combined_features.device)
            
            # Process through transformer encoder
            # Make sure attention_mask has proper shape for transformer
            transformer_output = self.transformer_encoder(padded_features, attention_mask)
            
            # Unpack the output back to the original node-level format
            unpacked_output = []
            for i, count in enumerate(nodes_per_doc):
                unpacked_output.append(transformer_output[i, :count, :])
            
            transformer_output = torch.cat(unpacked_output, dim=0)
        else:
            # If we only have a single document, we can process it directly
            # First, reshape to add batch dimension
            batched_features = combined_features.unsqueeze(0)
            
            # Process through transformer
            transformer_output = self.transformer_encoder(batched_features)
            
            # Remove batch dimension
            transformer_output = transformer_output.squeeze(0)
        
        # Final classification layer
        out = self.lin_out(transformer_output)
        
        return out
    
    def get_embeddings(self, data):
        """
        Extract different types of embeddings for visualization or analysis.
        
        Args:
            data: Heterogeneous graph data batch
            
        Returns:
            Dictionary containing different types of embeddings:
            - text_embeddings: Original BERT embeddings
            - image_embeddings: Original image region embeddings (if available)
            - spatial_embeddings: LayoutLM-style spatial embeddings
            - combined_initial: Initial combined embeddings
            - graph_embeddings: Output from the GNN layers
            - final_embeddings: Final embeddings after transformer processing
        """
        # Get pre-computed BERT embeddings
        text_emb = data['node'].x
        
        if text_emb.dim() == 3:
            text_emb = text_emb.squeeze(1)
        
        # Get image embeddings if available
        img_emb = None
        if self.use_image_features and hasattr(data['node'], 'img_x'):
            img_emb = data['node'].img_x
            if img_emb.dim() == 3:
                img_emb = img_emb.squeeze(1)
        
        # Get spatial embeddings
        spatial_emb = self._get_spatial_embeddings(data['node'].bbox)
        
        # Project embeddings
        projected_text = self.projection(text_emb)
        projected_img = self.image_projection(img_emb) if img_emb is not None else None
        
        # Concatenate and project
        if projected_img is not None:
            initial_combined = torch.cat([projected_text, projected_img, spatial_emb], dim=1)
        else:
            initial_combined = torch.cat([projected_text, spatial_emb], dim=1)
            
        initial_combined = self.concat_projection(initial_combined)
        initial_combined = F.relu(initial_combined)
        
        # Process through GNN (simplified from the forward method)
        x_dict = {'node': initial_combined}
        
        edge_index_dict = {
            ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_index,
            ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_index,
            ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_index
        }
        
        edge_attr_dict = None
        if self.use_edge_features:
            edge_attr_dict = {
                ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_attr,
                ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_attr,
                ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_attr
            }
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            if self.use_edge_features and edge_attr_dict is not None:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            
            x_dict = {key: self.gnn_layer_norms[i](F.relu(x)) for key, x in x_dict.items()}
        
        graph_embeddings = x_dict['node']
        
        # Combine embeddings
        combined_features = torch.cat([initial_combined, graph_embeddings], dim=1)
        combined_features = self.combination_projection(combined_features)
        combined_features = F.relu(combined_features)
        
        result = {
            'text_embeddings': text_emb,
            'spatial_embeddings': spatial_emb,
            'combined_initial': initial_combined,
            'graph_embeddings': graph_embeddings,
            'final_embeddings': combined_features
        }
        
        if img_emb is not None:
            result['image_embeddings'] = img_emb
            
        return result
    
from transformers import BertModel, BertConfig, BertTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from transformers import BertModel, BertConfig

class LayoutImageBertGNN(torch.nn.Module):
    """
    Combined model that integrates LayoutLM-inspired features, image features,
    Graph Neural Networks, and a BERT transformer for receipt information extraction.
    
    Supports multiple vision models as feature extractors through the flexible
    image embedding dimension parameter.
    """
    def __init__(self, hidden_channels, out_channels, 
                 image_embedding_dim=512,
                 num_gnn_layers=2, 
                 gnn_heads=8,
                 dropout=0.3, 
                 use_gat=True, 
                 use_edge_features=True,
                 use_image_features=True,
                 bert_model_name='bert-base-uncased',
                 freeze_bert=False,
                 max_seq_length=512):
        super(LayoutImageBertGNN, self).__init__()
        
        self.use_gat = use_gat
        self.use_edge_features = use_edge_features
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.use_image_features = use_image_features
        self.max_seq_length = max_seq_length
        self.image_embedding_dim = image_embedding_dim
        
        # BERT embedding size
        self.bert_model_name = bert_model_name
        self.bert_hidden_size = 768  # Standard size for BERT base
        
        # BERT model for transformer encoding
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        
        # Projection for pre-computed BERT text embeddings
        self.text_projection = nn.Linear(768, hidden_channels // 4)
        
        # Projection for pre-computed image embeddings
        if self.use_image_features:
            self.image_projection = nn.Linear(self.image_embedding_dim, hidden_channels // 4)
        
        # LayoutLM-style 2D position embeddings
        self.x_embedding = nn.Embedding(1001, hidden_channels // 8)  # 0-1000
        self.y_embedding = nn.Embedding(1001, hidden_channels // 8)  # 0-1000
        
        # Projection for concatenated features
        if use_image_features:
            total_concat_dim = hidden_channels // 4 + hidden_channels // 4 + hidden_channels // 2  # Projected text + image + 4 spatial embeddings
        else:
            total_concat_dim = hidden_channels // 4 + hidden_channels // 2  # Projected text + 4 spatial embeddings
            
        self.concat_projection = nn.Linear(total_concat_dim, hidden_channels)
        
        # Initialize heterogeneous graph layers
        self._init_hetero_layers(hidden_channels, num_gnn_layers, gnn_heads, dropout, use_gat)
        
        # Layer normalization for GNN outputs
        self.gnn_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_gnn_layers)
        ])
        
        # Projection layer to combine initial features with GNN outputs
        self.combination_projection = nn.Linear(hidden_channels * 2, self.bert_hidden_size)
        
        # Final classification layer
        self.lin_out = nn.Linear(self.bert_hidden_size, out_channels)
        
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
                    ('node', 'textual', 'node'): GATConv(
                        hidden_channels, hidden_channels // heads, 
                        heads=heads, dropout=dropout, edge_dim=1),
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
    
    def _create_attention_mask(self, num_nodes, batch_size, device):
        """
        Create an attention mask for BERT based on the document structure.
        Each node can attend to all other nodes in the same document.
        
        Args:
            num_nodes: List of number of nodes in each batch item
            batch_size: Number of documents in the batch
            device: Computation device
            
        Returns:
            A binary attention mask of shape [batch_size, max_nodes] (1 for tokens to attend to, 0 for padding)
        """
        max_nodes = max(num_nodes)
        attention_mask = torch.zeros(batch_size, max_nodes, device=device)
        
        for i in range(batch_size):
            # Allow attention for actual nodes (1s for valid positions, 0s for padding)
            attention_mask[i, :num_nodes[i]] = 1.0
        
        return attention_mask
    
    def forward(self, data):
        """
        Forward pass for the combined GNN and BERT model.
        
        Args:
            data: Heterogeneous graph data batch
            
        Returns:
            Classification predictions for each node
        """
        # Get pre-computed BERT embeddings
        text_emb = data['node'].x
        
        if text_emb.dim() == 3:
            text_emb = text_emb.squeeze(1)  # Remove the middle dimension
        
        # Project text embeddings
        text_emb_proj = self.text_projection(text_emb)
        
        # Get image embeddings if available
        img_emb = None
        img_emb_proj = None
        if self.use_image_features and hasattr(data['node'], 'img_x'):
            img_emb = data['node'].img_x
            if img_emb.dim() == 3:
                img_emb = img_emb.squeeze(1)
            # Project image embeddings
            img_emb_proj = self.image_projection(img_emb)
        
        # Get spatial embeddings
        spatial_emb = self._get_spatial_embeddings(data['node'].bbox)
        
        # Concatenate only text and image embeddings for GNN input (not spatial)
        if self.use_image_features and img_emb_proj is not None:
            gnn_input = torch.cat([text_emb_proj, img_emb_proj], dim=1)
        else:
            gnn_input = text_emb_proj
        
        # Project for GNN input
        gnn_input = F.relu(self.concat_projection(gnn_input))
        gnn_input = self.dropout_layer(gnn_input)
        
        # Initialize node features dict
        x_dict = {'node': gnn_input}
        
        # Edge indices dictionary
        edge_index_dict = {
            ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_index,
            ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_index,
            ('node', 'directed', 'node'): data[('node', 'directed', 'node')].edge_index
        }
        
        # Edge attributes dictionary (if using edge features)
        edge_attr_dict = None
        if self.use_edge_features:
            edge_attr_dict = {
                ('node', 'spatial', 'node'): data[('node', 'spatial', 'node')].edge_attr,
                ('node', 'textual', 'node'): data[('node', 'textual', 'node')].edge_attr,
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
            x_dict = {key: self.gnn_layer_norms[i](F.relu(x)) for key, x in x_dict.items()}
            
            # Apply dropout
            x_dict = {key: self.dropout_layer(x) for key, x in x_dict.items()}
        
        # Get the GNN output
        gnn_output = x_dict['node']
        
        # Combine text, image, spatial embeddings and GNN output
        # (instead of original_x and gnn_output)
        if self.use_image_features and img_emb_proj is not None:
            combined_features = torch.cat([text_emb_proj, img_emb_proj, spatial_emb, gnn_output], dim=1)
        else:
            combined_features = torch.cat([text_emb_proj, spatial_emb, gnn_output], dim=1)
        
        # Project to BERT hidden size
        combined_features = self.combination_projection(combined_features)
        combined_features = F.relu(combined_features)
        combined_features = self.dropout_layer(combined_features)
        
        # Prepare data for BERT processing
        # First, get the batch assignment for each node
        batch = data['node'].batch if hasattr(data['node'], 'batch') else None
        
        if batch is not None:
            # Count nodes per document in the batch
            unique_batch, counts = torch.unique(batch, return_counts=True)
            batch_size = len(unique_batch)
            nodes_per_doc = [counts[i].item() for i in range(batch_size)]
            max_nodes = min(max(nodes_per_doc), self.max_seq_length)  # Limit to BERT's max sequence length
            
            # Create a padded tensor for BERT input
            # Shape: [batch_size, max_nodes, hidden_dim]
            padded_features = torch.zeros(batch_size, max_nodes, combined_features.size(1), 
                                        device=combined_features.device)
            
            # Fill in the actual features
            node_idx = 0
            for i, count in enumerate(nodes_per_doc):
                # Truncate if longer than max_nodes
                actual_count = min(count, max_nodes)
                padded_features[i, :actual_count, :] = combined_features[node_idx:node_idx+actual_count, :]
                node_idx += count
            
            # Create attention mask for BERT (1 for actual tokens, 0 for padding)
            attention_mask = self._create_attention_mask(
                [min(n, max_nodes) for n in nodes_per_doc], 
                batch_size, 
                combined_features.device
            )
            
            # Process through BERT
            bert_outputs = self.bert_model(
                inputs_embeds=padded_features,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get the output from BERT
            bert_output = bert_outputs.last_hidden_state
            
            # Unpack the output back to the original node-level format
            unpacked_output = []
            node_idx = 0
            for i, count in enumerate(nodes_per_doc):
                actual_count = min(count, max_nodes)
                # Extract the BERT outputs for the actual nodes (not padding)
                node_outputs = bert_output[i, :actual_count, :]
                
                # If truncated, we need to handle missing nodes
                if actual_count < count:
                    # For truncated nodes, we'll use the original combined features
                    missing_nodes = combined_features[node_idx+actual_count:node_idx+count, :]
                    # Apply a linear layer to match dimensions if needed
                    if missing_nodes.size(1) != bert_output.size(2):
                        missing_nodes = self.lin_out.weight[:bert_output.size(2), :] @ missing_nodes.t()
                        missing_nodes = missing_nodes.t()
                    node_outputs = torch.cat([node_outputs, missing_nodes], dim=0)
                
                unpacked_output.append(node_outputs)
                node_idx += count
            
            transformer_output = torch.cat(unpacked_output, dim=0)
            
        else:
            # If we only have a single document, we can process it directly
            # First, reshape to add batch dimension
            batched_features = combined_features.unsqueeze(0)
            
            # Create attention mask (all 1s since there's no padding)
            attention_mask = torch.ones(1, batched_features.size(1), device=batched_features.device)
            
            # Truncate if too long for BERT
            if batched_features.size(1) > self.max_seq_length:
                batched_features = batched_features[:, :self.max_seq_length, :]
                attention_mask = attention_mask[:, :self.max_seq_length]
            
            # Process through BERT
            bert_outputs = self.bert_model(
                inputs_embeds=batched_features,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get the output from BERT
            transformer_output = bert_outputs.last_hidden_state.squeeze(0)
            
            # If we truncated, we need to handle the missing nodes
            if combined_features.size(0) > self.max_seq_length:
                # For truncated nodes, we'll use the original combined features
                missing_nodes = combined_features[self.max_seq_length:, :]
                # Apply a linear layer to match dimensions if needed
                if missing_nodes.size(1) != transformer_output.size(1):
                    missing_nodes = self.lin_out.weight[:transformer_output.size(1), :] @ missing_nodes.t()
                    missing_nodes = missing_nodes.t()
                transformer_output = torch.cat([transformer_output, missing_nodes], dim=0)
        
        # Final classification layer
        out = self.lin_out(transformer_output)
        
        return out