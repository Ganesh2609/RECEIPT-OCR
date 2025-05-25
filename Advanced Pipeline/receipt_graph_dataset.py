import json
import os
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from transformers import BertTokenizer, BertModel
from PIL import Image
import logging
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import math
import time
from shapely.geometry import Polygon, Point
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

# Import RoI Pooling implementation
from roi_pooling import RoIPool, RoIAlign
# Import feature extraction models
from feature_extraction_models import FeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReceiptGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for OCR receipt data with flexible feature extraction.
    
    Converts OCR annotations into multi-relational graphs where nodes are text elements
    and edges represent spatial proximity, text similarity, and directed relationships.
    
    Supports multiple vision models for feature extraction including ResNet, EfficientNet,
    ConvNeXt, Vision Transformers (ViT), Swin Transformer, CLIP, and LayoutLM.
    """
    def __init__(self, root, file_path, images_dir=None, transform=None, pre_transform=None, 
                 k_spatial=5, k_textual=3, 
                 spatial_threshold=None, textual_threshold=0.5,
                 max_seq_length=64, bert_model_name='bert-base-uncased',
                 vision_model_name='resnet18',
                 roi_output_size=(7, 7), use_roi_align=True,
                 force_reload=False, use_gpu=True):
        self.file_path = file_path
        self.images_dir = images_dir
        self.k_spatial = k_spatial
        self.k_textual = k_textual
        self.spatial_threshold = spatial_threshold
        self.textual_threshold = textual_threshold
        self.max_seq_length = max_seq_length
        self.bert_model_name = bert_model_name
        self.force_reload = force_reload
        self.vision_model_name = vision_model_name
        self.roi_output_size = roi_output_size
        self.use_roi_align = use_roi_align
        
        # Check if GPU is available and requested
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        logger.info(f"Using device: {self.device} for model embeddings")
        
        # Initialize BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        # Move model to device
        self.bert_model = self.bert_model.to(self.device)
        # Set BERT to eval mode - we won't train it
        self.bert_model.eval()
        
        # Initialize vision model for feature extraction
        logger.info(f"Initializing vision model: {vision_model_name}")
        self.vision_model, self.image_transform, self.feature_dim, \
        self.needs_fixed_size, self.input_size, self.is_transformer = \
            FeatureExtractor.create_model(vision_model_name, self.device)
            
        # Save feature dimension
        self.image_embedding_dim = self.feature_dim
        logger.info(f"Vision model initialized with feature dimension: {self.feature_dim}")
        
        # Initialize RoI Pooling/Align layer
        if use_roi_align:
            # RoI Align gives better alignment with the input image
            self.roi_layer = RoIAlign(output_size=roi_output_size)
        else:
            # RoI Pool is faster but less precise
            self.roi_layer = RoIPool(output_size=roi_output_size)
        
        super(ReceiptGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [os.path.basename(self.file_path)]
    
    @property
    def processed_file_names(self):
        base_name = os.path.basename(self.file_path).split(".")[0]
        model_name = self.vision_model_name.replace('_', '-')
        return [f'{base_name}_hetero_{model_name}_data.pt']
    
    def download(self):
        # No download needed, the data files should already exist
        pass
    def _extract_image_features(self, image):
        """
        Extract feature map from the entire image using the selected vision model.
        
        Uses mixed precision where supported for faster processing.
        
        Args:
            image: PIL Image of the receipt
                
        Returns:
            Tensor of feature map with shape [C, H, W]
        """
        # Transform the image
        img_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        
        # Use mixed precision if available (much faster on modern GPUs)
        use_amp = (self.device == 'cuda' and torch.cuda.is_available() and 
                hasattr(torch.cuda, 'amp') and
                torch.cuda.get_device_capability()[0] >= 7)  # Volta or newer
        
        # Extract features
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    feature_map = self.vision_model(img_tensor)
            else:
                feature_map = self.vision_model(img_tensor)
            
            # Remove batch dimension
            if feature_map.dim() == 4:
                feature_map = feature_map.squeeze(0)
            
        return feature_map
    
    def _extract_roi_features(self, feature_map, boxes, width, height):
        """
        Extract region features from the feature map using torchvision's optimized RoI operations.
        
        Args:
            feature_map: Feature map tensor with shape [C, H, W]
            boxes: List of bounding boxes in original image coordinates
            width: Original image width
            height: Original image height
            
        Returns:
            List of feature tensors for each region
        """
        from torchvision.ops import roi_align, roi_pool
        
        device = feature_map.device
        feature_height, feature_width = feature_map.shape[-2:]
        
        # Scale factors to convert from original image to feature map coordinates
        scale_h = feature_height / height
        scale_w = feature_width / width
        
        # Add batch dimension to feature map if not already present
        if feature_map.dim() == 3:
            feature_map = feature_map.unsqueeze(0)  # [1, C, H, W]
        
        # Convert boxes to the format expected by torchvision: [batch_idx, x1, y1, x2, y2]
        # where coordinates are in absolute feature map coordinates
        rois = []
        for box in boxes:
            # Extract coordinates to get top-left and bottom-right corners
            x_coords = [box[0], box[2], box[4], box[6]]
            y_coords = [box[1], box[3], box[5], box[7]]
            
            # Get top-left (x1, y1) and bottom-right (x2, y2) in original image coordinates
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # Scale to feature map coordinates
            x1 = x1 * scale_w
            y1 = y1 * scale_h
            x2 = x2 * scale_w
            y2 = y2 * scale_h
            
            # Add batch index (always 0 since we're processing one image at a time)
            rois.append([0, x1, y1, x2, y2])
        
        # Convert to tensor
        if not rois:
            # Return empty features if no boxes
            return [torch.zeros(feature_map.shape[1], self.roi_output_size[0], 
                            self.roi_output_size[1], device=device) 
                    for _ in range(len(boxes))]
        
        rois_tensor = torch.tensor(rois, dtype=torch.float32, device=device)
        
        # Use torchvision's efficient implementation of RoI operations
        if self.use_roi_align:
            # The spatial_scale is the ratio of feature map resolution to original image resolution
            regions = roi_align(feature_map, rois_tensor, output_size=self.roi_output_size, 
                            spatial_scale=1.0, sampling_ratio=2)
        else:
            regions = roi_pool(feature_map, rois_tensor, output_size=self.roi_output_size, 
                            spatial_scale=1.0)
        
        # Apply global average pooling to get feature vectors
        roi_embeddings = []
        for i in range(regions.size(0)):
            # Global average pooling: [C, H, W] -> [C]
            pooled = F.adaptive_avg_pool2d(regions[i].unsqueeze(0), (1, 1)).view(-1)
            roi_embeddings.append(pooled.cpu())
        
        return roi_embeddings
    
    def process(self):
        # Check if the file exists
        if not os.path.exists(self.file_path):
            # Try looking for it relative to the current directory
            current_dir_path = os.path.join(os.getcwd(), self.file_path)
            if os.path.exists(current_dir_path):
                self.file_path = current_dir_path
            else:
                # Try a few other common locations
                possible_paths = [
                    os.path.join(os.path.dirname(self.root), self.file_path),
                    os.path.join('..', self.file_path),
                    os.path.basename(self.file_path)
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.file_path = path
                        break
                else:
                    raise FileNotFoundError(f"Could not find data file: {self.file_path}. Please provide the correct path.")
        
        # Read the data
        logger.info(f"Reading data from {self.file_path}")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_start_time = time.time()
        data_list = []
        
        # Counters for filtering statistics
        total_receipts = 0
        filtered_small_annotations = 0
        filtered_sparse_documents = 0
        skipped_image_extraction = 0
        
        # Process in batches for better memory management
        batch_size = 30  # Process 30 receipts at a time
        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i:i+batch_size]
            batch_start_time = time.time()
            
            for j, line in enumerate(tqdm(batch_lines, desc=f"Processing receipts batch {i//batch_size+1}/{(len(lines)+batch_size-1)//batch_size}")):
                try:
                    total_receipts += 1
                    receipt_data = json.loads(line.strip())
                    
                    # Filter out small annotations (area < 50)
                    if 'annotations' in receipt_data:
                        original_annotation_count = len(receipt_data['annotations'])
                        width = receipt_data.get('width', 300)
                        height = receipt_data.get('height', 300)
                        
                        # Filter annotations based on area
                        filtered_annotations = []
                        for anno in receipt_data['annotations']:
                            if 'box' in anno and len(anno['box']) == 8:
                                # Extract coordinates to get top-left and bottom-right corners
                                x_coords = [anno['box'][0], anno['box'][2], anno['box'][4], anno['box'][6]]
                                y_coords = [anno['box'][1], anno['box'][3], anno['box'][5], anno['box'][7]]
                                
                                # Get top-left (x0, y0) and bottom-right (x1, y1)
                                x0, y0 = min(x_coords), min(y_coords)  
                                x1, y1 = max(x_coords), max(y_coords)
                                
                                # Calculate area
                                area = (x1 - x0) * (y1 - y0)
                                
                                # Keep annotation if area is >= 50
                                if area >= 50:
                                    filtered_annotations.append(anno)
                        
                        # Update the annotations in the receipt data
                        filtered_small_annotations += (original_annotation_count - len(filtered_annotations))
                        receipt_data['annotations'] = filtered_annotations
                    
                    # Skip documents with fewer than 2 annotations
                    if 'annotations' not in receipt_data or len(receipt_data['annotations']) < 2:
                        filtered_sparse_documents += 1
                        continue
                    
                    # Load image if images_dir is provided
                    receipt_image = None
                    feature_map = None
                    if self.images_dir and 'file_name' in receipt_data:
                        image_path = os.path.join(self.images_dir, receipt_data['file_name'])
                        if os.path.exists(image_path):
                            try:
                                # Load image
                                receipt_image = Image.open(image_path).convert('RGB')
                                
                                # Extract full image feature map
                                feature_map = self._extract_image_features(receipt_image)
                                
                            except Exception as e:
                                logger.warning(f"Error processing image {image_path}: {e}")
                        else:
                            logger.warning(f"Image not found: {image_path}")
                    
                    # If no image or feature map, increment counter
                    if receipt_image is None or feature_map is None:
                        skipped_image_extraction += 1
                    
                    # Create graph for the filtered document
                    graph_data = self._create_hetero_graph(receipt_data, receipt_image, feature_map)
                    
                    if self.pre_transform is not None:
                        graph_data = self.pre_transform(graph_data)
                    
                    # Store file name as metadata
                    if 'file_name' in receipt_data:
                        graph_data.metadata = {'file_name': receipt_data['file_name']}
                    
                    data_list.append(graph_data)
                except Exception as e:
                    logger.error(f"Error processing receipt {i+j}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            receipts_per_second = len(batch_lines) / batch_duration
            logger.info(f"Batch {i//batch_size+1} took {batch_duration:.2f}s ({receipts_per_second:.2f} receipts/s)")
            
            # Clear CUDA cache periodically to prevent OOM errors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Print filtering statistics
        logger.info(f"Total receipts processed: {total_receipts}")
        logger.info(f"Small annotations filtered out: {filtered_small_annotations}")
        logger.info(f"Documents with fewer than 2 annotations filtered out: {filtered_sparse_documents}")
        logger.info(f"Documents with skipped image extraction: {skipped_image_extraction}")
        logger.info(f"Final receipts in dataset: {len(data_list)}")
        
        logger.info(f"Successfully processed {len(data_list)} receipts in {total_duration:.2f}s ({len(data_list)/total_duration:.2f} receipts/s)")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _create_hetero_graph(self, receipt_data, image=None, feature_map=None):
        """Create a heterogeneous graph for a receipt with three edge types."""
        annotations = receipt_data.get('annotations', [])
        data = HeteroData()
        
        if not annotations:
            # Create an empty heterogeneous graph
            data['node'].x = torch.zeros((0, 768), dtype=torch.float)  # BERT embedding size
            data['node'].bbox = torch.zeros((0, 4), dtype=torch.long)
            data['node'].y = torch.zeros(0, dtype=torch.long)
            data['node'].img_x = torch.zeros((0, self.image_embedding_dim), dtype=torch.float)  # Image embedding size
            data['node', 'spatial', 'node'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data['node', 'spatial', 'node'].edge_attr = torch.zeros((0, 1), dtype=torch.float)
            data['node', 'textual', 'node'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data['node', 'textual', 'node'].edge_attr = torch.zeros((0, 1), dtype=torch.float)
            data['node', 'directed', 'node'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data['node', 'directed', 'node'].edge_attr = torch.zeros((0, 1), dtype=torch.float)
            return data
        
        # Get image dimensions
        width = receipt_data.get('width', 300)
        height = receipt_data.get('height', 300)
        
        # Extract text and boxes
        texts = [anno.get('text', '') for anno in annotations]
        boxes = [anno.get('box', [0, 0, 0, 0, 0, 0, 0, 0]) for anno in annotations]
        labels = [anno.get('label', 0) for anno in annotations]
        
        # Handle empty texts
        texts = [t if t else " " for t in texts]
        
        # Process bounding boxes to LayoutLM format
        normalized_boxes = [self._process_bbox(box, width, height) for box in boxes]
        
        # Sort boxes by reading order (top-to-bottom, left-to-right)
        boxes_with_idx = [(i, normalized_boxes[i][1], normalized_boxes[i][0]) for i in range(len(normalized_boxes))]  # (idx, y, x)
        sorted_indices = [item[0] for item in sorted(boxes_with_idx, key=lambda x: (x[1], x[2]))]
        
        # Reorder boxes, texts, labels based on sorted indices
        normalized_boxes = [normalized_boxes[i] for i in sorted_indices]
        original_boxes = [boxes[i] for i in sorted_indices]
        texts = [texts[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        
        # Process text and extract BERT embeddings in batches for efficiency
        start_time = time.time()
        text_embeddings = self._batch_process_texts(texts, batch_size=64)
        end_time = time.time()
        # logger.info(f"BERT embedding took {end_time - start_time:.2f} seconds")
        
        # Extract image embeddings using RoI pooling if image and feature map are available
        image_embeddings = []
        if image is not None and feature_map is not None:
            start_time = time.time()
            
            # Extract ROI features from the feature map
            image_embeddings = self._extract_roi_features(
                feature_map, original_boxes, width, height)
                
            end_time = time.time()
            # logger.info(f"RoI pooling took {end_time - start_time:.2f} seconds")
        else:
            # If no image or feature map, use zero vectors
            image_embeddings = [torch.zeros(self.image_embedding_dim) for _ in range(len(texts))]
        
        # The rest of the method is the same as before...
        # Calculate spatial centers and areas for edge creation
        centers = np.array([self._calculate_center(box) for box in normalized_boxes])
        areas = np.array([self._calculate_box_area(box) for box in normalized_boxes])
        
        # Create spatial edges using k-nearest neighbors
        k = min(self.k_spatial, len(centers)-1) if len(centers) > 1 else 0
        spatial_edges = self._compute_knn(centers, k)
        
        # Calculate actual distances for edge attributes
        spatial_distances = []
        if len(spatial_edges.shape) > 1 and spatial_edges.shape[1] > 0:
            for i in range(spatial_edges.shape[1]):
                src, dst = spatial_edges[0, i], spatial_edges[1, i]
                dist = np.sqrt(np.sum((centers[src] - centers[dst])**2))
                # Normalize by sqrt of area
                area_factor = np.sqrt(areas[src] + areas[dst])
                normalized_dist = dist / max(area_factor, 1e-6)
                spatial_distances.append(normalized_dist)
        
        # Apply spatial threshold if specified
        if self.spatial_threshold is not None and len(spatial_distances) > 0:
            spatial_distances = np.array(spatial_distances)
            mask = spatial_distances < self.spatial_threshold
            spatial_edges = spatial_edges[:, mask]
            spatial_distances = spatial_distances[mask]
        
        # Calculate text similarity
        if len(texts) > 1:
            vectorizer = TfidfVectorizer(min_df=1)
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                text_similarities = (tfidf_matrix * tfidf_matrix.T).toarray()
                np.fill_diagonal(text_similarities, 0)  # Remove self-connections
            except:
                # Fallback if TF-IDF fails
                text_similarities = np.zeros((len(texts), len(texts)))
        else:
            text_similarities = np.zeros((len(texts), len(texts)))
        
        # Create textual edges
        textual_edges = []
        similarity_values = []
        for i in range(len(texts)):
            similarities = text_similarities[i]
            # Apply similarity threshold
            valid_indices = np.where(similarities > self.textual_threshold)[0]
            # Sort by similarity and take top k
            if len(valid_indices) > 0:
                sorted_indices = np.argsort(similarities[valid_indices])
                top_indices = valid_indices[sorted_indices[-min(self.k_textual, len(valid_indices)):]]
                for j in top_indices:
                    if i != j:  # Avoid self-loops
                        textual_edges.append([i, j])
                        similarity_values.append(similarities[j])
        
        # Convert textual edges to tensor format
        if textual_edges:
            textual_edges = np.array(textual_edges).T
            similarity_values = np.array(similarity_values)
        else:
            textual_edges = np.zeros((2, 0), dtype=np.int64)
            similarity_values = np.array([])
        
        # Create additional directed edges with angle weights
        directed_edges = []
        directed_weights = []
        
        for i, box1 in enumerate(normalized_boxes):
            for j, box2 in enumerate(normalized_boxes):
                if i != j and self._is_connected(box1, box2, normalized_boxes):
                    directed_edges.append((i, j))
                    angle = self._compute_angle(box1, box2)
                    directed_weights.append(angle)
        
        # Convert directed edges to tensor format
        if directed_edges:
            directed_edge_index = torch.tensor(directed_edges).t()
            directed_edge_attr = torch.tensor(directed_weights).view(-1, 1)
        else:
            directed_edge_index = torch.zeros((2, 0), dtype=torch.long)
            directed_edge_attr = torch.zeros((0, 1))
        
        # Convert all data to tensors
        bbox_tensor = torch.tensor(normalized_boxes, dtype=torch.long)
        text_embeddings_tensor = torch.stack(text_embeddings)
        image_embeddings_tensor = torch.stack(image_embeddings)
        node_labels = torch.tensor(labels, dtype=torch.long)
        spatial_edge_index = torch.tensor(spatial_edges, dtype=torch.long)
        spatial_edge_attr = torch.tensor(spatial_distances, dtype=torch.float).view(-1, 1)
        textual_edge_index = torch.tensor(textual_edges, dtype=torch.long)
        textual_edge_attr = torch.tensor(similarity_values, dtype=torch.float).view(-1, 1)
        
        # Populate the heterogeneous data object
        data['node'].x = text_embeddings_tensor
        data['node'].img_x = image_embeddings_tensor  # Add image embeddings
        data['node'].bbox = bbox_tensor
        data['node'].y = node_labels
        data['node'].raw_text = texts  # Store original text for reference
        
        # Add all three types of edges
        data['node', 'spatial', 'node'].edge_index = spatial_edge_index
        data['node', 'spatial', 'node'].edge_attr = spatial_edge_attr
        data['node', 'textual', 'node'].edge_index = textual_edge_index
        data['node', 'textual', 'node'].edge_attr = textual_edge_attr
        data['node', 'directed', 'node'].edge_index = directed_edge_index
        data['node', 'directed', 'node'].edge_attr = directed_edge_attr
        
        return data
    
    # Rest of the methods (left the same for brevity)
    def _compute_knn(self, centers, k):
        """Compute k-nearest neighbors using scikit-learn."""
        n = centers.shape[0]
        
        # Handle edge cases
        if k <= 0 or n <= 1:
            return np.zeros((2, 0), dtype=np.int64)
        
        # Adjust k to be at most n-1 (can't be neighbors with yourself)
        k = min(k, n-1)
        
        # Initialize the NearestNeighbors model
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
        nn.fit(centers)
        
        # Query the model - returns distances and indices
        # We use k+1 because the first neighbor will be the point itself
        distances, indices = nn.kneighbors(centers)
        
        # Create edge list - skip the first index (which is the point itself)
        edges = []
        for i in range(n):
            for j in range(1, k+1):  # Skip the first neighbor (itself)
                edges.append((i, indices[i, j]))
        
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        
        return np.array(edges).T

    def _process_bbox(self, box, width, height):
        """Convert OCR bounding box to LayoutLM format."""
        # Extract coordinates to get top-left and bottom-right corners
        x_coords = [box[0], box[2], box[4], box[6]]
        y_coords = [box[1], box[3], box[5], box[7]]
        
        # Get top-left (x0, y0) and bottom-right (x1, y1)
        x0, y0 = min(x_coords), min(y_coords)  
        x1, y1 = max(x_coords), max(y_coords)  
        
        # Normalize to 0-1000 range as in LayoutLM
        x0_norm = min(max(0, int(x0 * 1000 / width)), 1000)
        y0_norm = min(max(0, int(y0 * 1000 / height)), 1000)
        x1_norm = min(max(0, int(x1 * 1000 / width)), 1000)
        y1_norm = min(max(0, int(y1 * 1000 / height)), 1000)
        
        return [x0_norm, y0_norm, x1_norm, y1_norm]
    
    def _batch_process_texts(self, texts, batch_size=64):
        """Process texts in batches to efficiently use GPU."""
        # Handle empty texts
        processed_texts = [t if t and not t.isspace() else "[UNK]" for t in texts]
        
        # Process in batches
        embeddings = []
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i+batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer.batch_encode_plus(
                batch_texts,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move tensors to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.bert_model(input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _calculate_center(self, box):
        """Calculate the center point of a bounding box."""
        # For 4-coordinate box (x0, y0, x1, y1)
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        return np.array([x_center, y_center])
    
    def _calculate_box_area(self, box):
        """Calculate the area of a bounding box."""
        # For 4-coordinate box (x0, y0, x1, y1)
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width * height
    
    def _compute_angle(self, box1, box2):
        """Compute angle between centers of two bounding boxes."""
        center1 = self._calculate_center(box1)
        center2 = self._calculate_center(box2)
        
        # Compute relative angle
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        return math.atan2(dy, dx)
    
    def _is_connected(self, box1, box2, all_boxes):
        """Check if two bounding boxes are connected without any other boxes in between."""
        # Create polygon connecting the two boxes
        polygon = Polygon([
            (box1[0], box1[1]),  # Top-left of box1
            (box1[0], box2[1]),  # Top-left x with box2 y
            (box2[2], box2[1]),  # Bottom-right of box2
            (box2[2], box1[1])   # Box2 x with box1 y
        ])
        
        # Check if any other box falls inside this polygon
        for other_box in all_boxes:
            if np.array_equal(other_box, box1) or np.array_equal(other_box, box2):
                continue
                
            if polygon.is_valid:
                x, y, x2, y2 = other_box
                points = [(x, y), (x2, y), (x2, y2), (x, y2)]
                
                # If any point of the other box is in the polygon,
                # the boxes are not directly connected
                if any(polygon.contains(Point(p)) for p in points):
                    return False
                    
        return True


def create_datasets(train_path, test_path, images_dir=None, root_dir="./data", 
                   k_spatial=5, k_textual=3,
                   spatial_threshold=None, textual_threshold=0.5,
                   max_seq_length=64, bert_model_name='bert-base-uncased',
                   vision_model_name='resnet18',
                   roi_output_size=(7, 7), use_roi_align=True,
                   force_reload=False, use_gpu=True):
    """
    Create and save the train and test datasets with RoI pooling for image features.
    
    Args:
        train_path: Path to the training data file
        test_path: Path to the test data file
        images_dir: Directory containing the receipt images
        root_dir: Directory to save the processed data
        k_spatial: Number of spatial neighbors
        k_textual: Number of textual neighbors
        spatial_threshold: Threshold for spatial connections (normalized by box area)
        textual_threshold: Threshold for textual similarity
        max_seq_length: Maximum sequence length for tokenization
        bert_model_name: BERT model name to use for tokenization
        vision_model_name: Model to use for image feature extraction
        roi_output_size: Size of the output from RoI pooling
        use_roi_align: Whether to use RoI Align instead of RoI Pool
        force_reload: Whether to force reprocessing of the data
        use_gpu: Whether to use GPU for embeddings
        
    Returns:
        train_dataset, test_dataset: The created datasets
    """
    # Get list of supported vision models
    supported_models = FeatureExtractor.get_supported_models()
    if vision_model_name not in supported_models:
        raise ValueError(f"Vision model {vision_model_name} not supported. "
                         f"Choose one of: {', '.join(supported_models)}")
    
    logger.info(f"Creating datasets using vision model: {vision_model_name}")
    
    # Create train dataset
    logger.info(f"Creating training dataset from {train_path}")
    train_dataset = ReceiptGraphDataset(
        root=os.path.join(root_dir, 'train'),
        file_path=train_path,
        images_dir=images_dir,
        k_spatial=k_spatial,
        k_textual=k_textual,
        spatial_threshold=spatial_threshold,
        textual_threshold=textual_threshold,
        max_seq_length=max_seq_length,
        bert_model_name=bert_model_name,
        vision_model_name=vision_model_name,
        roi_output_size=roi_output_size,
        use_roi_align=use_roi_align,
        force_reload=force_reload,
        use_gpu=use_gpu
    )
    
    # Create test dataset
    logger.info(f"Creating test dataset from {test_path}")
    test_dataset = ReceiptGraphDataset(
        root=os.path.join(root_dir, 'test'),
        file_path=test_path,
        images_dir=images_dir,
        k_spatial=k_spatial,
        k_textual=k_textual,
        spatial_threshold=spatial_threshold,
        textual_threshold=textual_threshold,
        max_seq_length=max_seq_length,
        bert_model_name=bert_model_name,
        vision_model_name=vision_model_name,
        roi_output_size=roi_output_size,
        use_roi_align=use_roi_align,
        force_reload=force_reload,
        use_gpu=use_gpu
    )
    
    return train_dataset, test_dataset


def analyze_dataset(dataset, num_samples=5):
    """
    Analyze the created dataset and print statistics.
    
    Args:
        dataset: The PyG dataset
        num_samples: Number of samples to analyze
    """
    # Basic statistics
    num_graphs = len(dataset)
    logger.info(f"Dataset contains {num_graphs} graphs")
    
    # Node statistics
    node_counts = []
    edge_counts = {'spatial': [], 'textual': [], 'directed': []}
    label_dist = {}
    
    for i in range(min(num_graphs, num_samples)):
        data = dataset[i]
        
        # For heterogeneous graphs
        if hasattr(data['node'], 'x'):
            num_nodes = data['node'].x.shape[0]
        else:
            num_nodes = 0
            
        spatial_edges = data['node', 'spatial', 'node'].edge_index.shape[1] if hasattr(data['node', 'spatial', 'node'], 'edge_index') else 0
        textual_edges = data['node', 'textual', 'node'].edge_index.shape[1] if hasattr(data['node', 'textual', 'node'], 'edge_index') else 0
        directed_edges = data['node', 'directed', 'node'].edge_index.shape[1] if hasattr(data['node', 'directed', 'node'], 'edge_index') else 0
        
        # Label distribution
        if hasattr(data['node'], 'y'):
            for label in data['node'].y.tolist():
                label_dist[label] = label_dist.get(label, 0) + 1
        
        node_counts.append(num_nodes)
        edge_counts['spatial'].append(spatial_edges)
        edge_counts['textual'].append(textual_edges)
        edge_counts['directed'].append(directed_edges)
        
        logger.info(f"Graph {i}: {num_nodes} nodes, {spatial_edges} spatial edges, {textual_edges} textual edges, {directed_edges} directed edges")
    
    # Print average statistics
    if node_counts:
        logger.info(f"Average number of nodes: {np.mean(node_counts):.2f}")
        logger.info(f"Average number of spatial edges: {np.mean(edge_counts['spatial']):.2f}")
        logger.info(f"Average number of textual edges: {np.mean(edge_counts['textual']):.2f}")
        logger.info(f"Average number of directed edges: {np.mean(edge_counts['directed']):.2f}")
    
    # Print label distribution
    logger.info("Label distribution:")
    label_map = {
        0: 'Ignore', 1: 'Store_name_value', 2: 'Store_name_key',
        3: 'Store_addr_value', 4: 'Store_addr_key', 5: 'Tel_value',
        6: 'Tel_key', 7: 'Date_value', 8: 'Date_key', 9: 'Time_value',
        10: 'Time_key', 11: 'Prod_item_value', 12: 'Prod_item_key',
        13: 'Prod_quantity_value', 14: 'Prod_quantity_key', 15: 'Prod_price_value',
        16: 'Prod_price_key', 17: 'Subtotal_value', 18: 'Subtotal_key',
        19: 'Tax_value', 20: 'Tax_key', 21: 'Tips_value', 22: 'Tips_key',
        23: 'Total_value', 24: 'Total_key', 25: 'Others'
    }
    
    for label, count in sorted(label_dist.items()):
        label_name = label_map.get(label, 'Unknown')
        logger.info(f"  {label} ({label_name}): {count}")


# Main execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process OCR receipt data into graph dataset')
    parser.add_argument('--train_file', type=str, default='train.txt', help='Path to the training data file')
    parser.add_argument('--test_file', type=str, default='test.txt', help='Path to the test data file')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory for processed data')
    parser.add_argument('--k_spatial', type=int, default=5, help='Number of spatial neighbors')
    parser.add_argument('--k_textual', type=int, default=3, help='Number of textual neighbors')
    parser.add_argument('--spatial_threshold', type=float, default=None, help='Threshold for spatial connections')
    parser.add_argument('--textual_threshold', type=float, default=0.5, help='Threshold for textual similarity')
    # Add LayoutLM-specific arguments
    parser.add_argument('--max_seq_length', type=int, default=64, help='Maximum sequence length for tokenization')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='BERT model name for tokenization')
    parser.add_argument('--visualize', action='store_true', help='Visualize some samples from the dataset')
    parser.add_argument('--num_vis', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU for BERT embeddings')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing text embeddings')
    
    args = parser.parse_args()
    
    # Create the datasets
    train_dataset, test_dataset = create_datasets(
        train_path=args.train_file,
        test_path=args.test_file,
        root_dir=args.output_dir,
        k_spatial=args.k_spatial,
        k_textual=args.k_textual,
        spatial_threshold=args.spatial_threshold,
        textual_threshold=args.textual_threshold,
        max_seq_length=args.max_seq_length,
        model_name=args.model_name
    )
    
    # Analyze the datasets
    logger.info("Analyzing training dataset:")
    analyze_dataset(train_dataset)
    
    logger.info("Analyzing test dataset:")
    analyze_dataset(test_dataset)
    
    # Visualize some samples if requested
    if args.visualize:
        logger.info(f"Visualizing {args.num_vis} samples from the training dataset")
        for i in range(min(args.num_vis, len(train_dataset))):
            plt = train_dataset.visualize_graph(i)
            plt.savefig(f"{args.output_dir}/train_graph_{i}.png")
            plt.close()
        
        logger.info(f"Visualizing {args.num_vis} samples from the test dataset")
        for i in range(min(args.num_vis, len(test_dataset))):
            plt = test_dataset.visualize_graph(i)
            plt.savefig(f"{args.output_dir}/test_graph_{i}.png")
            plt.close()
    
    logger.info("Dataset creation complete!")    
    
# python receipt_graph_dataset.py --train_file train.txt --test_file test.txt --output_dir ./data_layoutlm --max_seq_length 64 --model_name bert-base-uncased