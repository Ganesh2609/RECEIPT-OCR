import os
import json
import re
import time
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import math
from shapely.geometry import Polygon, Point
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SROIEDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for SROIE receipt data.
    
    Converts SROIE OCR annotations into multi-relational graphs where nodes are text elements
    and edges represent spatial proximity, text similarity, and directed relationships.
    
    Uses pre-computed BERT embeddings and adds LayoutLM-inspired spatial features.
    """
    def __init__(self, root, train=True, transform=None, pre_transform=None, 
                 k_spatial=5, k_textual=3, 
                 spatial_threshold=None, textual_threshold=0.5,
                 max_seq_length=64, model_name='bert-base-uncased',
                 force_reload=False, use_gpu=True):
        
        self.train = train
        # Set paths for the SROIE directory structure
        self.box_dir = os.path.join(root, 'train' if train else 'test', 'box')
        self.entity_dir = os.path.join(root, 'train' if train else 'test', 'entities')
        self.img_dir = os.path.join(root, 'train' if train else 'test', 'img')
        
        # Other initialization parameters
        self.k_spatial = k_spatial
        self.k_textual = k_textual
        self.spatial_threshold = spatial_threshold
        self.textual_threshold = textual_threshold
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.force_reload = force_reload
        
        # Simple label mapping for SROIE dataset
        self.sroie_label_map = {
            'company': 1,    # Company_name
            'address': 3,    # Address
            'date': 7,       # Date
            'total': 23,     # Total
            'other': 25      # Others
        }
        
        # Check if GPU is available and requested
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        logger.info(f"Using device: {self.device} for BERT embeddings")
        
        # Initialize BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        # Move model to device
        self.bert_model = self.bert_model.to(self.device)
        # Set BERT to eval mode - we won't train it
        self.bert_model.eval()
        
        # Adjust root_dir for processed files
        self.data_type = 'train' if train else 'test'
        root_dir = os.path.join(root, f'processed_{self.data_type}')
        
        super(SROIEDataset, self).__init__(root_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        """Define raw file names based on the dataset directory structure."""
        return ['box', 'entities', 'img']

    @property
    def processed_file_names(self):
        """Define processed file names."""
        return [f'sroie_{self.data_type}_graph_data.pt']
    
    def download(self):
        # No download needed, the data files should already exist
        pass
    
    def process(self):
        """Process SROIE dataset files into graph data."""
        # Create output directory
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Get all box files
        box_files = [f for f in os.listdir(self.box_dir) if f.endswith('.txt')]
        logger.info(f"Found {len(box_files)} files in {self.box_dir}")
        
        data_list = []
        
        for box_file in tqdm(box_files, desc=f"Processing {self.data_type} receipts"):
            try:
                # Get base file name (without extension)
                file_base = os.path.splitext(box_file)[0]
                
                # Parse box file
                box_path = os.path.join(self.box_dir, box_file)
                boxes, texts = self._parse_box_file(box_path)
                
                # If no valid boxes were found, skip this file
                if not boxes or not texts:
                    logger.warning(f"No valid boxes/texts found in {box_file}, skipping")
                    continue
                
                # Get corresponding entity file
                entity_file = f"{file_base}.txt"
                entity_path = os.path.join(self.entity_dir, entity_file)
                
                # Check if entity file exists
                if not os.path.exists(entity_path):
                    entity_path = os.path.join(self.entity_dir, box_file)  # Try with same filename
                    
                if os.path.exists(entity_path):
                    entity_data = self._parse_entity_file(entity_path)
                else:
                    logger.warning(f"Entity file not found for {box_file}, using empty entity data")
                    entity_data = {}
                
                # Get image dimensions
                img_file = f"{file_base}.jpg"  # Try jpg
                img_path = os.path.join(self.img_dir, img_file)
                
                # Check for different image extensions if not found
                if not os.path.exists(img_path):
                    for ext in ['.jpeg', '.png', '.tif', '.tiff']:
                        alt_img_file = f"{file_base}{ext}"
                        alt_img_path = os.path.join(self.img_dir, alt_img_file)
                        if os.path.exists(alt_img_path):
                            img_path = alt_img_path
                            break
                
                # Get image dimensions
                if os.path.exists(img_path):
                    from PIL import Image
                    img = Image.open(img_path)
                    width, height = img.size
                else:
                    # Estimate dimensions from bounding boxes if image not found
                    max_x = max([max(box[0], box[2], box[4], box[6]) for box in boxes])
                    max_y = max([max(box[1], box[3], box[5], box[7]) for box in boxes])
                    width, height = max_x + 50, max_y + 50  # Add padding
                    logger.warning(f"Image not found for {box_file}, using estimated dimensions: {width}x{height}")
                
                # Map entities to texts
                labels = self._map_entities_to_texts(texts, boxes, entity_data)
                
                # Create receipt data in the format expected by _create_hetero_graph
                receipt_data = {
                    'file_name': box_file,
                    'width': width,
                    'height': height,
                    'annotations': [
                        {'box': box, 'text': text, 'label': label}
                        for box, text, label in zip(boxes, texts, labels)
                    ]
                }
                
                # Create graph
                graph_data = self._create_hetero_graph(receipt_data)
                
                if self.pre_transform is not None:
                    graph_data = self.pre_transform(graph_data)
                
                # Store file name as metadata
                graph_data.metadata = {'file_name': box_file}
                
                data_list.append(graph_data)
                
                # Optional: Visualize entity mapping for debugging
                # if len(data_list) <= 5:  # Only visualize first 5 for efficiency
                #     self.visualize_entity_mapping(box_path, entity_path)
                
            except Exception as e:
                logger.error(f"Error processing receipt {box_file}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Successfully processed {len(data_list)} receipts")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def _parse_box_file(self, file_path):
        """Parse SROIE box file format."""
        boxes = []
        texts = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 9:  # Format: x1,y1,x2,y2,x3,y3,x4,y4,text
                    try:
                        # Convert the first 8 elements to integers for the bounding box
                        box = [int(float(p)) for p in parts[:8]]
                        text = ','.join(parts[8:])  # In case text contains commas
                        
                        # Skip empty boxes or invalid coordinates
                        if all(coord >= 0 for coord in box) and text.strip():
                            boxes.append(box)
                            texts.append(text.strip())
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing line in {file_path}: {line.strip()}, Error: {e}")
        except Exception as e:
            logger.error(f"Error reading box file {file_path}: {e}")
        
        return boxes, texts

    def _parse_entity_file(self, file_path):
        """Parse SROIE entity JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Try to parse as JSON
                try:
                    entity_data = json.loads(content)
                    return entity_data
                except json.JSONDecodeError:
                    # If not valid JSON, try to parse manually
                    logger.warning(f"Invalid JSON in {file_path}, trying manual parsing")
                    entity_data = {}
                    
                    # Look for patterns like "company: XYZ" or "company": "XYZ"
                    patterns = {
                        'company': r'(?:company|company_name)["\s:]+([^"]+?)(?:"|,|\n|$)',
                        'date': r'(?:date)["\s:]+([^"]+?)(?:"|,|\n|$)',
                        'address': r'(?:address)["\s:]+([^"]+?)(?:"|,|\n|$)',
                        'total': r'(?:total)["\s:]+([^"]+?)(?:"|,|\n|$)'
                    }
                    
                    for entity, pattern in patterns.items():
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            entity_data[entity] = match.group(1).strip()
                    
                    return entity_data
                    
        except Exception as e:
            logger.error(f"Error parsing entity file {file_path}: {e}")
            return {}
    
    def _fuzzy_match(self, text1, text2, threshold=0.7):
        """
        Perform fuzzy matching between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            threshold: Similarity threshold (0-1)
            
        Returns:
            Boolean indicating if texts match above threshold
        """
        # Direct matching
        if text1 == text2:
            return True
        
        # Normalize for comparison
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Substring matching
        if t1 in t2 or t2 in t1:
            return True
        
        # Calculate string similarity
        similarity = SequenceMatcher(None, t1, t2).ratio()
        return similarity >= threshold
    
    def _find_ngram_matches(self, text, target, n=3):
        """Find matches using n-grams."""
        words1 = text.lower().split()
        words2 = target.lower().split()
        
        if len(words1) < n or len(words2) < n:
            return self._fuzzy_match(text, target)
        
        # Generate n-grams
        ngrams1 = [' '.join(words1[i:i+n]) for i in range(len(words1)-n+1)]
        ngrams2 = [' '.join(words2[i:i+n]) for i in range(len(words2)-n+1)]
        
        # Check for matching n-grams
        for ng1 in ngrams1:
            for ng2 in ngrams2:
                if self._fuzzy_match(ng1, ng2, threshold=0.8):
                    return True
        
        return False
    
    def _map_entities_to_texts(self, texts, boxes, entity_data):
        """
        Map entities to multiple text lines using a sophisticated approach.
        
        Args:
            texts: List of OCR text lines
            boxes: List of corresponding bounding boxes
            entity_data: Dictionary of entity data from JSON
            
        Returns:
            List of labels for each text line
        """
        labels = [25] * len(texts)  # Default all to "Others" (25)
        
        # Helper to calculate vertical position
        def get_y_center(box):
            y_coords = [box[1], box[3], box[5], box[7]]
            return sum(y_coords) / len(y_coords)
        
        # Helper to sort by vertical position
        def get_reading_order(idx):
            # Get box vertical center
            y_center = get_y_center(boxes[idx])
            # Get box horizontal position (left edge)
            x_left = min(boxes[idx][0], boxes[idx][2], boxes[idx][4], boxes[idx][6])
            
            # Sort primarily by y (with some tolerance for same line)
            # and secondarily by x for items on the same line
            y_bin = int(y_center / 20)  # Group lines within ~20 pixels vertically
            return (y_bin, x_left)
        
        # Sort text boxes by reading order (top-to-bottom, left-to-right)
        sorted_indices = sorted(range(len(boxes)), key=get_reading_order)
        sorted_texts = [texts[i] for i in sorted_indices]
        
        # ADDRESS HANDLING
        if 'address' in entity_data:
            address = entity_data['address']
            # Split address by common delimiters
            address_parts = re.split(r'[,.;]', address)
            address_parts = [p.strip() for p in address_parts if p.strip()]
            
            # Find address components in text lines
            address_indices = []
            
            for idx, text in enumerate(texts):
                # Try different matching approaches
                for part in address_parts:
                    # Try exact substring match first
                    if part.lower() in text.lower():
                        address_indices.append(idx)
                        break
                    # Try n-gram matching for longer parts
                    elif len(part.split()) >= 3 and self._find_ngram_matches(text, part):
                        address_indices.append(idx)
                        break
                    # Try fuzzy matching for shorter parts
                    elif self._fuzzy_match(text, part, threshold=0.8):
                        address_indices.append(idx)
                        break
            
            # Check for address keywords that might not be in the entity data
            address_keywords = ['street', 'avenue', 'road', 'lane', 'drive', 'st', 'ave', 'rd', 
                                'blvd', 'boulevard', 'jalan', 'lorong', 'no.', 'postal', 'city']
            for idx, text in enumerate(texts):
                if idx not in address_indices:
                    text_lower = text.lower()
                    if any(keyword in text_lower for keyword in address_keywords):
                        address_indices.append(idx)
            
            # Look for postal codes
            postal_pattern = r'\b\d{5,6}\b'
            for idx, text in enumerate(texts):
                if idx not in address_indices and re.search(postal_pattern, text):
                    # Check if any address line is nearby vertically
                    if any(abs(get_y_center(boxes[idx]) - get_y_center(boxes[i])) < 100 for i in address_indices):
                        address_indices.append(idx)
            
            # Label all address lines
            for idx in address_indices:
                labels[idx] = self.sroie_label_map['address']  # Address
        
        # COMPANY NAME HANDLING
        if 'company' in entity_data:
            company = entity_data['company']
            # Split company name by parentheses, etc.
            company_parts = re.split(r'[()]', company)
            company_parts = [p.strip() for p in company_parts if p.strip()]
            
            company_indices = []
            
            # Look primarily at the top of the receipt
            top_candidates = sorted_indices[:min(10, len(sorted_indices))]
            
            for idx in top_candidates:
                text = texts[idx]
                # Try different matching approaches
                for part in company_parts:
                    if self._fuzzy_match(text, part, threshold=0.8):
                        company_indices.append(idx)
                        break
                    elif len(part.split()) >= 3 and self._find_ngram_matches(text, part):
                        company_indices.append(idx)
                        break
            
            # Look for company signifiers
            company_signifiers = ['ltd', 'limited', 'sdn', 'bhd', 'inc', 'corporation', 'corp', 'pte', 'llc', 'co.']
            for idx in top_candidates:
                if idx not in company_indices:
                    text_lower = texts[idx].lower()
                    if any(signifier in text_lower for signifier in company_signifiers):
                        company_indices.append(idx)
            
            # Label all company name lines
            for idx in company_indices:
                labels[idx] = self.sroie_label_map['company']  # Company_name
        
        # DATE HANDLING
        if 'date' in entity_data:
            date = entity_data['date']
            # Common date formats
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY or MM/DD/YYYY
                r'\d{2,4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # DD Mon YYYY
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}'  # Mon DD, YYYY
            ]
            
            for idx, text in enumerate(texts):
                # Direct match with the date string
                if date in text:
                    labels[idx] = self.sroie_label_map['date']  # Date
                    continue
                
                # Check various date patterns
                for pattern in date_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        date_match = re.search(pattern, text, re.IGNORECASE).group(0)
                        # If this matches our target date or is close
                        if self._fuzzy_match(date_match, date, threshold=0.7):
                            labels[idx] = self.sroie_label_map['date']  # Date
                            break
        
        # TOTAL HANDLING
        if 'total' in entity_data:
            total = entity_data['total']
            total_pattern = re.escape(total)  # Escape any special characters
            
            # Look for exact total matches
            for idx, text in enumerate(texts):
                if re.search(r'\b' + total_pattern + r'\b', text):
                    # Check if "TOTAL" appears in this line or just before
                    if 'TOTAL' in text.upper():
                        labels[idx] = self.sroie_label_map['total']  # Total
                    elif idx > 0 and 'TOTAL' in texts[idx-1].upper():
                        labels[idx] = self.sroie_label_map['total']  # Total
                    # Also consider position - totals are typically near the bottom
                    elif sorted_indices.index(idx) > len(sorted_indices) * 0.6:
                        labels[idx] = self.sroie_label_map['total']  # Total
            
            # Look for lines with "TOTAL" that have a number close to our target
            if total.replace('.', '').isdigit():  # If it's a numeric total
                total_value = float(total)
                for idx, text in enumerate(texts):
                    if 'TOTAL' in text.upper():
                        # Extract numbers from the text
                        numbers = re.findall(r'\d+\.\d+|\d+', text)
                        for num in numbers:
                            try:
                                value = float(num)
                                # If close to our target total (within 10%)
                                if abs(value - total_value) / total_value < 0.1:
                                    labels[idx] = self.sroie_label_map['total']  # Total
                                    break
                            except ValueError:
                                continue
        
        # Additional post-processing
        # Handle adjacent lines with same entity type
        for i in range(1, len(texts)):
            # If a line is unlabeled but adjacent to labeled lines of the same type
            if labels[i] == 25 and i > 0 and i < len(texts) - 1:
                # Check if surrounding lines have the same label and are vertically close
                prev_label = labels[i-1]
                next_label = labels[i+1] if i < len(texts) - 1 else 25
                
                if prev_label != 25 and prev_label == next_label:
                    # Check vertical proximity
                    if abs(get_y_center(boxes[i]) - get_y_center(boxes[i-1])) < 30:
                        labels[i] = prev_label
        
        return labels
    
    def _compute_knn(self, centers, k):
        """
        Compute k-nearest neighbors using scikit-learn.
        
        Args:
            centers: Array of center points [n_samples, n_features]
            k: Number of neighbors to find for each point
            
        Returns:
            Array of shape [2, n_edges] containing (source, target) pairs
        """
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
        """
        Convert OCR bounding box to LayoutLM format.
        
        Args:
            box: Original 8-point box coordinates [x1,y1,x2,y2,x3,y3,x4,y4]
            width: Image width
            height: Image height
            
        Returns:
            List of normalized coordinates [x0,y0,x1,y1] in LayoutLM format (0-1000 range)
        """
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
    
    def _process_text(self, text):
        """
        Process text and extract BERT embeddings directly.
        
        Args:
            text: Text string to process
            
        Returns:
            BERT embedding tensor for the text
        """
        # Handle empty text
        if not text or text.isspace():
            text = "[UNK]"
        
        # Tokenize and truncate/pad
        encoded = self.tokenizer.encode_plus(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to the appropriate device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Extract embeddings with no_grad to ensure no training
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            # Get [CLS] token embedding as text representation
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
        
        return embedding
    
    def _batch_process_texts(self, texts, batch_size=64):
        """
        Process texts in batches to efficiently use GPU.
        
        Args:
            texts: List of text strings to process
            batch_size: Batch size for processing
            
        Returns:
            List of BERT embedding tensors
        """
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
    
    def _create_hetero_graph(self, receipt_data):
        """Create a heterogeneous graph for a receipt with three edge types."""
        annotations = receipt_data.get('annotations', [])
        data = HeteroData()
        
        if not annotations:
            # Create an empty heterogeneous graph
            data['node'].x = torch.zeros((0, 768), dtype=torch.float)  # BERT embedding size
            data['node'].bbox = torch.zeros((0, 4), dtype=torch.long)
            data['node'].y = torch.zeros(0, dtype=torch.long)
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
        texts = [texts[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        
        # Process text and extract BERT embeddings in batches for efficiency
        # logger.info(f"Processing {len(texts)} text elements with BERT")
        start_time = time.time()
        text_embeddings = self._batch_process_texts(texts, batch_size=64)
        end_time = time.time()
        # logger.info(f"BERT embedding took {end_time - start_time:.2f} seconds")
        
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
        node_labels = torch.tensor(labels, dtype=torch.long)
        spatial_edge_index = torch.tensor(spatial_edges, dtype=torch.long)
        spatial_edge_attr = torch.tensor(spatial_distances, dtype=torch.float).view(-1, 1)
        textual_edge_index = torch.tensor(textual_edges, dtype=torch.long)
        textual_edge_attr = torch.tensor(similarity_values, dtype=torch.float).view(-1, 1)
        
        # Populate the heterogeneous data object
        data['node'].x = text_embeddings_tensor
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
    
    def visualize_entity_mapping(self, box_file, entity_file, output_dir="./visualizations"):
        """
        Create a visual representation of the entity mapping.
        
        Args:
            box_file: Path to the box file
            entity_file: Path to the entity file
            output_dir: Directory to save visualizations
        """
        # Parse files
        boxes, texts = self._parse_box_file(box_file)
        entity_data = self._parse_entity_file(entity_file)
        
        # Get image path
        file_base = os.path.splitext(os.path.basename(box_file))[0]
        
        # Try different image extensions
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            path = os.path.join(self.img_dir, f"{file_base}{ext}")
            if os.path.exists(path):
                img_path = path
                break
                
        if not img_path:
            logger.warning(f"Image not found for {file_base}, skipping visualization")
            return
        
        # Map entities to texts
        labels = self._map_entities_to_texts(texts, boxes, entity_data)
        
        # Load image
        img = Image.open(img_path)
        fig, ax = plt.subplots(figsize=(12, 18))
        ax.imshow(img)
        
        # Define colors for different entity types
        colors = {
            1: 'red',       # Company
            3: 'blue',      # Address
            7: 'green',     # Date
            23: 'purple',   # Total
            25: 'gray'      # Others
        }
        
        # Reverse label map
        label_names = {
            1: 'Company',
            3: 'Address',
            7: 'Date',
            23: 'Total',
            25: 'Others'
        }
        
        # Draw boxes with colors based on entity type
        for box, text, label in zip(boxes, texts, labels):
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            
            # Create polygon
            polygon = patches.Polygon([
                (x1, y1), (x2, y2), (x3, y3), (x4, y4)
            ], closed=True, fill=False, edgecolor=colors.get(label, 'gray'), linewidth=2)
            
            ax.add_patch(polygon)
            
            # Add text label if not "Others"
            if label != 25:
                entity_type = label_names.get(label, "Unknown")
                ax.text(x1, y1-10, f"{entity_type}: {text[:20]}...", 
                       color='white', bbox={'facecolor': colors.get(label), 'alpha': 0.8})
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{file_base}_mapping.png")
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")

# Main execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process SROIE receipt data into graph dataset')
    parser.add_argument('--data_dir', type=str, default='./SROIE2019', 
                        help='Path to the SROIE dataset directory')
    parser.add_argument('--output_dir', type=str, default='./data_sroie', 
                        help='Output directory for processed data')
    parser.add_argument('--k_spatial', type=int, default=5, 
                        help='Number of spatial neighbors')
    parser.add_argument('--k_textual', type=int, default=3, 
                        help='Number of textual neighbors')
    parser.add_argument('--spatial_threshold', type=float, default=None, 
                        help='Threshold for spatial connections')
    parser.add_argument('--textual_threshold', type=float, default=0.5, 
                        help='Threshold for textual similarity')
    parser.add_argument('--max_seq_length', type=int, default=64, 
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', 
                        help='BERT model name for tokenization')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize some samples from the dataset')
    parser.add_argument('--num_vis', type=int, default=3, 
                        help='Number of samples to visualize')
    parser.add_argument('--use_gpu', action='store_true', default=True, 
                        help='Use GPU for BERT embeddings')
    parser.add_argument('--force_reload', action='store_true', 
                        help='Force reprocessing of the data')
    
    args = parser.parse_args()
    
    # Create the datasets
    logger.info("Creating training dataset...")
    train_dataset = SROIEDataset(
        root=args.data_dir,
        train=True,
        k_spatial=args.k_spatial,
        k_textual=args.k_textual,
        spatial_threshold=args.spatial_threshold,
        textual_threshold=args.textual_threshold,
        max_seq_length=args.max_seq_length,
        model_name=args.model_name,
        force_reload=args.force_reload,
        use_gpu=args.use_gpu
    )
    
    logger.info("Creating test dataset...")
    test_dataset = SROIEDataset(
        root=args.data_dir,
        train=False,
        k_spatial=args.k_spatial,
        k_textual=args.k_textual,
        spatial_threshold=args.spatial_threshold,
        textual_threshold=args.textual_threshold,
        max_seq_length=args.max_seq_length,
        model_name=args.model_name,
        force_reload=args.force_reload,
        use_gpu=args.use_gpu
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    
    # Visualize some samples if requested
    if args.visualize:
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        logger.info(f"Visualizing {min(args.num_vis, len(train_dataset))} samples from training set")
        for i in range(min(args.num_vis, len(train_dataset))):
            # Get file name from metadata
            file_name = train_dataset[i].metadata['file_name']
            box_path = os.path.join(args.data_dir, 'train', 'box', file_name)
            entity_path = os.path.join(args.data_dir, 'train', 'entities', file_name)
            
            # Visualize entity mapping
            train_dataset.visualize_entity_mapping(box_path, entity_path, output_dir=vis_dir)
    
    logger.info("Dataset creation complete!")