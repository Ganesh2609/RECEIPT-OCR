# LayoutLM-GNN: Receipt Information Extraction

![Sample Result](sample.png)

A comprehensive deep learning project for information extraction from receipts using a combination of LayoutLM-inspired features, Graph Neural Networks, and advanced computer vision techniques.

## 🎯 Project Overview

This project implements a novel approach to receipt information extraction by combining:

1. **LayoutLM-inspired features**: Utilizing spatial and textual information from document images
2. **Graph Neural Networks**: Modeling relationships between text elements in receipts
3. **Multi-relational graphs**: Capturing spatial proximity, textual similarity, and directed relationships
4. **Advanced Pipeline**: Incorporating image denoising, perspective correction, and multi-modal processing

The model classifies text elements in receipts into semantic categories such as store name, address, product items, prices, totals, etc.

## 🏗️ Architecture Overview

### Core Pipeline
The project consists of two main approaches:

#### 1. Advanced Pipeline (Latest)
- **Image Denoising**: Restormer-based transformer architecture for document enhancement
- **Perspective Correction**: Automatic document rectification and bounding box adjustment
- **Multi-Modal Processing**: Vision transformers (ViT, CLIP, ConvNeXt) for image feature extraction
- **Graph Construction**: Multi-relational heterogeneous graphs with spatial, textual, and directed edges
- **Model Variants**: 
  - LayoutGNN (Base model)
  - LayoutImageTransformerGNN (With image features)
  - LayoutImageBertGNN (With BERT transformer integration)

#### 2. Original Pipeline
- **Document Denoising**: Restormer transformer model
- **OCR Processing**: DocTR framework (DBNet + CRNN)
- **Key Information Extraction**: SDMGR (Spatial Dual Modality Graph Reasoning)
- **Natural Language Queries**: Gemini Pro LLM for SQL generation

## 📊 Performance Metrics

### Advanced Pipeline Results
- **Classification Accuracy**: 90.57% on WildReceipt dataset
- **Best Model Performance**: LayoutImageBertGNN with vision features
- **Processing Speed**: ~2-3 seconds per receipt
- **Supported Vision Models**: ResNet, EfficientNet, ConvNeXt, ViT, Swin, CLIP

### Original Pipeline Results
- **Denoising Performance**: MSE reduced to 0.0116 by epoch 22
- **OCR Accuracy**: 76.97% using DocTR
- **KIE F1 Score**: 93% using SDMGR
- **End-to-end Processing Time**: ~2.5 seconds per receipt

## 🗂️ Data Structure

The project uses a custom data format for receipts, stored in text files (train.txt and test.txt). Each line represents a single receipt in JSON format:

```json
{
  "file_name": "path/to/image.jpeg",
  "height": 1200,
  "width": 1600,
  "annotations": [
    {
      "box": [x1, y1, x2, y2, x3, y3, x4, y4],
      "text": "SAFEWAY",
      "label": 1
    }
  ]
}
```

### Label Mapping

The model classifies text elements into 26 categories:

| Label | Description | Label | Description |
|-------|-------------|-------|-------------|
| 0 | Ignore | 13 | Prod_quantity_value |
| 1 | Store_name_value | 14 | Prod_quantity_key |
| 2 | Store_name_key | 15 | Prod_price_value |
| 3 | Store_addr_value | 16 | Prod_price_key |
| 4 | Store_addr_key | 17 | Subtotal_value |
| 5 | Tel_value | 18 | Subtotal_key |
| 6 | Tel_key | 19 | Tax_value |
| 7 | Date_value | 20 | Tax_key |
| 8 | Date_key | 21 | Tips_value |
| 9 | Time_value | 22 | Tips_key |
| 10 | Time_key | 23 | Total_value |
| 11 | Prod_item_value | 24 | Total_key |
| 12 | Prod_item_key | 25 | Others |

## 📁 Project Structure

```
LayoutLM-GNN/
├── Advanced Pipeline/           # Latest implementation
│   ├── layoutlm_gnn.py         # Base LayoutGNN model
│   ├── layoutlm_transformer_gnn.py  # Transformer variants
│   ├── receipt_graph_dataset.py     # Dataset processing
│   ├── feature_extraction_models.py # Vision model support
│   ├── train_layoutlm_gnn.py        # Training script
│   ├── train_bert_gnn.py            # BERT-GNN training
│   ├── roi_pooling.py               # RoI operations
│   ├── bbox-visualization.py        # Visualization tools
│   └── classification_report.txt    # Performance results
├── Denoising restformer/       # Image enhancement
│   ├── Restformer.py           # Main architecture
│   ├── RestformerBlocks.py     # Transformer blocks
│   ├── dataset.py              # Data loading
│   ├── trainer.py              # Training utilities
│   └── Results/                # Training visualizations
├── SDMGR/                      # Spatial Dual Modality
│   ├── VisionModel.py          # Vision processing
│   ├── transformer.py          # Text processing
│   ├── PreprocessingModule.py  # Multimodal fusion
│   └── unet.py                 # U-Net architecture
├── KIE/                        # Key Information Extraction
│   ├── model.py                # GCN model
│   ├── graph.py                # Graph construction
│   └── dataset.py              # SROIE dataset
└── LLM/                        # Language Model Integration
    ├── app.py                  # Flask API
    ├── gemini_functions.py     # LLM integration
    ├── database_functions.py   # SQL operations
    └── processing_functions.py # Query processing
```

## 🛠️ Technical Stack

### Deep Learning Frameworks
- **PyTorch**: Primary framework
- **PyTorch Geometric**: Graph neural networks
- **Transformers**: BERT, LayoutLM integration
- **Torchvision**: Vision models

### Vision Models Supported
- **ResNet**: ResNet18, ResNet50
- **EfficientNet**: B0, B3 variants
- **ConvNeXt**: ConvNeXt Small
- **Vision Transformers**: ViT-B/16
- **Swin Transformer**: Swin-T
- **CLIP**: CLIP-ViT-Base

### Key Libraries
- **NetworkX**: Graph operations
- **scikit-learn**: Machine learning utilities
- **OpenCV**: Image processing
- **PIL/Pillow**: Image handling
- **Flask**: Web API
- **SQLAlchemy**: Database operations

## 🚀 Installation & Setup

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# CUDA support (optional but recommended)
nvidia-smi
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/layoutlm-gnn.git
cd RECEIPT-OCR

# Install dependencies
pip install torch torch-geometric transformers networkx matplotlib seaborn scikit-learn tqdm opencv-python pillow flask sqlalchemy
```

### Additional Requirements
```bash
# For advanced features
pip install torchvision timm sentence-transformers google-generativeai
```

## 💻 Usage

### Quick Start
```bash
# Prepare your data
python Advanced\ Pipeline/receipt_graph_dataset.py \
    --train_file train.txt \
    --test_file test.txt \
    --output_dir ./data_layoutlm

# Train the model
python Advanced\ Pipeline/train_layoutlm_gnn.py \
    --data_dir ./data_layoutlm \
    --train_file train.txt \
    --test_file test.txt \
    --use_gat \
    --attention_heads 8 \
    --hidden_channels 256 \
    --num_layers 3 \
    --use_edge_features \
    --class_weighting \
    --learning_rate 5e-5 \
    --epochs 100 \
    --output_dir ./results_layoutlm
```

### Advanced Training with Image Features
```bash
# Train with vision features
python Advanced\ Pipeline/train_bert_gnn.py \
    --data_dir ./data_layoutlm \
    --train_file train.txt \
    --test_file test.txt \
    --images_dir ./receipt_images \
    --vision_model vit \
    --use_image_features \
    --bert_model bert-base-uncased \
    --max_seq_length 512 \
    --hidden_channels 256 \
    --gnn_layers 2 \
    --use_gat \
    --gnn_heads 8 \
    --dropout 0.3 \
    --learning_rate 3e-5 \
    --scheduler cosine \
    --output_dir ./results_vision_bert_gnn
```

### Resume Training
```bash
# Resume from checkpoint
python Advanced\ Pipeline/train_layoutlm_gnn.py \
    --data_dir ./data_layoutlm \
    --train_file train.txt \
    --test_file test.txt \
    --resume \
    --output_dir ./results_layoutlm
```

## 🔬 Model Architecture Details

### LayoutGNN (Base Model)
- **Text Processing**: Pre-computed BERT embeddings
- **Spatial Features**: LayoutLM-style 2D positional embeddings
- **Graph Structure**: 
  - Spatial edges based on proximity
  - Textual edges based on semantic similarity
  - Directed edges for sequential information
- **GNN Architecture**: Graph Attention Networks (GAT) with edge features

### LayoutImageTransformerGNN
- **Multimodal Input**: Text + Image + Spatial features
- **Vision Processing**: CNN/Transformer feature extraction with RoI pooling
- **Graph Processing**: Multi-relational heterogeneous graphs
- **Global Attention**: Custom transformer encoder for document-level reasoning

### LayoutImageBertGNN
- **BERT Integration**: Full BERT transformer for sequence modeling
- **Image Features**: Vision model integration with RoI alignment
- **Flexible Architecture**: Supports freezing/fine-tuning BERT layers
- **Advanced Features**: Differential learning rates for BERT vs. other components

## 📈 Training Results

### Classification Performance
```
                     precision    recall  f1-score   support
             Ignore     0.9856    0.9876    0.9866       969
   Store_name_value     0.8217    0.8190    0.8203       591
   Store_addr_value     0.8599    0.9016    0.8802       701
          Tel_value     0.9196    0.9533    0.9362       300
        Total_value     0.7347    0.6441    0.6864       576
        Total_key      0.8818    0.7691    0.8216       485
             Others     0.8876    0.9069    0.8971      6540

           accuracy                         0.9057     19267
```

### Denoising Results
The Restformer model achieved consistent improvement:
- **Epoch 1**: Training Loss = 0.0223
- **Epoch 10**: Training Loss = 0.0206
- **Epoch 22**: Training Loss = 0.0116

## 🔧 Advanced Features

### Multi-Vision Model Support
```python
# Supported vision models
supported_models = [
    'resnet18', 'resnet50', 
    'efficientnet_b0', 'efficientnet_b3',
    'convnext_small', 
    'vit', 'swin',
    'clip'
]
```

### Flexible Training Options
- **Learning Rate Scheduling**: Cosine, plateau, step, exponential
- **Class Weighting**: Automatic imbalanced class handling
- **Mixed Precision**: Automatic mixed precision training
- **Checkpoint Management**: Automatic saving and resuming

### Visualization Tools
- **Graph Visualization**: NetworkX-based graph plotting
- **Bounding Box Visualization**: Before/after perspective correction
- **Training Metrics**: Loss curves, confusion matrices
- **Classification Reports**: Detailed performance analysis

## 🎨 Visualization Examples

The project includes comprehensive visualization tools:
- **Document Graphs**: Spatial and textual relationships
- **Bounding Box Overlays**: OCR detection visualization
- **Training Progress**: Loss and accuracy curves
- **Confusion Matrices**: Classification performance analysis

## 🔍 API Integration

### Flask Web API
```python
# Start the API server
python LLM/app.py

# Upload images
curl -X POST -F "image=@receipt.jpg" http://localhost:5000/upload

# Query database
curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "Show me all receipts from last month"}' \
     http://localhost:5000/query
```

### LLM Integration
The system supports natural language queries through Gemini Pro:
- **SQL Generation**: Convert English to SQL queries
- **Plot Generation**: Create visualizations from queries
- **Image Retrieval**: Fetch specific receipt images

## 🧪 Experimental Features

### Perspective Correction
The advanced pipeline includes automatic perspective correction:
- **Homography Estimation**: SIFT/ORB feature matching
- **Bounding Box Transformation**: Automatic coordinate adjustment
- **Quality Improvement**: Enhanced OCR accuracy

## 🔮 Future Enhancements

1. **End-to-End Training**: Joint optimization of all components
2. **Real-time Processing**: Optimized inference pipeline
3. **Mobile Deployment**: TensorFlow Lite/ONNX conversion
4. **Multi-language Support**: Extend beyond English receipts
5. **Active Learning**: Semi-supervised learning for new domains