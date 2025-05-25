import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_t, Swin_T_Weights
)
from transformers import (
    CLIPProcessor, CLIPVisionModel
)

class FeatureExtractor:
    """
    Factory class to create and configure different feature extraction models.
    
    Supports multiple vision models including ResNet, EfficientNet, ConvNeXt,
    Vision Transformers (ViT), Swin Transformer, CLIP, and LayoutLM.
    """
    @staticmethod
    def create_model(model_name, device='cuda'):
        """
        Create a feature extraction model based on the model name.
        
        Args:
            model_name: Name of the model (e.g., 'resnet18', 'vit', 'clip')
            device: Device to move the model to ('cuda' or 'cpu')
            
        Returns:
            model: Feature extraction model
            transform: Preprocessing transforms
            feature_dim: Dimension of the output features
            needs_fixed_size: Whether the model requires fixed-size input
            input_size: Required input size if needs_fixed_size is True
            is_transformer: Whether the model uses a transformer architecture
        """
        # Default parameters
        needs_fixed_size = True
        is_transformer = False
        
        # ResNet models
        if model_name == 'resnet18':
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model = nn.Sequential(*list(model.children())[:-2])  # Remove avg pool and FC
            feature_dim = 512
            input_size = (224, 224)
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        elif model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            model = nn.Sequential(*list(model.children())[:-2])
            feature_dim = 2048
            input_size = (224, 224)
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        # EfficientNet models
        elif model_name == 'efficientnet_b0':
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # Remove classifier
            model.classifier = nn.Identity()
            feature_dim = 1280
            input_size = (224, 224)
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        elif model_name == 'efficientnet_b3':
            model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            model.classifier = nn.Identity()
            feature_dim = 1536
            input_size = (300, 300)
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        # ConvNeXt models
        elif model_name == 'convnext_small':
            model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            # Remove classifier
            model.classifier = nn.Identity()
            feature_dim = 768
            input_size = (224, 224)
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        # Vision Transformer models
        elif model_name == 'vit':
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            # Modified forward to return patch embeddings
            original_forward = model.forward
            
            def new_forward(x):
                # Run model up to the point where we get patch embeddings
                x = model._process_input(x)
                n = x.shape[0]
                
                # Drop class token
                # x = model.dropout(x)
                x = model.encoder(x)
                
                # Reshape to feature map (remove sequence dimension, organize as spatial grid)
                # For ViT-B/16, we have 14x14 patches for 224x224 images, each with dimension 768
                # Shape will be [batch_size, 768, 14, 14]
                grid_size = int(x.shape[1] ** 0.5)
                # Skip class token at position 0
                x = x[:, 1:, :]
                x = x.reshape(n, grid_size, grid_size, -1).permute(0, 3, 1, 2)
                return x
                
            model.forward = new_forward
            feature_dim = 768
            input_size = (224, 224)
            is_transformer = True
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        # Swin Transformer
        elif model_name == 'swin':
            model = swin_t(weights=Swin_T_Weights.DEFAULT)
            # Remove head
            model.head = nn.Identity()
            feature_dim = 768
            input_size = (224, 224)
            is_transformer = True
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        # CLIP Vision Encoder
        elif model_name == 'clip':
            model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            # We need to modify the forward to output spatial features
            original_forward = model.forward
            
            def new_forward(pixel_values):
                # Shape of embedded patches: [batch_size, n_patches, hidden_size]
                outputs = original_forward(pixel_values)
                last_hidden_state = outputs.last_hidden_state
                
                # Skip the [CLS] token and reshape to spatial grid for feature map
                # CLIP uses 7x7 patches for 224x224 images with the base model
                batch_size = last_hidden_state.shape[0]
                grid_size = 7
                hidden_size = 768
                
                # Skip CLS token and reshape
                patch_embeddings = last_hidden_state[:, 1:, :]
                feature_map = patch_embeddings.reshape(batch_size, grid_size, grid_size, hidden_size)
                feature_map = feature_map.permute(0, 3, 1, 2)  # [B, C, H, W]
                
                return feature_map
                
            model.forward = new_forward
            feature_dim = 768
            input_size = (224, 224)
            is_transformer = True
            
            # CLIP requires specific preprocessing
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])
            
            
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        return model, transform, feature_dim, needs_fixed_size, input_size, is_transformer

    @staticmethod
    def get_feature_map(model, image, transform, device='cuda', is_transformer=False):
        """
        Extract feature map from an image using the provided model.
        
        Args:
            model: Feature extraction model
            image: PIL Image
            transform: Preprocessing transforms
            device: Device to process on
            is_transformer: Whether the model is a transformer architecture
            
        Returns:
            Feature map tensor
        """
        # Apply transforms
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            if is_transformer:
                feature_map = model(img_tensor)
            else:
                feature_map = model(img_tensor)
            
        # Remove batch dimension
        feature_map = feature_map.squeeze(0)
        
        return feature_map

    @staticmethod
    def get_supported_models():
        """Return a list of supported model names"""
        return [
            'resnet18', 'resnet50', 
            'efficientnet_b0', 'efficientnet_b3',
            'convnext_small', 
            'vit', 'swin',
            'clip', 'layoutlm'
        ]