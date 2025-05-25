import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoIPool(nn.Module):
    """
    Region of Interest Pooling layer for extracting features from specified regions in a feature map.
    
    Args:
        output_size: Size of the output features after pooling (height, width)
    """
    def __init__(self, output_size):
        super(RoIPool, self).__init__()
        self.output_size = output_size
        
    def forward(self, features, rois):
        """
        Extracts features from specified regions.
        
        Args:
            features: Feature map tensor of shape [C, H, W]
            rois: List of region coordinates [x1, y1, x2, y2] normalized to [0, 1]
            
        Returns:
            Pooled features for each ROI with shape [N, C, output_size[0], output_size[1]]
            where N is the number of ROIs
        """
        device = features.device
        feature_h, feature_w = features.shape[-2:]
        
        # Initialize output tensor
        n_rois = len(rois)
        output = torch.zeros(n_rois, features.shape[0], 
                            self.output_size[0], self.output_size[1], 
                            device=device)
        
        for i, roi in enumerate(rois):
            # Convert normalized coordinates [0, 1] to feature map coordinates
            x1, y1, x2, y2 = roi
            x1 = int(x1 * feature_w)
            x2 = int(x2 * feature_w)
            y1 = int(y1 * feature_h)
            y2 = int(y2 * feature_h)
            
            # Ensure valid coordinates
            x1 = max(0, min(x1, feature_w - 1))
            x2 = max(x1 + 1, min(x2, feature_w))
            y1 = max(0, min(y1, feature_h - 1))
            y2 = max(y1 + 1, min(y2, feature_h))
            
            # Extract ROI from feature map
            roi_features = features[:, y1:y2, x1:x2]
            
            # Apply adaptive max pooling to get fixed size output
            if roi_features.numel() > 0:  # Check if ROI is not empty
                pooled = F.adaptive_max_pool2d(
                    roi_features.unsqueeze(0),  # Add batch dimension
                    self.output_size
                ).squeeze(0)  # Remove batch dimension
                output[i] = pooled
        
        return output

class RoIAlign(nn.Module):
    """
    Region of Interest Align layer for extracting features from specified regions in a feature map
    using bilinear interpolation for better alignment with the input image.
    
    Args:
        output_size: Size of the output features after pooling (height, width)
        spatial_scale: Scale factor for mapping from input image to feature map coordinates
    """
    def __init__(self, output_size, spatial_scale=1.0):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        
    def forward(self, features, rois):
        """
        Extracts features from specified regions using bilinear interpolation.
        
        Args:
            features: Feature map tensor of shape [C, H, W]
            rois: List of region coordinates [x1, y1, x2, y2] normalized to [0, 1]
            
        Returns:
            Aligned features for each ROI with shape [N, C, output_size[0], output_size[1]]
            where N is the number of ROIs
        """
        device = features.device
        feature_h, feature_w = features.shape[-2:]
        
        # Initialize output tensor
        n_rois = len(rois)
        output = torch.zeros(n_rois, features.shape[0], 
                            self.output_size[0], self.output_size[1], 
                            device=device)
                            
        for i, roi in enumerate(rois):
            # Convert normalized coordinates [0, 1] to feature map coordinates
            x1, y1, x2, y2 = roi
            x1 = x1 * feature_w * self.spatial_scale
            x2 = x2 * feature_w * self.spatial_scale
            y1 = y1 * feature_h * self.spatial_scale
            y2 = y2 * feature_h * self.spatial_scale
            
            # Calculate bin size
            bin_size_h = (y2 - y1) / self.output_size[0]
            bin_size_w = (x2 - x1) / self.output_size[1]
            
            # Skip if ROI is too small
            if bin_size_h <= 0 or bin_size_w <= 0:
                continue
            
            # Generate sampling grid
            # For each output pixel, calculate 4 sampling points in the feature map
            for oh in range(self.output_size[0]):
                for ow in range(self.output_size[1]):
                    # Calculate center of bin
                    y = y1 + (oh + 0.5) * bin_size_h
                    x = x1 + (ow + 0.5) * bin_size_w
                    
                    # Ensure coordinates are in bounds
                    y = max(0, min(y, feature_h - 1))
                    x = max(0, min(x, feature_w - 1))
                    
                    # Convert to normalized (-1, 1) coordinates for grid_sample
                    y_norm = (y / (feature_h - 1)) * 2 - 1
                    x_norm = (x / (feature_w - 1)) * 2 - 1
                    
                    # Use bilinear interpolation to sample the feature map
                    # This is a simplified version - actual RoIAlign uses multiple samples per bin
                    sample_points = torch.tensor([[y_norm, x_norm]], device=device)
                    for c in range(features.shape[0]):
                        output[i, c, oh, ow] = self._bilinear_interpolate(
                            features[c], x, y, feature_h, feature_w)
        
        return output
    
    def _bilinear_interpolate(self, feature_map, x, y, height, width):
        """Bilinear interpolation for a single point and channel"""
        # Get the four surrounding points
        # Use math.floor instead of torch.floor for Python floats
        x0 = int(math.floor(x))
        x1 = min(x0 + 1, width - 1)
        y0 = int(math.floor(y))
        y1 = min(y0 + 1, height - 1)
        
        # Get weights
        wx = x - x0
        wy = y - y0
        wx0 = 1 - wx
        wx1 = wx
        wy0 = 1 - wy
        wy1 = wy
        
        # Sample values
        v00 = feature_map[y0, x0]
        v01 = feature_map[y0, x1]
        v10 = feature_map[y1, x0]
        v11 = feature_map[y1, x1]
        
        # Interpolate
        value = wx0 * wy0 * v00 + wx1 * wy0 * v01 + wx0 * wy1 * v10 + wx1 * wy1 * v11
        return value