import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

def visualize_bboxes(original_img_path, rectified_img_path, original_anno_path, updated_anno_path=None):
    """
    Visualize bounding boxes on original and rectified images.
    
    Parameters:
    -----------
    original_img_path : str
        Path to the original warped image
    rectified_img_path : str
        Path to the rectified/unwrapped image
    original_anno_path : str
        Path to the original annotation file
    updated_anno_path : str, optional
        Path to the updated annotation file. If None, will just show original
        boxes on both images (for comparison)
    """
    # Load images
    original_img = cv2.imread(original_img_path)
    rectified_img = cv2.imread(rectified_img_path)
    
    # Convert to RGB (matplotlib uses RGB)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    rectified_img = cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB)
    
    # Load annotations
    with open(original_anno_path, 'r') as f:
        original_anno = json.load(f)
    
    if updated_anno_path:
        with open(updated_anno_path, 'r') as f:
            updated_anno = json.load(f)
    else:
        # If no updated annotations, use the original for both
        updated_anno = original_anno
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Display images
    ax1.imshow(original_img)
    ax1.set_title('Original Image with Original Boxes')
    ax1.axis('off')
    
    ax2.imshow(rectified_img)
    ax2.set_title('Rectified Image with Updated Boxes')
    ax2.axis('off')
    
    # Plot original boxes on original image
    for anno in original_anno['annotations']:
        box = anno['box']
        text = anno.get('text', '')
        
        # Create points from box coordinates
        points = np.array([[box[i], box[i+1]] for i in range(0, len(box), 2)])
        
        # Generate a random color for each box
        color = np.random.rand(3,)
        
        # Create and add the polygon patch to the first axis
        polygon = Polygon(points, fill=False, edgecolor=color, linewidth=2)
        ax1.add_patch(polygon)
        
        # Add text label near the box
        # centroid_x = np.mean(points[:, 0])
        # centroid_y = np.mean(points[:, 1])
        # ax1.text(centroid_x, centroid_y, text, fontsize=8, 
        #          color='white', bbox=dict(facecolor=color, alpha=0.7))
    
    # Plot updated boxes on rectified image
    for anno in updated_anno['annotations']:
        box = anno['box']
        text = anno.get('text', '')
        
        # Create points from box coordinates
        points = np.array([[box[i], box[i+1]] for i in range(0, len(box), 2)])
        
        # Generate a random color for each box
        color = np.random.rand(3,)
        
        # Create and add the polygon patch to the second axis
        polygon = Polygon(points, fill=False, edgecolor=color, linewidth=2)
        ax2.add_patch(polygon)
        
        # # Add text label near the box
        # centroid_x = np.mean(points[:, 0])
        # centroid_y = np.mean(points[:, 1])
        # ax2.text(centroid_x, centroid_y, text, fontsize=8, 
        #          color='white', bbox=dict(facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('bbox_visualization.png', dpi=300)
    plt.show()

# Function to display a single image with bounding boxes for easier verification
def visualize_single_image(img_path, anno_path, title="Image with Bounding Boxes"):
    """
    Visualize bounding boxes on a single image.
    
    Parameters:
    -----------
    img_path : str
        Path to the image
    anno_path : str
        Path to the annotation file
    title : str
        Title for the plot
    """
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load annotations
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display image
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    
    # Create a color map for different label types
    label_colors = {}
    
    # Plot boxes
    for anno_item in anno['annotations']:
        box = anno_item['box']
        text = anno_item.get('text', '')
        label = anno_item.get('label', 0)
        
        # Create points from box coordinates
        points = np.array([[box[i], box[i+1]] for i in range(0, len(box), 2)])
        
        # Use consistent colors for the same label
        if label not in label_colors:
            label_colors[label] = np.random.rand(3,)
        color = label_colors[label]
        
        # Create and add the polygon patch
        polygon = Polygon(points, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(polygon)
        
        # # Add text label near the box
        # centroid_x = np.mean(points[:, 0])
        # centroid_y = np.mean(points[:, 1])
        # ax.text(centroid_x, centroid_y, text, fontsize=8, 
        #         color='white', bbox=dict(facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Paths to images and annotations - update these with your actual paths
    original_img_path = r'C:\Users\Arun\pytorch\datasets\bills\wildreceipt\image_files\Image_2\0\74eb6a846a45613ebbd2379bcb1516b7227bbb0b.jpeg'
    rectified_img_path = "unwrapped_img.png"  # Replace with your unwrapped image path
    original_anno_path = "annotation.json"
    updated_anno_path = "updated_annotation.json"  # Path to your updated annotations
    
    # Visualize both images with their respective boxes
    visualize_bboxes(original_img_path, rectified_img_path, original_anno_path, updated_anno_path)
    
    # Optionally visualize just the rectified image with updated boxes for closer inspection
    visualize_single_image(rectified_img_path, updated_anno_path, "Rectified Image with Updated Boxes")