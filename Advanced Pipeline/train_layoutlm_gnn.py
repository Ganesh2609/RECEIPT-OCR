import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
import seaborn as sns
from tqdm import tqdm
import argparse
import gc
import psutil
import json
from torch.utils.data import ConcatDataset, Subset, random_split

# Import our dataset and model
from receipt_graph_dataset import ReceiptGraphDataset
from sroie_dataset import SROIEDataset
from layoutlm_gnn import LayoutGNN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#------------------------------------------------------------
# Memory Monitoring Functions
#------------------------------------------------------------

def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    result = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
        result.append(f"GPU {i}: Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    
    return "\n".join(result)

def log_memory_usage(phase=""):
    """Log current memory usage with optional phase label."""
    # Get process info
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**2  # MB
    
    # Log CPU RAM usage
    logger.info(f"{phase} RAM Usage: {ram_usage:.2f} MB")
    
    # Log GPU memory if available
    if torch.cuda.is_available():
        logger.info(f"{phase} GPU Memory:\n{get_gpu_memory_info()}")

def clear_memory():
    """Clear memory by releasing cached tensors and running garbage collection."""
    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()
    
    # Log memory after clearing
    logger.debug("Memory cleared")

#------------------------------------------------------------
# Model Analysis Functions
#------------------------------------------------------------

def display_model_info(model, sample_data=None, device=None, depth=10, verbose=1, skip_memory_usage=False):
    """
    Display basic information about the model without using torchinfo.
    
    Args:
        model: The PyTorch model
        Other args kept for compatibility but not used
    """
    # Custom model summary without relying on torchinfo
    try:
        logger.info("Generating custom model summary...")
        
        # Count total and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary_str = []
        summary_str.append(f"\nModel Summary for {model.__class__.__name__}:")
        summary_str.append(f"Total parameters: {total_params:,}")
        summary_str.append(f"Trainable parameters: {trainable_params:,}")
        summary_str.append(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        summary_str.append("\nParameter counts by layer:")
        
        # Get parameter counts by top-level modules
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary_str.append(f"  {name}: {module_params:,} params ({module_trainable:,} trainable)")
            
            # If verbose, show second level
            if verbose > 0:
                for subname, submodule in module.named_children():
                    submodule_params = sum(p.numel() for p in submodule.parameters())
                    submodule_trainable = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
                    summary_str.append(f"    {subname}: {submodule_params:,} params ({submodule_trainable:,} trainable)")
        
        # Add model structure
        summary_str.append("\nModel structure:")
        summary_str.append(str(model))
        
        result = "\n".join(summary_str)
        logger.info("Model summary complete")
        return result
    
    except Exception as e:
        logger.error(f"Error generating custom model summary: {e}")
        return str(model)

#------------------------------------------------------------
# Checkpoint Management Functions
#------------------------------------------------------------

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pt'):
    """
    Save checkpoint including model, optimizer, scheduler state, and training history.
    
    Args:
        state: Dictionary containing state to save
        is_best: Boolean indicating if this is the best model so far
        output_dir: Directory to save checkpoints
        filename: Filename for the checkpoint
    """
    # Save the checkpoint
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(state, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # If this is the best model, save a copy as best model
    if is_best:
        best_path = os.path.join(output_dir, 'best_model_checkpoint.pt')
        torch.save(state, best_path)
        logger.info(f"Best model checkpoint saved to {best_path}")
        
        # Also save just the model state dict for easy loading
        best_model_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(state['model_state_dict'], best_model_path)
        logger.info(f"Best model state dict saved to {best_model_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load the state dict into
        optimizer: Optimizer to load the state dict into
        scheduler: Optional scheduler to load the state dict into
        
    Returns:
        Dictionary containing loaded training history and metadata
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found at {checkpoint_path}")
        return None
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Return training metadata and history
    return {
        'start_epoch': checkpoint['epoch'] + 1,
        'best_val_acc': checkpoint['best_val_acc'],
        'train_losses': checkpoint['train_losses'],
        'val_accuracies': checkpoint['val_accuracies'],
        'learning_rates': checkpoint['learning_rates']
    }

def save_training_history(history, output_dir):
    """
    Save training history to a JSON file for later analysis.
    
    Args:
        history: Dictionary containing training metrics
        output_dir: Directory to save the history file
    """
    history_path = os.path.join(output_dir, 'training_history.json')
    
    # Convert numpy values to Python native types for JSON serialization
    serializable_history = {}
    for key, value in history.items():
        if isinstance(value, list) and value and isinstance(value[0], (np.float32, np.float64, np.int32, np.int64)):
            serializable_history[key] = [float(v) for v in value]
        else:
            serializable_history[key] = value
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f)
    
    logger.info(f"Training history saved to {history_path}")

def load_training_history(output_dir):
    """
    Load training history from a JSON file.
    
    Args:
        output_dir: Directory containing the history file
        
    Returns:
        Dictionary containing training metrics or None if file doesn't exist
    """
    history_path = os.path.join(output_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    logger.info(f"Training history loaded from {history_path}")
    return history

#------------------------------------------------------------
# Training & Evaluation Functions
#------------------------------------------------------------

def train(model, train_loader, optimizer, device, class_weights=None):
    """Train the model for one epoch with memory optimization."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)
        
        # Get target labels - always use heterogeneous format
        labels = batch['node'].y
        
        # Compute loss
        if class_weights is not None:
            loss = F.cross_entropy(out, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(out, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate statistics
        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Clear memory for batch
        del batch, out, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / total_samples

def evaluate(model, loader, device):
    """Evaluate the model on the given data loader with memory optimization."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch)
            pred = out.argmax(dim=1)
            
            # Get target labels - always use heterogeneous format
            labels = batch['node'].y
            
            # Compute accuracy
            batch_correct = (pred == labels).sum().item()
            batch_total = labels.size(0)
            correct += batch_correct
            total += batch_total
            
            # Store predictions and labels for detailed metrics
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Clear memory for batch
            del batch, out, pred, labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return correct / total, all_preds, all_labels

#------------------------------------------------------------
# Visualization Functions
#------------------------------------------------------------

def plot_confusion_matrix(true_labels, predictions, class_names=None, figsize=(12, 10)):
    """Plot confusion matrix with seaborn."""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt

def plot_training_curves(train_losses, val_accuracies, learning_rates=None):
    """Plot training loss, validation accuracy, and optionally learning rates."""
    if learning_rates:
        plt.figure(figsize=(18, 5))
        n_plots = 3
    else:
        plt.figure(figsize=(12, 5))
        n_plots = 2
    
    plt.subplot(1, n_plots, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, n_plots, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    if learning_rates:
        plt.subplot(1, n_plots, 3)
        plt.plot(learning_rates)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    return plt

#------------------------------------------------------------
# Main Function
#------------------------------------------------------------

def main():
    """Main function to run the training script with memory management."""
    parser = argparse.ArgumentParser(description='Train LayoutLM-GNN on receipt dataset')
    
    # Dataset arguments
    parser.add_argument('--dataset_type', type=str, default='wildreceipt', choices=['wildreceipt', 'sroie'], help='Dataset type to use (wildreceipt or SROIE)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--train_file', type=str, default='train.txt', help='Training file name')
    parser.add_argument('--test_file', type=str, default='test.txt', help='Test file name')
    parser.add_argument('--max_seq_length', type=int, default=265, help='Maximum sequence length for tokenization')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--use_gpu_for_bert', action='store_true', default=True, help='Use GPU for BERT embeddings')
    parser.add_argument('--process_batch_size', type=int, default=32, help='Batch size for BERT processing')
    
    # Model architecture arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='BERT model to use')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden channels')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--use_gat', action='store_true', help='Use Graph Attention Networks')
    parser.add_argument('--attention_heads', type=int, default=8, help='Number of attention heads for GAT')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--use_edge_features', action='store_true', help='Use edge features')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--class_weighting', action='store_true', help='Use class weights for loss function')
    parser.add_argument('--output_dir', type=str, default='./results_layoutlm', help='Output directory')
    
    # Learning rate scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine', 
                    choices=['step', 'plateau', 'cosine', 'exponential'],
                    help='Learning rate scheduler type')
    parser.add_argument('--scheduler_warmup_steps', type=int, default=1000, 
                    help='Number of warmup steps')
    parser.add_argument('--scheduler_step_size', type=int, default=10, 
                    help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, 
                    help='Multiplicative factor for StepLR and ExponentialLR')
    parser.add_argument('--scheduler_patience', type=int, default=3, 
                    help='Patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6, 
                    help='Minimum learning rate for schedulers')
    
    # Model information arguments
    parser.add_argument('--print_model_summary', action='store_true', help='Print detailed model summary')
    parser.add_argument('--model_summary_depth', type=int, default=10, help='Depth for model summary')
    parser.add_argument('--skip_memory_usage', action='store_true', help='Skip memory usage calculation in summary')
    
    # Memory management arguments
    parser.add_argument('--monitor_memory', action='store_true', help='Monitor memory usage during training')
    parser.add_argument('--memory_log_frequency', type=int, default=5, help='Log memory every N epochs')
    
    # Checkpoint arguments
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--keep_n_checkpoints', type=int, default=1, 
                        help='Number of recent checkpoints to keep (in addition to best)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initial memory logging if requested
    if args.monitor_memory:
        log_memory_usage("Initial")
    
    #------------------------------------------------------------
    # Data Loading
    #------------------------------------------------------------
    
    # Load datasets
    # logger.info("Loading datasets...")
    if args.dataset_type == 'sroie':
        logger.info("Loading SROIE dataset...")
        train_dataset = SROIEDataset(
            root=args.data_dir,
            train=True,
            k_spatial=args.k_spatial if hasattr(args, 'k_spatial') else 5,
            k_textual=args.k_textual if hasattr(args, 'k_textual') else 3,
            max_seq_length=args.max_seq_length,
            model_name=args.model_name,
            use_gpu=args.use_gpu_for_bert
        )
        
        test_dataset = SROIEDataset(
            root=args.data_dir,
            train=False,
            k_spatial=args.k_spatial if hasattr(args, 'k_spatial') else 5,
            k_textual=args.k_textual if hasattr(args, 'k_textual') else 3,
            max_seq_length=args.max_seq_length,
            model_name=args.model_name,
            use_gpu=args.use_gpu_for_bert
        )
    else:
        logger.info("Loading WildReceipt dataset...")
        train_dataset = ReceiptGraphDataset(
            root=args.data_dir,
            file_path=args.train_file,
            max_seq_length=args.max_seq_length,
            model_name=args.model_name,
            use_gpu=args.use_gpu_for_bert
        )
        
        test_dataset = ReceiptGraphDataset(
            root=args.data_dir,
            file_path=args.test_file,
            max_seq_length=args.max_seq_length,
            model_name=args.model_name,
            use_gpu=args.use_gpu_for_bert
        )
        

    
    # Split training data into train and validation
    num_train = int(len(train_dataset) * 0.8)
    num_val = len(train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val])
    
    logger.info(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Log memory after loading datasets
    if args.monitor_memory:
        log_memory_usage("After data loading")
    
    #------------------------------------------------------------
    # Class Weights Calculation
    #------------------------------------------------------------
    
    # Count number of classes from both train and test
    # This code should replace the existing class extraction section in main()
    # Add this after loading the datasets and before creating the model

    # This code should replace the existing class extraction section in main()
    # Add this after loading the datasets and before creating the model

    # Add more robust logging
    logger.info("Extracting class information from datasets...")

    # Updated class extraction function with better error handling and debugging
    def get_classes_from_dataset(dataset, dataset_name="dataset"):
        """
        Extract classes from dataset with better error handling and debugging.
        
        Args:
            dataset: The dataset to extract classes from
            dataset_name: Name of the dataset for logging
            
        Returns:
            set: Set of class labels
        """
        all_labels = []
        
        try:
            # First try direct access if available
            if hasattr(dataset, 'data') and hasattr(dataset.data, 'node') and hasattr(dataset.data.node, 'y'):
                logger.info(f"Extracting labels directly from {dataset_name}.data.node.y")
                return set(dataset.data.node.y.tolist())
            
            # If it's a Subset, access the dataset attribute
            if hasattr(dataset, 'dataset'):
                orig_dataset = dataset.dataset
                if hasattr(orig_dataset, 'data') and hasattr(orig_dataset.data, 'node') and hasattr(orig_dataset.data.node, 'y'):
                    logger.info(f"Extracting labels from {dataset_name}.dataset.data.node.y")
                    return set(orig_dataset.data.node.y.tolist())
            
            # Otherwise iterate through dataset
            logger.info(f"Iterating through {dataset_name} to extract labels")
            for i in range(len(dataset)):
                try:
                    data = dataset[i]
                    
                    # Print data structure for the first item to help diagnose
                    if i == 0:
                        logger.info(f"Sample data structure keys for {dataset_name}: {type(data)}")
                        if hasattr(data, '__dict__'):
                            logger.info(f"Sample data attributes: {dir(data)}")
                        if isinstance(data, dict):
                            logger.info(f"Sample data dictionary keys: {data.keys()}")
                        
                    if hasattr(data, 'node') and hasattr(data.node, 'y'):
                        all_labels.extend(data.node.y.tolist())
                    elif hasattr(data, 'y') and data.y is not None:
                        # Try direct y attribute
                        all_labels.extend(data.y.tolist() if hasattr(data.y, 'tolist') else [data.y])
                    elif isinstance(data, dict) and 'node' in data and 'y' in data['node']:
                        # Dictionary structure
                        all_labels.extend(data['node']['y'].tolist() if hasattr(data['node']['y'], 'tolist') else [data['node']['y']])
                    elif hasattr(data, 'data') and hasattr(data.data, 'node') and hasattr(data.data.node, 'y'):
                        # Nested data structure
                        all_labels.extend(data.data.node.y.tolist())
                except Exception as e:
                    logger.warning(f"Error accessing item {i} in {dataset_name}: {e}")
                    
                    # If we're having trouble, try to examine the data structure more deeply
                    if i == 0:
                        try:
                            import json
                            data = dataset[i]
                            logger.info(f"Detailed structure of first item in {dataset_name}:")
                            if hasattr(data, '__dict__'):
                                logger.info(f"Object attributes: {dir(data)}")
                            if isinstance(data, dict):
                                logger.info(f"Dictionary keys at root level: {data.keys()}")
                                if 'node' in data:
                                    logger.info(f"'node' keys: {data['node'].keys() if isinstance(data['node'], dict) else 'Not a dict'}")
                        except:
                            logger.warning(f"Could not inspect data structure in detail")
        except Exception as e:
            logger.error(f"Error extracting classes from {dataset_name}: {e}")
        
        return set(all_labels)

    # Get the original train dataset (handling possible Subset objects)
    orig_train_dataset = getattr(train_dataset, 'dataset', train_dataset)

    # Try to extract classes from both datasets with better logging
    logger.info(f"Train dataset type: {type(train_dataset)}")
    logger.info(f"Train dataset length: {len(train_dataset)}")
    classes_train = get_classes_from_dataset(orig_train_dataset, "orig_train_dataset")
    logger.info(f"Extracted classes from train dataset: {classes_train}")

    logger.info(f"Test dataset type: {type(test_dataset)}")
    logger.info(f"Test dataset length: {len(test_dataset)}")
    classes_test = get_classes_from_dataset(test_dataset, "test_dataset") 
    logger.info(f"Extracted classes from test dataset: {classes_test}")

    # Combine classes from both datasets
    all_classes = classes_train.union(classes_test)
    logger.info(f"Combined classes: {all_classes}")

    # Protect against empty class sets
    if not all_classes:
        # Fallback - assume a standard set of classes for your specific domain
        # Adjust this based on your dataset (e.g., SROIE typically has 5 classes)
        logger.warning("No classes found! Defaulting to predefined class count")
        if args.dataset_type == 'sroie':
            out_channels = 5  # Default for SROIE
        else:
            out_channels = 26  # Default for WildReceipt
        logger.warning(f"Using default number of output channels: {out_channels}")
    else:
        # Calculate output channels normally if we have classes
        out_channels = max(all_classes) + 1
        logger.info(f"Calculated output channels: {out_channels}")
    # Calculate class weights if requested
    class_weights = None
    if args.class_weighting:
        logger.info("Computing class weights for balanced loss...")
        
        # Collect all labels from training set
        all_labels = []
        
        # Extract all labels from the dataset
        for i in range(len(orig_train_dataset)):
            data = orig_train_dataset[i]
            if hasattr(data, 'node'):
                all_labels.extend(data['node'].y.tolist())
            elif hasattr(data, 'data') and hasattr(data.data, 'node'):
                all_labels.extend(data.data['node'].y.tolist())
        
        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        unique_classes = np.unique(all_labels)
        if len(unique_classes) > 1:
            try:
                weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
                class_weights = torch.FloatTensor(weights).to(device)
                logger.info(f"Class weights: {weights}")
            except Exception as e:
                logger.warning(f"Error computing class weights: {e}. Using unweighted loss.")
    
    #------------------------------------------------------------
    # Model Creation and Analysis
    #------------------------------------------------------------
    
    # Create model - Always heterogeneous
    logger.info("Creating LayoutLM-GNN model...")
    model = LayoutGNN(
        hidden_channels=args.hidden_channels,
        out_channels=out_channels,
        num_layers=args.num_layers,
        heads=args.attention_heads,
        dropout=args.dropout,
        use_gat=args.use_gat,
        use_edge_features=args.use_edge_features
    ).to(device)
    
    # Log memory after model creation
    if args.monitor_memory:
        log_memory_usage("After model creation")
    
    # Display model architecture and memory usage
    if args.print_model_summary:
        logger.info("Generating model information...")
        try:
            model_summary = display_model_info(
                model,
                verbose=2 if not args.skip_memory_usage else 1
            )
            
            # Save model summary to file
            with open(os.path.join(args.output_dir, 'model_summary.txt'), 'w') as f:
                f.write(model_summary)
                
            logger.info("Model summary saved to file")
        except Exception as e:
            logger.error(f"Error generating model summary: {e}")
            logger.info("Continuing without model summary...")
    else:
        logger.info("Skipping detailed model summary (use --print_model_summary to show)")
        # Print basic parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    #------------------------------------------------------------
    # Optimizer and Learning Rate Scheduler
    #------------------------------------------------------------
    
    lr = float(args.learning_rate)
    
    # Optimizer with all parameters (BERT is already fixed in graph construction)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = None
    if args.scheduler:
        logger.info(f"Using {args.scheduler} learning rate scheduler")
        
        if args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=args.scheduler_step_size,
                gamma=args.scheduler_gamma
            )
        elif args.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',  # Since we're tracking validation accuracy
                factor=args.scheduler_gamma,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
                verbose=True
            )
        elif args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.scheduler_min_lr
            )
        elif args.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=args.scheduler_gamma
            )
    
    #------------------------------------------------------------
    # Resume from Checkpoint (if requested)
    #------------------------------------------------------------
    
    start_epoch = 0
    train_losses = []
    val_accuracies = []
    learning_rates = []
    best_val_acc = 0
    checkpoint_files = []
    
    if args.resume:
        # Check for existing checkpoint
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pt')
        
        # Try to load the checkpoint
        checkpoint_data = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        
        if checkpoint_data:
            # Resume from the loaded checkpoint
            start_epoch = checkpoint_data['start_epoch']
            best_val_acc = checkpoint_data['best_val_acc']
            train_losses = checkpoint_data['train_losses']
            val_accuracies = checkpoint_data['val_accuracies']
            learning_rates = checkpoint_data['learning_rates']
            
            logger.info(f"Resuming training from epoch {start_epoch} with best validation accuracy: {best_val_acc:.4f}")
            
            # List existing checkpoints for rotation
            for filename in os.listdir(args.output_dir):
                if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                    checkpoint_files.append(os.path.join(args.output_dir, filename))
        else:
            logger.info("No checkpoint found, starting training from scratch.")
    
    #------------------------------------------------------------
    # Training Loop
    #------------------------------------------------------------
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Log memory before epoch if requested
        if args.monitor_memory and epoch % args.memory_log_frequency == 0:
            log_memory_usage(f"Before epoch {epoch+1}")
        
        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Train
        train_loss = train(model, train_loader, optimizer, device, class_weights)
        train_losses.append(train_loss)
        
        # Validate
        val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)
        
        logger.info(f"Epoch: {epoch+1:03d}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Step the scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                # For ReduceLROnPlateau, we need to pass the validation accuracy
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        # Save checkpoint every checkpoint_freq epochs or if it's the last epoch
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
            # Create checkpoint state
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'learning_rates': learning_rates,
            }
            
            # Add scheduler state if it exists
            if scheduler is not None:
                checkpoint_state['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save the latest checkpoint
            save_checkpoint(checkpoint_state, is_best, args.output_dir)
            
            # Also save an epoch-specific checkpoint
            epoch_checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint_state, epoch_checkpoint_path)
            checkpoint_files.append(epoch_checkpoint_path)
            
            # Keep only the N most recent checkpoints (plus the best one, which is saved separately)
            if len(checkpoint_files) > args.keep_n_checkpoints:
                checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
                while len(checkpoint_files) > args.keep_n_checkpoints:
                    # Don't delete if it's the most recent one
                    oldest_checkpoint = checkpoint_files.pop(0)
                    if os.path.exists(oldest_checkpoint) and oldest_checkpoint != epoch_checkpoint_path:
                        os.remove(oldest_checkpoint)
                        logger.info(f"Removed old checkpoint: {oldest_checkpoint}")
        
        # Save the training history separately for easy access
        save_training_history({
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates,
            'best_val_acc': best_val_acc,
            'current_epoch': epoch + 1
        }, args.output_dir)
        
        # Clear memory after epoch
        clear_memory()
        
        # Log memory after epoch if requested
        if args.monitor_memory and epoch % args.memory_log_frequency == 0:
            log_memory_usage(f"After epoch {epoch+1}")
    
    # Plot training curves
    plt = plot_training_curves(train_losses, val_accuracies, learning_rates)
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.close()  # Close figure to free memory
    
    #------------------------------------------------------------
    # Final Evaluation
    #------------------------------------------------------------
    
    # Load best model for evaluation
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        logger.warning("Best model file not found, using current model state")
    
    # Log memory before final evaluation
    if args.monitor_memory:
        log_memory_usage("Before final evaluation")
    
    # Final test evaluation
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Define class names
    if args.dataset_type == 'sroie':
        logger.info("Using SROIE label schema (5 classes)")
        sroie_label_map = {
            1: 'Company_name',
            3: 'Address',
            7: 'Date',
            23: 'Total',
            25: 'Others'
        }
        
        # For visualization and reports
        class_names = sroie_label_map
    else:
        # Original wildreceipt label map
        class_names = {
            0: 'Ignore', 1: 'Store_name_value', 2: 'Store_name_key',
            3: 'Store_addr_value', 4: 'Store_addr_key', 5: 'Tel_value',
            6: 'Tel_key', 7: 'Date_value', 8: 'Date_key', 9: 'Time_value',
            10: 'Time_key', 11: 'Prod_item_value', 12: 'Prod_item_key',
            13: 'Prod_quantity_value', 14: 'Prod_quantity_key', 15: 'Prod_price_value',
            16: 'Prod_price_key', 17: 'Subtotal_value', 18: 'Subtotal_key',
            19: 'Tax_value', 20: 'Tax_key', 21: 'Tips_value', 22: 'Tips_key',
            23: 'Total_value', 24: 'Total_key', 25: 'Others'
        }
        
    # Get class names for the labels actually present in the test set
    present_classes = sorted(list(set(test_labels)))
    class_names_present = [class_names.get(i, f"Class_{i}") for i in present_classes]
    
    # Print classification report
    report = classification_report(
        test_labels, test_preds, 
        target_names=[class_names.get(i, f"Class_{i}") for i in present_classes],
        digits=4
    )
    logger.info(f"Classification Report:\n{report}")
    
    # Save report to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    plt = plot_confusion_matrix(test_labels, test_preds, class_names=class_names_present)
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()  # Close figure to free memory
    
    # Final memory clearing
    clear_memory()
    
    # Log final memory usage
    if args.monitor_memory:
        log_memory_usage("Final")
    
    logger.info(f"Results saved to {args.output_dir}")

#------------------------------------------------------------
# Main Execution
#------------------------------------------------------------

if __name__ == "__main__":
    main()
        
# Example command:
# python train_layoutlm_gnn.py --data_dir ./data_layoutlm --train_file train.txt --test_file test.txt --use_gat --attention_heads 8 --hidden_channels 256 --num_layers 3 --use_edge_features --class_weighting --learning_rate 5e-5 --scheduler plateau --scheduler_patience 5 --scheduler_gamma 0.5 --epochs 100 --output_dir ./results_layoutlm --print_model_summary

# To resume training:
# python train_layoutlm_gnn.py --data_dir ./data_layoutlm --train_file train.txt --test_file test.txt --use_gat --attention_heads 8 --hidden_channels 256 --num_layers 3 --use_edge_features --class_weighting --learning_rate 5e-5 --scheduler plateau --scheduler_patience 5 --scheduler_gamma 0.5 --epochs 100 --output_dir ./results_layoutlm --print_model_summary --resume

# Example command line for SROIE dataset:
# python train_layoutlm_gnn.py --data_dir C:\Users\Arun\pytorch\datasets\bills\SROIE2019 --dataset_type sroie --use_gat --attention_heads 8 --hidden_channels 256 --num_layers 3 --use_edge_features --class_weighting --learning_rate 5e-5 --scheduler plateau --scheduler_patience 5 --scheduler_gamma 0.5 --epochs 50 --output_dir ./results_sroie