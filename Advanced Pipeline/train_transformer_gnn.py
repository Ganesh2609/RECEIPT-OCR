import os
import argparse
import logging
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import json
import gc

# Import our dataset and model
from receipt_graph_dataset import ReceiptGraphDataset
from layoutlm_transformer_gnn import LayoutImageTransformerGNN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(model, train_loader, optimizer, device, class_weights=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch)
        
        # Get target labels
        labels = batch['node'].y
        
        # Compute loss
        if class_weights is not None:
            loss = torch.nn.functional.cross_entropy(out, labels, weight=class_weights)
        else:
            loss = torch.nn.functional.cross_entropy(out, labels)
        
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
    """Evaluate the model on the given data loader."""
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
            
            # Get target labels
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

def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pt'):
    """Save checkpoint."""
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
    """Load a training checkpoint."""
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
    """Save training history to a JSON file."""
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

def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description='Train LayoutImageTransformerGNN on receipt dataset')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='./data_layoutlm', help='Data directory')
    parser.add_argument('--train_file', type=str, default='train.txt', help='Training file name')
    parser.add_argument('--test_file', type=str, default='test.txt', help='Test file name')
    parser.add_argument('--images_dir', type=str, default=None, help='Directory containing receipt images')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    # Image embedding arguments
    parser.add_argument('--image_model', type=str, default='resnet18', 
                    choices=['resnet18', 'resnet50'], help='Model to use for image embeddings')
    parser.add_argument('--image_embedding_dim', type=int, default=512, 
                    help='Dimension of image embeddings (512 for ResNet18, 2048 for ResNet50)')
    parser.add_argument('--use_image_features', action='store_true', 
                    help='Whether to use image features in the model')
    
    # Model architecture arguments
    parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden channels')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--transformer_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--use_gat', action='store_true', help='Use Graph Attention Networks')
    parser.add_argument('--gnn_heads', type=int, default=8, help='Number of attention heads for GAT')
    parser.add_argument('--transformer_heads', type=int, default=8, help='Number of attention heads for transformer')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--use_edge_features', action='store_true', help='Use edge features')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--class_weighting', action='store_true', help='Use class weights')
    parser.add_argument('--output_dir', type=str, default='./results_transformer_gnn', help='Output directory')
    
    # Learning rate scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine', 
                    choices=['step', 'plateau', 'cosine', 'exponential', 'none'],
                    help='Learning rate scheduler type')
    parser.add_argument('--scheduler_step_size', type=int, default=10, 
                    help='Step size for StepLR scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, 
                    help='Gamma for StepLR and ExponentialLR')
    parser.add_argument('--scheduler_patience', type=int, default=3, 
                    help='Patience for ReduceLROnPlateau')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6, 
                    help='Minimum learning rate for schedulers')
    
    # Checkpoint arguments
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    #------------------------------------------------------------
    # Data Loading
    #------------------------------------------------------------
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = ReceiptGraphDataset(
        root=args.data_dir,
        file_path=args.train_file,
        images_dir=args.images_dir,
        image_model_name=args.image_model,
        image_embedding_dim=args.image_embedding_dim
    )
    
    test_dataset = ReceiptGraphDataset(
        root=args.data_dir,
        file_path=args.test_file,
        images_dir=args.images_dir,
        image_model_name=args.image_model,
        image_embedding_dim=args.image_embedding_dim
    )
    
    # Split training data into train and validation
    num_train = int(len(train_dataset) * 0.8)
    num_val = len(train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val])
    
    logger.info(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    #------------------------------------------------------------
    # Class Weights Calculation
    #------------------------------------------------------------
    
    def get_classes_from_dataset(dataset):
        """Extract classes from dataset."""
        if hasattr(dataset, 'data'):
            # Direct access to data
            return set(dataset.data['node'].y.tolist())
        else:
            # Need to iterate through dataset
            all_labels = []
            for i in range(len(dataset)):
                data = dataset[i]
                if hasattr(data, 'node'):
                    all_labels.extend(data['node'].y.tolist())
                elif hasattr(data, 'data') and hasattr(data.data, 'node'):
                    all_labels.extend(data.data['node'].y.tolist())
            return set(all_labels)
    
    # Get class counts for model and class weighting
    orig_train_dataset = getattr(train_dataset, 'dataset', train_dataset)
    classes_train = get_classes_from_dataset(orig_train_dataset)
    classes_test = get_classes_from_dataset(test_dataset)
    
    # Get maximum class ID for output dimension
    out_channels = max(max(classes_train), max(classes_test)) + 1
    logger.info(f"Output classes: {out_channels}")
    
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
                logger.info(f"Class weights calculated for {len(unique_classes)} classes")
            except Exception as e:
                logger.warning(f"Error computing class weights: {e}. Using unweighted loss.")
    
    #------------------------------------------------------------
    # Model Creation
    #------------------------------------------------------------
    
    # Create model
    logger.info("Creating LayoutImageTransformerGNN model...")
    model = LayoutImageTransformerGNN(
        hidden_channels=args.hidden_channels,
        out_channels=out_channels,
        image_embedding_dim=args.image_embedding_dim,
        num_gnn_layers=args.gnn_layers,
        num_transformer_layers=args.transformer_layers,
        gnn_heads=args.gnn_heads,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        use_gat=args.use_gat,
        use_edge_features=args.use_edge_features,
        use_image_features=args.use_image_features
    ).to(device)
    
    # Print model summary
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Using {args.gnn_layers} GNN layers and {args.transformer_layers} transformer layers")
    if args.use_image_features:
        logger.info(f"Using image features from {args.image_model} with dimension {args.image_embedding_dim}")
    else:
        logger.info("Not using image features")
    
    #------------------------------------------------------------
    # Optimizer and Scheduler
    #------------------------------------------------------------
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = None
    if args.scheduler != 'none':
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
        else:
            logger.info("No checkpoint found, starting training from scratch.")
    
    #------------------------------------------------------------
    # Training Loop
    #------------------------------------------------------------
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
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
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1 or is_best:
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
            
            # Save the checkpoint
            save_checkpoint(checkpoint_state, is_best, args.output_dir)
        
        # Save the training history
        save_training_history({
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates,
            'best_val_acc': best_val_acc,
            'current_epoch': epoch + 1
        }, args.output_dir)
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Plot training curves
    plt = plot_training_curves(train_losses, val_accuracies, learning_rates)
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    plt.close()
    
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
    
    # Final test evaluation
    test_acc, test_preds, test_labels = evaluate(model, test_loader, device)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Define class names
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
    
    # Find unique classes present in the test data
    unique_classes = sorted(list(set(test_labels)))
    class_names_present = [class_names.get(i, f"Class_{i}") for i in unique_classes]
    
    # Print classification report
    report = classification_report(
        test_labels, test_preds, 
        labels=unique_classes,
        target_names=class_names_present,
        digits=4
    )
    logger.info(f"Classification Report:\n{report}")
    
    # Save report to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    plt = plot_confusion_matrix(test_labels, test_preds, class_names=class_names_present)
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    plt.close()
    
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()

# python train_transformer_gnn.py --data_dir ./data_layoutlm --train_file train.txt --test_file test.txt --hidden_channels 256 --gnn_layers 2 --transformer_layers 2 --use_gat --gnn_heads 8 --transformer_heads 8 --dropout 0.3 --use_edge_features --learning_rate 5e-5 --scheduler plateau --class_weighting --output_dir ./results_transformer_gnn

# Example usage with image features:
# python train_transformer_gnn.py --data_dir ./data_layoutlm --train_file train.txt --test_file test.txt 
#   --images_dir ./receipt_images --image_model resnet18 --image_embedding_dim 512 --use_image_features
#   --hidden_channels 256 --gnn_layers 2 --transformer_layers 2 --use_gat --gnn_heads 8 
#   --transformer_heads 8 --dropout 0.3 --use_edge_features --learning_rate 5e-5 
#   --scheduler plateau --class_weighting --output_dir ./results_image_transformer_gnn

# python train_transformer_gnn.py --data_dir ./data_layoutlm --train_file train.txt --test_file test.txt --images_dir ./receipt_images --use_image_features --image_model resnet18 --image_embedding_dim 2048 --hidden_channels 512 --gnn_layers 3 --transformer_layers 4 --use_gat --gnn_heads 8 --transformer_heads 8 --dropout 0.2 --use_edge_features --learning_rate 3e-5 --scheduler cosine --class_weighting --output_dir ./results_advanced_image_model
