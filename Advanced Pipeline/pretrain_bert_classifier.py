import os
import json
import torch
import numpy as np
import logging
import argparse
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReceiptDataset(Dataset):
    """Dataset for receipt text classification."""
    
    def __init__(self, file_path, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            file_path: Path to the receipt data file
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []
        
        # Load and process the data
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Processing receipts"):
            receipt_data = json.loads(line.strip())
            annotations = receipt_data.get('annotations', [])
            
            for annotation in annotations:
                text = annotation.get('text', '')
                label = annotation.get('label', 0)
                
                # Skip empty text or too short text
                if not text or len(text) < 1:
                    continue
                
                self.texts.append(text)
                self.labels.append(label)
        
        logger.info(f"Loaded {len(self.texts)} text elements")
        
        # Get label distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            logger.info(f"Label {label}: {count} instances")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_bert_classifier(args):
    """
    Train BERT for receipt text element classification.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # Define label mapping
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
    
    # Number of labels is the max label value + 1
    num_labels = max(label_map.keys()) + 1
    logger.info(f"Using {num_labels} labels for classification")
    
    # Create model
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=True  # We need hidden states for embeddings later
    )
    model = model.to(device)
    
    # Load datasets
    train_dataset = ReceiptDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    
    test_dataset = ReceiptDataset(
        file_path=args.test_file,
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size
    )
    
    # Set up optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * args.epochs
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    logger.info("Starting training...")
    
    best_accuracy = 0.0
    train_losses = []
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Clear gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
        
        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Find unique classes present in the data
        unique_classes = sorted(list(set(all_labels)))
        
        # Get detailed classification report with only classes present in the data
        report = classification_report(
            all_labels, all_preds,
            labels=unique_classes,
            target_names=[label_map.get(i, f"Class_{i}") for i in unique_classes],
            digits=4
        )
        logger.info(f"Classification Report:\n{report}")
        
        # Save the report to file
        with open(os.path.join(args.output_dir, f'epoch_{epoch+1}_report.txt'), 'w') as f:
            f.write(report)
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            logger.info(f"New best accuracy: {best_accuracy:.4f}. Saving model...")
            
            # Save the model
            model_save_path = os.path.join(args.output_dir, 'best_model')
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # Save configuration for later use
            with open(os.path.join(model_save_path, 'config.json'), 'w') as f:
                json.dump({
                    'num_labels': num_labels,
                    'label_map': label_map,
                    'max_seq_length': args.max_seq_length
                }, f)
    
    # Save the final model
    logger.info("Training complete. Saving final model...")
    final_model_path = os.path.join(args.output_dir, 'final_model')
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save training losses
    with open(os.path.join(args.output_dir, 'training_losses.json'), 'w') as f:
        json.dump(train_losses, f)
    
    logger.info(f"Best test accuracy: {best_accuracy:.4f}")
    logger.info(f"Model saved to {args.output_dir}")
    
    return model, tokenizer

def extract_bert_embeddings(model, tokenizer, file_path, output_file, device, max_length=128):
    """
    Extract BERT embeddings from the fine-tuned model for later use.
    
    Args:
        model: Fine-tuned BERT model
        tokenizer: BERT tokenizer
        file_path: Path to the receipt data file
        output_file: Path to save the embeddings
        device: Computation device (CPU/GPU)
        max_length: Maximum sequence length
    """
    logger.info(f"Extracting embeddings from {file_path}...")
    
    # Process the data
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_embeddings = []
    receipt_indices = []
    
    for receipt_idx, line in enumerate(tqdm(lines, desc="Processing receipts")):
        receipt_data = json.loads(line.strip())
        file_name = receipt_data.get('file_name', f'receipt_{receipt_idx}')
        annotations = receipt_data.get('annotations', [])
        
        receipt_embeddings = []
        
        for anno_idx, annotation in enumerate(annotations):
            text = annotation.get('text', '')
            label = annotation.get('label', 0)
            box = annotation.get('box', [0, 0, 0, 0, 0, 0, 0, 0])
            
            # Skip empty text
            if not text or len(text) < 1:
                # Add zero embedding for empty text
                empty_embedding = np.zeros(768)  # BERT hidden size
                receipt_embeddings.append({
                    'embedding': empty_embedding,
                    'text': text,
                    'label': label,
                    'box': box
                })
                continue
            
            # Tokenize
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get the embedding from the last hidden state of the [CLS] token
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]
                cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()[0]
                
                receipt_embeddings.append({
                    'embedding': cls_embedding,
                    'text': text,
                    'label': label,
                    'box': box
                })
        
        # Store the receipt's embeddings
        all_embeddings.append({
            'file_name': file_name,
            'embeddings': receipt_embeddings
        })
        
        receipt_indices.append(receipt_idx)
    
    # Save the embeddings
    logger.info(f"Saving embeddings to {output_file}...")
    np.save(output_file, {
        'embeddings': all_embeddings,
        'receipt_indices': receipt_indices
    })
    
    logger.info(f"Extracted embeddings for {len(all_embeddings)} receipts")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Pretrain BERT for receipt text classification')
    
    # Input/output arguments
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to the training data file')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to the test data file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the trained model and results')
    
    # BERT model arguments
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='BERT model to use')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='Maximum sequence length for tokenization')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps for learning rate scheduler')
    
    # Embedding extraction arguments
    parser.add_argument('--extract_embeddings', action='store_true',
                        help='Extract embeddings after training')
    parser.add_argument('--embedding_output_dir', type=str, default=None,
                        help='Directory to save the extracted embeddings')
    
    args = parser.parse_args()
    
    # Train the BERT classifier
    model, tokenizer = train_bert_classifier(args)
    
    # Extract embeddings if requested
    if args.extract_embeddings:
        embedding_output_dir = args.embedding_output_dir or os.path.join(args.output_dir, 'embeddings')
        os.makedirs(embedding_output_dir, exist_ok=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract embeddings for train and test sets
        extract_bert_embeddings(
            model=model,
            tokenizer=tokenizer,
            file_path=args.train_file,
            output_file=os.path.join(embedding_output_dir, 'train_embeddings.npy'),
            device=device,
            max_length=args.max_seq_length
        )
        
        extract_bert_embeddings(
            model=model,
            tokenizer=tokenizer,
            file_path=args.test_file,
            output_file=os.path.join(embedding_output_dir, 'test_embeddings.npy'),
            device=device,
            max_length=args.max_seq_length
        )

if __name__ == "__main__":
    main()
    
    
# python pretrain_bert_classifier.py --train_file train.txt --test_file test.txt --output_dir ./bert_pretrained --bert_model bert-base-uncased --max_seq_length 128 --batch_size 16 --learning_rate 2e-5 --epochs 4 --extract_embeddings --embedding_output_dir ./bert_pretrained/embeddings