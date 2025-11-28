"""
Enhanced Hybrid BERT + Wav2Vec2 Model Training V2
- 15 epochs with early stopping (patience=5)
- GPU training with comprehensive logging
- Training curve generation
- Model checkpointing
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer, 
    Wav2Vec2Model,
    Wav2Vec2Processor,
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coach.audio_features import get_wav2vec_features

# Find latest v2 files or use today's timestamp
from glob import glob
import os

# Try to find existing v2 files
existing_files = glob('data/bert_train_v2_*.pt')
if existing_files:
    # Extract timestamp from latest file
    latest_file = max(existing_files, key=os.path.getmtime)
    V2_SUFFIX = latest_file.split('bert_train_')[1].replace('.pt', '')
    TIMESTAMP = V2_SUFFIX.split('_')[1]
else:
    # Use today's timestamp if no files found
    TIMESTAMP = datetime.now().strftime("%Y%m%d")
    V2_SUFFIX = f"v2_{TIMESTAMP}"

# Setup enhanced logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'training_{V2_SUFFIX}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("ENHANCED HYBRID BERT + WAV2VEC2 TRAINING V2")
logger.info(f"Log file: {log_file}")
logger.info(f"Timestamp: {TIMESTAMP}")
logger.info("="*80)


class HybridSpeechDataset(Dataset):
    """Dataset for Hybrid BERT + Wav2Vec2 training"""
    
    def __init__(self, data_path, extract_features=True):
        logger.info(f"Loading dataset from {data_path}...")
        self.data = torch.load(data_path)
        self.extract_features = extract_features
        
        logger.info(f"Dataset loaded: {len(self.data['input_ids'])} samples")
        
        # Cache Wav2Vec2 features if extracting
        if extract_features and 'audio_paths' in self.data:
            logger.info("Extracting Wav2Vec2 features...")
            self.wav2vec_features = []
            for audio_path in tqdm(self.data['audio_paths'], desc='Extracting audio features'):
                if os.path.exists(audio_path):
                    features = get_wav2vec_features(audio_path)
                    self.wav2vec_features.append(torch.tensor(features, dtype=torch.float32))
                else:
                    self.wav2vec_features.append(torch.zeros(768, dtype=torch.float32))
            self.wav2vec_features = torch.stack(self.wav2vec_features)
            logger.info(f"[OK] Cached {len(self.wav2vec_features)} Wav2Vec2 feature vectors")
        else:
            self.wav2vec_features = None
        
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'labels': self.data['labels'][idx]
        }
        
        if self.wav2vec_features is not None:
            item['wav2vec_features'] = self.wav2vec_features[idx]
        else:
            item['wav2vec_features'] = torch.zeros(768, dtype=torch.float32)
        
        return item


class HybridSpeechModel(nn.Module):
    """Hybrid BERT + Wav2Vec2 Model"""
    
    def __init__(self, num_labels=3, bert_frozen=False):
        super().__init__()
        
        # BERT for text
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels,
            problem_type='multi_label_classification'
        )
        
        if bert_frozen:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Wav2Vec2 for audio (frozen)
        logger.info("Loading Wav2Vec2 for accent-robust features...")
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.wav2vec.eval()
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_labels)
        )
        
        self.bert.classifier = nn.Identity()
    
    def forward(self, input_ids, attention_mask, wav2vec_features):
        # BERT text features
        bert_outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # Mean pooling
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        bert_pooled = sum_embeddings / sum_mask
        
        # Wav2Vec2 features
        if wav2vec_features.dim() == 1:
            wav2vec_features = wav2vec_features.unsqueeze(0)
        
        # Fuse
        combined = torch.cat([bert_pooled, wav2vec_features], dim=1)
        logits = self.fusion(combined)
        
        return logits


class HybridSpeechTrainerV2:
    """Enhanced Trainer for Hybrid BERT + Wav2Vec2 model"""
    
    def __init__(self, model_dir=None, num_labels=3):
        if model_dir is None:
            model_dir = f'coach/models/hybrid_bert_wav2vec_{V2_SUFFIX}'
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        logger.info(f"Initializing Hybrid BERT + Wav2Vec2 model on {self.device}...")
        self.model = HybridSpeechModel(num_labels=num_labels, bert_frozen=False).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        logger.info("[OK] Hybrid model loaded")
        
        # Training history for plots
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_metrics_history = {'filler': [], 'clarity': [], 'pacing': []}
        self.val_metrics_history = {'filler': [], 'clarity': [], 'pacing': []}
    
    def train(self, train_data_path, val_data_path, epochs=15, batch_size=8, lr=2e-5):
        """Train hybrid model with comprehensive logging"""
        logger.info("="*80)
        logger.info("TRAINING CONFIGURATION")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {lr}")
        logger.info(f"  Early stopping patience: 5")
        logger.info(f"  Device: {self.device}")
        logger.info("="*80)
        
        # Load datasets
        logger.info("Loading datasets...")
        train_dataset = HybridSpeechDataset(train_data_path, extract_features=True)
        val_dataset = HybridSpeechDataset(val_data_path, extract_features=True)
        
        # Calculate class weights for imbalanced data BEFORE creating DataLoader
        # Load all training labels to calculate weights
        logger.info("Calculating class weights for imbalanced data...")
        train_data = torch.load(train_data_path)
        train_labels_all = train_data['labels'].numpy()
        
        # Calculate positive class frequency for each label
        pos_freq = train_labels_all.mean(axis=0)  # [filler, clarity, pacing]
        neg_freq = 1 - pos_freq
        
        # Calculate weights: weight = neg_freq / pos_freq (higher weight for rare classes)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        class_weights = neg_freq / (pos_freq + epsilon)
        
        logger.info(f"Class weights calculated:")
        logger.info(f"  Filler: {class_weights[0]:.3f} (pos_freq: {pos_freq[0]:.3f}, neg_freq: {neg_freq[0]:.3f})")
        logger.info(f"  Clarity: {class_weights[1]:.3f} (pos_freq: {pos_freq[1]:.3f}, neg_freq: {neg_freq[1]:.3f})")
        logger.info(f"  Pacing: {class_weights[2]:.3f} (pos_freq: {pos_freq[2]:.3f}, neg_freq: {neg_freq[2]:.3f})")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Create weighted loss function with Focal Loss for better handling of hard examples
        # Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        # Combines with class weights for maximum effectiveness
        
        class FocalBCELoss(nn.Module):
            """Focal Loss combined with class weights for imbalanced multi-label classification"""
            def __init__(self, pos_weight, alpha=0.25, gamma=2.0):
                super().__init__()
                self.pos_weight = pos_weight
                self.alpha = alpha
                self.gamma = gamma
                self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
            
            def forward(self, logits, targets):
                # Standard BCE loss
                bce_loss = self.bce(logits, targets)
                
                # Get probabilities
                p = torch.sigmoid(logits)
                
                # Calculate p_t (probability of true class)
                p_t = p * targets + (1 - p) * (1 - targets)
                
                # Focal term: (1 - p_t)^gamma
                focal_weight = (1 - p_t) ** self.gamma
                
                # Apply focal weight
                focal_loss = self.alpha * focal_weight * bce_loss
                
                return focal_loss.mean()
        
        # Use Focal Loss for better performance on imbalanced data
        pos_weight = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        criterion = FocalBCELoss(pos_weight=pos_weight, alpha=0.25, gamma=2.0)
        logger.info(f"[OK] Using Focal Loss with class weights (alpha=0.25, gamma=2.0)")
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_epoch = 0
        
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*80}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in tqdm(train_loader, desc='Training', leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                wav2vec_features = batch['wav2vec_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask, wav2vec_features)
                loss = criterion(logits, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().detach().numpy()
                preds = (probs > 0.5).astype(int)
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validating', leave=False):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    wav2vec_features = batch['wav2vec_features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits = self.model(input_ids, attention_mask, wav2vec_features)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
                    val_probs.extend(probs)
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_preds = np.array(train_preds)
            train_labels = np.array(train_labels)
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            val_probs = np.array(val_probs)
            
            # Per-class metrics
            label_names = ['filler', 'clarity', 'pacing']
            train_metrics = {}
            val_metrics = {}
            
            logger.info(f"\nTrain Loss: {avg_train_loss:.4f}")
            logger.info(f"Val Loss: {avg_val_loss:.4f}")
            
            for i, name in enumerate(label_names):
                # Train metrics
                train_acc = accuracy_score(train_labels[:, i], train_preds[:, i])
                train_metrics[name] = {'accuracy': train_acc}
                
                # Val metrics
                val_acc = accuracy_score(val_labels[:, i], val_preds[:, i])
                val_auc = roc_auc_score(val_labels[:, i], val_probs[:, i])
                val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                    val_labels[:, i], val_preds[:, i], average='binary', zero_division=0
                )
                val_metrics[name] = {
                    'accuracy': val_acc,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1,
                    'auc': val_auc
                }
                
                logger.info(f"\n{name.capitalize()}:")
                logger.info(f"  Train Acc: {train_acc:.4f}")
                logger.info(f"  Val Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            
            # Store history
            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)
            for name in label_names:
                self.train_metrics_history[name].append(train_metrics[name]['accuracy'])
                self.val_metrics_history[name].append(val_metrics[name]['accuracy'])
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                self.save_model()
                logger.info(f"\n[OK] Best model saved! (Val Loss: {best_val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"\n[!] No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    logger.info(f"\n[STOP] Early stopping triggered at epoch {epoch + 1}")
                    logger.info(f"Best model was at epoch {best_epoch} with val loss {best_val_loss:.4f}")
                    break
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info("="*80)
        
        # Load best model
        self.load_best_model()
        
        # Generate training curves
        self.plot_training_curves()
    
    def save_model(self):
        """Save hybrid model"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_dir / 'model.pth')
        self.tokenizer.save_pretrained(self.model_dir)
        logger.debug(f"Model saved to {self.model_dir}")
    
    def load_best_model(self):
        """Load best model"""
        model_path = self.model_dir / 'model.pth'
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded best model from {model_path}")
    
    def plot_training_curves(self):
        """Generate training curve plots"""
        logger.info("Generating training curves...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0, 0].plot(self.train_loss_history, label='Train Loss', marker='o')
        axes[0, 0].plot(self.val_loss_history, label='Val Loss', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves per class
        for i, name in enumerate(['filler', 'clarity', 'pacing']):
            row = (i + 1) // 2
            col = (i + 1) % 2
            if row < 2 and col < 2:
                axes[row, col].plot(self.train_metrics_history[name], label=f'Train {name}', marker='o')
                axes[row, col].plot(self.val_metrics_history[name], label=f'Val {name}', marker='s')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].set_title(f'{name.capitalize()} Accuracy')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        plot_path = reports_dir / f'training_curves_{V2_SUFFIX}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OK] Training curves saved to {plot_path}")
    
    def evaluate(self, test_data_path):
        """Comprehensive evaluation on test set"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATION ON TEST SET")
        logger.info("="*80)
        
        test_dataset = HybridSpeechDataset(test_data_path, extract_features=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        logger.info("Running inference on test set...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                wav2vec_features = batch['wav2vec_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask, wav2vec_features)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        logger.info("\n" + "="*80)
        logger.info("TEST SET RESULTS (Hybrid BERT + Wav2Vec2)")
        logger.info("="*80)
        
        # Calculate metrics per class
        label_names = ['filler', 'clarity', 'pacing']
        metrics_dict = {}
        
        for i, name in enumerate(label_names):
            # Calculate all metrics
            acc = accuracy_score(all_labels[:, i], all_preds[:, i])
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[:, i], all_preds[:, i], average='binary', zero_division=0
            )
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            
            metrics_dict[name] = {
                'accuracy': float(acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'auc': float(auc)
            }
            
            logger.info(f"\n{name.capitalize()}:")
            logger.info(f"  Accuracy:  {acc:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
            logger.info(f"  AUC-ROC:   {auc:.4f}")
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(all_labels, all_preds, target_names=label_names))
        
        # Save metrics to JSON
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        metrics_path = reports_dir / f'metrics_{V2_SUFFIX}.json'
        
        metrics_output = {
            'bert': metrics_dict,
            'generated_at': datetime.now().isoformat(),
            'model_version': V2_SUFFIX
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_output, f, indent=2)
        
        logger.info(f"\n[OK] Metrics saved to {metrics_path}")
        
        return metrics_dict


def main():
    """Main entry point"""
    data_dir = Path('data')
    
    train_path = data_dir / f'bert_train_{V2_SUFFIX}.pt'
    val_path = data_dir / f'bert_val_{V2_SUFFIX}.pt'
    test_path = data_dir / f'bert_test_{V2_SUFFIX}.pt'
    
    # Check if files exist
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please run preprocessing first: python data/preprocess_v2.py")
        return
    
    if not val_path.exists():
        logger.error(f"Validation data not found: {val_path}")
        return
    
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        return
    
    logger.info("="*80)
    logger.info("HYBRID BERT + WAV2VEC2 TRAINING V2")
    logger.info("="*80)
    
    # Train hybrid model
    trainer = HybridSpeechTrainerV2()
    trainer.train(train_path, val_path, epochs=10, batch_size=8, lr=2e-5)
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("RUNNING FINAL EVALUATION")
    logger.info("="*80)
    trainer.evaluate(test_path)
    
    logger.info("\n" + "="*80)
    logger.info("[OK] Hybrid training pipeline complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()


