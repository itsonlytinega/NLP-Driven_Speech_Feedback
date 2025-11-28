"""
Hybrid BERT + Wav2Vec2 Model for Multi-Label Speech Analysis
DEFENSE: Fuses text (BERT) + audio (Wav2Vec2) for accent-robust, superior performance
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
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coach.audio_features import get_wav2vec_features

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridSpeechDataset(Dataset):
    """
    Dataset for Hybrid BERT + Wav2Vec2 training
    DEFENSE: Loads both text (BERT) and audio (Wav2Vec2) features
    """
    
    def __init__(self, data_path, extract_features=True):
        self.data = torch.load(data_path)
        self.extract_features = extract_features
        
        # Cache Wav2Vec2 features if extracting
        if extract_features and 'audio_paths' in self.data:
            logger.info("Extracting Wav2Vec2 features...")
            self.wav2vec_features = []
            for audio_path in tqdm(self.data['audio_paths'], desc='Extracting audio features'):
                if os.path.exists(audio_path):
                    features = get_wav2vec_features(audio_path)
                    self.wav2vec_features.append(torch.tensor(features, dtype=torch.float32))
                else:
                    # Zero features if audio missing
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
        
        # Add Wav2Vec2 features if available
        if self.wav2vec_features is not None:
            item['wav2vec_features'] = self.wav2vec_features[idx]
        else:
            # Zero features as fallback
            item['wav2vec_features'] = torch.zeros(768, dtype=torch.float32)
        
        return item


class HybridSpeechModel(nn.Module):
    """
    Hybrid BERT + Wav2Vec2 Model
    DEFENSE: Why this architecture?
    - BERT: 768-dim text features (semantic understanding)
    - Wav2Vec2: 768-dim audio features (accent-invariant, phonetic)
    - Fusion: Concatenate → Linear layer → 3 labels
    - Result: Superior to pure Wav2Vec2 (faster) or pure BERT (accent-robust)
    """
    
    def __init__(self, num_labels=3, bert_frozen=False):
        super().__init__()
        
        # BERT for text
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels,
            problem_type='multi_label_classification'
        )
        
        # Freeze BERT if requested (faster training)
        if bert_frozen:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Wav2Vec2 for audio (pre-trained, not fine-tuned)
        logger.info("Loading Wav2Vec2 for accent-robust features...")
        self.wav2vec = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
        self.wav2vec.eval()  # Use in eval mode (no fine-tuning)
        for param in self.wav2vec.parameters():
            param.requires_grad = False  # Freeze Wav2Vec2
        
        # Fusion layer: BERT (768) + Wav2Vec2 (768) → 768 → 3
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_labels)
        )
        
        # BERT already has classifier, but we'll use our fusion instead
        # Replace BERT's classifier with identity
        self.bert.classifier = nn.Identity()
    
    def forward(self, input_ids, attention_mask, wav2vec_features):
        """
        Forward pass: BERT text + Wav2Vec2 audio → fused features → predictions
        """
        # BERT text features (get sequence output, then pool)
        bert_outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # Pool: mean pooling over sequence length
        # Use attention_mask to ignore padding
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        bert_pooled = sum_embeddings / sum_mask  # [batch, 768]
        
        # Wav2Vec2 features (already extracted, shape: [batch, 768])
        if wav2vec_features.dim() == 1:
            wav2vec_features = wav2vec_features.unsqueeze(0)
        
        # Fuse: concatenate BERT + Wav2Vec2
        combined = torch.cat([bert_pooled, wav2vec_features], dim=1)  # [batch, 1536]
        
        # Fusion → predictions
        logits = self.fusion(combined)  # [batch, 3]
        
        return logits


class HybridSpeechTrainer:
    """Trainer for Hybrid BERT + Wav2Vec2 model"""
    
    def __init__(self, model_dir='coach/models/hybrid_bert_wav2vec', num_labels=3):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing Hybrid BERT + Wav2Vec2 model on {self.device}...")
        
        # Load hybrid model
        self.model = HybridSpeechModel(num_labels=num_labels, bert_frozen=False).to(self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        logger.info("[OK] Hybrid model loaded")
    
    def train(self, train_data_path, val_data_path, epochs=5, batch_size=8, lr=2e-5):
        """
        Train hybrid model
        DEFENSE: Smaller batch size (8) due to memory requirements (BERT + Wav2Vec2)
        """
        
        # Load datasets
        logger.info("Loading datasets...")
        train_dataset = HybridSpeechDataset(train_data_path, extract_features=True)
        val_dataset = HybridSpeechDataset(val_data_path, extract_features=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer (only train fusion + BERT classifier, Wav2Vec2 frozen)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*50}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch in tqdm(train_loader, desc='Training'):
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
                
                # Predictions
                probs = torch.sigmoid(logits).cpu().detach().numpy()
                preds = (probs > 0.5).astype(int)
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validating'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    wav2vec_features = batch['wav2vec_features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    logits = self.model(input_ids, attention_mask, wav2vec_features)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    # Predictions
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())
            
            # Metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"\nTrain Loss: {avg_train_loss:.4f}")
            logger.info(f"Val Loss: {avg_val_loss:.4f}")
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                logger.info("[OK] Model saved!")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("\n[OK] Training complete!")
        self.load_best_model()
    
    def save_model(self):
        """Save hybrid model"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_dir / 'model.pth')
        self.tokenizer.save_pretrained(self.model_dir)
        logger.info(f"Saved model to {self.model_dir}")
    
    def load_best_model(self):
        """Load best model"""
        model_path = self.model_dir / 'model.pth'
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded best model from {model_path}")
    
    def evaluate(self, test_data_path):
        """Evaluate on test set"""
        
        test_dataset = HybridSpeechDataset(test_data_path, extract_features=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
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
        
        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        logger.info("\n" + "="*50)
        logger.info("Test Set Results (Hybrid BERT + Wav2Vec2)")
        logger.info("="*50)
        
        # Per-class metrics
        label_names = ['filler', 'clarity', 'pacing']
        for i, name in enumerate(label_names):
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            acc = (all_preds[:, i] == all_labels[:, i]).mean()
            logger.info(f"{name.capitalize()}: AUC={auc:.3f}, Acc={acc:.3f}")
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(all_labels, all_preds, target_names=label_names))


def main():
    """Main entry point"""
    
    data_dir = Path('data')
    
    train_path = data_dir / 'bert_train.pt'
    test_path = data_dir / 'bert_test.pt'
    
    # Split train into train/val
    train_data = torch.load(train_path)
    val_size = len(train_data['input_ids']) // 5
    
    val_data = {
        'input_ids': train_data['input_ids'][:val_size],
        'attention_mask': train_data['attention_mask'][:val_size],
        'labels': train_data['labels'][:val_size],
        'audio_paths': train_data['audio_paths'][:val_size]
    }
    
    train_data_reduced = {
        'input_ids': train_data['input_ids'][val_size:],
        'attention_mask': train_data['attention_mask'][val_size:],
        'labels': train_data['labels'][val_size:],
        'audio_paths': train_data['audio_paths'][val_size:]
    }
    
    val_path = data_dir / 'bert_val.pt'
    torch.save(val_data, val_path)
    torch.save(train_data_reduced, train_path)
    
    # Train hybrid model
    logger.info("="*60)
    logger.info("HYBRID BERT + WAV2VEC2 TRAINING")
    logger.info("="*60)
    
    trainer = HybridSpeechTrainer()
    trainer.train(train_path, val_path, epochs=5, batch_size=8, lr=2e-5)
    
    # Evaluate
    trainer.evaluate(test_path)
    
    logger.info("\n[OK] Hybrid training pipeline complete!")


if __name__ == '__main__':
    main()
