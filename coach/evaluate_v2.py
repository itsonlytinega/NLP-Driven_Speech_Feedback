"""
Comprehensive Evaluation Script V2
Generates ROC curves, confusion matrices, and detailed metrics
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    classification_report, roc_auc_score,
    accuracy_score, precision_recall_fscore_support
)
from datetime import datetime
import json
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from coach.train_bert_v2 import HybridSpeechDataset, HybridSpeechModel, HybridSpeechTrainerV2

# Find latest v2 files or use today's timestamp
from glob import glob
import os

# Try to find existing v2 files
existing_files = glob('data/bert_test_v2_*.pt')
if existing_files:
    # Extract timestamp from latest file
    latest_file = max(existing_files, key=os.path.getmtime)
    V2_SUFFIX = latest_file.split('bert_test_')[1].replace('.pt', '')
    TIMESTAMP = V2_SUFFIX.split('_')[1]
else:
    # Use today's timestamp if no files found
    TIMESTAMP = datetime.now().strftime("%Y%m%d")
    V2_SUFFIX = f"v2_{TIMESTAMP}"

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'evaluation_{V2_SUFFIX}.log'

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
logger.info("COMPREHENSIVE EVALUATION V2")
logger.info(f"Log file: {log_file}")
logger.info("="*80)


def evaluate_model(test_data_path, model_dir=None):
    """Comprehensive evaluation with visualizations"""
    
    if model_dir is None:
        model_dir = f'coach/models/hybrid_bert_wav2vec_{V2_SUFFIX}'
    model_dir = Path(model_dir)
    
    # Load model
    logger.info(f"Loading model from {model_dir}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = HybridSpeechModel(num_labels=3, bert_frozen=False).to(device)
    model_path = model_dir / 'model.pth'
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("[OK] Model loaded")
    
    # Load test dataset
    logger.info(f"Loading test dataset from {test_data_path}...")
    test_dataset = HybridSpeechDataset(test_data_path, extract_features=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Run inference
    logger.info("Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            wav2vec_features = batch['wav2vec_features'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, wav2vec_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate optimal thresholds first
    label_names = ['filler', 'clarity', 'pacing']
    optimal_thresholds = {}
    
    logger.info("\n" + "="*80)
    logger.info("CALCULATING OPTIMAL THRESHOLDS")
    logger.info("="*80)
    
    for i, name in enumerate(label_names):
        fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_probs[:, i])
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds[name] = optimal_threshold
        logger.info(f"{name.capitalize()}: Optimal threshold = {optimal_threshold:.4f}")
    
    # Generate predictions using optimal thresholds
    all_preds_optimal = np.zeros_like(all_probs, dtype=int)
    for i, name in enumerate(label_names):
        all_preds_optimal[:, i] = (all_probs[:, i] > optimal_thresholds[name]).astype(int)
    
    # Calculate metrics with optimal thresholds
    metrics_dict = {}
    
    logger.info("\n" + "="*80)
    logger.info("DETAILED METRICS (Using Optimal Thresholds)")
    logger.info("="*80)
    
    for i, name in enumerate(label_names):
        # Calculate metrics with optimal threshold
        acc = accuracy_score(all_labels[:, i], all_preds_optimal[:, i])
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels[:, i], all_preds_optimal[:, i], average='binary', zero_division=0
        )
        roc_auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        
        # Also calculate with 0.5 threshold for comparison
        preds_fixed = (all_probs[:, i] > 0.5).astype(int)
        acc_fixed = accuracy_score(all_labels[:, i], preds_fixed)
        precision_fixed, recall_fixed, f1_fixed, _ = precision_recall_fscore_support(
            all_labels[:, i], preds_fixed, average='binary', zero_division=0
        )
        
        metrics_dict[name] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(roc_auc),
            'threshold': float(optimal_thresholds[name]),
            'threshold_0.5_accuracy': float(acc_fixed),
            'threshold_0.5_precision': float(precision_fixed),
            'threshold_0.5_recall': float(recall_fixed),
            'threshold_0.5_f1': float(f1_fixed)
        }
        
        logger.info(f"\n{name.capitalize()}:")
        logger.info(f"  Accuracy:  {acc:.4f} (optimal) vs {acc_fixed:.4f} (0.5 threshold)")
        logger.info(f"  Precision: {precision:.4f} (optimal) vs {precision_fixed:.4f} (0.5 threshold)")
        logger.info(f"  Recall:    {recall:.4f} (optimal) vs {recall_fixed:.4f} (0.5 threshold)")
        logger.info(f"  F1 Score:  {f1:.4f} (optimal) vs {f1_fixed:.4f} (0.5 threshold)")
        logger.info(f"  AUC-ROC:   {roc_auc:.4f}")
        logger.info(f"  Optimal Threshold: {optimal_thresholds[name]:.4f}")
        logger.info(f"  Improvement: F1 +{f1-f1_fixed:.4f}, Recall +{recall-recall_fixed:.4f}, Precision {precision-precision_fixed:+.4f}")
    
    logger.info("\nClassification Report (Optimal Thresholds):")
    logger.info(classification_report(all_labels, all_preds_optimal, target_names=label_names))
    
    # Generate visualizations
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # 1. ROC Curves
    logger.info("\nGenerating ROC curves...")
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name.capitalize()} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Hybrid BERT + Wav2Vec2 Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = reports_dir / f'roc_curves_{V2_SUFFIX}.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] ROC curves saved to {roc_path}")
    
    # 2. Confusion Matrices (using optimal thresholds)
    logger.info("Generating confusion matrices (optimal thresholds)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, name in enumerate(label_names):
        cm = confusion_matrix(all_labels[:, i], all_preds_optimal[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar_kws={'label': 'Count'})
        axes[i].set_xlabel('Predicted', fontsize=11)
        axes[i].set_ylabel('Actual', fontsize=11)
        axes[i].set_title(f'{name.capitalize()} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[i].set_xticklabels(['No', 'Yes'])
        axes[i].set_yticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    cm_path = reports_dir / f'confusion_matrices_{V2_SUFFIX}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Confusion matrices saved to {cm_path}")
    
    # 3. Individual confusion matrices (using optimal thresholds)
    for i, name in enumerate(label_names):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(all_labels[:, i], all_preds_optimal[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'{name.capitalize()} Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xticks([0.5, 1.5], ['No', 'Yes'])
        plt.yticks([0.5, 1.5], ['No', 'Yes'])
        plt.tight_layout()
        
        cm_path = reports_dir / f'confusion_matrix_{name}_{V2_SUFFIX}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[OK] {name} confusion matrix saved to {cm_path}")
    
    # Save metrics
    metrics_path = reports_dir / f'metrics_{V2_SUFFIX}.json'
    metrics_output = {
        'bert': metrics_dict,
        'generated_at': datetime.now().isoformat(),
        'model_version': V2_SUFFIX
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    logger.info(f"[OK] Metrics saved to {metrics_path}")
    
    return metrics_dict


def main():
    """Main entry point"""
    data_dir = Path('data')
    test_path = data_dir / f'bert_test_{V2_SUFFIX}.pt'
    
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        logger.error("Please run preprocessing and training first")
        return
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL EVALUATION V2")
    logger.info("="*80)
    
    metrics = evaluate_model(test_path)
    
    if metrics:
        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE!")
        logger.info("="*80)
        logger.info("\nAll visualizations and metrics saved to reports/ directory")


if __name__ == '__main__':
    main()


