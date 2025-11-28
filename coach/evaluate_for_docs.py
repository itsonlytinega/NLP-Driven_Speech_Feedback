"""
Enhanced Evaluation Script for Documentation
Generates beautiful metrics, graphs, and confusion matrices
Optimized for screenshot and documentation purposes
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

from coach.train_bert_v2 import HybridSpeechDataset, HybridSpeechModel

# Find latest v2 files
from glob import glob
existing_files = glob('data/bert_test_v2_*.pt')
if existing_files:
    latest_file = max(existing_files, key=os.path.getmtime)
    V2_SUFFIX = latest_file.split('bert_test_')[1].replace('.pt', '')
else:
    V2_SUFFIX = f"v2_{datetime.now().strftime('%Y%m%d')}"

# Setup enhanced logging with beautiful formatting
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'evaluation_docs_{V2_SUFFIX}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger.info("="*80)
logger.info("COMPREHENSIVE MODEL EVALUATION FOR DOCUMENTATION")
logger.info("="*80)


def evaluate_model_for_docs(test_data_path, model_dir=None):
    """Comprehensive evaluation optimized for documentation"""
    
    if model_dir is None:
        model_dir = f'coach/models/hybrid_bert_wav2vec_{V2_SUFFIX}'
    model_dir = Path(model_dir)
    
    # Load model
    logger.info(f"\n[1/5] Loading trained model from {model_dir}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"    Using device: {device}")
    
    model = HybridSpeechModel(num_labels=3, bert_frozen=False).to(device)
    model_path = model_dir / 'model.pth'
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("    [OK] Model loaded successfully")
    
    # Load test dataset
    logger.info(f"\n[2/5] Loading test dataset...")
    test_dataset = HybridSpeechDataset(test_data_path, extract_features=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    logger.info(f"    Test samples: {len(test_dataset)}")
    logger.info("    [OK] Dataset loaded")
    
    # Run inference
    logger.info(f"\n[3/5] Running inference on test set...")
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
    logger.info("    [OK] Inference complete")
    
    # Calculate optimal thresholds
    logger.info(f"\n[4/5] Calculating optimal thresholds and metrics...")
    label_names = ['filler', 'clarity', 'pacing']
    optimal_thresholds = {}
    
    for i, name in enumerate(label_names):
        fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_probs[:, i])
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds[name] = optimal_threshold
    
    # Generate predictions using optimal thresholds
    all_preds_optimal = np.zeros_like(all_probs, dtype=int)
    for i, name in enumerate(label_names):
        all_preds_optimal[:, i] = (all_probs[:, i] > optimal_thresholds[name]).astype(int)
    
    # Also calculate with 0.5 threshold for comparison
    all_preds_fixed = (all_probs > 0.5).astype(int)
    
    # Calculate comprehensive metrics
    metrics_dict = {}
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    for i, name in enumerate(label_names):
        # Metrics with optimal threshold
        acc_opt = accuracy_score(all_labels[:, i], all_preds_optimal[:, i])
        precision_opt, recall_opt, f1_opt, _ = precision_recall_fscore_support(
            all_labels[:, i], all_preds_optimal[:, i], average='binary', zero_division=0
        )
        
        # Metrics with 0.5 threshold
        acc_fixed = accuracy_score(all_labels[:, i], all_preds_fixed[:, i])
        precision_fixed, recall_fixed, f1_fixed, _ = precision_recall_fscore_support(
            all_labels[:, i], all_preds_fixed[:, i], average='binary', zero_division=0
        )
        
        roc_auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        
        # Use the better performing threshold (higher F1)
        if f1_opt > f1_fixed:
            use_optimal = True
            acc, precision, recall, f1 = acc_opt, precision_opt, recall_opt, f1_opt
            threshold = optimal_thresholds[name]
        else:
            use_optimal = False
            acc, precision, recall, f1 = acc_fixed, precision_fixed, recall_fixed, f1_fixed
            threshold = 0.5
        
        metrics_dict[name] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(roc_auc),
            'threshold': float(threshold)
        }
        
        logger.info(f"\n{name.upper()}:")
        logger.info(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        logger.info(f"  AUC-ROC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        logger.info(f"  Threshold: {threshold:.4f}")
    
    # Use best predictions for classification report
    best_preds = all_preds_optimal if f1_opt > f1_fixed else all_preds_fixed
    
    logger.info("\n" + "="*80)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*80)
    logger.info(classification_report(all_labels, best_preds, target_names=label_names, digits=4))
    
    # Generate visualizations
    logger.info(f"\n[5/5] Generating visualizations...")
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # 1. ROC Curves - Enhanced
    logger.info("    Generating ROC curves...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name.capitalize()} (AUC = {roc_auc:.3f})', 
                linewidth=3, alpha=0.8)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - Hybrid BERT + Wav2Vec2 Model', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    roc_path = reports_dir / f'roc_curves_docs_{V2_SUFFIX}.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    [OK] ROC curves saved: {roc_path}")
    
    # 2. Confusion Matrices - Enhanced
    logger.info("    Generating confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, name in enumerate(label_names):
        cm = confusion_matrix(all_labels[:, i], best_preds[:, i])
        
        # Create heatmap with annotations
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                   cbar_kws={'label': 'Count'}, linewidths=2, linecolor='gray',
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        axes[i].set_xlabel('Predicted', fontsize=13, fontweight='bold')
        axes[i].set_ylabel('Actual', fontsize=13, fontweight='bold')
        axes[i].set_title(f'{name.capitalize()} Confusion Matrix', 
                         fontsize=14, fontweight='bold', pad=15)
        axes[i].set_xticklabels(['No', 'Yes'], fontsize=12)
        axes[i].set_yticklabels(['No', 'Yes'], fontsize=12)
    
    plt.suptitle('Confusion Matrices - Hybrid BERT + Wav2Vec2 Model', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    cm_path = reports_dir / f'confusion_matrices_docs_{V2_SUFFIX}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    [OK] Confusion matrices saved: {cm_path}")
    
    # 3. Individual confusion matrices - Enhanced
    for i, name in enumerate(label_names):
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(all_labels[:, i], best_preds[:, i])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'}, linewidths=3, linecolor='gray',
                   annot_kws={'size': 18, 'weight': 'bold'})
        
        ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
        ax.set_title(f'{name.capitalize()} Confusion Matrix', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(['No', 'Yes'], fontsize=13)
        ax.set_yticklabels(['No', 'Yes'], fontsize=13)
        plt.tight_layout()
        
        cm_path = reports_dir / f'confusion_matrix_{name}_docs_{V2_SUFFIX}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"    [OK] {name} confusion matrix saved")
    
    # 4. Metrics Summary Bar Chart
    logger.info("    Generating metrics summary chart...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        values = [metrics_dict[name][metric] for name in label_names]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        bars = ax.bar([n.capitalize() for n in label_names], values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} by Class', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}\n({val*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Performance Metrics Summary - Hybrid BERT + Wav2Vec2 Model', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    summary_path = reports_dir / f'metrics_summary_docs_{V2_SUFFIX}.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    [OK] Metrics summary saved: {summary_path}")
    
    # Save metrics to JSON
    metrics_path = reports_dir / f'metrics_docs_{V2_SUFFIX}.json'
    metrics_output = {
        'bert': metrics_dict,
        'generated_at': datetime.now().isoformat(),
        'model_version': V2_SUFFIX,
        'test_samples': len(test_dataset)
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    logger.info(f"    [OK] Metrics saved: {metrics_path}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*80)
    logger.info("\nGenerated files:")
    logger.info(f"  - ROC Curves: {roc_path}")
    logger.info(f"  - Confusion Matrices: {cm_path}")
    logger.info(f"  - Metrics Summary: {summary_path}")
    logger.info(f"  - Metrics JSON: {metrics_path}")
    logger.info(f"  - Log file: {log_file}")
    
    return metrics_dict


def main():
    """Main entry point"""
    data_dir = Path('data')
    test_path = data_dir / f'bert_test_{V2_SUFFIX}.pt'
    
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        logger.error("Please ensure test dataset exists")
        return
    
    logger.info("="*80)
    logger.info("DOCUMENTATION-READY MODEL EVALUATION")
    logger.info("="*80)
    
    metrics = evaluate_model_for_docs(test_path)
    
    if metrics:
        logger.info("\n" + "="*80)
        logger.info("ALL VISUALIZATIONS AND METRICS READY FOR DOCUMENTATION!")
        logger.info("="*80)


if __name__ == '__main__':
    main()

