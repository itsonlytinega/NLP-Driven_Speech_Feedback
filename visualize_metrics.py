"""
Visualization script for BERT model performance metrics.
Generates comprehensive charts from metrics JSON files.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

def load_metrics(json_path):
    """Load metrics from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_comparison_bar_chart(metrics, output_dir):
    """Create bar charts comparing metrics across tasks."""
    tasks = ['filler', 'clarity', 'pacing']
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = [metrics['bert'][task][metric] for task in tasks]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax.bar(tasks, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Created: metric_comparison_bars.png")

def create_grouped_bar_chart(metrics, output_dir):
    """Create grouped bar chart showing all metrics for each task."""
    tasks = ['filler', 'clarity', 'pacing']
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    x = np.arange(len(tasks))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, metric in enumerate(metric_names):
        values = [metrics['bert'][task][metric] for task in tasks]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, values, width, label=metric.capitalize(), alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('BERT Model Performance Metrics by Task', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in tasks])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grouped_metrics_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Created: grouped_metrics_chart.png")

def create_heatmap(metrics, output_dir):
    """Create heatmap of metrics across tasks."""
    tasks = ['filler', 'clarity', 'pacing']
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    data = []
    for task in tasks:
        row = [metrics['bert'][task][metric] for metric in metric_names]
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu', 
                xticklabels=[m.capitalize() for m in metric_names],
                yticklabels=[t.capitalize() for t in tasks],
                cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1,
                linewidths=1, linecolor='black')
    
    ax.set_title('BERT Model Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Created: metrics_heatmap.png")

def create_radar_chart(metrics, output_dir):
    """Create radar chart for each task."""
    tasks = ['filler', 'clarity', 'pacing']
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Number of variables
    N = len(metric_names)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (task, ax, color) in enumerate(zip(tasks, axes, colors)):
        values = [metrics['bert'][task][metric] for metric in metric_names]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=task.capitalize())
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metric_names], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f'{task.capitalize()} Performance', fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for angle, value, metric in zip(angles[:-1], values[:-1], metric_names):
            ax.text(angle, value + 0.05, f'{value:.3f}', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Created: radar_charts.png")

def create_summary_dashboard(metrics, output_dir):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    tasks = ['filler', 'clarity', 'pacing']
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # 1. Grouped bar chart (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(tasks))
    width = 0.15
    for i, metric in enumerate(metric_names):
        values = [metrics['bert'][task][metric] for task in tasks]
        offset = (i - 2) * width
        ax1.bar(x + offset, values, width, label=metric.capitalize(), alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Task', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.capitalize() for t in tasks])
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Heatmap (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    data = np.array([[metrics['bert'][task][metric] for metric in metric_names] for task in tasks])
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu', 
                xticklabels=[m.capitalize() for m in metric_names],
                yticklabels=[t.capitalize() for t in tasks],
                cbar_kws={'label': 'Score'}, ax=ax2, vmin=0, vmax=1,
                linewidths=0.5, linecolor='black')
    ax2.set_title('Heatmap', fontsize=13, fontweight='bold')
    
    # 3. AUC comparison (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    auc_values = [metrics['bert'][task]['auc'] for task in tasks]
    bars = ax3.bar(tasks, auc_values, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
    ax3.set_title('AUC by Task', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, auc_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. F1 Score comparison (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    f1_values = [metrics['bert'][task]['f1'] for task in tasks]
    bars = ax4.bar(tasks, f1_values, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax4.set_title('F1 Score by Task', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, 1.05)
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, f1_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Accuracy comparison (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    acc_values = [metrics['bert'][task]['accuracy'] for task in tasks]
    bars = ax5.bar(tasks, acc_values, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax5.set_title('Accuracy by Task', fontsize=13, fontweight='bold')
    ax5.set_ylim(0, 1.05)
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, acc_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Precision-Recall comparison (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    precision_values = [metrics['bert'][task]['precision'] for task in tasks]
    recall_values = [metrics['bert'][task]['recall'] for task in tasks]
    x = np.arange(len(tasks))
    width = 0.35
    ax6.bar(x - width/2, precision_values, width, label='Precision', alpha=0.8, edgecolor='black')
    ax6.bar(x + width/2, recall_values, width, label='Recall', alpha=0.8, edgecolor='black')
    ax6.set_xlabel('Task', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax6.set_title('Precision vs Recall', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([t.capitalize() for t in tasks])
    ax6.set_ylim(0, 1.05)
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Threshold values (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    threshold_values = [metrics['bert'][task]['threshold'] for task in tasks]
    bars = ax7.bar(tasks, threshold_values, color=colors, alpha=0.8, edgecolor='black')
    ax7.set_ylabel('Threshold', fontsize=11, fontweight='bold')
    ax7.set_title('Optimal Thresholds', fontsize=13, fontweight='bold')
    ax7.set_ylim(0, 1.05)
    ax7.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, threshold_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Model info text (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    info_text = f"""
    Model Version: {metrics.get('model_version', 'N/A')}
    Test Samples: {metrics.get('test_samples', 'N/A')}
    Generated: {metrics.get('generated_at', 'N/A')}
    
    Best Performance:
    • Filler: {metrics['bert']['filler']['f1']:.3f} F1
    • Clarity: {metrics['bert']['clarity']['f1']:.3f} F1
    • Pacing: {metrics['bert']['pacing']['f1']:.3f} F1
    """
    ax8.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('BERT Model Performance Dashboard', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(output_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Created: comprehensive_dashboard.png")

def main():
    """Main function to generate all visualizations."""
    # Get the metrics file path
    metrics_file = Path('reports/metrics_docs_v2_20251119.json')
    
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found!")
        return
    
    # Load metrics
    print(f"Loading metrics from {metrics_file}...")
    metrics = load_metrics(metrics_file)
    
    # Create output directory
    output_dir = Path('reports/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations in {output_dir}...\n")
    
    # Generate all visualizations
    create_comparison_bar_chart(metrics, output_dir)
    create_grouped_bar_chart(metrics, output_dir)
    create_heatmap(metrics, output_dir)
    create_radar_chart(metrics, output_dir)
    create_summary_dashboard(metrics, output_dir)
    
    print(f"\n[SUCCESS] All visualizations created successfully!")
    print(f"[INFO] Output directory: {output_dir}")

if __name__ == '__main__':
    main()

