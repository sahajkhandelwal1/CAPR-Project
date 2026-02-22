"""
Visualization script to compare original vs diffused risk scores.
Shows the effectiveness of the spatial buffering algorithm.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_diffusion_comparison():
    """Create comparison plots showing the impact of risk diffusion."""
    
    # Load diffused scores
    df = pd.read_csv('data/processed/edge_risk_scores_diffused.csv')
    
    original_scores = df['risk_score_original'].values
    buffered_scores = df['risk_score_buffered'].values
    changes = df['risk_score_change'].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Risk Diffusion & Spatial Buffering Results', fontsize=16, fontweight='bold')
    
    # 1. Score distribution comparison
    axes[0, 0].hist(original_scores, bins=50, alpha=0.7, label='Original', color='red', density=True)
    axes[0, 0].hist(buffered_scores, bins=50, alpha=0.7, label='Buffered', color='blue', density=True)
    axes[0, 0].set_xlabel('Risk Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Risk Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Before vs after scatter plot
    sample_indices = np.random.choice(len(df), 5000, replace=False)  # Sample for visibility
    axes[0, 1].scatter(original_scores[sample_indices], buffered_scores[sample_indices], 
                      alpha=0.5, s=1, c='purple')
    max_score = max(np.max(original_scores), np.max(buffered_scores))
    axes[0, 1].plot([0, max_score], [0, max_score], 'r--', label='No change line')
    axes[0, 1].set_xlabel('Original Risk Score')
    axes[0, 1].set_ylabel('Buffered Risk Score')
    axes[0, 1].set_title('Original vs Buffered Scores (Sample)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Change distribution
    axes[1, 0].hist(changes, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', label='No change')
    axes[1, 0].set_xlabel('Risk Score Change')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Risk Score Changes')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Statistics summary table
    axes[1, 1].axis('off')
    
    # Create statistics table
    stats_data = [
        ['Metric', 'Original', 'Buffered', 'Improvement'],
        ['Mean Risk', f'{np.mean(original_scores):.1f}', f'{np.mean(buffered_scores):.1f}', 
         f'{np.mean(buffered_scores) - np.mean(original_scores):.1f}'],
        ['Std Deviation', f'{np.std(original_scores):.1f}', f'{np.std(buffered_scores):.1f}', 
         f'{np.std(buffered_scores) - np.std(original_scores):.1f}'],
        ['Max Risk', f'{np.max(original_scores):.1f}', f'{np.max(buffered_scores):.1f}', 
         f'{np.max(buffered_scores) - np.max(original_scores):.1f}'],
        ['Min Risk Edges', f'{np.sum(original_scores <= 1.1):,}', f'{np.sum(buffered_scores <= 1.1):,}', 
         f'{np.sum(buffered_scores <= 1.1) - np.sum(original_scores <= 1.1):,}'],
        ['', '', '', ''],
        ['Edges Improved', '', f'{np.sum(changes > 0):,}', f'{np.sum(changes > 0)/len(changes)*100:.1f}%'],
        ['Mean |Change|', '', f'{np.mean(np.abs(changes)):.2f}', '']
    ]
    
    # Create table
    table = axes[1, 1].table(cellText=stats_data, cellLoc='center', loc='center',
                           colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Diffusion Impact Summary', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('visualization')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'risk_diffusion_comparison.png', dpi=300, bbox_inches='tight')
    
    print("📊 Visualization saved to 'visualization/risk_diffusion_comparison.png'")
    
    return fig

if __name__ == "__main__":
    create_diffusion_comparison()
    plt.show()
