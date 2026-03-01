#!/usr/bin/env python3
"""
Ultra-Simple Crime Risk Diffusion Visualization
Shows just 3 key stages for maximum clarity
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

def create_ultra_simple_diffusion():
    """Create the clearest possible diffusion demonstration"""
    
    # Create figure with just 3 key stages
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Crime Risk Diffusion: Key Stages', fontsize=20, fontweight='bold')
    
    # Define the three key stages
    stages = [
        {'risks': [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 
         'title': 'Initial Crime\nLocation', 
         'description': 'Single high-risk\ncrime hotspot'},
        
        {'risks': [0.0, 0.05, 0.4, 1.6, 0.4, 0.05, 0.0], 
         'title': 'Risk Spreads\nto Neighbors', 
         'description': 'Origin stays high,\nneighbors gain risk'},
        
        {'risks': [0.2, 0.5, 1.0, 1.3, 1.0, 0.5, 0.2], 
         'title': 'Final Diffused\nState', 
         'description': 'Widespread influence,\norigin still dangerous'}
    ]
    
    # Custom colormap for clear risk visualization
    colors = ['#000066', '#0066CC', '#66CCFF', '#FFFF66', '#FF6600', '#CC0000']
    risk_cmap = LinearSegmentedColormap.from_list('ultra_simple', colors, N=100)
    
    max_risk = 2.0  # Known maximum for consistent scaling
    
    for ax, stage in zip(axes, stages):
        risk_values = stage['risks']
        
        # Draw network edges as thick black lines
        for i in range(6):
            ax.plot([i, i+1], [0, 0], 'k-', linewidth=4, alpha=0.7)
        
        # Draw nodes with risk-based visualization
        for node, risk in enumerate(risk_values):
            # Color and size based on risk
            color_intensity = risk / max_risk
            color = risk_cmap(color_intensity)
            size = 200 + risk * 600  # Large, visible nodes
            
            # Special styling for crime origin (node 3)
            if node == 3:
                edge_color = 'darkred'
                edge_width = 4
                marker = 'o'
            else:
                edge_color = 'black'
                edge_width = 2
                marker = 'o'
            
            ax.scatter(node, 0, c=[color], s=size, marker=marker,
                      edgecolors=edge_color, linewidth=edge_width, zorder=3)
            
            # Large, clear risk value labels positioned at top
            if risk > 0.01:  # Only show non-zero risks
                ax.text(node, 0.6, f'{risk:.1f}', 
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.15", facecolor='white', 
                                edgecolor='black', linewidth=1, alpha=0.9))
        
        # Clean up the plot - move to very top with minimal bottom space
        ax.set_xlim(-0.8, 6.8)
        ax.set_ylim(-0.8, 1.0)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Label nodes at bottom with minimal spacing
        for i in range(7):
            ax.text(i, -0.4, f'N{i}', ha='center', va='center', 
                   fontsize=9, color='gray')
        
        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add arrows between stages
        if ax != axes[-1]:  # Not the last subplot
            # Add arrow pointing to next stage
            ax.annotate('', xy=(7.2, 0), xytext=(6.8, 0),
                       arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    # Add smaller colorbar at very bottom with maximum separation
    sm = plt.cm.ScalarMappable(cmap=risk_cmap, norm=plt.Normalize(vmin=0, vmax=max_risk))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=30, pad=0.15, location='bottom')
    cbar.set_label('Crime Risk Level', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.18, 1, 0.99])
    return fig

def create_comparison_chart():
    """Create a side-by-side comparison of conservative vs additive diffusion"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Conservative vs. Additive Diffusion Models', fontsize=18, fontweight='bold')
    
    # Plot parameters
    nodes = list(range(7))
    
    # Conservative diffusion (wrong for crime modeling)
    ax1.set_title('❌ Conservative Diffusion (Unrealistic for Crime)', fontsize=14, color='red', fontweight='bold')
    
    conservative_stages = [
        [0, 0, 0, 2.0, 0, 0, 0],      # Initial
        [0, 0, 0.3, 1.4, 0.3, 0, 0], # Risk moves away from origin
        [0, 0.2, 0.5, 0.6, 0.5, 0.2, 0] # Origin becomes low-risk
    ]
    
    colors_conservative = ['red', 'orange', 'yellow']
    labels_conservative = ['Initial', 'Step 5', 'Step 10']
    
    for i, (risks, color, label) in enumerate(zip(conservative_stages, colors_conservative, labels_conservative)):
        ax1.plot(nodes, risks, 'o-', color=color, linewidth=3, markersize=8, 
                label=label, alpha=0.8)
    
    ax1.set_ylabel('Risk Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(3, 1.5, 'Crime origin\nbecomes safer\n(Unrealistic!)', 
             ha='center', va='center', fontsize=11, 
             bbox=dict(boxstyle="round", facecolor='pink', alpha=0.8))
    
    # Additive diffusion (correct for crime modeling)
    ax2.set_title('✅ Additive Diffusion (Realistic for Crime)', fontsize=14, color='green', fontweight='bold')
    
    additive_stages = [
        [0, 0, 0, 2.0, 0, 0, 0],        # Initial
        [0, 0.05, 0.4, 1.6, 0.4, 0.05, 0], # Risk spreads but origin stays high
        [0.2, 0.5, 1.0, 1.3, 1.0, 0.5, 0.2] # Widespread but origin still dangerous
    ]
    
    colors_additive = ['red', 'orange', 'darkred']
    labels_additive = ['Initial', 'Step 5', 'Step 10']
    
    for i, (risks, color, label) in enumerate(zip(additive_stages, colors_additive, labels_additive)):
        ax2.plot(nodes, risks, 'o-', color=color, linewidth=3, markersize=8, 
                label=label, alpha=0.8)
    
    ax2.set_xlabel('Node Position')
    ax2.set_ylabel('Risk Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(3, 1.8, 'Crime origin\nstays dangerous\n(Realistic!)', 
             ha='center', va='center', fontsize=11, 
             bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate ultra-simple diffusion visualizations"""
    print("Creating ultra-simple crime risk diffusion visualizations...")
    
    # Create ultra-simple 3-stage visualization
    fig1 = create_ultra_simple_diffusion()
    ultra_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/ultra_simple_diffusion.png'
    fig1.savefig(ultra_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Create comparison chart
    fig2 = create_comparison_chart()
    comparison_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/diffusion_comparison.png'
    fig2.savefig(comparison_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    print(f"Ultra-simple diffusion visualizations saved:")
    print(f"  Ultra-simple 3-stage: {ultra_path}")
    print(f"  Comparison chart: {comparison_path}")
    print("\nThese visualizations show:")
    print("  ✓ Clear 3-stage progression (Initial → Spreading → Final)")
    print("  ✓ Large, visible nodes and risk values")
    print("  ✓ Conservative vs additive model comparison")
    print("  ✓ Judge-friendly maximum clarity")
    print("  ✓ Key insight prominently displayed")

if __name__ == "__main__":
    main()
