#!/usr/bin/env python3
"""
Fixed Ultra-Simple Crime Risk Diffusion Visualization
Diagrams positioned at the VERY TOP with colorbar at bottom
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

def create_ultra_simple_diffusion_fixed():
    """Create the clearest possible diffusion demonstration with top positioning"""
    
    # Create figure with explicit positioning
    fig = plt.figure(figsize=(15, 8))
    
    # Create grid layout - top 92% for diagrams, bottom 2% for tiny colorbar
    gs = gridspec.GridSpec(2, 3, figure=fig, 
                          height_ratios=[20, 0.8],  # Diagrams 25x bigger than colorbar
                          hspace=0.8, wspace=0.3)
    
    # Title at very top
    fig.suptitle('Crime Risk Diffusion: Key Stages', fontsize=20, fontweight='bold', y=0.95)
    
    # Define the three key stages
    stages = [
        {'risks': [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 
         'title': 'Initial Crime\nLocation'},
        
        {'risks': [0.0, 0.05, 0.4, 1.6, 0.4, 0.05, 0.0], 
         'title': 'Risk Spreads\nto Neighbors'},
        
        {'risks': [0.2, 0.5, 1.0, 1.3, 1.0, 0.5, 0.2], 
         'title': 'Final Diffused\nState'}
    ]
    
    # Custom colormap
    colors = ['#000066', '#0066CC', '#66CCFF', '#FFFF66', '#FF6600', '#CC0000']
    risk_cmap = LinearSegmentedColormap.from_list('ultra_simple', colors, N=100)
    
    max_risk = 2.0
    
    # Create subplots in top row
    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        
        stage = stages[i]
        risk_values = stage['risks']
        
        # Draw network edges
        for j in range(6):
            ax.plot([j, j+1], [0, 0], 'k-', linewidth=4, alpha=0.7)
        
        # Draw nodes with risk-based visualization
        for node, risk in enumerate(risk_values):
            color_intensity = risk / max_risk
            color = risk_cmap(color_intensity)
            size = 200 + risk * 600
            
            if node == 3:
                edge_color = 'darkred'
                edge_width = 4
            else:
                edge_color = 'black'
                edge_width = 2
            
            ax.scatter(node, 0, c=[color], s=size, marker='o',
                      edgecolors=edge_color, linewidth=edge_width, zorder=3)
            
            # Risk value labels positioned much higher with more space from nodes
            if risk > 0.01:
                ax.text(node, 0.9, f'{risk:.1f}', 
                       ha='center', va='center', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.15", facecolor='white', 
                                edgecolor='black', linewidth=1, alpha=0.9))
        
        # Node labels at bottom with much more space from nodes
        for j in range(7):
            ax.text(j, -0.9, f'N{j}', ha='center', va='center', 
                   fontsize=10, color='gray', fontweight='bold')
        
        # Configure subplot with much bigger box for spacing
        ax.set_title(stage['title'], fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-1.4, 1.4)  # Much bigger vertical space for spacing
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Create colorbar subplot in bottom row, only center column for narrow width
    cbar_ax = fig.add_subplot(gs[1, 1])  # Only middle column instead of spanning all
    
    # Create much smaller and narrower colorbar
    sm = plt.cm.ScalarMappable(cmap=risk_cmap, norm=plt.Normalize(vmin=0, vmax=max_risk))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', shrink=0.8, aspect=15)
    cbar.set_label('Crime Risk Level', fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=7)  # Smaller tick labels
    
    # Adjust overall layout with minimal space for colorbar
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    
    return fig

def create_simple_diffusion_fixed():
    """Create 6-step diffusion with fixed top positioning"""
    
    # Create figure with explicit positioning
    fig = plt.figure(figsize=(15, 10))
    
    # Create grid layout - top 92% for diagrams, bottom 2% for tiny colorbar
    gs = gridspec.GridSpec(3, 3, figure=fig, 
                          height_ratios=[8, 8, 0.6],  # Two big rows of plots, tiny colorbar
                          hspace=0.5, wspace=0.3)
    
    # Title at very top
    fig.suptitle('Simplified Crime Risk Diffusion Process\nKey Steps Showing Origin Preservation', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Define stages
    stages_data = [
        ([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0], 'Initial\n(Step 0)'),
        ([0.0, 0.0, 0.2, 1.8, 0.2, 0.0, 0.0], 'Early Spread\n(Step 1)'),
        ([0.0, 0.05, 0.4, 1.6, 0.4, 0.05, 0.0], 'Growing\n(Step 3)'),
        ([0.02, 0.15, 0.6, 1.5, 0.6, 0.15, 0.02], 'Expanding\n(Step 6)'),
        ([0.08, 0.3, 0.8, 1.4, 0.8, 0.3, 0.08], 'Widespread\n(Step 10)'),
        ([0.2, 0.5, 1.0, 1.3, 1.0, 0.5, 0.2], 'Final State\n(Step 15)')
    ]
    
    # Custom colormap
    colors = ['#000066', '#0066CC', '#66CCFF', '#FFFF66', '#FF6600', '#CC0000']
    risk_cmap = LinearSegmentedColormap.from_list('simple_risk', colors, N=100)
    
    max_risk = 2.0
    
    # Create subplots in top two rows (2x3 grid)
    axes = []
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        risk_values, label = stages_data[i]
        
        # Draw network edges
        for j in range(6):
            ax.plot([j, j+1], [0, 0], 'k-', linewidth=3, alpha=0.6)
        
        # Draw nodes
        for node, risk in enumerate(risk_values):
            color_intensity = risk / max_risk
            color = risk_cmap(color_intensity)
            size = 150 + risk * 400
            
            edge_color = 'darkred' if node == 3 and risk > 1.0 else 'black'
            edge_width = 3 if node == 3 and risk > 1.0 else 2
            
            ax.scatter(node, 0, c=[color], s=size, 
                      edgecolors=edge_color, linewidth=edge_width, zorder=3)
            
            # Risk value labels positioned much higher with more space from nodes
            if risk > 0.01:
                ax.text(node, 0.7, f'{risk:.2f}', 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))
        
        # Node numbers with much more space from nodes
        for j in range(7):
            ax.text(j, -0.7, f'{j}', ha='center', va='center', 
                   fontsize=10, color='gray', fontweight='bold')
        
        # Configure subplot with bigger box for more spacing
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-1.1, 1.1)  # Much bigger vertical space for spacing
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Create colorbar subplot in bottom row, only center column for narrow width
    cbar_ax = fig.add_subplot(gs[2, 1])  # Only middle column instead of spanning all
    
    # Create much smaller and narrower colorbar
    sm = plt.cm.ScalarMappable(cmap=risk_cmap, norm=plt.Normalize(vmin=0, vmax=max_risk))
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', shrink=0.8, aspect=15)
    cbar.set_label('Crime Risk Level', fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=7)  # Smaller tick labels
    
    # Adjust layout with minimal space for colorbar
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    
    return fig

def main():
    """Generate fixed diffusion visualizations"""
    print("Creating FIXED ultra-simple crime risk diffusion visualizations...")
    
    # Create ultra-simple 3-stage visualization
    fig1 = create_ultra_simple_diffusion_fixed()
    ultra_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/ultra_simple_diffusion_fixed.png'
    fig1.savefig(ultra_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Create 6-step visualization
    fig2 = create_simple_diffusion_fixed()
    simple_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/simple_diffusion_steps_fixed.png'
    fig2.savefig(simple_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    print(f"FIXED diffusion visualizations saved:")
    print(f"  Ultra-simple 3-stage: {ultra_path}")
    print(f"  Simple 6-step: {simple_path}")
    print("\nThese visualizations have:")
    print("  ✓ Diagrams positioned at VERY TOP")
    print("  ✓ Colorbar at bottom with clear separation")
    print("  ✓ No overlapping elements")
    print("  ✓ Clean, professional layout")
    print("  ✓ Maximum space for the diffusion process")

if __name__ == "__main__":
    main()
