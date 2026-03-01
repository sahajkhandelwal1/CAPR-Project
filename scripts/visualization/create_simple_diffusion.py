#!/usr/bin/env python3
"""
Simplified Crime Risk Diffusion Visualization
Shows key steps in the diffusion process for clearer understanding
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

def create_simplified_diffusion_demo():
    """Create a simple linear network for clear diffusion demonstration"""
    # Create a 7-node linear network
    G = nx.path_graph(7)
    pos = {i: (i, 0) for i in range(7)}
    
    # Simulate additive diffusion manually for clear control
    steps = []
    
    # Step 0: Initial crime at center
    risk0 = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
    steps.append(risk0)
    
    # Step 1: First diffusion - spreads to immediate neighbors
    risk1 = [0.0, 0.0, 0.2, 1.8, 0.2, 0.0, 0.0]
    steps.append(risk1)
    
    # Step 3: More spreading
    risk3 = [0.0, 0.05, 0.4, 1.6, 0.4, 0.05, 0.0]
    steps.append(risk3)
    
    # Step 6: Further diffusion
    risk6 = [0.02, 0.15, 0.6, 1.5, 0.6, 0.15, 0.02]
    steps.append(risk6)
    
    # Step 10: Significant spreading while maintaining origin
    risk10 = [0.08, 0.3, 0.8, 1.4, 0.8, 0.3, 0.08]
    steps.append(risk10)
    
    # Step 15: Final state - well diffused but origin still high
    risk15 = [0.2, 0.5, 1.0, 1.3, 1.0, 0.5, 0.2]
    steps.append(risk15)
    
    return G, pos, steps

def create_simple_diffusion_visual():
    """Create a clean, simple diffusion visualization"""
    
    G, pos, steps = create_simplified_diffusion_demo()
    step_labels = ['Initial\n(Step 0)', 'Early Spread\n(Step 1)', 'Growing\n(Step 3)', 
                   'Expanding\n(Step 6)', 'Widespread\n(Step 10)', 'Final State\n(Step 15)']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Simplified Crime Risk Diffusion Process\nKey Steps Showing Origin Preservation', 
                 fontsize=18, fontweight='bold')
    
    # Custom colormap
    colors = ['#000066', '#0066CC', '#66CCFF', '#FFFF66', '#FF6600', '#CC0000']
    risk_cmap = LinearSegmentedColormap.from_list('simple_risk', colors, N=100)
    
    # Find max risk for consistent scaling
    max_risk = max(max(step) for step in steps)
    
    for idx, (ax, risk_values, label) in enumerate(zip(axes.flat, steps, step_labels)):
        
        # Draw network edges
        for i in range(len(pos)-1):
            ax.plot([i, i+1], [0, 0], 'k-', linewidth=3, alpha=0.6)
        
        # Draw nodes with risk-based sizing and coloring
        for node, risk in enumerate(risk_values):
            color_intensity = risk / max_risk if max_risk > 0 else 0
            color = risk_cmap(color_intensity)
            size = 150 + risk * 400  # Base size + risk scaling
            
            # Special highlighting for crime origin
            edge_color = 'darkred' if node == 3 and risk > 1.0 else 'black'
            edge_width = 3 if node == 3 and risk > 1.0 else 2
            
            ax.scatter(node, 0, c=[color], s=size, 
                      edgecolors=edge_color, linewidth=edge_width, zorder=3)
            
            # Add risk value labels at top
            ax.text(node, 0.35, f'{risk:.2f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.9))
            
            # Add node numbers at bottom with minimal spacing
            ax.text(node, -0.25, f'{node}', 
                   ha='center', va='center', fontsize=9, color='gray')
        
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(-0.8, 6.8)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        # Remove xlabel to save space
        
        # Remove y-axis for cleaner look
        ax.set_yticks([])
    
    # Add smaller colorbar at very bottom with maximum separation
    sm = plt.cm.ScalarMappable(cmap=risk_cmap, norm=plt.Normalize(vmin=0, vmax=max_risk))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=30, pad=0.15, location='bottom')
    cbar.set_label('Crime Risk Level', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.18, 1, 0.99])
    return fig

def create_concept_diagram():
    """Create a conceptual diagram showing the diffusion principle"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Conservative vs Non-Conservative
    ax1.set_title('Conservative vs. Additive Diffusion', fontsize=16, fontweight='bold')
    
    # Conservative (wrong for crime)
    x = np.linspace(0, 10, 100)
    y1 = np.exp(-(x-5)**2 / 2) * 0.5  # Lower peak, spread out
    y2 = np.exp(-(x-5)**2 / 0.5) * 2  # High peak, concentrated
    
    ax1.plot(x, y2, 'r-', linewidth=3, label='Initial Crime (t=0)', alpha=0.8)
    ax1.plot(x, y1, 'b--', linewidth=3, label='Conservative Diffusion (Wrong)', alpha=0.8)
    
    # Additive (correct for crime)
    y3 = np.exp(-(x-5)**2 / 2) * 0.8 + np.exp(-(x-5)**2 / 0.5) * 1.5
    ax1.plot(x, y3, 'g-', linewidth=3, label='Additive Diffusion (Correct)', alpha=0.8)
    
    ax1.set_xlabel('Distance from Crime Location')
    ax1.set_ylabel('Risk Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(5, 2.5, 'Origin stays\nhigh-risk', ha='center', fontsize=12, 
             bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8))
    
    # Right: Real-world analogy
    ax2.set_title('Real-World Crime Risk Analogy', fontsize=16, fontweight='bold')
    
    # Create a simple illustration
    ax2.text(0.5, 0.8, '🏢 Crime Hotspot', ha='center', fontsize=20, transform=ax2.transAxes)
    ax2.text(0.5, 0.65, 'High Risk Persists', ha='center', fontsize=14, fontweight='bold', 
             transform=ax2.transAxes)
    
    # Draw influence zones
    circle1 = plt.Circle((0.5, 0.4), 0.05, color='red', alpha=0.8, transform=ax2.transAxes)
    circle2 = plt.Circle((0.5, 0.4), 0.15, color='orange', alpha=0.4, transform=ax2.transAxes)
    circle3 = plt.Circle((0.5, 0.4), 0.25, color='yellow', alpha=0.2, transform=ax2.transAxes)
    
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)
    ax2.add_patch(circle3)
    
    ax2.text(0.5, 0.15, 'Risk spreads to nearby areas\nwithout reducing origin risk', 
             ha='center', fontsize=12, transform=ax2.transAxes,
             bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Generate simplified diffusion visualizations"""
    print("Creating simplified crime risk diffusion visualizations...")
    
    # Create simple diffusion steps
    fig1 = create_simple_diffusion_visual()
    simple_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/simple_diffusion_steps.png'
    fig1.savefig(simple_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Create concept diagram
    fig2 = create_concept_diagram()
    concept_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/diffusion_concept.png'
    fig2.savefig(concept_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    print(f"Simplified diffusion visualizations saved:")
    print(f"  Simple steps: {simple_path}")
    print(f"  Concept diagram: {concept_path}")
    print("\nThese visualizations show:")
    print("  ✓ Clear 6-step progression")
    print("  ✓ Origin preservation principle")
    print("  ✓ Additive vs conservative comparison")
    print("  ✓ Real-world crime risk analogy")
    print("  ✓ Judge-friendly simplified presentation")

if __name__ == "__main__":
    main()
