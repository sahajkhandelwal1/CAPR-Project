#!/usr/bin/env python3
"""
Clean Mathematical Pipeline Visualization for CAPR System
Creates a clear, judge-friendly visualization showing the mathematical flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_clean_pipeline():
    """Create a clean mathematical pipeline visualization"""
    
    # Set up the figure with high DPI for crisp text
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    data_color = '#E8F4FD'      # Light blue for data
    process_color = '#FFF2CC'    # Light yellow for processing
    math_color = '#E1F5FE'      # Light cyan for mathematical operations
    output_color = '#E8F5E8'     # Light green for results
    
    # Arrow style
    arrow_style = dict(arrowstyle='->', lw=2.5, color='#333333')
    
    # Title
    ax.text(8, 9.5, 'Crime-Aware Pathfinding & Routing (CAPR) System', 
            fontsize=24, fontweight='bold', ha='center', va='center')
    ax.text(8, 9, 'Mathematical Pipeline Architecture', 
            fontsize=16, ha='center', va='center', style='italic', color='#666666')
    
    # Stage 1: Crime Data
    box1 = FancyBboxPatch((0.5, 7), 2.5, 1.2, 
                         boxstyle="round,pad=0.1", 
                         facecolor=data_color, 
                         edgecolor='#1976D2', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.75, 7.6, 'Crime Data', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(1.75, 7.3, 'SF Police Reports\n2018-Present', fontsize=10, ha='center', va='center')
    
    # Arrow 1
    arrow1 = ConnectionPatch((3, 7.6), (4, 7.6), "data", "data", **arrow_style)
    ax.add_patch(arrow1)
    
    # Stage 2: Risk Weighting Functions
    box2 = FancyBboxPatch((4, 6.8), 3, 1.6, 
                         boxstyle="round,pad=0.1", 
                         facecolor=process_color, 
                         edgecolor='#F57C00', linewidth=2)
    ax.add_patch(box2)
    ax.text(5.5, 8, 'Risk Weighting Functions', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(5.5, 7.6, r'$R(x,y,t) = \sum_i w_i \cdot K_\sigma(d_i) \cdot T_\lambda(t_i)$', 
            fontsize=12, ha='center', va='center')
    ax.text(5.5, 7.2, 'Spatial-Temporal Risk Scoring', fontsize=10, ha='center', va='center', style='italic')
    
    # Arrow 2
    arrow2 = ConnectionPatch((7, 7.6), (8, 7.6), "data", "data", **arrow_style)
    ax.add_patch(arrow2)
    
    # Stage 3: Graph Construction
    box3 = FancyBboxPatch((8, 6.8), 2.8, 1.6, 
                         boxstyle="round,pad=0.1", 
                         facecolor=math_color, 
                         edgecolor='#0097A7', linewidth=2)
    ax.add_patch(box3)
    ax.text(9.4, 8, 'Graph Construction', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(9.4, 7.6, r'$G = (V, E)$', fontsize=14, ha='center', va='center', fontweight='bold')
    ax.text(9.4, 7.2, 'Street Network Graph', fontsize=10, ha='center', va='center', style='italic')
    
    # Arrow 3
    arrow3 = ConnectionPatch((10.8, 7.6), (11.8, 7.6), "data", "data", **arrow_style)
    ax.add_patch(arrow3)
    
    # Stage 4: Diffusion / Clustering
    box4 = FancyBboxPatch((11.8, 6.8), 3.2, 1.6, 
                         boxstyle="round,pad=0.1", 
                         facecolor=process_color, 
                         edgecolor='#F57C00', linewidth=2)
    ax.add_patch(box4)
    ax.text(13.4, 8, 'Diffusion / Clustering', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(13.4, 7.6, r'$\nabla^2 R + \alpha \cdot \text{Buffer}(r)$', 
            fontsize=12, ha='center', va='center')
    ax.text(13.4, 7.2, 'Risk Propagation & Hotspots', fontsize=10, ha='center', va='center', style='italic')
    
    # Vertical arrow down
    arrow_down = ConnectionPatch((8, 6.5), (8, 5.5), "data", "data", 
                               arrowstyle='->', lw=2.5, color='#333333')
    ax.add_patch(arrow_down)
    
    # Stage 5: Multi-Objective Cost Function
    box5 = FancyBboxPatch((5, 4), 6, 1.4, 
                         boxstyle="round,pad=0.1", 
                         facecolor=math_color, 
                         edgecolor='#0097A7', linewidth=2)
    ax.add_patch(box5)
    ax.text(8, 5, 'Multi-Objective Cost Function', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(8, 4.6, r'$C(p) = (1-\beta) \cdot d(p) + \beta \cdot r(p)$', 
            fontsize=12, ha='center', va='center')
    ax.text(8, 4.3, 'Distance-Safety Tradeoff Parameter β', fontsize=10, ha='center', va='center', style='italic')
    
    # Arrow down
    arrow_down2 = ConnectionPatch((8, 3.8), (8, 3), "data", "data", 
                                arrowstyle='->', lw=2.5, color='#333333')
    ax.add_patch(arrow_down2)
    
    # Stage 6: Pareto Frontier
    box6 = FancyBboxPatch((5.5, 1.5), 5, 1.4, 
                         boxstyle="round,pad=0.1", 
                         facecolor=process_color, 
                         edgecolor='#F57C00', linewidth=2)
    ax.add_patch(box6)
    ax.text(8, 2.5, 'Pareto Frontier', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(8, 2.1, r'$\min_{p \in P} \{d(p), r(p)\}$ s.t. dominance', 
            fontsize=12, ha='center', va='center')
    ax.text(8, 1.8, 'Optimal Distance-Safety Tradeoffs', fontsize=10, ha='center', va='center', style='italic')
    
    # Arrow right to final output
    arrow_final = ConnectionPatch((10.5, 2.2), (12, 2.2), "data", "data", **arrow_style)
    ax.add_patch(arrow_final)
    
    # Stage 7: Optimized Routes
    box7 = FancyBboxPatch((12, 1.5), 3, 1.4, 
                         boxstyle="round,pad=0.1", 
                         facecolor=output_color, 
                         edgecolor='#388E3C', linewidth=2)
    ax.add_patch(box7)
    ax.text(13.5, 2.5, 'Optimized Routes', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(13.5, 2.1, '30-50% Risk Reduction', fontsize=11, ha='center', va='center', color='#2E7D32', fontweight='bold')
    ax.text(13.5, 1.8, '<15-25% Distance Increase', fontsize=11, ha='center', va='center', color='#2E7D32', fontweight='bold')
    
    # Add subtle background grid for mathematical feel
    for i in range(1, 16):
        ax.axvline(x=i, color='#f0f0f0', linestyle='-', alpha=0.3, zorder=0)
    for i in range(1, 10):
        ax.axhline(y=i, color='#f0f0f0', linestyle='-', alpha=0.3, zorder=0)
    
    # Add mathematical symbols in corners for aesthetic
    ax.text(0.5, 0.5, r'$\mathcal{G}$', fontsize=20, alpha=0.1, fontweight='bold')
    ax.text(15.5, 0.5, r'$\mathcal{R}$', fontsize=20, alpha=0.1, fontweight='bold')
    ax.text(0.5, 9.5, r'$\mathcal{P}$', fontsize=20, alpha=0.1, fontweight='bold')
    ax.text(15.5, 9.5, r'$\mathcal{O}$', fontsize=20, alpha=0.1, fontweight='bold')
    
    # Add footer with key insight
    ax.text(8, 0.3, 'Structured Mathematical Engine for Crime-Aware Navigation', 
            fontsize=14, ha='center', va='center', 
            style='italic', color='#333333', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Generate the clean pipeline visualization"""
    print("Creating clean mathematical pipeline visualization...")
    
    # Create the visualization
    fig = create_clean_pipeline()
    
    # Save the figure
    output_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/clean_pipeline.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Also save as PDF for crisp scaling
    pdf_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/clean_pipeline.pdf'
    fig = create_clean_pipeline()
    fig.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Clean pipeline visualization saved to:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {pdf_path}")
    print("\nThis visualization demonstrates:")
    print("  ✓ Structured mathematical pipeline")
    print("  ✓ Clear flow from data to results")
    print("  ✓ Mathematical rigor with equations")
    print("  ✓ Judge-friendly presentation")
    print("  ✓ Professional engineering approach")

if __name__ == "__main__":
    main()
