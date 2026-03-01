#!/usr/bin/env python3
"""
Simplified Algorithm Flow for CAPR System
Creates a streamlined view focusing on the core algorithmic steps
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_algorithm_flow():
    """Create a simplified algorithm flow visualization"""
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define colors
    input_color = '#E3F2FD'      # Light blue
    algo_color = '#FFF3E0'       # Light orange
    math_color = '#F3E5F5'       # Light purple
    result_color = '#E8F5E8'     # Light green
    
    # Arrow style
    arrow_style = dict(arrowstyle='->', lw=3, color='#2C3E50')
    
    # Title
    ax.text(7, 7.5, 'CAPR Algorithm Pipeline', 
            fontsize=22, fontweight='bold', ha='center', va='center')
    ax.text(7, 7, 'From Crime Data to Safe Routes', 
            fontsize=14, ha='center', va='center', style='italic', color='#34495E')
    
    # Step 1: Crime Data Input
    box1 = FancyBboxPatch((1, 5.5), 2.5, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=input_color, 
                         edgecolor='#1976D2', linewidth=2)
    ax.add_patch(box1)
    ax.text(2.25, 6, 'Crime Data', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(2.25, 5.7, '(Time, Location)', fontsize=10, ha='center', va='center')
    
    # Arrow 1
    arrow1 = ConnectionPatch((3.5, 6), (4.5, 6), "data", "data", **arrow_style)
    ax.add_patch(arrow1)
    
    # Step 2: Risk Scoring
    box2 = FancyBboxPatch((4.5, 5.5), 2.5, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=algo_color, 
                         edgecolor='#FF9800', linewidth=2)
    ax.add_patch(box2)
    ax.text(5.75, 6, 'Risk Scoring', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5.75, 5.7, r'$R(x,y) = f(crimes)$', fontsize=10, ha='center', va='center')
    
    # Arrow 2
    arrow2 = ConnectionPatch((7, 6), (8, 6), "data", "data", **arrow_style)
    ax.add_patch(arrow2)
    
    # Step 3: Graph Weighting
    box3 = FancyBboxPatch((8, 5.5), 2.5, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=math_color, 
                         edgecolor='#9C27B0', linewidth=2)
    ax.add_patch(box3)
    ax.text(9.25, 6, 'Graph Weighting', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(9.25, 5.7, r'$w_{ij} = d + \alpha \cdot r$', fontsize=10, ha='center', va='center')
    
    # Arrow 3 (down)
    arrow3 = ConnectionPatch((9.25, 5.5), (9.25, 4.5), "data", "data", **arrow_style)
    ax.add_patch(arrow3)
    
    # Step 4: Multi-Objective Optimization
    box4 = FancyBboxPatch((7.5, 3.5), 3.5, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=math_color, 
                         edgecolor='#9C27B0', linewidth=2)
    ax.add_patch(box4)
    ax.text(9.25, 4, 'Multi-Objective Optimization', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(9.25, 3.7, r'$\min\{distance, risk\}$', fontsize=10, ha='center', va='center')
    
    # Arrow 4 (left)
    arrow4 = ConnectionPatch((7.5, 4), (6.5, 4), "data", "data", **arrow_style)
    ax.add_patch(arrow4)
    
    # Step 5: Pareto Routes
    box5 = FancyBboxPatch((4, 3.5), 2.5, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=result_color, 
                         edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(box5)
    ax.text(5.25, 4, 'Pareto Routes', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(5.25, 3.7, 'Optimal Tradeoffs', fontsize=10, ha='center', va='center')
    
    # Arrow 5 (left)
    arrow5 = ConnectionPatch((4, 4), (3, 4), "data", "data", **arrow_style)
    ax.add_patch(arrow5)
    
    # Step 6: Safe Navigation
    box6 = FancyBboxPatch((0.5, 3.5), 2.5, 1, 
                         boxstyle="round,pad=0.1", 
                         facecolor=result_color, 
                         edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(box6)
    ax.text(1.75, 4, 'Safe Navigation', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(1.75, 3.7, 'User Routes', fontsize=10, ha='center', va='center')
    
    # Add performance metrics box
    perf_box = FancyBboxPatch((4, 1.5), 6, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#FFF8E1', 
                             edgecolor='#FFC107', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(7, 2.4, 'Performance Results', fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text(7, 2, '• 30-50% Risk Reduction', fontsize=11, ha='center', va='center', color='#388E3C')
    ax.text(7, 1.7, '• <15-25% Distance Increase', fontsize=11, ha='center', va='center', color='#388E3C')
    
    # Add algorithm complexity note
    ax.text(7, 0.8, 'Algorithm: Dijkstra + Multi-Objective Optimization', 
            fontsize=12, ha='center', va='center', 
            style='italic', color='#5D4037', fontweight='bold')
    ax.text(7, 0.4, 'Time Complexity: O(E log V) per route', 
            fontsize=10, ha='center', va='center', color='#5D4037')
    
    plt.tight_layout()
    return fig

def main():
    """Generate the simplified algorithm flow"""
    print("Creating simplified algorithm flow visualization...")
    
    # Create the visualization
    fig = create_algorithm_flow()
    
    # Save the figure
    output_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/algorithm_flow.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Also save as PDF
    pdf_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/algorithm_flow.pdf'
    fig = create_algorithm_flow()
    fig.savefig(pdf_path, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Algorithm flow visualization saved to:")
    print(f"  PNG: {output_path}")
    print(f"  PDF: {pdf_path}")
    print("\nThis visualization shows:")
    print("  ✓ Simplified algorithm pipeline")
    print("  ✓ Clear step-by-step flow")
    print("  ✓ Performance metrics")
    print("  ✓ Algorithm complexity")

if __name__ == "__main__":
    main()
