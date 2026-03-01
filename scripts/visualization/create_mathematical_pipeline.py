#!/usr/bin/env python3
"""
CAPR Mathematical Pipeline Visualization
Creates a clean, structured diagram showing the mathematical engine architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import matplotlib.patches as patches
import numpy as np

def create_capr_mathematical_pipeline():
    """Create a clean mathematical pipeline visualization."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define color scheme
    colors = {
        'data': '#E3F2FD',        # Light blue
        'processing': '#FFF3E0',   # Light orange
        'math': '#E8F5E8',        # Light green
        'optimization': '#F3E5F5', # Light purple
        'output': '#FFEBEE',       # Light red
        'border_data': '#1976D2',
        'border_processing': '#F57C00',
        'border_math': '#388E3C',
        'border_optimization': '#7B1FA2',
        'border_output': '#D32F2F'
    }
    
    # Title
    ax.text(8, 9.5, 'CAPR: Crime-Aware Pedestrian Routing Engine', 
           fontsize=20, fontweight='bold', ha='center', color='#1565C0')
    ax.text(8, 9, 'Mathematical Pipeline Architecture', 
           fontsize=14, ha='center', style='italic', color='#424242')
    
    # Step 1: Crime Data Input
    step1 = FancyBboxPatch(
        (0.5, 7), 2.5, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors['data'],
        edgecolor=colors['border_data'],
        linewidth=2
    )
    ax.add_patch(step1)
    ax.text(1.75, 7.6, 'Crime Data\nInput', fontsize=12, fontweight='bold', 
           ha='center', va='center', color=colors['border_data'])
    ax.text(1.75, 6.5, 'Incident Reports\nSpatial Coordinates', fontsize=9, 
           ha='center', va='center', color='#424242')
    
    # Arrow 1
    arrow1 = patches.FancyArrowPatch((3, 7.6), (4, 7.6),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#424242', linewidth=2)
    ax.add_patch(arrow1)
    
    # Step 2: Risk Weighting Functions
    step2 = FancyBboxPatch(
        (4, 7), 2.5, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors['processing'],
        edgecolor=colors['border_processing'],
        linewidth=2
    )
    ax.add_patch(step2)
    ax.text(5.25, 7.6, 'Risk Weighting\nFunctions', fontsize=12, fontweight='bold', 
           ha='center', va='center', color=colors['border_processing'])
    ax.text(5.25, 6.5, 'Spatial Aggregation\nTemporal Decay', fontsize=9, 
           ha='center', va='center', color='#424242')
    
    # Arrow 2
    arrow2 = patches.FancyArrowPatch((6.5, 7.6), (7.5, 7.6),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#424242', linewidth=2)
    ax.add_patch(arrow2)
    
    # Step 3: Graph Construction
    step3 = FancyBboxPatch(
        (7.5, 7), 2.5, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors['math'],
        edgecolor=colors['border_math'],
        linewidth=2
    )
    ax.add_patch(step3)
    ax.text(8.75, 7.8, 'Graph Construction', fontsize=12, fontweight='bold', 
           ha='center', va='center', color=colors['border_math'])
    ax.text(8.75, 7.4, 'G = (V, E)', fontsize=14, fontweight='bold', 
           ha='center', va='center', color=colors['border_math'])
    ax.text(8.75, 6.5, 'V: Street Intersections\nE: Weighted Edges', fontsize=9, 
           ha='center', va='center', color='#424242')
    
    # Arrow 3
    arrow3 = patches.FancyArrowPatch((10, 7.6), (11, 7.6),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#424242', linewidth=2)
    ax.add_patch(arrow3)
    
    # Step 4: Diffusion/Clustering
    step4 = FancyBboxPatch(
        (11, 7), 2.5, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors['processing'],
        edgecolor=colors['border_processing'],
        linewidth=2
    )
    ax.add_patch(step4)
    ax.text(12.25, 7.6, 'Diffusion &\nClustering', fontsize=12, fontweight='bold', 
           ha='center', va='center', color=colors['border_processing'])
    ax.text(12.25, 6.5, 'Spatial Smoothing\nRisk Propagation', fontsize=9, 
           ha='center', va='center', color='#424242')
    
    # Arrow 4 (downward)
    arrow4 = patches.FancyArrowPatch((12.25, 6.8), (12.25, 5.8),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#424242', linewidth=2)
    ax.add_patch(arrow4)
    
    # Step 5: Multi-Objective Cost Function
    step5 = FancyBboxPatch(
        (10.5, 4.5), 3.5, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors['optimization'],
        edgecolor=colors['border_optimization'],
        linewidth=2
    )
    ax.add_patch(step5)
    ax.text(12.25, 5.3, 'Multi-Objective Cost Function', fontsize=12, fontweight='bold', 
           ha='center', va='center', color=colors['border_optimization'])
    ax.text(12.25, 4.9, 'c(e) = β·d(e) + (1-β)·r(e)', fontsize=11, fontweight='bold', 
           ha='center', va='center', color=colors['border_optimization'])
    ax.text(12.25, 4.2, 'β ∈ [0,1]: Safety-Distance Tradeoff', fontsize=9, 
           ha='center', va='center', color='#424242')
    
    # Arrow 5 (leftward)
    arrow5 = patches.FancyArrowPatch((10.5, 5.1), (9, 5.1),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#424242', linewidth=2)
    ax.add_patch(arrow5)
    
    # Step 6: Pareto Frontier
    step6 = FancyBboxPatch(
        (6, 4.5), 3, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors['optimization'],
        edgecolor=colors['border_optimization'],
        linewidth=2
    )
    ax.add_patch(step6)
    ax.text(7.5, 5.3, 'Pareto Frontier', fontsize=12, fontweight='bold', 
           ha='center', va='center', color=colors['border_optimization'])
    ax.text(7.5, 4.9, 'Optimal Tradeoffs', fontsize=10, fontweight='bold', 
           ha='center', va='center', color=colors['border_optimization'])
    ax.text(7.5, 4.2, '32% Risk Reduction\n23% Distance Increase', fontsize=9, 
           ha='center', va='center', color='#424242')
    
    # Arrow 6 (leftward)
    arrow6 = patches.FancyArrowPatch((6, 5.1), (4.5, 5.1),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='#424242', linewidth=2)
    ax.add_patch(arrow6)
    
    # Step 7: Optimized Routes (Final Output)
    step7 = FancyBboxPatch(
        (1.5, 4.5), 3, 1.2,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor=colors['border_output'],
        linewidth=2
    )
    ax.add_patch(step7)
    ax.text(3, 5.3, 'Optimized Routes', fontsize=12, fontweight='bold', 
           ha='center', va='center', color=colors['border_output'])
    ax.text(3, 4.9, 'Safe Navigation Paths', fontsize=10, fontweight='bold', 
           ha='center', va='center', color=colors['border_output'])
    ax.text(3, 4.2, 'Real-time Route Generation', fontsize=9, 
           ha='center', va='center', color='#424242')
    
    # Add mathematical formulations box
    math_box = FancyBboxPatch(
        (0.5, 1.5), 6, 2,
        boxstyle="round,pad=0.15",
        facecolor='#F5F5F5',
        edgecolor='#424242',
        linewidth=1.5
    )
    ax.add_patch(math_box)
    
    ax.text(3.5, 3.2, 'Mathematical Framework', fontsize=14, fontweight='bold', 
           ha='center', color='#1565C0')
    
    # Mathematical formulations
    math_text = """
Objective Function:    min f(P) = β·D(P) + (1-β)·R(P)

Risk Function:         R(P) = Σ r(eᵢ) ∀eᵢ ∈ P
                             eᵢ

Distance Function:     D(P) = Σ d(eᵢ) ∀eᵢ ∈ P  
                             eᵢ

Constraints:           P ∈ Γ(s,t), β ∈ [0,1]
    """
    
    ax.text(3.5, 2.3, math_text, fontsize=10, ha='center', va='center', 
           fontfamily='monospace', color='#424242')
    
    # Add performance metrics box
    perf_box = FancyBboxPatch(
        (9.5, 1.5), 6, 2,
        boxstyle="round,pad=0.15",
        facecolor='#E8F5E8',
        edgecolor=colors['border_math'],
        linewidth=1.5
    )
    ax.add_patch(perf_box)
    
    ax.text(12.5, 3.2, 'Experimental Results', fontsize=14, fontweight='bold', 
           ha='center', color=colors['border_math'])
    
    results_text = """
Dataset:               150 Route Trials
Algorithm:             Dijkstra's + Multi-objective

Performance Metrics:
• Risk Reduction:      32.0% average
• Distance Increase:   22.8% average  
• Success Rate:        70% of routes
• Pareto Efficiency:   6 optimal points
    """
    
    ax.text(12.5, 2.3, results_text, fontsize=10, ha='center', va='center', 
           fontfamily='monospace', color='#424242')
    
    # Add system characteristics
    ax.text(8, 0.5, 'System Characteristics: Real-time Processing • Scalable Architecture • Mathematical Rigor • Validated Performance', 
           fontsize=11, ha='center', style='italic', color='#666666')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = 'visualization/capr_mathematical_pipeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"CAPR Mathematical Pipeline saved to: {output_path}")
    
    plt.close()

def create_simplified_pipeline():
    """Create a more compact version for presentations."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(7, 5.5, 'CAPR Mathematical Engine', 
           fontsize=18, fontweight='bold', ha='center', color='#1565C0')
    
    # Pipeline steps (horizontal layout)
    steps = [
        {'x': 1, 'text': 'Crime\nData', 'math': '', 'color': '#E3F2FD', 'border': '#1976D2'},
        {'x': 3, 'text': 'Risk\nWeighting', 'math': 'f(x,t)', 'color': '#FFF3E0', 'border': '#F57C00'},
        {'x': 5, 'text': 'Graph\nConstruction', 'math': 'G=(V,E)', 'color': '#E8F5E8', 'border': '#388E3C'},
        {'x': 7, 'text': 'Diffusion\nClustering', 'math': '∇²r', 'color': '#FFF3E0', 'border': '#F57C00'},
        {'x': 9, 'text': 'Multi-Objective\nOptimization', 'math': 'β·d + (1-β)·r', 'color': '#F3E5F5', 'border': '#7B1FA2'},
        {'x': 11, 'text': 'Pareto\nFrontier', 'math': '32% ↓ Risk', 'color': '#F3E5F5', 'border': '#7B1FA2'},
        {'x': 13, 'text': 'Optimized\nRoutes', 'math': 'P*', 'color': '#FFEBEE', 'border': '#D32F2F'}
    ]
    
    for i, step in enumerate(steps):
        # Draw step box
        box = FancyBboxPatch(
            (step['x']-0.6, 2.5), 1.2, 1.5,
            boxstyle="round,pad=0.08",
            facecolor=step['color'],
            edgecolor=step['border'],
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add step text
        ax.text(step['x'], 3.6, step['text'], fontsize=10, fontweight='bold', 
               ha='center', va='center', color=step['border'])
        
        # Add mathematical notation
        if step['math']:
            ax.text(step['x'], 3, step['math'], fontsize=9, fontweight='bold', 
                   ha='center', va='center', color='#424242', fontfamily='monospace')
        
        # Add arrows (except for last step)
        if i < len(steps) - 1:
            arrow = patches.FancyArrowPatch(
                (step['x'] + 0.6, 3.25), (steps[i+1]['x'] - 0.6, 3.25),
                arrowstyle='->', mutation_scale=15, 
                color='#424242', linewidth=1.5
            )
            ax.add_patch(arrow)
    
    # Add bottom summary
    ax.text(7, 1.5, 'Structured • Mathematical • Optimized', 
           fontsize=14, fontweight='bold', ha='center', color='#1565C0')
    ax.text(7, 1, '150 Trial Validation • 32% Risk Reduction • Real-time Performance', 
           fontsize=11, ha='center', style='italic', color='#666666')
    
    plt.tight_layout()
    
    # Save the simplified version
    output_path = 'visualization/capr_pipeline_simplified.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Simplified CAPR Pipeline saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Creating CAPR Mathematical Pipeline Visualizations...")
    create_capr_mathematical_pipeline()
    create_simplified_pipeline()
    print("Pipeline visualizations created successfully!")
