#!/usr/bin/env python3
"""
Create a simple visual aid showing maps stacking up on top of each other.
Clean, minimal design without technical jargon.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

def create_map_stack_visual():
    """Create a clean visual showing maps stacking up."""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('white')
    
    # Colors for each layer
    colors = [
        '#E8F4FD',  # Light blue for base map
        '#FFE6E6',  # Light red for crime data
        '#FFF2CC',  # Light yellow for risk analysis
        '#E8F5E8'   # Light green for smart routes
    ]
    
    border_colors = [
        '#4A90C2',  # Blue
        '#D63384',  # Red
        '#FFC107',  # Yellow
        '#28A745'   # Green
    ]
    
    labels = [
        'Base Map',
        'Crime Data',
        'Risk Analysis',
        'Smart Routes'
    ]
    
    # Left panel - Side view (stacking)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    
    # Draw stacked maps from bottom to top
    for i, (color, border_color, label) in enumerate(zip(colors, border_colors, labels)):
        y_pos = 2 + i * 1.2
        thickness = 0.3
        
        # Main map rectangle
        rect = FancyBboxPatch(
            (2, y_pos), 6, thickness,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=border_color,
            linewidth=2,
            alpha=0.9
        )
        ax1.add_patch(rect)
        
        # Add shadow/depth effect
        shadow = FancyBboxPatch(
            (2.1, y_pos - 0.05), 6, thickness,
            boxstyle="round,pad=0.02",
            facecolor='gray',
            alpha=0.2,
            zorder=i-1
        )
        ax1.add_patch(shadow)
        
        # Label
        ax1.text(1.5, y_pos + thickness/2, label, 
                fontsize=12, fontweight='bold',
                ha='right', va='center',
                color=border_color)
    
    # Add arrows showing the stacking process
    for i in range(3):
        y_start = 2.8 + i * 1.2
        y_end = y_start + 0.7
        ax1.annotate('', xy=(9.2, y_end), xytext=(9.2, y_start),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#666666'))
    
    ax1.set_title('How CAPR Builds Up', fontsize=16, fontweight='bold', pad=20)
    ax1.text(5, 1.2, 'Each layer adds intelligence to the map', 
            fontsize=11, ha='center', style='italic', color='#555555')
    ax1.axis('off')
    
    # Right panel - 3D perspective view
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    
    # 3D perspective parameters
    perspective_offset_x = 0.4
    perspective_offset_y = 0.3
    
    # Draw maps in 3D perspective from bottom to top
    for i, (color, border_color, label) in enumerate(zip(colors, border_colors, labels)):
        base_x = 2
        base_y = 2 + i * 0.6
        width = 5
        height = 3
        
        # Calculate 3D offsets
        offset_x = i * perspective_offset_x
        offset_y = i * perspective_offset_y
        
        # Draw the top face of the map
        top_face = FancyBboxPatch(
            (base_x + offset_x, base_y + offset_y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=border_color,
            linewidth=2,
            alpha=0.9,
            zorder=10+i
        )
        ax2.add_patch(top_face)
        
        # Draw the side face for 3D effect
        if i > 0:  # Don't draw side for bottom layer
            # Right side
            side_points = np.array([
                [base_x + width + offset_x, base_y + offset_y],
                [base_x + width + offset_x - perspective_offset_x, base_y + offset_y - perspective_offset_y],
                [base_x + width + offset_x - perspective_offset_x, base_y + height + offset_y - perspective_offset_y],
                [base_x + width + offset_x, base_y + height + offset_y]
            ])
            side_patch = patches.Polygon(side_points, closed=True, 
                                       facecolor=color, edgecolor=border_color,
                                       alpha=0.7, linewidth=1.5, zorder=8+i)
            ax2.add_patch(side_patch)
            
            # Front side
            front_points = np.array([
                [base_x + offset_x, base_y + offset_y],
                [base_x + offset_x - perspective_offset_x, base_y + offset_y - perspective_offset_y],
                [base_x + width + offset_x - perspective_offset_x, base_y + offset_y - perspective_offset_y],
                [base_x + width + offset_x, base_y + offset_y]
            ])
            front_patch = patches.Polygon(front_points, closed=True,
                                        facecolor=color, edgecolor=border_color,
                                        alpha=0.7, linewidth=1.5, zorder=8+i)
            ax2.add_patch(front_patch)
    
    # Add labels for 3D view
    for i, (color, border_color, label) in enumerate(zip(colors, border_colors, labels)):
        base_y = 2 + i * 0.6
        offset_y = i * perspective_offset_y
        ax2.text(7.5, base_y + 1.5 + offset_y, label,
                fontsize=11, fontweight='bold',
                ha='left', va='center',
                color=border_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax2.set_title('Layered Intelligence', fontsize=16, fontweight='bold', pad=20)
    ax2.text(5, 1, 'Building smarter navigation step by step', 
            fontsize=11, ha='center', style='italic', color='#555555')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/simple_map_stack.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Simple map stack visual saved to: {output_path}")
    
    plt.close()


def create_animated_stack_visual():
    """Create a step-by-step visual showing how maps stack up."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    # Colors and labels
    colors = ['#E8F4FD', '#FFE6E6', '#FFF2CC', '#E8F5E8']
    border_colors = ['#4A90C2', '#D63384', '#FFC107', '#28A745']
    labels = ['Base Map', 'Crime Data', 'Risk Analysis', 'Smart Routes']
    descriptions = [
        'Start with a basic city map',
        'Add crime incident data',
        'Analyze risk patterns',
        'Generate safe routes'
    ]
    
    for step in range(4):
        ax = axes[step]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        # Draw all layers up to current step
        for i in range(step + 1):
            y_pos = 2 + i * 0.8
            
            # Map rectangle with 3D effect
            rect = FancyBboxPatch(
                (2 + i*0.2, y_pos + i*0.15), 6, 0.4,
                boxstyle="round,pad=0.02",
                facecolor=colors[i],
                edgecolor=border_colors[i],
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(rect)
            
            # Add subtle shadow
            shadow = Rectangle(
                (2.1 + i*0.2, y_pos + i*0.15 - 0.05), 6, 0.4,
                facecolor='gray', alpha=0.15
            )
            ax.add_patch(shadow)
        
        # Current layer label (highlighted)
        ax.text(5, 6.5, labels[step], 
               fontsize=14, fontweight='bold', ha='center',
               color=border_colors[step],
               bbox=dict(boxstyle="round,pad=0.5", facecolor=colors[step], alpha=0.8))
        
        # Description
        ax.text(5, 1, descriptions[step], 
               fontsize=11, ha='center', style='italic', color='#555555')
        
        # Step number
        ax.text(1, 7, f"Step {step + 1}", 
               fontsize=12, fontweight='bold', color='#333333')
        
        ax.axis('off')
    
    fig.suptitle('Building the CAPR System Layer by Layer', 
                fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    # Save the step-by-step visualization
    output_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/map_stack_steps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Step-by-step map stack visual saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    print("Creating simple map stack visuals...")
    create_map_stack_visual()
    create_animated_stack_visual()
    print("Visual aids created successfully!")
