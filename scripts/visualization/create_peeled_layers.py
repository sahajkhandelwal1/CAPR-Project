#!/usr/bin/env python3
"""
Create a "peeled layers" visual showing how CAPR works.
Shows one map with corners peeled back to reveal underlying layers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch
import numpy as np

def create_peeled_layers_visual():
    """Create a visual showing map layers being peeled back."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor('white')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # Layer information
    layers = [
        {'color': '#E8F4FD', 'border': '#4A90C2', 'label': 'Base Map', 'desc': 'Streets & Buildings'},
        {'color': '#FFE6E6', 'border': '#D63384', 'label': 'Crime Data', 'desc': 'Incident Reports'},
        {'color': '#FFF2CC', 'border': '#FFC107', 'label': 'Risk Analysis', 'desc': 'Safety Patterns'},
        {'color': '#E8F5E8', 'border': '#28A745', 'label': 'Smart Routes', 'desc': 'Safe Navigation'}
    ]
    
    # Main map dimensions
    map_x, map_y = 2, 2
    map_width, map_height = 6, 5
    
    # Draw base layers (bottom to top)
    for i, layer in enumerate(layers):
        if i < 3:  # Don't draw the top layer yet
            rect = FancyBboxPatch(
                (map_x, map_y), map_width, map_height,
                boxstyle="round,pad=0.02",
                facecolor=layer['color'],
                edgecolor=layer['border'],
                linewidth=2,
                alpha=0.8,
                zorder=i
            )
            ax.add_patch(rect)
    
    # Create peeled-back corners for top 3 layers
    peel_positions = [
        {'corner': 'top-right', 'x_peel': 6, 'y_peel': 6, 'layer_idx': 3},
        {'corner': 'top-left', 'x_peel': 3, 'y_peel': 5.5, 'layer_idx': 2},
        {'corner': 'bottom-right', 'x_peel': 5.5, 'y_peel': 3, 'layer_idx': 1}
    ]
    
    for i, peel in enumerate(peel_positions):
        layer = layers[peel['layer_idx']]
        
        # Create the main part of the layer (with corner cut out)
        if peel['corner'] == 'top-right':
            # Top layer with top-right corner peeled
            main_points = np.array([
                [map_x, map_y],
                [map_x + map_width, map_y],
                [map_x + map_width, map_y + map_height - 1.5],
                [map_x + map_width - 1.5, map_y + map_height],
                [map_x, map_y + map_height]
            ])
            peel_points = np.array([
                [map_x + map_width - 1.5, map_y + map_height],
                [map_x + map_width, map_y + map_height - 1.5],
                [peel['x_peel'] + 1, peel['y_peel'] + 0.8],
                [peel['x_peel'] + 0.2, peel['y_peel'] + 1.5]
            ])
            
        elif peel['corner'] == 'top-left':
            main_points = np.array([
                [map_x + 1, map_y],
                [map_x + map_width, map_y],
                [map_x + map_width, map_y + map_height],
                [map_x, map_y + map_height - 1],
                [map_x, map_y]
            ])
            peel_points = np.array([
                [map_x, map_y + map_height - 1],
                [map_x + 1, map_y + map_height],
                [peel['x_peel'] - 0.5, peel['y_peel'] + 1.2],
                [peel['x_peel'] - 1.2, peel['y_peel'] + 0.5]
            ])
            
        else:  # bottom-right
            main_points = np.array([
                [map_x, map_y],
                [map_x + map_width - 1, map_y],
                [map_x + map_width, map_y + 1],
                [map_x + map_width, map_y + map_height],
                [map_x, map_y + map_height]
            ])
            peel_points = np.array([
                [map_x + map_width - 1, map_y],
                [map_x + map_width, map_y + 1],
                [peel['x_peel'] + 0.8, peel['y_peel'] - 0.5],
                [peel['x_peel'] + 0.5, peel['y_peel'] - 0.8]
            ])
        
        # Draw main part of layer
        main_patch = Polygon(main_points, closed=True,
                           facecolor=layer['color'], edgecolor=layer['border'],
                           linewidth=2, alpha=0.9, zorder=10+i)
        ax.add_patch(main_patch)
        
        # Draw peeled corner with shadow effect
        peel_patch = Polygon(peel_points, closed=True,
                           facecolor=layer['color'], edgecolor=layer['border'],
                           linewidth=2, alpha=0.9, zorder=15+i)
        ax.add_patch(peel_patch)
        
        # Add shadow under peeled corner
        shadow_points = peel_points + np.array([0.1, -0.1])
        shadow_patch = Polygon(shadow_points, closed=True,
                             facecolor='gray', alpha=0.3, zorder=14+i)
        ax.add_patch(shadow_patch)
        
        # Add label near the peeled corner
        label_x, label_y = peel['x_peel'], peel['y_peel']
        ax.text(label_x, label_y, f"{layer['label']}\n{layer['desc']}", 
               fontsize=10, fontweight='bold', ha='center', va='center',
               color=layer['border'],
               bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                        edgecolor=layer['border'], alpha=0.9))
        
        # Add arrow pointing to the layer
        arrow_start_x = label_x + (0.8 if 'right' in peel['corner'] else -0.8)
        arrow_start_y = label_y + (0.3 if 'top' in peel['corner'] else -0.3)
        arrow_end_x = peel_points[0, 0]
        arrow_end_y = peel_points[0, 1]
        
        ax.annotate('', xy=(arrow_end_x, arrow_end_y), 
                   xytext=(arrow_start_x, arrow_start_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, 
                                 color=layer['border'], alpha=0.7))
    
    # Add title and subtitle
    ax.text(6, 9, 'How CAPR Works', fontsize=20, fontweight='bold', 
           ha='center', color='#333333')
    ax.text(6, 8.4, 'Multiple layers of intelligence for safer navigation', 
           fontsize=12, ha='center', style='italic', color='#666666')
    
    # Add base layer label (since it's not peeled)
    ax.text(1, 4.5, f"{layers[0]['label']}\n{layers[0]['desc']}", 
           fontsize=10, fontweight='bold', ha='center', va='center',
           color=layers[0]['border'],
           bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                    edgecolor=layers[0]['border'], alpha=0.9))
    
    # Add arrow to base layer
    ax.annotate('', xy=(2.2, 4.5), xytext=(1.8, 4.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, 
                             color=layers[0]['border'], alpha=0.7))
    
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/peeled_layers_visual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Peeled layers visual saved to: {output_path}")
    
    plt.close()


def create_clean_concept_visual():
    """Create a super clean concept visual with minimal text."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # Simple color scheme
    colors = ['#f8f9fa', '#fff3cd', '#f8d7da', '#d1ecf1']
    borders = ['#6c757d', '#856404', '#721c24', '#0c5460']
    labels = ['Map', 'Crime', 'Risk', 'Routes']
    
    # Draw simple stacked rectangles
    for i in range(4):
        y_pos = 2 + i * 0.8
        
        # Main rectangle
        rect = FancyBboxPatch(
            (3, y_pos), 4, 0.6,
            boxstyle="round,pad=0.02",
            facecolor=colors[i],
            edgecolor=borders[i],
            linewidth=2,
            zorder=4-i
        )
        ax.add_patch(rect)
        
        # Label
        ax.text(2.5, y_pos + 0.3, labels[i], 
               fontsize=14, fontweight='bold', ha='right', va='center',
               color=borders[i])
        
        # Plus sign between layers
        if i < 3:
            ax.text(7.5, y_pos + 0.6, '+', 
                   fontsize=20, fontweight='bold', ha='center', va='center',
                   color='#666666')
    
    # Arrow pointing to result
    ax.annotate('', xy=(5, 1.5), xytext=(5, 0.8),
               arrowprops=dict(arrowstyle='->', lw=3, color='#28a745'))
    
    # Result
    ax.text(5, 0.3, 'Safer Navigation', 
           fontsize=16, fontweight='bold', ha='center', va='center',
           color='#28a745',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='#d4edda', 
                    edgecolor='#28a745', linewidth=2))
    
    # Title
    ax.text(5, 7.2, 'Crime-Aware Routing', 
           fontsize=18, fontweight='bold', ha='center', color='#333333')
    
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/clean_concept_visual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Clean concept visual saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    print("Creating additional map visuals...")
    create_peeled_layers_visual()
    create_clean_concept_visual()
    print("Additional visuals created successfully!")
