#!/usr/bin/env python3
"""
Create a simple visual showing maps stacking up on top of each other
A clean visual aid without technical descriptions - just showing the layering concept
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_map_stack_visual():
    """Create a visual showing maps stacking up like physical layers"""
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define layer positions and properties
    layers = [
        {
            'name': 'Base Map',
            'y_pos': 1.5,
            'color': '#E8F4F8',
            'edge_color': '#2E86AB',
            'shadow_offset': 0.05,
            'description': 'Streets & Geography'
        },
        {
            'name': 'Crime Data',
            'y_pos': 3.0,
            'color': '#FFE5E5',
            'edge_color': '#D32F2F',
            'shadow_offset': 0.1,
            'description': 'Incident Locations'
        },
        {
            'name': 'Risk Analysis',
            'y_pos': 4.5,
            'color': '#FFF3E0',
            'edge_color': '#F57C00',
            'shadow_offset': 0.15,
            'description': 'Safety Zones'
        },
        {
            'name': 'Smart Routes',
            'y_pos': 6.0,
            'color': '#E8F5E8',
            'edge_color': '#388E3C',
            'shadow_offset': 0.2,
            'description': 'Optimized Paths'
        }
    ]
    
    # Draw each layer
    for i, layer in enumerate(layers):
        # Shadow
        shadow = FancyBboxPatch(
            (1.5 + layer['shadow_offset'], layer['y_pos'] - layer['shadow_offset']), 
            7, 1.2,
            boxstyle="round,pad=0.05",
            facecolor='#CCCCCC',
            alpha=0.3,
            zorder=i*2
        )
        ax.add_patch(shadow)
        
        # Main layer
        layer_patch = FancyBboxPatch(
            (1.5, layer['y_pos']), 
            7, 1.2,
            boxstyle="round,pad=0.05",
            facecolor=layer['color'],
            edgecolor=layer['edge_color'],
            linewidth=2,
            zorder=i*2+1
        )
        ax.add_patch(layer_patch)
        
        # Layer name
        ax.text(5, layer['y_pos'] + 0.6, layer['name'], 
                ha='center', va='center', fontsize=16, fontweight='bold',
                color=layer['edge_color'], zorder=i*2+2)
        
        # Description
        ax.text(5, layer['y_pos'] + 0.2, layer['description'], 
                ha='center', va='center', fontsize=12,
                color=layer['edge_color'], alpha=0.8, zorder=i*2+2)
    
    # Add arrows showing the stacking
    for i in range(len(layers)-1):
        start_y = layers[i]['y_pos'] + 1.4
        end_y = layers[i+1]['y_pos'] - 0.2
        
        ax.annotate('', xy=(5, end_y), xytext=(5, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#666666', alpha=0.7))
    
    # Title
    ax.text(5, 8.5, 'CAPR System', 
            ha='center', va='center', fontsize=24, fontweight='bold',
            color='#2C3E50')
    ax.text(5, 8.0, 'Crime-Aware Path Routing', 
            ha='center', va='center', fontsize=16,
            color='#34495E')
    
    # Subtitle
    ax.text(5, 0.5, 'Each layer builds upon the previous to create safer routes', 
            ha='center', va='center', fontsize=12, style='italic',
            color='#7F8C8D')
    
    plt.tight_layout()
    return fig

def create_isometric_stack_visual():
    """Create an isometric view of the map stack"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Isometric transformation parameters
    angle = np.pi/6  # 30 degrees
    
    def iso_transform(x, y, z):
        """Transform 3D coordinates to isometric 2D"""
        iso_x = (x - z) * np.cos(angle)
        iso_y = (x + z) * np.sin(angle) + y
        return iso_x, iso_y
    
    # Define layers in 3D space
    layers = [
        {
            'name': 'Base Map',
            'z': 0,
            'color': '#E8F4F8',
            'edge_color': '#2E86AB',
            'label': 'Streets & Places'
        },
        {
            'name': 'Crime Layer',
            'z': 0.8,
            'color': '#FFE5E5',
            'edge_color': '#D32F2F',
            'label': 'Crime Incidents'
        },
        {
            'name': 'Risk Layer',
            'z': 1.6,
            'color': '#FFF3E0',
            'edge_color': '#F57C00',
            'label': 'Safety Analysis'
        },
        {
            'name': 'Route Layer',
            'z': 2.4,
            'color': '#E8F5E8',
            'edge_color': '#388E3C',
            'label': 'Smart Paths'
        }
    ]
    
    # Base rectangle in 3D
    base_width = 4
    base_height = 3
    base_x = 2
    base_y = 2
    
    for layer in layers:
        z = layer['z']
        
        # Define rectangle corners in 3D
        corners_3d = [
            (base_x, base_y, z),
            (base_x + base_width, base_y, z),
            (base_x + base_width, base_y + base_height, z),
            (base_x, base_y + base_height, z)
        ]
        
        # Transform to 2D isometric
        corners_2d = [iso_transform(x, y, z) for x, y, z in corners_3d]
        
        # Create polygon
        polygon = plt.Polygon(corners_2d, 
                            facecolor=layer['color'], 
                            edgecolor=layer['edge_color'],
                            linewidth=2,
                            alpha=0.9)
        ax.add_patch(polygon)
        
        # Add side faces for depth
        if z > 0:
            # Right side
            right_side = [
                iso_transform(base_x + base_width, base_y, z-0.8),
                iso_transform(base_x + base_width, base_y, z),
                iso_transform(base_x + base_width, base_y + base_height, z),
                iso_transform(base_x + base_width, base_y + base_height, z-0.8)
            ]
            right_poly = plt.Polygon(right_side, 
                                   facecolor=layer['color'], 
                                   edgecolor=layer['edge_color'],
                                   alpha=0.6)
            ax.add_patch(right_poly)
            
            # Front side
            front_side = [
                iso_transform(base_x, base_y + base_height, z-0.8),
                iso_transform(base_x, base_y + base_height, z),
                iso_transform(base_x + base_width, base_y + base_height, z),
                iso_transform(base_x + base_width, base_y + base_height, z-0.8)
            ]
            front_poly = plt.Polygon(front_side, 
                                   facecolor=layer['color'], 
                                   edgecolor=layer['edge_color'],
                                   alpha=0.7)
            ax.add_patch(front_poly)
        
        # Add label
        center_x, center_y = iso_transform(base_x + base_width/2, base_y + base_height/2, z)
        ax.text(center_x, center_y, layer['name'], 
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=layer['edge_color'])
        
        # Add description to the side
        desc_x, desc_y = iso_transform(base_x + base_width + 1, base_y + base_height/2, z)
        ax.text(desc_x, desc_y, layer['label'], 
                ha='left', va='center', fontsize=10,
                color=layer['edge_color'])
    
    # Title
    ax.text(6, 9, 'CAPR Map Layers', 
            ha='center', va='center', fontsize=22, fontweight='bold',
            color='#2C3E50')
    ax.text(6, 8.5, 'Stacked for Crime-Aware Routing', 
            ha='center', va='center', fontsize=14,
            color='#34495E')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create both visualizations
    print("Creating map stack visuals...")
    
    # Simple stacked view
    fig1 = create_map_stack_visual()
    fig1.savefig('/Users/Sahaj/CAPR Project/CAPR-Project/visualization/map_layers_stack.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    
    # Isometric view
    fig2 = create_isometric_stack_visual()
    fig2.savefig('/Users/Sahaj/CAPR Project/CAPR-Project/visualization/map_layers_isometric.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Map stack visuals created:")
    print("   - visualization/map_layers_stack.png (Simple stacked view)")
    print("   - visualization/map_layers_isometric.png (3D isometric view)")
    
    plt.show()
