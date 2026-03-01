#!/usr/bin/env python3
"""
Edge-Focused Crime Diffusion Visualization
Shows how crime risk specifically diffuses along network edges
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

def create_edge_diffusion_demo():
    """Create a detailed edge-focused diffusion demonstration"""
    
    # Create a small, clear network for demonstration
    G = nx.Graph()
    
    # Define nodes in a cross pattern
    nodes = {
        'center': (5, 5),
        'north': (5, 8),
        'south': (5, 2),
        'east': (8, 5),
        'west': (2, 5),
        'ne': (7, 7),
        'nw': (3, 7),
        'se': (7, 3),
        'sw': (3, 3)
    }
    
    # Add nodes and edges
    for name, pos in nodes.items():
        G.add_node(name, pos=pos)
    
    # Create edges - star pattern from center
    center_edges = [('center', 'north'), ('center', 'south'), 
                   ('center', 'east'), ('center', 'west')]
    diagonal_edges = [('center', 'ne'), ('center', 'nw'), 
                     ('center', 'se'), ('center', 'sw')]
    
    G.add_edges_from(center_edges)
    G.add_edges_from(diagonal_edges)
    
    # Add some peripheral connections
    G.add_edge('north', 'ne')
    G.add_edge('north', 'nw')
    G.add_edge('east', 'ne')
    G.add_edge('east', 'se')
    
    return G, nodes

def calculate_edge_risks(G, nodes, crime_node, diffusion_steps=6, alpha=0.2):
    """Calculate risk values for edges during additive diffusion process"""
    
    # Initialize node risks
    node_risks = {node: 0.0 for node in G.nodes()}
    node_risks[crime_node] = 2.0  # Higher initial crime value
    
    # Track diffusion over time
    diffusion_history = []
    edge_risk_history = []
    
    for step in range(diffusion_steps):
        # Store current state
        diffusion_history.append(node_risks.copy())
        
        # Calculate edge risks (average of connected nodes)
        edge_risks = {}
        for edge in G.edges():
            node1, node2 = edge
            edge_risks[edge] = (node_risks[node1] + node_risks[node2]) / 2
        edge_risk_history.append(edge_risks.copy())
        
        # Additive diffusion step (not conservative)
        new_risks = node_risks.copy()  # Start with current risks
        
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors and node_risks[node] > 0.01:
                # Spread risk to neighbors additively
                spread_per_neighbor = alpha * node_risks[node] / len(neighbors)
                for neighbor in neighbors:
                    new_risks[neighbor] += spread_per_neighbor * 0.6  # Partial spreading
        
        # Apply slight decay but maintain crime origin
        for node in G.nodes():
            new_risks[node] *= 0.95  # Slight decay
        
        # Maintain high risk at crime origin
        new_risks[crime_node] = max(new_risks[crime_node], 1.5)
        
        node_risks = new_risks
    
    return diffusion_history, edge_risk_history

def create_edge_diffusion_visualization():
    """Create the main edge diffusion visualization"""
    
    G, nodes = create_edge_diffusion_demo()
    diffusion_history, edge_risk_history = calculate_edge_risks(G, nodes, 'center')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Edge-Level Crime Risk Diffusion with Laplacian Smoothing', 
                 fontsize=18, fontweight='bold')
    
    # Custom colormap
    colors = ['#000033', '#000066', '#0066CC', '#66CCFF', '#FFFF66', '#FF6600', '#CC0000']
    risk_cmap = LinearSegmentedColormap.from_list('edge_risk', colors, N=100)
    
    steps_to_show = [0, 1, 2, 3, 4, 5]
    
    for idx, step in enumerate(steps_to_show):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        node_risks = diffusion_history[step]
        edge_risks = edge_risk_history[step]
        
        # Draw edges with risk-based colors and widths
        max_edge_risk = max(edge_risks.values()) if edge_risks.values() else 1.0
        
        for edge in G.edges():
            node1, node2 = edge
            pos1 = nodes[node1]
            pos2 = nodes[node2]
            
            # Edge risk determines color and width
            risk = edge_risks[edge]
            risk_normalized = risk / max_edge_risk if max_edge_risk > 0 else 0
            color = risk_cmap(risk_normalized)
            width = 2 + risk_normalized * 8  # Width between 2 and 10
            
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                   color=color, linewidth=width, alpha=0.8, zorder=1)
            
            # Add risk value label on edge
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            ax.text(mid_x, mid_y, f'{risk:.2f}', fontsize=8, 
                   ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8),
                   zorder=3)
        
        # Draw nodes
        for node, pos in nodes.items():
            risk = node_risks[node]
            size = 100 + risk * 300
            color = 'red' if node == 'center' and step == 0 else risk_cmap(risk)
            
            ax.scatter(pos[0], pos[1], c=[color], s=size, 
                      edgecolors='black', linewidth=2, zorder=2)
            
            # Add node labels
            ax.text(pos[0], pos[1] + 0.8, f'{risk:.2f}', 
                   ha='center', va='center', fontweight='bold', fontsize=10)
        
        ax.set_title(f'Diffusion Step {step}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=risk_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8)
    cbar.set_label('Edge Risk Level', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_diffusion_mechanism_diagram():
    """Create a diagram explaining the diffusion mechanism"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: Before diffusion
    ax1.set_title('Before Diffusion: Crime at Single Location', 
                  fontsize=14, fontweight='bold')
    
    # Simple 3-node network
    positions = [(1, 2), (3, 2), (5, 2)]
    risks = [0.0, 1.0, 0.0]  # Crime only at middle node
    
    # Draw edges
    for i in range(len(positions)-1):
        ax1.plot([positions[i][0], positions[i+1][0]], 
                [positions[i][1], positions[i+1][1]], 
                'k-', linewidth=3)
    
    # Draw nodes
    colors = ['lightblue', 'red', 'lightblue']
    for i, (pos, risk, color) in enumerate(zip(positions, risks, colors)):
        ax1.scatter(pos[0], pos[1], c=color, s=300, 
                   edgecolors='black', linewidth=2)
        ax1.text(pos[0], pos[1]-0.5, f'Risk: {risk}', 
                ha='center', fontweight='bold')
        ax1.text(pos[0], pos[1]+0.5, f'Node {i+1}', 
                ha='center', fontweight='bold')
    
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Right subplot: After diffusion
    ax2.set_title('After Diffusion: Risk Spreads to Neighbors', 
                  fontsize=14, fontweight='bold')
    
    # After diffusion - additive spreading
    new_risks = [0.4, 1.2, 0.4]  # Origin maintains higher risk, neighbors gain risk
    colors = ['orange', 'darkred', 'orange']
    
    # Draw edges with risk-based width
    for i in range(len(positions)-1):
        edge_risk = (new_risks[i] + new_risks[i+1]) / 2
        width = 3 + edge_risk * 5
        ax2.plot([positions[i][0], positions[i+1][0]], 
                [positions[i][1], positions[i+1][1]], 
                'purple', linewidth=width, alpha=0.7)
    
    # Draw nodes
    for i, (pos, risk, color) in enumerate(zip(positions, new_risks, colors)):
        ax2.scatter(pos[0], pos[1], c=color, s=300, 
                   edgecolors='black', linewidth=2)
        ax2.text(pos[0], pos[1]-0.5, f'Risk: {risk:.1f}', 
                ha='center', fontweight='bold')
        ax2.text(pos[0], pos[1]+0.5, f'Node {i+1}', 
                ha='center', fontweight='bold')
    
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 4)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Add mathematical explanation
    fig.text(0.5, 0.1, 
             r'Additive Diffusion: $r_i^{(t+1)} = r_i^{(t)} + \alpha \sum_j \frac{w_{ij} \cdot r_j^{(t)}}{deg(j)}$' + 
             '\n' + r'Origin Preservation: $r_{origin} = \max(r_{origin}, r_{min})$ | Edge Risk: $r_{edge} = \frac{r_i + r_j}{2}$',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Generate edge diffusion visualizations"""
    print("Creating edge-focused diffusion visualizations...")
    
    # Create edge diffusion visualization
    fig1 = create_edge_diffusion_visualization()
    edge_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/edge_diffusion_detailed.png'
    fig1.savefig(edge_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Create mechanism diagram
    fig2 = create_diffusion_mechanism_diagram()
    mechanism_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/diffusion_mechanism.png'
    fig2.savefig(mechanism_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    print(f"Edge diffusion visualizations saved:")
    print(f"  Detailed edge diffusion: {edge_path}")
    print(f"  Diffusion mechanism: {mechanism_path}")
    print("\nThese visualizations show:")
    print("  ✓ Risk diffusion along network edges")
    print("  ✓ Edge risk calculation and visualization")
    print("  ✓ Step-by-step diffusion process")
    print("  ✓ Mathematical mechanism explanation")
    print("  ✓ Clear before/after comparison")

if __name__ == "__main__":
    main()
