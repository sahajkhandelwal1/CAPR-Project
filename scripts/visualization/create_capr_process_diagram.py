"""
CAPR System Process Layer Visualization

Creates a side-view diagram showing how we progressively build layers
on top of the base street network to create the final crime-aware routing engine.

Shows the transformation pipeline:
1. Base OSM Street Network
2. + Pedestrian Graph Processing  
3. + Crime Data Integration
4. + Risk Scoring & Aggregation
5. + Spatial Risk Diffusion
6. + Multi-Objective Routing Engine

Each layer adds functionality and intelligence to create the final system.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import pandas as pd


def create_capr_process_diagram():
    """Create a comprehensive process layer diagram showing CAPR system evolution."""
    
    # Create figure with specific aspect ratio for side view
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Color scheme for different layers
    colors = {
        'base': '#E8E8E8',          # Light gray for base layer
        'network': '#FFD700',        # Gold for network processing
        'crime': '#FF6B6B',          # Red for crime data
        'scoring': '#4ECDC4',        # Teal for scoring
        'diffusion': '#45B7D1',      # Blue for diffusion
        'routing': '#96CEB4',        # Green for final routing
        'text': '#2C3E50',           # Dark blue for text
        'arrow': '#34495E',          # Dark gray for arrows
        'highlight': '#E74C3C'       # Red for highlights
    }
    
    # Layer specifications (from bottom to top)
    layers = [
        {
            'name': 'Multi-Objective Safety Routing Engine',
            'description': 'β-parameterized pathfinding\n• Pareto frontier analysis\n• Real-time route optimization\n• Safety vs. distance tradeoffs',
            'color': colors['routing'],
            'height': 1.5,
            'data_points': ['Route requests', 'β parameters', 'Optimal paths'],
            'algorithms': ['Dijkstra variant', 'Multi-objective optimization', 'Pareto analysis']
        },
        {
            'name': 'Enhanced Spatial Risk Diffusion Layer',
            'description': 'Graph-based risk propagation\n• Adaptive diffusion (α=0.3)\n• High-risk core preservation\n• Regional smoothing',
            'color': colors['diffusion'],
            'height': 1.5,
            'data_points': ['Edge adjacency', 'Diffused scores', 'Spatial gradients'],
            'algorithms': ['Graph Laplacian diffusion', 'Risk amplification', 'Curve normalization']
        },
        {
            'name': 'Crime Risk Scoring & Aggregation Layer',
            'description': 'Edge-level danger computation\n• R = s̄ × log(1 + n)\n• Normalized to [1-100] scale\n• Crime-to-edge mapping',
            'color': colors['scoring'],
            'height': 1.5,
            'data_points': ['Crime clusters', 'Aggregated scores', 'Risk statistics'],
            'algorithms': ['Spatial aggregation', 'Risk normalization', 'Statistical modeling']
        },
        {
            'name': 'Crime Data Integration Layer',
            'description': 'SF Police incident processing\n• 565,783 incidents\n• Severity scoring\n• Spatial indexing',
            'color': colors['crime'],
            'height': 1.5,
            'data_points': ['Raw incidents', 'Scored crimes', 'Spatial mapping'],
            'algorithms': ['Data cleaning', 'Category scoring', 'Geocoding validation']
        },
        {
            'name': 'Pedestrian Network Processing Layer',
            'description': 'OSM graph enhancement\n• 51,726 walkable edges\n• UTM projection\n• Geometry processing',
            'color': colors['network'],
            'height': 1.5,
            'data_points': ['Network topology', 'Edge geometries', 'Node coordinates'],
            'algorithms': ['Graph simplification', 'CRS transformation', 'Topology validation']
        },
        {
            'name': 'Base OpenStreetMap Data Layer',
            'description': 'Raw street network\n• San Francisco region\n• Street geometries\n• Basic connectivity',
            'color': colors['base'],
            'height': 1.2,
            'data_points': ['Raw OSM data', 'Street segments', 'Intersections'],
            'algorithms': ['Data extraction', 'Format conversion', 'Initial validation']
        }
    ]
    
    # Calculate positions
    total_height = sum(layer['height'] for layer in layers)
    y_start = 0.5
    y_positions = []
    current_y = y_start
    
    for layer in layers:
        y_positions.append(current_y)
        current_y += layer['height']
    
    # Draw layers as 3D-looking blocks
    layer_width = 12
    layer_start_x = 1
    depth_offset = 0.15
    
    for i, (layer, y_pos) in enumerate(zip(layers, y_positions)):
        height = layer['height']
        
        # Main rectangle (front face)
        main_rect = patches.Rectangle(
            (layer_start_x, y_pos), layer_width, height,
            facecolor=layer['color'], edgecolor='black', linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(main_rect)
        
        # 3D depth effect (top and side faces)
        if i < len(layers) - 1:  # Don't add depth to top layer
            # Top face
            top_face = patches.Polygon([
                (layer_start_x, y_pos + height),
                (layer_start_x + depth_offset, y_pos + height + depth_offset),
                (layer_start_x + layer_width + depth_offset, y_pos + height + depth_offset),
                (layer_start_x + layer_width, y_pos + height)
            ], facecolor=layer['color'], edgecolor='black', linewidth=1, alpha=0.9)
            ax.add_patch(top_face)
            
            # Right face
            right_face = patches.Polygon([
                (layer_start_x + layer_width, y_pos),
                (layer_start_x + layer_width + depth_offset, y_pos + depth_offset),
                (layer_start_x + layer_width + depth_offset, y_pos + height + depth_offset),
                (layer_start_x + layer_width, y_pos + height)
            ], facecolor=layer['color'], edgecolor='black', linewidth=1, alpha=0.6)
            ax.add_patch(right_face)
        
        # Layer title
        ax.text(layer_start_x + layer_width/2, y_pos + height - 0.2, layer['name'],
               ha='center', va='top', fontsize=12, fontweight='bold', color=colors['text'])
        
        # Layer description
        ax.text(layer_start_x + 0.3, y_pos + height - 0.5, layer['description'],
               ha='left', va='top', fontsize=9, color=colors['text'])
        
        # Data flow indicators (right side)
        data_y = y_pos + height/2
        for j, data_point in enumerate(layer['data_points']):
            ax.text(layer_start_x + layer_width + 0.5, data_y - j*0.3, f"• {data_point}",
                   ha='left', va='center', fontsize=8, color=colors['text'], style='italic')
        
        # Algorithms (left side)
        algo_y = y_pos + height/2
        for j, algo in enumerate(layer['algorithms']):
            ax.text(layer_start_x - 0.1, algo_y - j*0.3, f"⚙️ {algo}",
                   ha='right', va='center', fontsize=8, color=colors['text'], weight='bold')
    
    # Add data flow arrows between layers
    arrow_x = layer_start_x + layer_width/2
    for i in range(len(y_positions) - 1):
        start_y = y_positions[i] + layers[i]['height']
        end_y = y_positions[i + 1]
        
        # Main upward arrow
        ax.annotate('', xy=(arrow_x, end_y), xytext=(arrow_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=3, color=colors['arrow']))
        
        # Data processing indicator
        mid_y = (start_y + end_y) / 2
        ax.text(arrow_x + 0.2, mid_y, 'Processing & Enhancement',
               ha='left', va='center', fontsize=8, color=colors['arrow'],
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add side annotations showing the progression
    progression_x = layer_start_x + layer_width + 4.5
    progression_items = [
        ('Raw Data', '📊 Base street network'),
        ('Processing', '🔧 Graph optimization'),
        ('Data Integration', '📍 Crime mapping'),
        ('Intelligence', '🧠 Risk computation'),
        ('Enhancement', '🌊 Spatial diffusion'),
        ('Application', '🎯 Smart routing')
    ]
    
    for i, (stage, description) in enumerate(progression_items):
        stage_y = y_positions[i] + layers[i]['height']/2
        
        # Stage indicator
        stage_box = patches.FancyBboxPatch(
            (progression_x, stage_y - 0.3), 3.5, 0.6,
            boxstyle="round,pad=0.1", facecolor=layers[i]['color'], 
            edgecolor='black', alpha=0.7
        )
        ax.add_patch(stage_box)
        
        ax.text(progression_x + 1.75, stage_y, f"{stage}\n{description}",
               ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text'])
    
    # Add title and key metrics
    ax.text(8, total_height + 1.5, 'Crime-Aware Pedestrian Routing (CAPR)\nSystem Architecture & Process Flow',
           ha='center', va='center', fontsize=18, fontweight='bold', color=colors['text'])
    
    # Add key metrics box
    metrics_text = """Key System Metrics:
📊 565,783 crime incidents processed
🛣️  51,726 street network edges
📈 Risk scores: 6.4 → 35.0 mean
🎯 Multi-objective optimization
⚡ Real-time route computation
📐 3D visualization ready"""
    
    metrics_box = patches.FancyBboxPatch(
        (layer_start_x + layer_width + 6, total_height - 1), 4, 2,
        boxstyle="round,pad=0.2", facecolor='lightblue', 
        edgecolor='blue', alpha=0.3
    )
    ax.add_patch(metrics_box)
    
    ax.text(layer_start_x + layer_width + 8, total_height, metrics_text,
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add input/output indicators
    # Input arrow (bottom)
    ax.annotate('Raw OSM Data\nSF Police Reports', 
               xy=(layer_start_x - 2, y_positions[0] + layers[0]['height']/2),
               xytext=(layer_start_x - 4, y_positions[0] + layers[0]['height']/2),
               ha='center', va='center', fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['highlight']),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Output arrow (top)
    ax.annotate('Crime-Aware\nRouting API', 
               xy=(layer_start_x + layer_width + 2, y_positions[-1] + layers[-1]['height']/2),
               xytext=(layer_start_x + layer_width + 4, y_positions[-1] + layers[-1]['height']/2),
               ha='center', va='center', fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['highlight']),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Set axis properties
    ax.set_xlim(-5, 20)
    ax.set_ylim(0, total_height + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the diagram
    output_path = Path('visualization/capr_process_layers_diagram.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ CAPR Process Layer Diagram saved: {output_path}")
    return output_path


def create_data_flow_summary():
    """Create a summary showing data transformations at each layer."""
    
    print("\n🔄 CAPR SYSTEM DATA FLOW SUMMARY")
    print("=" * 60)
    
    flow_stages = [
        {
            'stage': 'Layer 1: Raw OSM Data',
            'input': 'OpenStreetMap XML/PBF files',
            'processing': 'Extract pedestrian-accessible ways and nodes',
            'output': 'Raw street network graph',
            'size': '~10MB raw data'
        },
        {
            'stage': 'Layer 2: Network Processing',
            'input': 'Raw street graph',
            'processing': 'Simplify, project to UTM, validate geometry',
            'output': 'Clean pedestrian graph (10,819 nodes, 51,726 edges)',
            'size': '~5MB GraphML'
        },
        {
            'stage': 'Layer 3: Crime Integration',
            'input': 'SF Police incident reports + Clean graph',
            'processing': 'Filter, score severity, geocode, map to edges',
            'output': 'Crime-incident database with spatial mapping',
            'size': '565,783 incidents → ~50MB processed'
        },
        {
            'stage': 'Layer 4: Risk Aggregation',
            'input': 'Mapped crime incidents',
            'processing': 'Aggregate by edge: R = s̄ × log(1 + n), normalize [1-100]',
            'output': 'Edge risk scores (mean: 6.4, sparse distribution)',
            'size': '51,726 risk scores → ~2MB CSV'
        },
        {
            'stage': 'Layer 5: Spatial Diffusion',
            'input': 'Sparse edge risk scores',
            'processing': 'Graph diffusion, risk amplification, curve fitting',
            'output': 'Enhanced risk scores (mean: 35.0, realistic distribution)',
            'size': '51,726 enhanced scores → ~3MB CSV'
        },
        {
            'stage': 'Layer 6: Routing Engine',
            'input': 'Risk-weighted graph + route queries',
            'processing': 'Multi-objective optimization (β-parameterized)',
            'output': 'Optimal safe routes with Pareto analysis',
            'size': 'Real-time API responses'
        }
    ]
    
    for i, stage in enumerate(flow_stages, 1):
        print(f"\n📍 {stage['stage']}:")
        print(f"   Input:      {stage['input']}")
        print(f"   Processing: {stage['processing']}")
        print(f"   Output:     {stage['output']}")
        print(f"   Data Size:  {stage['size']}")
        
        if i < len(flow_stages):
            print("   ↓ ↓ ↓")
    
    print(f"\n🎯 FINAL SYSTEM CAPABILITIES:")
    print("   • Real-time safety-aware route planning")
    print("   • Multi-objective optimization (safety vs. distance)")
    print("   • Pareto frontier analysis for decision support")
    print("   • 3D visualization of crime risk landscape")
    print("   • Scalable to other cities and crime datasets")


if __name__ == "__main__":
    # Create the process layer diagram
    diagram_path = create_capr_process_diagram()
    
    # Print data flow summary
    create_data_flow_summary()
    
    print(f"\n🎨 Process diagram saved to: {diagram_path}")
    print("\n✅ CAPR System Process Documentation Complete!")
