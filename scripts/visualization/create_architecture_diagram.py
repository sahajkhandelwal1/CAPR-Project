"""
Create a side-view architectural diagram showing the CAPR system technology stack.

This diagram complements the process diagram by showing:
1. The technology stack from bottom to top
2. How each layer builds on the previous one
3. The APIs and interfaces between layers
4. The data transformation at each level
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_architecture_stack():
    """Create a technology stack diagram showing how CAPR is built in layers."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    
    # Define the technology stack layers (bottom to top)
    layers = [
        {
            'name': 'Data Foundation Layer',
            'components': ['OpenStreetMap', 'SF Crime Database', 'PostGIS/SQLite'],
            'tech': 'Raw Data Sources',
            'color': '#34495e',
            'height': 1.2,
            'y_pos': 0.5
        },
        {
            'name': 'Data Processing Layer', 
            'components': ['OSMnx', 'Pandas', 'GeoPandas', 'NetworkX'],
            'tech': 'Python Geospatial Stack',
            'color': '#3498db',
            'height': 1.2,
            'y_pos': 2
        },
        {
            'name': 'Graph Intelligence Layer',
            'components': ['NetworkX Graphs', 'NumPy Arrays', 'Spatial Indexing'],
            'tech': 'Graph Data Structures',
            'color': '#e74c3c',
            'height': 1.2,
            'y_pos': 3.5
        },
        {
            'name': 'Risk Analysis Layer',
            'components': ['Crime Aggregation', 'Risk Diffusion', 'Spatial Smoothing'],
            'tech': 'Mathematical Models',
            'color': '#f39c12',
            'height': 1.2,
            'y_pos': 5
        },
        {
            'name': 'Routing Engine Layer',
            'components': ['Dijkstra Algorithm', 'Multi-objective Optimization', 'Pareto Analysis'],
            'tech': 'Algorithmic Core',
            'color': '#27ae60',
            'height': 1.2,
            'y_pos': 6.5
        },
        {
            'name': 'Application Interface Layer',
            'components': ['Python API', 'Matplotlib Viz', '3D Models', 'Web Interface'],
            'tech': 'User Interfaces',
            'color': '#9b59b6',
            'height': 1.2,
            'y_pos': 8
        }
    ]
    
    # Draw each layer as a 3D block
    layer_width = 10
    layer_x = 2
    depth = 0.2
    
    for i, layer in enumerate(layers):
        # Main rectangle
        rect = FancyBboxPatch(
            (layer_x, layer['y_pos']), layer_width, layer['height'],
            boxstyle="round,pad=0.05",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # 3D effect - top face
        if i < len(layers) - 1:
            top_face = patches.Polygon([
                (layer_x, layer['y_pos'] + layer['height']),
                (layer_x + depth, layer['y_pos'] + layer['height'] + depth),
                (layer_x + layer_width + depth, layer['y_pos'] + layer['height'] + depth),
                (layer_x + layer_width, layer['y_pos'] + layer['height'])
            ], facecolor=layer['color'], edgecolor='black', alpha=0.9, linewidth=1)
            ax.add_patch(top_face)
            
            # 3D effect - right face
            right_face = patches.Polygon([
                (layer_x + layer_width, layer['y_pos']),
                (layer_x + layer_width + depth, layer['y_pos'] + depth),
                (layer_x + layer_width + depth, layer['y_pos'] + layer['height'] + depth),
                (layer_x + layer_width, layer['y_pos'] + layer['height'])
            ], facecolor=layer['color'], edgecolor='black', alpha=0.6, linewidth=1)
            ax.add_patch(right_face)
        
        # Layer name
        ax.text(layer_x + layer_width/2, layer['y_pos'] + layer['height'] - 0.15,
               layer['name'], ha='center', va='top', fontsize=12, fontweight='bold',
               color='white')
        
        # Technology type
        ax.text(layer_x + layer_width/2, layer['y_pos'] + layer['height'] - 0.45,
               layer['tech'], ha='center', va='center', fontsize=10,
               color='white', style='italic')
        
        # Components
        components_text = ' • '.join(layer['components'])
        ax.text(layer_x + layer_width/2, layer['y_pos'] + 0.3,
               components_text, ha='center', va='center', fontsize=9,
               color='white', fontweight='bold')
    
    # Add arrows showing dependencies
    arrow_x = layer_x + layer_width/2
    for i in range(len(layers) - 1):
        start_y = layers[i]['y_pos'] + layers[i]['height']
        end_y = layers[i + 1]['y_pos']
        
        ax.annotate('', xy=(arrow_x, end_y), xytext=(arrow_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add side annotations showing data flow
    data_flow_x = layer_x + layer_width + 1.5
    data_flows = [
        ('Raw street\n& crime data', 0.5, '#7f8c8d'),
        ('Cleaned\ngraph data', 2, '#2980b9'),  
        ('Risk-weighted\nnetwork', 3.5, '#c0392b'),
        ('Intelligence\nscores', 5, '#d35400'),
        ('Optimal\nroutes', 6.5, '#229954'),
        ('User\ninterface', 8, '#8e44ad')
    ]
    
    for text, y_pos, color in data_flows:
        ax.text(data_flow_x, y_pos + 0.6, text, ha='left', va='center',
               fontsize=10, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=color, alpha=0.8))
    
    # Add input/output indicators
    # Input (bottom left)
    ax.annotate('DATA SOURCES\n\n📊 OpenStreetMap\n🚔 Crime Reports\n📍 Geospatial Data',
               xy=(layer_x - 0.5, layers[0]['y_pos'] + layers[0]['height']/2),
               xytext=(layer_x - 2.5, layers[0]['y_pos'] + layers[0]['height']/2),
               ha='center', va='center', fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    # Output (top right)
    ax.annotate('APPLICATIONS\n\n🎯 Safe Routing API\n📱 Mobile App\n🌐 Web Interface\n📊 Analytics Dashboard',
               xy=(layer_x + layer_width + 0.5, layers[-1]['y_pos'] + layers[-1]['height']/2),
               xytext=(layer_x + layer_width + 2.5, layers[-1]['y_pos'] + layers[-1]['height']/2),
               ha='center', va='center', fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', lw=2, color='green'),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9))
    
    # Title
    ax.text(7, 9.5, 'CAPR System: Technology Stack Architecture',
           ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(7, 9.1, 'Crime-Aware Pedestrian Routing - Layered Technology Implementation',
           ha='center', va='center', fontsize=12, style='italic')
    
    # Add technology badges
    tech_badges = [
        (0.5, 8.5, '🐍\nPython', 'lightblue'),
        (0.5, 7.5, '🗺️\nOSMnx', 'lightcyan'),
        (0.5, 6.5, '📊\nPandas', 'lightgreen'),
        (0.5, 5.5, '🔗\nNetworkX', 'lightyellow'),
        (0.5, 4.5, '🧮\nNumPy', 'lightcoral'),
        (0.5, 3.5, '📐\nMatplotlib', 'lightpink'),
    ]
    
    for x, y, text, color in tech_badges:
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig

def create_api_interface_diagram():
    """Create a diagram showing the API interfaces between layers."""
    
    print("\n🔗 CAPR SYSTEM API INTERFACES")
    print("=" * 50)
    
    interfaces = [
        {
            'layer': 'Data Foundation → Processing',
            'interface': 'File I/O APIs',
            'methods': ['OSM XML/PBF readers', 'CSV loaders', 'Database connectors'],
            'data_format': 'Raw files → DataFrames'
        },
        {
            'layer': 'Processing → Graph Intelligence',
            'interface': 'NetworkX Graph API',
            'methods': ['graph.add_edge()', 'graph.nodes[id]', 'nx.shortest_path()'],
            'data_format': 'DataFrames → Graph objects'
        },
        {
            'layer': 'Graph Intelligence → Risk Analysis',
            'interface': 'Mathematical Functions',
            'methods': ['risk_aggregation()', 'spatial_diffusion()', 'score_normalization()'],
            'data_format': 'Graph attributes → Risk scores'
        },
        {
            'layer': 'Risk Analysis → Routing Engine',
            'interface': 'Optimization API',
            'methods': ['multi_objective_dijkstra()', 'pareto_frontier()', 'route_scoring()'],
            'data_format': 'Risk scores → Route weights'
        },
        {
            'layer': 'Routing Engine → Application',
            'interface': 'Public API',
            'methods': ['find_route(start, end, beta)', 'get_risk_map()', 'analyze_routes()'],
            'data_format': 'Route requests → JSON responses'
        }
    ]
    
    for i, interface in enumerate(interfaces, 1):
        print(f"\n{i}. {interface['layer']}:")
        print(f"   Interface: {interface['interface']}")
        print(f"   Methods:   {', '.join(interface['methods'])}")
        print(f"   Data Flow: {interface['data_format']}")

def main():
    """Generate the architecture stack diagram."""
    
    # Create output directory
    output_dir = Path("visualization")
    output_dir.mkdir(exist_ok=True)
    
    print("Creating CAPR architecture stack diagram...")
    
    # Generate the diagram
    fig = create_architecture_stack()
    
    # Save the diagram
    output_path = output_dir / "capr_architecture_stack.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Architecture diagram saved: {output_path}")
    
    # Show API interfaces
    create_api_interface_diagram()
    
    print(f"\n🏗️ ARCHITECTURE HIGHLIGHTS:")
    print("  • 6-layer technology stack")
    print("  • Clean separation of concerns")
    print("  • Modular, extensible design")
    print("  • Standard Python geospatial ecosystem")
    print("  • APIs for easy integration")
    print("  • Scalable to other cities/datasets")
    
    plt.show()

if __name__ == "__main__":
    main()
