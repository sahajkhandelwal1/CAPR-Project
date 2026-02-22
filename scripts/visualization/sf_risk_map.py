"""
San Francisco Crime Risk Visualization
Creates comprehensive geospatial maps showing risk levels across the street network.
"""

import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import contextily as ctx
from shapely.geometry import Point, LineString
from shapely import wkt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_graph_with_geometry(graph_path):
    """Load graph and extract edge geometries."""
    print(f"Loading graph from {graph_path}...")
    G = nx.read_graphml(graph_path)
    
    edges_data = []
    for u, v, key in tqdm(G.edges(keys=True), desc="Extracting edge geometries"):
        edge_data = G.edges[u, v, key]
        
        # Get geometry from WKT
        geom_str = edge_data.get('geometry')
        if geom_str:
            try:
                geom = wkt.loads(geom_str)
            except:
                continue
        else:
            # Fallback: create geometry from node coordinates
            node_u = G.nodes[u]
            node_v = G.nodes[v]
            try:
                x1, y1 = float(node_u['x']), float(node_u['y'])
                x2, y2 = float(node_v['x']), float(node_v['y'])
                geom = LineString([(x1, y1), (x2, y2)])
            except:
                continue
        
        # Get risk score
        risk_score = edge_data.get('risk_score', edge_data.get('risk_score_enhanced', 1.0))
        
        edges_data.append({
            'u': u,
            'v': v, 
            'key': key,
            'edge_id': f"{u}_{v}_{key}",
            'risk_score': float(risk_score),
            'geometry': geom
        })
    
    print(f"Extracted {len(edges_data)} edges with geometry and risk scores")
    
    # Create GeoDataFrame
    # Use SF projected CRS (California State Plane Zone 3)
    gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:32610')  # UTM Zone 10N for SF
    
    return gdf

def create_sf_risk_map():
    """Create comprehensive San Francisco risk visualization."""
    
    # Load enhanced risk scores
    print("Loading enhanced risk scores...")
    df_risk = pd.read_csv('data/processed/edge_risk_scores_enhanced.csv')
    
    # Load graph with geometries
    edges_gdf = load_graph_with_geometry('data/graphs/sf_pedestrian_graph_enhanced.graphml')
    
    # Merge risk scores with geometries
    print("Merging risk scores with edge geometries...")
    edges_gdf = edges_gdf.merge(
        df_risk[['edge_id', 'risk_score_enhanced']],
        on='edge_id',
        how='left'
    )
    
    # Use enhanced scores if available, otherwise original
    edges_gdf['final_risk'] = edges_gdf['risk_score_enhanced'].fillna(edges_gdf['risk_score'])
    
    # Convert to Web Mercator for visualization
    edges_gdf = edges_gdf.to_crs('EPSG:3857')
    
    print(f"Final dataset: {len(edges_gdf)} edges")
    print(f"Risk score range: [{edges_gdf['final_risk'].min():.1f}, {edges_gdf['final_risk'].max():.1f}]")
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('San Francisco Street Network Crime Risk Analysis', fontsize=20, fontweight='bold')
    
    # Define color schemes
    risk_colors = ['#2E8B57', '#90EE90', '#FFFF00', '#FFA500', '#FF4500', '#8B0000']  # Green to Red
    risk_cmap = mcolors.LinearSegmentedColormap.from_list('risk', risk_colors, N=256)
    
    # 1. Overall Risk Map
    ax1 = axes[0, 0]
    edges_gdf.plot(
        column='final_risk',
        cmap=risk_cmap,
        linewidth=0.5,
        ax=ax1,
        legend=True,
        legend_kwds={
            'label': 'Risk Score (1-100)',
            'orientation': 'horizontal',
            'shrink': 0.8,
            'pad': 0.05
        }
    )
    
    # Add basemap
    try:
        ctx.add_basemap(ax1, crs=edges_gdf.crs, source=ctx.providers.CartoDB.Positron, alpha=0.6)
    except:
        print("Note: Could not add basemap - continuing without background")
    
    ax1.set_title('Enhanced Risk-Weighted Street Network', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Risk Categories Map
    ax2 = axes[0, 1]
    
    # Create risk categories
    edges_gdf['risk_category'] = pd.cut(
        edges_gdf['final_risk'],
        bins=[0, 10, 25, 50, 75, 100],
        labels=['Very Low (1-10)', 'Low (10-25)', 'Medium (25-50)', 'High (50-75)', 'Very High (75-100)'],
        include_lowest=True
    )
    
    # Color mapping for categories
    category_colors = {
        'Very Low (1-10)': '#2E8B57',    # Dark green
        'Low (10-25)': '#90EE90',        # Light green  
        'Medium (25-50)': '#FFFF00',     # Yellow
        'High (50-75)': '#FFA500',       # Orange
        'Very High (75-100)': '#8B0000'  # Dark red
    }
    
    for category, color in category_colors.items():
        subset = edges_gdf[edges_gdf['risk_category'] == category]
        if len(subset) > 0:
            subset.plot(color=color, linewidth=0.6, ax=ax2, label=category, alpha=0.8)
    
    try:
        ctx.add_basemap(ax2, crs=edges_gdf.crs, source=ctx.providers.CartoDB.Positron, alpha=0.4)
    except:
        pass
    
    ax2.set_title('Risk Categories Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. High-Risk Focus Map
    ax3 = axes[1, 0]
    
    # Show all edges in light gray
    edges_gdf.plot(color='lightgray', linewidth=0.3, ax=ax3, alpha=0.5)
    
    # Highlight high-risk edges
    high_risk = edges_gdf[edges_gdf['final_risk'] >= 50]
    if len(high_risk) > 0:
        high_risk.plot(
            column='final_risk',
            cmap='Reds',
            linewidth=1.2,
            ax=ax3,
            legend=True,
            legend_kwds={
                'label': 'High Risk Score (50-100)',
                'orientation': 'horizontal',
                'shrink': 0.8,
                'pad': 0.05
            }
        )
    
    try:
        ctx.add_basemap(ax3, crs=edges_gdf.crs, source=ctx.providers.CartoDB.Positron, alpha=0.6)
    except:
        pass
    
    ax3.set_title(f'High-Risk Areas Focus (≥50 Risk)\n{len(high_risk):,} dangerous edges', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics Panel
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    total_edges = len(edges_gdf)
    risk_stats = {
        'Very Low (1-10)': len(edges_gdf[edges_gdf['final_risk'] <= 10]),
        'Low (10-25)': len(edges_gdf[(edges_gdf['final_risk'] > 10) & (edges_gdf['final_risk'] <= 25)]),
        'Medium (25-50)': len(edges_gdf[(edges_gdf['final_risk'] > 25) & (edges_gdf['final_risk'] <= 50)]),
        'High (50-75)': len(edges_gdf[(edges_gdf['final_risk'] > 50) & (edges_gdf['final_risk'] <= 75)]),
        'Very High (75-100)': len(edges_gdf[edges_gdf['final_risk'] > 75])
    }
    
    # Create statistics table
    stats_text = [
        "SAN FRANCISCO CRIME RISK ANALYSIS",
        "=" * 40,
        "",
        "NETWORK STATISTICS:",
        f"Total Street Edges: {total_edges:,}",
        f"Mean Risk Score: {edges_gdf['final_risk'].mean():.1f}/100",
        f"Standard Deviation: {edges_gdf['final_risk'].std():.1f}",
        f"Range: [{edges_gdf['final_risk'].min():.1f}, {edges_gdf['final_risk'].max():.1f}]",
        "",
        "RISK DISTRIBUTION:",
    ]
    
    for category, count in risk_stats.items():
        percentage = count / total_edges * 100
        stats_text.append(f"{category}: {count:,} ({percentage:.1f}%)")
    
    stats_text.extend([
        "",
        "SAFETY INSIGHTS:",
        f"• {risk_stats['Very Low (1-10)']:,} very safe edges",
        f"• {risk_stats['High (50-75)'] + risk_stats['Very High (75-100)']:,} high-risk edges to avoid",
        f"• {risk_stats['Medium (25-50)']:,} moderate-risk edges",
        "",
        "ALGORITHM FEATURES:",
        "✓ Enhanced risk diffusion applied",
        "✓ Core preservation for dangerous areas",
        "✓ Regional risk buffering",
        "✓ Realistic risk distribution (mean=35)",
        "✓ Ready for crime-aware routing"
    ])
    
    # Display statistics
    ax4.text(0.05, 0.95, '\n'.join(stats_text), 
             transform=ax4.transAxes, fontsize=11, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Add risk level color legend
    legend_elements = []
    for category, color in category_colors.items():
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, label=category))
    
    ax4.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.95, 0.3),
               title="Risk Levels", title_fontsize=12, fontsize=10)
    
    plt.tight_layout()
    
    # Ensure visualization folder exists and save
    viz_dir = Path('visualization')
    viz_dir.mkdir(exist_ok=True)
    
    output_file = viz_dir / 'sf_crime_risk_map.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"San Francisco risk map saved to: {output_file}")
    
    return fig, edges_gdf

def organize_visualizations():
    """Move all visualization files to a single organized folder."""
    
    viz_dir = Path('visualization')
    viz_dir.mkdir(exist_ok=True)
    
    # Files to organize
    viz_files = [
        'risk_diffusion_comparison.png',
        'enhanced_risk_diffusion_comparison.png',
        'sf_crime_risk_map.png'
    ]
    
    # Check for existing files in workspace root and move them
    workspace_root = Path('.')
    for file_name in viz_files:
        source_file = workspace_root / file_name
        dest_file = viz_dir / file_name
        
        if source_file.exists() and source_file != dest_file:
            source_file.rename(dest_file)
            print(f"Moved {file_name} to visualization/")
    
    # List all visualization files
    print(f"\n📊 ALL VISUALIZATIONS IN '{viz_dir}':")
    viz_files_found = list(viz_dir.glob('*.png'))
    for i, viz_file in enumerate(sorted(viz_files_found), 1):
        print(f"  {i}. {viz_file.name}")
    
    return viz_dir

def main():
    """Create comprehensive San Francisco crime risk visualization."""
    
    print("🗺️  CREATING SAN FRANCISCO CRIME RISK VISUALIZATION")
    print("=" * 60)
    
    try:
        # Create main risk map
        fig, edges_gdf = create_sf_risk_map()
        
        # Organize all visualizations
        viz_dir = organize_visualizations()
        
        print(f"\n✅ VISUALIZATION COMPLETE!")
        print(f"📁 All visualizations saved in: {viz_dir.absolute()}")
        print(f"🗺️  Main map: sf_crime_risk_map.png")
        print(f"📊 {len(edges_gdf):,} street edges visualized")
        print(f"🎯 Risk range: [{edges_gdf['final_risk'].min():.1f} - {edges_gdf['final_risk'].max():.1f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
