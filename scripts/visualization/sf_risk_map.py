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
    
    # Create the visualization - single panel only
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define color schemes
    risk_colors = ['#2E8B57', '#90EE90', '#FFFF00', '#FFA500', '#FF4500', '#8B0000']  # Green to Red
    risk_cmap = mcolors.LinearSegmentedColormap.from_list('risk', risk_colors, N=256)
    
    # Create subtle but noticeable line widths based on risk level
    edges_gdf['line_width'] = edges_gdf['final_risk'].apply(
        lambda x: 1.2 if x >= 80 else    # Very high risk - slightly thick lines
                  0.9 if x >= 60 else    # High risk - medium lines  
                  0.6 if x >= 40 else    # Medium risk - slightly thin lines
                  0.4                    # Low risk - thin lines
    )
    
    # Plot different risk levels with different line widths for emphasis
    # Plot low-medium risk areas first (thin lines)
    low_medium_risk = edges_gdf[edges_gdf['final_risk'] < 60]
    if len(low_medium_risk) > 0:
        low_medium_risk.plot(
            column='final_risk',
            cmap=risk_cmap,
            linewidth=low_medium_risk['line_width'],
            ax=ax,
            alpha=0.7
        )
    
    # Plot high risk areas (medium-thick lines)
    high_risk = edges_gdf[(edges_gdf['final_risk'] >= 60) & (edges_gdf['final_risk'] < 80)]
    if len(high_risk) > 0:
        high_risk.plot(
            column='final_risk',
            cmap=risk_cmap,
            linewidth=high_risk['line_width'],
            ax=ax,
            alpha=0.8
        )
    
    # Plot very high risk areas last (thick lines) to make them most prominent
    very_high_risk = edges_gdf[edges_gdf['final_risk'] >= 80]
    if len(very_high_risk) > 0:
        very_high_risk.plot(
            column='final_risk',
            cmap=risk_cmap,
            linewidth=very_high_risk['line_width'],
            ax=ax,
            alpha=0.9
        )
    
    # Add a colorbar legend manually for better control
    sm = plt.cm.ScalarMappable(cmap=risk_cmap, norm=plt.Normalize(
        vmin=edges_gdf['final_risk'].min(), 
        vmax=edges_gdf['final_risk'].max()
    ))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
    cbar.set_label('Risk Score (1-100)', fontsize=12, fontweight='bold')
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=edges_gdf.crs, source=ctx.providers.CartoDB.Positron, alpha=0.6)
    except:
        print("Note: Could not add basemap - continuing without background")
    
    ax.set_title('Enhanced Risk-Weighted Street Network', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    
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
