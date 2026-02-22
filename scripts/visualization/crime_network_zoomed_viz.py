#!/usr/bin/env python3
"""
Zoomed-in visualization of pedestrian network with crime/safety points.

Shows a detailed view of a specific San Francisco area with:
- Pedestrian graph nodes (intersections)
- Street edges 
- Crime incident locations mapped to the network
- Safety risk indicators
"""

import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
from collections import defaultdict

class CrimeNetworkVisualizer:
    """Creates zoomed visualizations of crime data mapped to pedestrian network."""
    
    def __init__(self, graph_path, nodes_path, edges_path, crime_mapping_path):
        self.graph_path = Path(graph_path)
        self.nodes_path = Path(nodes_path)
        self.edges_path = Path(edges_path)
        self.crime_mapping_path = Path(crime_mapping_path)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load all required data files."""
        print("📊 Loading network and crime data...")
        
        # Load graph
        self.G = nx.read_graphml(self.graph_path)
        print(f"   📈 Loaded graph: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        
        # Load geographic data
        self.nodes_gdf = gpd.read_file(self.nodes_path)
        self.edges_gdf = gpd.read_file(self.edges_path)
        print(f"   🗺️  Loaded geography: {len(self.nodes_gdf)} nodes, {len(self.edges_gdf)} edges")
        
        # Load crime mapping
        self.crime_mapping = pd.read_csv(self.crime_mapping_path)
        print(f"   🚨 Loaded crime mapping: {len(self.crime_mapping)} incidents")
        
        # Calculate crime density per edge
        self.calculate_crime_density()
    
    def calculate_crime_density(self):
        """Calculate crime incidents per edge for risk visualization (less sampling for better coverage)."""
        print("🔍 Calculating crime density per network edge (optimized for coverage)...")
        
        # Use more data for better crime coverage
        sample_size = min(200000, len(self.crime_mapping))  # Process up to 200k records for better coverage
        crime_sample = self.crime_mapping.sample(n=sample_size, random_state=42)
        print(f"   📊 Processing {len(crime_sample)} crime records (sampled from {len(self.crime_mapping)})")
        
        # Count crimes per edge
        edge_crime_counts = defaultdict(int)
        
        for _, row in crime_sample.iterrows():
            edge_u = row['edge_u']
            edge_v = row['edge_v'] 
            edge_key = row['edge_key']
            
            if pd.notna(edge_u) and pd.notna(edge_v) and pd.notna(edge_key):
                edge_id = (int(edge_u), int(edge_v), int(edge_key))
                edge_crime_counts[edge_id] += 1
        
        # Add crime counts to edges GeoDataFrame
        self.edges_gdf['crime_count'] = 0
        
        for idx, row in self.edges_gdf.iterrows():
            edge_id = (row['u'], row['v'], row['key'])
            crime_count = edge_crime_counts.get(edge_id, 0)
            self.edges_gdf.at[idx, 'crime_count'] = crime_count
        
        print(f"   📊 Crime statistics:")
        print(f"      • Total edges with crimes: {sum(1 for c in edge_crime_counts.values() if c > 0)}")
        print(f"      • Max crimes per edge: {max(edge_crime_counts.values()) if edge_crime_counts else 0}")
        print(f"      • Mean crimes per edge: {np.mean(list(edge_crime_counts.values())) if edge_crime_counts else 0:.2f}")
    
    def get_crime_locations(self, bbox):
        """Get crime edge mappings within bounding box (optimized for better coverage)."""
        print("   🔍 Finding crimes in bounding box...")
        
        # Use more data for better crime coverage in the visualization area
        sample_size = min(50000, len(self.crime_mapping))  # Increased for better coverage
        crime_sample = self.crime_mapping.sample(n=sample_size, random_state=42)
        
        print(f"   📊 Processing {len(crime_sample)} crime incidents (sampled from {len(self.crime_mapping)})")
        
        crime_edges_in_area = []
        crime_scores = []
        
        for _, row in crime_sample.iterrows():
            edge_u = row['edge_u']
            edge_v = row['edge_v'] 
            edge_key = row['edge_key']
            
            if pd.notna(edge_u) and pd.notna(edge_v) and pd.notna(edge_key):
                # Find if this edge is in our area
                edge_mask = ((self.edges_gdf['u'] == int(edge_u)) & 
                           (self.edges_gdf['v'] == int(edge_v)) & 
                           (self.edges_gdf['key'] == int(edge_key)))
                
                if edge_mask.any():
                    edge_geom = self.edges_gdf[edge_mask].iloc[0]['geometry']
                    
                    # Check if edge intersects with bounding box
                    if hasattr(edge_geom, 'bounds'):
                        minx, miny, maxx, maxy = edge_geom.bounds
                        if (minx <= bbox['east'] and maxx >= bbox['west'] and 
                            miny <= bbox['north'] and maxy >= bbox['south']):
                            
                            # Get a representative point on the edge
                            if hasattr(edge_geom, 'interpolate'):
                                crime_point = edge_geom.interpolate(0.5, normalized=True)
                                crime_edges_in_area.append(crime_point)
                                
                                # Get crime score if available
                                score = row.get('score', 1.0)  # Default to 1.0 if no score
                                crime_scores.append(score)
        
        print(f"   ✅ Found {len(crime_edges_in_area)} crime locations in area")
        return crime_edges_in_area, crime_scores
    
    def create_zoomed_visualization(self, area_name="Mission District", 
                                  bbox=None, output_path=None):
        """Create a zoomed-in visualization of the network with crime data."""
        
        if bbox is None:
            # Default to Mission District area
            bbox = {
                'west': 551000,   # UTM coordinates
                'east': 553000,
                'south': 4175000,
                'north': 4177000
            }
        
        print(f"\n🎯 Creating zoomed visualization for {area_name}...")
        print(f"   📦 Bounding box: {bbox}")
        
        # Filter nodes and edges to the bounding box
        nodes_in_area = self.nodes_gdf[
            (self.nodes_gdf.geometry.x >= bbox['west']) &
            (self.nodes_gdf.geometry.x <= bbox['east']) &
            (self.nodes_gdf.geometry.y >= bbox['south']) &
            (self.nodes_gdf.geometry.y <= bbox['north'])
        ]
        
        edges_in_area = self.edges_gdf[
            (self.edges_gdf.bounds.minx >= bbox['west']) &
            (self.edges_gdf.bounds.maxx <= bbox['east']) &
            (self.edges_gdf.bounds.miny >= bbox['south']) &
            (self.edges_gdf.bounds.maxy <= bbox['north'])
        ]
        
        # Get crime points in the area
        crime_points, crime_scores = self.get_crime_locations(bbox)
        
        print(f"   📊 Area statistics:")
        print(f"      • Nodes: {len(nodes_in_area)}")
        print(f"      • Edges: {len(edges_in_area)}")
        print(f"      • Crime incidents: {len(crime_points)}")
        
        # Create the visualization with wider figure to accommodate side statistics
        fig, (ax, ax_stats) = plt.subplots(1, 2, figsize=(24, 12), facecolor='white',
                                           gridspec_kw={'width_ratios': [4, 1]})
        
        # Plot edges with crime-based coloring
        if len(edges_in_area) > 0:
            # Plot edges with no crimes (safe streets) in light gray
            safe_edges = edges_in_area[edges_in_area['crime_count'] == 0]
            if len(safe_edges) > 0:
                safe_edges.plot(
                    ax=ax,
                    color='lightgray',
                    linewidth=2.0,
                    alpha=0.6,
                    zorder=1,
                    label=f'Safe streets ({len(safe_edges)})'
                )
            
            # Plot edges with crimes in gradient - fix color mapping
            risky_edges = edges_in_area[edges_in_area['crime_count'] > 0]
            if len(risky_edges) > 0:
                # Use log scale for better color differentiation when there's high variance
                max_crimes = risky_edges['crime_count'].max()
                min_crimes = risky_edges['crime_count'].min()
                
                print(f"   🎨 Crime color mapping: {min_crimes} to {max_crimes} incidents")
                
                # Use a more dramatic colormap with better scaling
                risky_edges.plot(
                    ax=ax,
                    column='crime_count',
                    cmap='Reds',  # Better colormap for crime visualization
                    linewidth=3.5,
                    alpha=0.9,
                    zorder=2,
                    legend=True,
                    legend_kwds={
                        'shrink': 0.8,
                        'label': f'Crime incidents per street\n(Range: {min_crimes}-{max_crimes})',
                        'orientation': 'horizontal',
                        'pad': 0.02
                    },
                    vmin=min_crimes,  # Ensure full color range is used
                    vmax=max_crimes
                )
        
        # Plot intersection nodes
        if len(nodes_in_area) > 0:
            nodes_in_area.plot(
                ax=ax,
                color='navy',
                markersize=30,  # Larger nodes for visibility
                alpha=0.8,
                zorder=4,
                label=f'Intersections ({len(nodes_in_area)})',
                edgecolor='white',
                linewidth=1.5
            )
        
        # Plot crime incident locations with severity-based coloring
        if crime_points and len(crime_scores) > 0:
            crime_x = [p.x for p in crime_points]
            crime_y = [p.y for p in crime_points]
            
            # Use log scale for better color distribution
            log_scores = np.log1p(crime_scores)  # log(1 + score) to handle zeros
            vmin, vmax = log_scores.min(), log_scores.max()
            
            scatter = ax.scatter(
                crime_x, crime_y,
                c=log_scores,
                cmap='Reds',
                s=25,  # Smaller points to avoid overwhelming
                alpha=0.7,
                vmin=vmin,
                vmax=vmax,
                zorder=5,
                label=f'Crime incidents ({len(crime_points)})',
                marker='o',
                edgecolors='darkred',
                linewidths=0.3
            )
            
            # Add colorbar for crime severity
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20, pad=0.02)
            cbar.set_label('Crime Severity (log scale)', rotation=270, labelpad=20, fontsize=10)
        elif crime_points:
            # Fallback if no scores available
            crime_x = [p.x for p in crime_points]
            crime_y = [p.y for p in crime_points]
            ax.scatter(
                crime_x, crime_y,
                c='red',
                s=25,
                alpha=0.6,
                zorder=5,
                label=f'Crime incidents ({len(crime_points)})',
                marker='o',
                edgecolors='darkred',
                linewidths=0.3
            )
        
        # Add the context inset map
        self.create_context_inset(ax, bbox, area_name)
        
        # Formatting
        ax.set_xlim(bbox['west'], bbox['east'])
        ax.set_ylim(bbox['south'], bbox['north'])
        
        ax.set_title(
            f'Crime-Aware Pedestrian Network: {area_name}\n'
            f'{len(nodes_in_area)} intersections, {len(edges_in_area)} street segments, '
            f'{len(crime_points)} crime incidents',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        ax.set_xlabel('UTM Easting (meters)', fontsize=12)
        ax.set_ylabel('UTM Northing (meters)', fontsize=12)
        
        # Add legend positioned at bottom to avoid overlap
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        # Add grid for reference
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Calculate accurate safety statistics (only for edges in this specific area)
        total_edges = len(edges_in_area)
        if total_edges > 0:
            safe_edges_count = len(edges_in_area[edges_in_area['crime_count'] == 0])
            risky_edges_count = len(edges_in_area[edges_in_area['crime_count'] > 0])
            safety_pct = (safe_edges_count / total_edges * 100)
            avg_crimes_per_risky_edge = edges_in_area[edges_in_area['crime_count'] > 0]['crime_count'].mean() if risky_edges_count > 0 else 0
            max_crimes_edge = edges_in_area['crime_count'].max()
            total_crimes_in_area = edges_in_area['crime_count'].sum()
        else:
            safe_edges_count = 0
            risky_edges_count = 0
            safety_pct = 0
            avg_crimes_per_risky_edge = 0
            max_crimes_edge = 0
            total_crimes_in_area = 0
        
        # Create comprehensive statistics panel in the right subplot
        ax_stats.axis('off')
        
        # Safety summary with detailed breakdown
        stats_text = (
            f"AREA SAFETY ANALYSIS\n"
            f"{'='*30}\n\n"
            f"📍 {area_name}\n"
            f"🗺️  Area: {(bbox['east']-bbox['west'])/1000:.1f} × {(bbox['north']-bbox['south'])/1000:.1f} km\n\n"
            
            f"STREET SAFETY:\n"
            f"• Total streets: {total_edges:,}\n"
            f"• Safe streets: {safe_edges_count:,} ({safety_pct:.1f}%)\n"
            f"• At-risk streets: {risky_edges_count:,} ({100-safety_pct:.1f}%)\n\n"
            
            f"CRIME INCIDENTS:\n"
            f"• Total incidents: {len(crime_points):,}\n"
            f"• Network total: {total_crimes_in_area:,}\n"
            f"• Max per street: {max_crimes_edge:,}\n"
            f"• Avg per risky street: {avg_crimes_per_risky_edge:.1f}\n\n"
            
            f"NETWORK STRUCTURE:\n"
            f"• Intersections: {len(nodes_in_area):,}\n"
            f"• Connectivity: High\n"
            f"• Type: Pedestrian\n\n"
            
            f"SAFETY RATING:\n"
        )
        
        # Add safety rating based on percentage
        if safety_pct >= 80:
            rating = "🟢 VERY SAFE"
            rating_desc = "Most streets are safe"
        elif safety_pct >= 60:
            rating = "🟡 MODERATELY SAFE"
            rating_desc = "Mixed safety conditions"
        elif safety_pct >= 40:
            rating = "🟠 CAUTION ADVISED"
            rating_desc = "Many risky streets"
        else:
            rating = "🔴 HIGH RISK"
            rating_desc = "Majority of streets have crimes"
            
        stats_text += f"• {rating}\n• {rating_desc}\n\n"
        
        # Add recommendations
        if risky_edges_count > safe_edges_count:
            recommendation = "⚠️  Use alternative routes\n⚠️  Avoid peak crime times"
        else:
            recommendation = "✅ Generally safe for walking\n✅ Standard precautions advised"
            
        stats_text += f"RECOMMENDATIONS:\n• {recommendation}"
        
        # Display the statistics
        ax_stats.text(0.05, 0.95, stats_text, 
                     transform=ax_stats.transAxes, 
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8, edgecolor='navy'),
                     fontsize=11, fontfamily='monospace')
        
        # Remove axis spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Adjust layout to accommodate inset (skip tight_layout to avoid warning)
        fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
        
        # Save the visualization
        if output_path is None:
            output_path = f"data/graphs/visualizations/crime_network_zoomed_{area_name.lower().replace(' ', '_')}.png"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"   ✅ Saved visualization: {output_path}")
        
        # Close the plot to free memory
        plt.close()
        
        return output_path
    
    def create_context_inset(self, ax_main, bbox, area_name):
        """Create a small inset map showing the zoomed area in context of the full city."""
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # Create inset axes for the context map (positioned in bottom left)
        ax_inset = inset_axes(ax_main, width="25%", height="25%", 
                             bbox_to_anchor=(0.02, 0.02, 1, 1),
                             bbox_transform=ax_main.transAxes, loc='lower left')
        
        # Get full SF bounds
        full_bounds = self.edges_gdf.total_bounds
        sf_bbox = {
            'west': full_bounds[0], 'east': full_bounds[2],
            'south': full_bounds[1], 'north': full_bounds[3]
        }
        
        # Plot simplified full SF network in the inset (sample for performance)
        sample_edges = self.edges_gdf.sample(n=min(5000, len(self.edges_gdf)), random_state=42)
        sample_edges.plot(ax=ax_inset, color='lightgray', linewidth=0.1, alpha=0.5)
        
        # Highlight the zoomed area with a rectangle
        from matplotlib.patches import Rectangle
        zoom_rect = Rectangle(
            (bbox['west'], bbox['south']), 
            bbox['east'] - bbox['west'], 
            bbox['north'] - bbox['south'],
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.3
        )
        ax_inset.add_patch(zoom_rect)
        
        # Format the inset
        ax_inset.set_xlim(sf_bbox['west'], sf_bbox['east'])
        ax_inset.set_ylim(sf_bbox['south'], sf_bbox['north'])
        ax_inset.set_title(f'{area_name} Location', fontsize=8, fontweight='bold')
        ax_inset.tick_params(labelsize=6)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        
        # Add a border around the inset
        for spine in ax_inset.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        
        return ax_inset
    
    def create_multiple_area_views(self):
        """Create visualizations for multiple interesting SF neighborhoods."""
        
        areas = {
            "Mission District": {
                'west': 551000, 'east': 553000,
                'south': 4175000, 'north': 4177000
            },
            "Financial District": {
                'west': 551500, 'east': 553500,
                'south': 4180000, 'north': 4182000
            },
            "Castro District": {
                'west': 550000, 'east': 552000,
                'south': 4176000, 'north': 4178000
            },
            "Chinatown": {
                'west': 551000, 'east': 553000,
                'south': 4180500, 'north': 4182500
            }
        }
        
        output_paths = []
        
        for area_name, bbox in areas.items():
            try:
                output_path = self.create_zoomed_visualization(area_name, bbox)
                output_paths.append(output_path)
            except Exception as e:
                print(f"   ⚠️  Failed to create visualization for {area_name}: {e}")
        
        return output_paths


def main():
    """Main execution function."""
    
    # Define file paths
    graph_path = "data/graphs/sf_pedestrian_graph_projected.graphml"
    nodes_path = "data/graphs/sf_pedestrian_nodes_projected.geojson"
    edges_path = "data/graphs/sf_pedestrian_edges_projected.geojson"
    crime_mapping_path = "scripts/data_optimization/data to map/points_to_graph.csv"
    
    print("🚀 CRIME-AWARE NETWORK VISUALIZATION")
    print("=" * 50)
    
    try:
        # Create visualizer
        visualizer = CrimeNetworkVisualizer(
            graph_path=graph_path,
            nodes_path=nodes_path,
            edges_path=edges_path,
            crime_mapping_path=crime_mapping_path
        )
        
        # Create a single focused visualization (Mission District)
        visualizer.create_zoomed_visualization(
            area_name="Mission District",
            bbox={
                'west': 551000,   # UTM Zone 10N coordinates
                'east': 553000,
                'south': 4175000,
                'north': 4177000
            }
        )
        
        # Create additional area visualizations
        print("\n🏙️ Creating additional neighborhood views...")
        
        # Financial District
        visualizer.create_zoomed_visualization(
            area_name="Financial District", 
            bbox={
                'west': 551500,
                'east': 553500,
                'south': 4180000, 
                'north': 4182000
            }
        )
        
        # Castro District
        visualizer.create_zoomed_visualization(
            area_name="Castro District",
            bbox={
                'west': 550000,
                'east': 552000, 
                'south': 4176000,
                'north': 4178000
            }
        )
        
        print("\n🎯 SUCCESS: Zoomed visualization created!")
        print("   📊 Shows intersection nodes, street segments, and crime locations")
        print("   🎨 Color-coded by crime density for safety assessment")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
