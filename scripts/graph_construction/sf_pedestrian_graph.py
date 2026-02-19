"""
San Francisco Pedestrian Graph Construction Module

Builds a projected pedestrian network graph for SF that serves as the foundation
for crime-risk weighting and multi-objective routing.

This module ONLY constructs the graph structure - no routing or risk modeling.
"""

import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from shapely.geometry import Point
import random
import time
from collections import Counter
from scipy import stats

# Configure OSMnx (OSMnx v1.0+ uses settings instead of config)
try:
    ox.settings.use_cache = True
    ox.settings.log_console = True
    ox.settings.timeout = 300  # Increase timeout to 5 minutes
    ox.settings.overpass_rate_limit = True  # Enable rate limiting
    ox.settings.overpass_wait_duration = 10  # Wait 10 seconds between requests
except AttributeError:
    # Older version compatibility
    pass

class SFPedestrianGraphConstructor:
    """Constructs and validates SF pedestrian network graph."""
    
    def __init__(self, output_dir="data/graphs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Graph storage
        self.G_raw = None           # Original lat/lon graph
        self.G_projected = None     # UTM projected graph
        self.nodes_gdf = None       # Nodes as GeoDataFrame
        self.edges_gdf = None       # Edges as GeoDataFrame
        
        # Validation metrics
        self.metrics = {}
        
    def define_geographic_scope(self):
        """Define San Francisco boundaries for graph retrieval."""
        print("🗺️  Defining geographic scope...")
        
        # Option 1: Use city boundaries
        self.place_name = "San Francisco, California, USA"
        
        # Option 2: Use bounding box (more controlled)
        # These coordinates cover the main SF peninsula
        self.bbox = {
            'north': 37.8324,   # North boundary
            'south': 37.7049,   # South boundary  
            'east': -122.3482,  # East boundary
            'west': -122.5143   # West boundary
        }
        
        print(f"   📍 Target area: {self.place_name}")
        print(f"   📦 Bounding box: {self.bbox}")
        
        return True
    
    def retrieve_pedestrian_network(self, use_bbox=False):
        """Retrieve walkable street network from OpenStreetMap."""
        print("\n🚶 Retrieving pedestrian network from OpenStreetMap...")
        
        # Import time for retry delays
        import time
        
        max_retries = 3
        retry_delay = 30  # Start with 30 seconds
        
        for attempt in range(max_retries):
            try:
                if use_bbox:
                    # Use smaller bounding box to avoid timeout
                    print(f"   📦 Using bounding box method (attempt {attempt + 1}/{max_retries})...")
                    # Smaller area around central SF
                    bbox_small = {
                        'north': 37.8050,   # Reduced area
                        'south': 37.7400,   
                        'east': -122.3800,  
                        'west': -122.4800   
                    }
                    
                    self.G_raw = ox.graph_from_bbox(
                        bbox_small['north'],
                        bbox_small['south'], 
                        bbox_small['east'],
                        bbox_small['west'],
                        network_type='walk',
                        simplify=True
                    )
                else:
                    # Use place name method (often more reliable)
                    print(f"   🏙️ Using place name method (attempt {attempt + 1}/{max_retries})...")
                    self.G_raw = ox.graph_from_place(
                        self.place_name,
                        network_type='walk',
                        simplify=True
                    )
                
                print(f"   ✅ Retrieved graph: {len(self.G_raw.nodes)} nodes, {len(self.G_raw.edges)} edges")
                
                # Verify pedestrian-only content
                self._validate_pedestrian_content()
                
                return True
                
            except Exception as e:
                print(f"   ⚠️ Attempt {attempt + 1} failed: {e}")
                
                if "504" in str(e) or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"   ⏳ Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"   ❌ All retry attempts failed due to server timeouts")
                        print(f"   💡 Suggestion: Try running again later when Overpass API is less busy")
                        return False
                else:
                    print(f"   ❌ Non-timeout error: {e}")
                    return False
        
        return False
    
    def _validate_pedestrian_content(self):
        """Validate that retrieved network contains only walkable paths."""
        print("   🔍 Validating pedestrian-only content...")
        
        # Convert to GeoDataFrames for analysis
        nodes_temp, edges_temp = ox.graph_to_gdfs(self.G_raw)
        
        # Check highway types
        if 'highway' in edges_temp.columns:
            highway_types = edges_temp['highway'].apply(self._normalize_highway_type).unique()
            
            # Pedestrian-friendly types
            pedestrian_types = {
                'footway', 'pedestrian', 'path', 'steps', 'residential', 
                'living_street', 'service', 'tertiary', 'secondary', 'primary',
                'unclassified', 'track', 'cycleway', 'bridleway'
            }
            
            # Check for unwanted types
            unwanted_types = {'motorway', 'trunk', 'motorway_link', 'trunk_link'}
            found_unwanted = set(highway_types) & unwanted_types
            
            if found_unwanted:
                print(f"   ⚠️  WARNING: Found vehicle-only roads: {found_unwanted}")
            else:
                print(f"   ✅ All highway types are pedestrian-compatible")
            
            print(f"   📊 Found {len(highway_types)} different street types")
    
    def _normalize_highway_type(self, highway_val):
        """Normalize highway type values (handle lists/arrays)."""
        if isinstance(highway_val, (list, tuple)):
            return highway_val[0] if highway_val else 'unknown'
        return str(highway_val) if highway_val else 'unknown'
    
    def project_to_metric_system(self):
        """Project graph to UTM coordinate system for metric operations."""
        print("\n📐 Projecting to metric coordinate system...")
        
        try:
            # Project to appropriate UTM zone (SF is in UTM Zone 10N)
            self.G_projected = ox.project_graph(self.G_raw)
            
            # Verify projection
            crs = self.G_projected.graph.get('crs', 'Unknown')
            print(f"   ✅ Projected to CRS: {crs}")
            
            # Convert to GeoDataFrames in projected coordinates
            self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.G_projected)
            
            # Verify edge lengths are in meters
            if 'length' in self.edges_gdf.columns:
                sample_length = self.edges_gdf['length'].iloc[0]
                print(f"   📏 Sample edge length: {sample_length:.2f} meters")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error during projection: {e}")
            return False
    
    def validate_graph_structure(self):
        """Validate and compute structural metrics."""
        print("\n🔍 Validating graph structure...")
        
        G = self.G_projected
        
        # Basic metrics
        self.metrics = {
            'num_nodes': len(G.nodes),
            'num_edges': len(G.edges),
            'is_directed': G.is_directed(),
            'is_connected': nx.is_connected(G.to_undirected()),
        }
        
        # Connectivity analysis
        if not self.metrics['is_connected']:
            components = list(nx.connected_components(G.to_undirected()))
            largest_component = max(components, key=len)
            
            self.metrics['num_components'] = len(components)
            self.metrics['largest_component_size'] = len(largest_component)
            self.metrics['largest_component_pct'] = len(largest_component) / len(G.nodes) * 100
            
            print(f"   📊 Network has {len(components)} components")
            print(f"   📊 Largest component: {len(largest_component)} nodes ({self.metrics['largest_component_pct']:.1f}%)")
            
            # Extract largest component if needed
            if self.metrics['largest_component_pct'] > 90:
                print("   ✅ Largest component dominates - keeping full graph")
            else:
                print("   🔧 Extracting largest connected component...")
                self.G_projected = G.subgraph(largest_component).copy()
                self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.G_projected)
                self.metrics['extracted_largest'] = True
        else:
            print("   ✅ Graph is fully connected")
            self.metrics['extracted_largest'] = False
        
        # Edge length statistics
        if 'length' in self.edges_gdf.columns:
            lengths = self.edges_gdf['length']
            self.metrics.update({
                'total_length_km': lengths.sum() / 1000,
                'avg_edge_length_m': lengths.mean(),
                'median_edge_length_m': lengths.median(),
                'min_edge_length_m': lengths.min(),
                'max_edge_length_m': lengths.max(),
                'std_edge_length_m': lengths.std()
            })
        
        # Network density and degree statistics
        degrees = dict(self.G_projected.degree())
        degree_values = list(degrees.values())
        
        self.metrics.update({
            'network_density': nx.density(self.G_projected),
            'avg_degree': np.mean(degree_values),
            'max_degree': max(degree_values),
            'min_degree': min(degree_values)
        })
        
        # Print summary
        print(f"   📊 Final graph: {self.metrics['num_nodes']} nodes, {self.metrics['num_edges']} edges")
        print(f"   📊 Total length: {self.metrics.get('total_length_km', 0):.1f} km")
        print(f"   📊 Average edge: {self.metrics.get('avg_edge_length_m', 0):.1f} meters")
        print(f"   📊 Network density: {self.metrics['network_density']:.6f}")
        
        return True
    
    def save_graph_data(self):
        """Save graph and validation data to files."""
        print("\n💾 Saving graph data...")
        
        try:
            # Save NetworkX graph (GraphML format)
            graph_path = self.output_dir / "sf_pedestrian_graph_projected.graphml"
            ox.save_graphml(self.G_projected, filepath=graph_path)
            print(f"   ✅ Saved graph: {graph_path}")
            
            # Save GeoDataFrames (GeoJSON format)
            nodes_path = self.output_dir / "sf_pedestrian_nodes_projected.geojson"
            edges_path = self.output_dir / "sf_pedestrian_edges_projected.geojson"
            
            self.nodes_gdf.to_file(nodes_path, driver='GeoJSON')
            self.edges_gdf.to_file(edges_path, driver='GeoJSON')
            
            print(f"   ✅ Saved nodes: {nodes_path}")
            print(f"   ✅ Saved edges: {edges_path}")
            
            # Save validation metrics
            metrics_path = self.output_dir / "validation_metrics.json"
            with open(metrics_path, 'w') as f:
                # Convert numpy types for JSON serialization
                metrics_json = {}
                for k, v in self.metrics.items():
                    if isinstance(v, (np.integer, np.floating)):
                        metrics_json[k] = float(v)
                    else:
                        metrics_json[k] = v
                json.dump(metrics_json, f, indent=2)
            
            print(f"   ✅ Saved metrics: {metrics_path}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error saving data: {e}")
            return False
    
    def create_static_visualizations(self):
        """Create static validation visualizations."""
        print("\n📊 Creating static validation visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set figure parameters
        plt.style.use('default')
        
        # Visualization 1: Full Network Overview
        self._create_network_overview(viz_dir)
        
        # Visualization 2: Edge Length Distribution Map  
        self._create_edge_length_map(viz_dir)
        
        # Visualization 2b: Alternative Edge Length Views (SKIPPED - performance issues)
        print("   ⏭️  Skipping alternative edge length views (performance optimization)")
        
        # Visualization 3: Example Shortest Path
        self._create_shortest_path_example(viz_dir)
        
        # Analytical Visualizations
        self._create_analytical_visualizations(viz_dir)
        
        print(f"   ✅ All visualizations saved to: {viz_dir}")
    
    def _create_network_overview(self, viz_dir):
        """Visualization 1: Complete network overview."""
        print("   📊 Creating network overview...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='white')
        
        # Plot edges (thin gray lines) - bottom layer
        self.edges_gdf.plot(
            ax=ax,
            color='gray',
            linewidth=0.5,
            alpha=0.7,
            zorder=1
        )
        
        # Plot nodes (small points) - top layer for visibility
        self.nodes_gdf.plot(
            ax=ax,
            color='red',
            markersize=1.2,  # Slightly larger for better visibility
            alpha=0.9,       # More opaque for better visibility
            zorder=2         # On top of edges
        )
        
        # Formatting
        consolidation_info = ""
        if self.metrics.get('num_nodes_before_consolidation', 0) > 0:
            original_nodes = self.metrics['num_nodes_before_consolidation']
            reduction_pct = (original_nodes - self.metrics['num_nodes']) / original_nodes * 100
            consolidation_info = f" (consolidated from {original_nodes:,}, {reduction_pct:.1f}% reduction)"
        
        ax.set_title(
            'San Francisco Pedestrian Network (Consolidated Intersections)\n'
            f'{self.metrics["num_nodes"]:,} nodes{consolidation_info}, {self.metrics["num_edges"]:,} edges, '
            f'{self.metrics.get("total_length_km", 0):.1f} km total',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('UTM Easting (meters)', fontsize=12)
        ax.set_ylabel('UTM Northing (meters)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Remove axis spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        # Save
        output_path = viz_dir / "01_network_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Saved: {output_path}")
    
    def _create_edge_length_map(self, viz_dir):
        """Visualization 2: Edge lengths color-coded."""
        print("   📊 Creating edge length distribution map...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='white')
        
        # Handle outliers for better visualization
        lengths = self.edges_gdf['length']
        
        # Use percentiles to handle outliers
        p95 = lengths.quantile(0.95)
        p5 = lengths.quantile(0.05)
        
        # Cap extreme values for better color distribution
        lengths_capped = lengths.clip(upper=p95)
        
        # Create a copy for plotting with capped lengths
        edges_plot = self.edges_gdf.copy()
        edges_plot['length_viz'] = lengths_capped
        
        # Color edges by capped length with better colormap
        edges_plot.plot(
            ax=ax,
            column='length_viz',
            cmap='plasma',  # Better colormap than viridis for this data
            linewidth=1.0,
            alpha=0.8,
            legend=True,
            legend_kwds={
                'shrink': 0.8, 
                'label': f'Edge Length (meters)\n(Capped at 95th percentile: {p95:.0f}m)',
                'orientation': 'vertical'
            },
            zorder=1  # Bottom layer
        )
        
        # Plot nodes (smaller, subtle) - on top for visibility
        self.nodes_gdf.plot(
            ax=ax,
            color='red',
            markersize=0.5,  # Slightly larger for better visibility
            alpha=0.8,       # More opaque
            zorder=2         # On top of edges
        )
        
        # Add statistics text box
        stats_text = (
            f"Length Statistics:\n"
            f"Mean: {lengths.mean():.1f}m\n"
            f"Median: {lengths.median():.1f}m\n"
            f"95th percentile: {p95:.1f}m\n"
            f"Max: {lengths.max():.1f}m\n"
            f"Edges > 200m: {(lengths > 200).sum():,}"
        )
        
        # Add text box with statistics
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        # Formatting
        ax.set_title(
            'Edge Length Distribution (Consolidated Network)\n'
            f'Showing {len(edges_plot):,} street segments with improved color scale',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('UTM Easting (meters)', fontsize=12)
        ax.set_ylabel('UTM Northing (meters)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Remove axis spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        # Save
        output_path = viz_dir / "02_edge_length_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Saved: {output_path}")
        print(f"      📊 Used 'plasma' colormap with 95th percentile capping ({p95:.0f}m)")
    
    def _create_alternative_edge_views(self, viz_dir):
        """Create alternative edge length visualizations for better analysis."""
        print("   📊 Creating alternative edge length views...")
        
        lengths = self.edges_gdf['length']
        
        # Create categorical bins for discrete visualization
        def categorize_length(length):
            if length < 20:
                return 'Very Short (<20m)'
            elif length < 50:
                return 'Short (20-50m)'
            elif length < 100:
                return 'Medium (50-100m)'
            elif length < 200:
                return 'Long (100-200m)'
            else:
                return 'Very Long (>200m)'
        
        edges_categorical = self.edges_gdf.copy()
        edges_categorical['length_category'] = lengths.apply(categorize_length)
        
        # Create subplot with multiple views
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')
        fig.suptitle('Edge Length Analysis - Multiple Views', fontsize=16, fontweight='bold')
        
        # View 1: Categorical coloring
        ax1 = axes[0, 0]
        category_colors = {
            'Very Short (<20m)': 'blue',
            'Short (20-50m)': 'green', 
            'Medium (50-100m)': 'orange',
            'Long (100-200m)': 'red',
            'Very Long (>200m)': 'purple'
        }
        
        for category, color in category_colors.items():
            subset = edges_categorical[edges_categorical['length_category'] == category]
            if len(subset) > 0:
                subset.plot(ax=ax1, color=color, linewidth=0.8, alpha=0.7, label=category)
        
        ax1.set_title('Categorical Edge Lengths', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # View 2: Log scale coloring
        ax2 = axes[0, 1]
        edges_log = self.edges_gdf.copy()
        edges_log['length_log'] = np.log10(lengths.clip(lower=1))  # Avoid log(0)
        
        edges_log.plot(
            ax=ax2,
            column='length_log',
            cmap='coolwarm',
            linewidth=0.8,
            alpha=0.7,
            legend=True,
            legend_kwds={'shrink': 0.6, 'label': 'Log10(Length)'}
        )
        
        ax2.set_title('Log-Scale Edge Lengths', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # View 3: Percentile-based coloring
        ax3 = axes[1, 0]
        def percentile_category(length):
            if length <= lengths.quantile(0.25):
                return '0-25th percentile'
            elif length <= lengths.quantile(0.50):
                return '25-50th percentile'
            elif length <= lengths.quantile(0.75):
                return '50-75th percentile'
            elif length <= lengths.quantile(0.90):
                return '75-90th percentile'
            else:
                return '90-100th percentile'
        
        edges_pct = self.edges_gdf.copy()
        edges_pct['percentile_category'] = lengths.apply(percentile_category)
        
        pct_colors = {
            '0-25th percentile': '#440154',    # Dark purple
            '25-50th percentile': '#3b528b',   # Dark blue
            '50-75th percentile': '#21908c',   # Teal
            '75-90th percentile': '#5dc863',   # Green
            '90-100th percentile': '#fde725'   # Yellow
        }
        
        for category, color in pct_colors.items():
            subset = edges_pct[edges_pct['percentile_category'] == category]
            if len(subset) > 0:
                subset.plot(ax=ax3, color=color, linewidth=0.8, alpha=0.7, label=category)
        
        ax3.set_title('Percentile-Based Edge Lengths', fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # View 4: Outliers highlighted
        ax4 = axes[1, 1]
        p90 = lengths.quantile(0.90)
        p99 = lengths.quantile(0.99)
        
        # Base network in gray
        self.edges_gdf.plot(ax=ax4, color='lightgray', linewidth=0.3, alpha=0.5)
        
        # Highlight different ranges
        long_edges = self.edges_gdf[lengths > p90]
        very_long_edges = self.edges_gdf[lengths > p99]
        
        if len(long_edges) > 0:
            long_edges.plot(ax=ax4, color='orange', linewidth=1.5, alpha=0.8, label=f'90th+ percentile (>{p90:.0f}m)')
        
        if len(very_long_edges) > 0:
            very_long_edges.plot(ax=ax4, color='red', linewidth=2.0, alpha=0.9, label=f'99th+ percentile (>{p99:.0f}m)')
        
        ax4.set_title('Outlier Edge Lengths Highlighted', fontweight='bold')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Format all subplots
        for ax in axes.flat:
            ax.set_xlabel('UTM Easting (m)', fontsize=10)
            ax.set_ylabel('UTM Northing (m)', fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        
        # Save
        output_path = viz_dir / "02b_edge_length_alternatives.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Saved: {output_path}")
        
        # Create length distribution histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Regular histogram
        ax1.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Edge Length (meters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Edge Length Distribution (Linear Scale)')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean: {lengths.mean():.1f}m')
        ax1.axvline(lengths.median(), color='orange', linestyle='--', label=f'Median: {lengths.median():.1f}m')
        ax1.legend()
        
        # Log-scale histogram
        ax2.hist(lengths, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Edge Length (meters)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Edge Length Distribution (Log Scale)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean: {lengths.mean():.1f}m')
        ax2.axvline(lengths.median(), color='orange', linestyle='--', label=f'Median: {lengths.median():.1f}m')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save histogram
        hist_path = viz_dir / "02c_edge_length_histogram.png"
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Saved: {hist_path}")
    
    def _create_shortest_path_example(self, viz_dir):
        """Visualization 3: Example shortest path routing."""
        print("   📊 Creating shortest path example...")
        
        # Select random origin and destination
        nodes = list(self.G_projected.nodes())
        origin = random.choice(nodes)
        destination = random.choice(nodes)
        
        # Ensure different nodes
        attempts = 0
        while origin == destination and attempts < 10:
            destination = random.choice(nodes)
            attempts += 1
        
        try:
            # Calculate shortest path
            shortest_path = nx.shortest_path(
                self.G_projected, 
                origin, 
                destination, 
                weight='length'
            )
            
            # Calculate path metrics
            path_length = nx.shortest_path_length(
                self.G_projected,
                origin,
                destination, 
                weight='length'
            )
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 12), facecolor='white')
            
            # Plot background network (faded) - bottom layer
            self.edges_gdf.plot(
                ax=ax,
                color='lightgray',
                linewidth=0.3,
                alpha=0.5,
                zorder=1
            )
            
            # Plot path edges
            path_edges = []
            for i in range(len(shortest_path) - 1):
                u, v = shortest_path[i], shortest_path[i + 1]
                # Find edge in GeoDataFrame
                edge_mask = (self.edges_gdf['u'] == u) & (self.edges_gdf['v'] == v)
                if not edge_mask.any():
                    # Try reverse direction for undirected representation
                    edge_mask = (self.edges_gdf['u'] == v) & (self.edges_gdf['v'] == u)
                
                if edge_mask.any():
                    path_edges.append(self.edges_gdf[edge_mask].iloc[0])
            
            # Plot path - middle layer
            if path_edges:
                path_gdf = gpd.GeoDataFrame(path_edges)
                path_gdf.plot(
                    ax=ax,
                    color='red',
                    linewidth=3,
                    alpha=0.9,
                    zorder=2
                )
            
            # Plot origin and destination - top layer for maximum visibility
            origin_point = Point(self.G_projected.nodes[origin]['x'], self.G_projected.nodes[origin]['y'])
            dest_point = Point(self.G_projected.nodes[destination]['x'], self.G_projected.nodes[destination]['y'])
            
            # Origin (green) - larger and with border for visibility
            ax.scatter(origin_point.x, origin_point.y, c='green', s=150, zorder=10, 
                      label='Origin', edgecolors='darkgreen', linewidths=2)
            # Destination (blue) - larger and with border for visibility
            ax.scatter(dest_point.x, dest_point.y, c='blue', s=150, zorder=10, 
                      label='Destination', edgecolors='darkblue', linewidths=2)
            
            # Formatting
            ax.set_title(
                'Example Shortest Path Routing\n'
                f'Distance: {path_length:.0f}m, Nodes traversed: {len(shortest_path)}',
                fontsize=14,
                fontweight='bold'
            )
            ax.set_xlabel('UTM Easting (meters)', fontsize=12)
            ax.set_ylabel('UTM Northing (meters)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Remove axis spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            plt.tight_layout()
            
            # Save
            output_path = viz_dir / "03_shortest_path_example.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"      ✅ Saved: {output_path}")
            print(f"      📊 Path: {path_length:.0f}m via {len(shortest_path)} nodes")
            
        except Exception as e:
            print(f"      ⚠️  Could not create shortest path example: {e}")
    
    def _create_analytical_visualizations(self, viz_dir):
        """Create comprehensive analytical visualizations for professional presentation."""
        print("   📊 Creating analytical visualizations...")
        
        # 1️⃣ Edge Length Distribution Histogram
        self._create_edge_length_histogram(viz_dir)
        
        # 2️⃣ Degree Distribution Plot
        self._create_degree_distribution_plot(viz_dir)
        
        # 3️⃣ Network Summary Dashboard
        self._create_network_summary_dashboard(viz_dir)
        
        # 4️⃣ Connected Component Analysis
        self._create_component_analysis(viz_dir)
        
        # 5️⃣ Shortest Path Runtime Analysis
        self._create_runtime_analysis(viz_dir)
        
        # 6️⃣ Spatial Density Analysis
        self._create_spatial_density_analysis(viz_dir)
        
        print("   ✅ All analytical visualizations complete!")
    
    def _create_edge_length_histogram(self, viz_dir):
        """1️⃣ Professional edge length distribution analysis."""
        print("      📈 Creating edge length distribution histogram...")
        
        lengths = self.edges_gdf['length']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Edge Length Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Linear scale histogram
        n, bins, patches = ax1.hist(lengths, bins=50, alpha=0.7, color='skyblue', 
                                   edgecolor='black', density=True)
        
        # Overlay normal distribution
        mu, sigma = lengths.mean(), lengths.std()
        x = np.linspace(lengths.min(), lengths.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')
        
        # Statistical annotations
        ax1.axvline(mu, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mu:.1f}m')
        ax1.axvline(lengths.median(), color='orange', linestyle='--', alpha=0.8, 
                   label=f'Median: {lengths.median():.1f}m')
        ax1.axvline(mu + sigma, color='purple', linestyle=':', alpha=0.6, 
                   label=f'μ+σ: {mu+sigma:.1f}m')
        ax1.axvline(mu - sigma, color='purple', linestyle=':', alpha=0.6, 
                   label=f'μ-σ: {mu-sigma:.1f}m')
        
        ax1.set_xlabel('Edge Length (meters)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Linear Scale Distribution', fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Log scale histogram for outliers
        ax2.hist(lengths, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Edge Length (meters)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Log Scale (Outlier Analysis)', fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add summary statistics box
        stats_text = (
            f"Statistical Summary:\n"
            f"Count: {len(lengths):,} edges\n"
            f"Mean: {mu:.2f}m\n"
            f"Median: {lengths.median():.2f}m\n"
            f"Std Dev: {sigma:.2f}m\n"
            f"Min: {lengths.min():.2f}m\n"
            f"Max: {lengths.max():.2f}m\n"
            f"95th %ile: {lengths.quantile(0.95):.2f}m\n"
            f"Skewness: {stats.skew(lengths):.2f}\n"
            f"Kurtosis: {stats.kurtosis(lengths):.2f}"
        )
        
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9, fontfamily='monospace')
        
        plt.tight_layout()
        output_path = viz_dir / "03_edge_length_histogram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"         ✅ Saved: {output_path}")
    
    def _create_degree_distribution_plot(self, viz_dir):
        """2️⃣ Node degree distribution analysis."""
        print("      📊 Creating degree distribution analysis...")
        
        degrees = dict(self.G_projected.degree())
        degree_values = list(degrees.values())
        degree_counts = Counter(degree_values)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Network Degree Analysis (Consolidated Intersections)', fontsize=16, fontweight='bold')
        
        # Degree histogram
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]
        
        ax1.bar(degrees_sorted, counts, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Node Degree', fontsize=12)
        ax1.set_ylabel('Number of Nodes', fontsize=12)
        ax1.set_title('Degree Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Annotate intersection types
        for i, (deg, count) in enumerate(zip(degrees_sorted, counts)):
            if deg <= 6:  # Only annotate common degrees
                intersection_type = {
                    1: "Dead End", 2: "Continuation", 3: "T-Junction", 
                    4: "4-Way Cross", 5: "5-Way", 6: "Complex"
                }.get(deg, f"{deg}-Way")
                ax1.text(deg, count + max(counts)*0.01, f'{intersection_type}\n({count:,})', 
                        ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Cumulative degree distribution
        total_nodes = sum(counts)
        cumulative_pct = []
        running_sum = 0
        for count in counts:
            running_sum += count
            cumulative_pct.append(running_sum / total_nodes * 100)
        
        ax2.plot(degrees_sorted, cumulative_pct, 'o-', color='darkorange', linewidth=2)
        ax2.set_xlabel('Node Degree', fontsize=12)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax2.set_title('Cumulative Degree Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Degree vs Network Position (sample)
        if len(self.nodes_gdf) > 1000:
            sample_nodes = self.nodes_gdf.sample(1000, random_state=42)
        else:
            sample_nodes = self.nodes_gdf
            
        # Get degrees only for nodes that exist in both the graph and sample
        valid_node_ids = set(degrees.keys()) & set(sample_nodes.index)
        sample_nodes_valid = sample_nodes.loc[list(valid_node_ids)]
        sample_degrees = [degrees[idx] for idx in sample_nodes_valid.index]
        
        # Use geometry coordinates
        x_coords = [geom.x for geom in sample_nodes_valid.geometry]
        y_coords = [geom.y for geom in sample_nodes_valid.geometry]
        
        scatter = ax3.scatter(x_coords, y_coords, c=sample_degrees, 
                             cmap='viridis', alpha=0.8, s=35, zorder=2,
                             edgecolors='black', linewidths=0.3)
        
        plt.colorbar(scatter, ax=ax3, label='Node Degree')
        ax3.set_xlabel('UTM Easting (m)', fontsize=12)
        ax3.set_ylabel('UTM Northing (m)', fontsize=12)
        ax3.set_title('Spatial Distribution of Node Degrees', fontweight='bold')
        
        # Network topology summary
        ax4.axis('off')
        
        avg_degree = np.mean(degree_values)
        max_degree_node = max(degrees.items(), key=lambda x: x[1])
        
        topology_text = (
            f"Network Topology Analysis\n"
            f"{'='*30}\n\n"
            f"Total Intersections: {len(degrees):,}\n"
            f"Average Degree: {avg_degree:.2f}\n"
            f"Max Degree: {max(degree_values)} (Node {max_degree_node[0]})\n"
            f"Min Degree: {min(degree_values)}\n\n"
            f"Intersection Types:\n"
            f"• Dead Ends (deg=1): {degree_counts.get(1, 0):,} ({degree_counts.get(1, 0)/len(degrees)*100:.1f}%)\n"
            f"• Continuations (deg=2): {degree_counts.get(2, 0):,} ({degree_counts.get(2, 0)/len(degrees)*100:.1f}%)\n"
            f"• T-Junctions (deg=3): {degree_counts.get(3, 0):,} ({degree_counts.get(3, 0)/len(degrees)*100:.1f}%)\n"
            f"• 4-Way Crosses (deg=4): {degree_counts.get(4, 0):,} ({degree_counts.get(4, 0)/len(degrees)*100:.1f}%)\n"
            f"• Complex (deg≥5): {sum(v for k, v in degree_counts.items() if k >= 5):,}\n\n"
            f"Graph Properties:\n"
            f"• Density: {nx.density(self.G_projected):.6f}\n"
            f"• Is Connected: {'Yes' if nx.is_connected(self.G_projected.to_undirected()) else 'No'}\n"
            f"• Network Type: {'Directed' if self.G_projected.is_directed() else 'Undirected'}"
        )
        
        ax4.text(0.05, 0.95, topology_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
        
        plt.tight_layout()
        output_path = viz_dir / "04_degree_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"         ✅ Saved: {output_path}")
    
    def _create_network_summary_dashboard(self, viz_dir):
        """3️⃣ Professional network summary dashboard."""
        print("      📉 Creating network summary dashboard...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('San Francisco Pedestrian Network Dashboard (Consolidated)', fontsize=18, fontweight='bold')
        
        # Bar chart of key metrics
        metrics = ['Nodes', 'Edges', 'Length (km)', 'Avg Degree', 'Density×10⁶']
        values = [
            self.metrics['num_nodes'],
            self.metrics['num_edges'],
            self.metrics['total_length_km'],
            self.metrics['avg_degree'],
            self.metrics['network_density'] * 1e6  # Scale density for visibility
        ]
        
        bars = ax1.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_ylabel('Count / Value', fontsize=12)
        ax1.set_title('Key Network Metrics', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if value > 1000:
                label = f'{value:,.0f}'
            else:
                label = f'{value:.2f}'
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        # Edge length distribution pie chart
        lengths = self.edges_gdf['length']
        length_categories = {
            'Very Short\n(<20m)': len(lengths[lengths < 20]),
            'Short\n(20-50m)': len(lengths[(lengths >= 20) & (lengths < 50)]),
            'Medium\n(50-100m)': len(lengths[(lengths >= 50) & (lengths < 100)]),
            'Long\n(100-200m)': len(lengths[(lengths >= 100) & (lengths < 200)]),
            'Very Long\n(≥200m)': len(lengths[lengths >= 200])
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        wedges, texts, autotexts = ax2.pie(length_categories.values(), 
                                          labels=length_categories.keys(),
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        ax2.set_title('Edge Length Distribution', fontweight='bold')
        
        # Network growth simulation (conceptual)
        cumulative_length = np.cumsum(sorted(lengths))
        percentiles = np.linspace(0, 100, len(cumulative_length))
        
        ax3.plot(percentiles, cumulative_length / 1000, color='darkblue', linewidth=2)
        ax3.set_xlabel('Percentage of Edges (%)', fontsize=12)
        ax3.set_ylabel('Cumulative Length (km)', fontsize=12)
        ax3.set_title('Network Length Accumulation', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Fill area under curve
        ax3.fill_between(percentiles, cumulative_length / 1000, alpha=0.3, color='lightblue')
        
        # Summary statistics table
        ax4.axis('off')
        
        # Create table data
        original_nodes = self.metrics.get('num_nodes_before_consolidation', self.metrics['num_nodes'])
        consolidation_pct = self.metrics.get('consolidation_reduction_pct', 0)
        
        table_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Total Nodes', f'{self.metrics["num_nodes"]:,}', f'Intersections (was {original_nodes:,})'],
            ['Total Edges', f'{self.metrics["num_edges"]:,}', 'Street segments'],
            ['Consolidation', f'{consolidation_pct:.1f}%', 'Node reduction achieved'],
            ['Network Length', f'{self.metrics["total_length_km"]:.1f} km', 'Total walkable distance'],
            ['Average Edge', f'{self.metrics["avg_edge_length_m"]:.1f} m', 'Typical block length'],
            ['Connectivity', f'{self.metrics["avg_degree"]:.1f}', 'Streets per intersection'],
            ['Min Edge Length', f'{self.metrics["min_edge_length_m"]:.1f} m', 'Shortest segment'],
            ['Max Edge Length', f'{self.metrics["max_edge_length_m"]:.1f} m', 'Longest segment'],
            ['Network Density', f'{self.metrics["network_density"]:.2e}', 'Connection ratio']
        ]
        
        # Create table
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='left', loc='center', colWidths=[0.3, 0.3, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(table_data)):
            for j in range(3):
                cell = table[i, j]
                if i == 0:  # Header
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
        
        plt.tight_layout()
        output_path = viz_dir / "05_network_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"         ✅ Saved: {output_path}")
    
    def _create_component_analysis(self, viz_dir):
        """4️⃣ Connected component analysis."""
        print("      📍 Creating connected component analysis...")
        
        # Analyze connected components
        G_undirected = self.G_projected.to_undirected()
        components = list(nx.connected_components(G_undirected))
        component_sizes = [len(comp) for comp in components]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Connected Component Analysis', fontsize=16, fontweight='bold')
        
        if len(components) > 1:
            # Component size distribution
            ax1.hist(component_sizes, bins=min(20, len(components)), 
                    alpha=0.7, color='lightcoral', edgecolor='black')
            ax1.set_xlabel('Component Size (nodes)', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title(f'Component Size Distribution ({len(components)} components)', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Largest components bar chart
            largest_components = sorted(component_sizes, reverse=True)[:10]
            ax2.bar(range(1, len(largest_components)+1), largest_components, 
                   color='steelblue', alpha=0.7)
            ax2.set_xlabel('Component Rank', fontsize=12)
            ax2.set_ylabel('Size (nodes)', fontsize=12)
            ax2.set_title('Top 10 Largest Components', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
        else:
            # Single component analysis
            ax1.text(0.5, 0.5, 'Network is Fully Connected\n(Single Component)', 
                    transform=ax1.transAxes, ha='center', va='center',
                    fontsize=16, fontweight='bold', color='darkgreen',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
            ax1.set_title('Component Connectivity Status', fontweight='bold')
            
            # Show network efficiency metrics instead
            largest_component = components[0]
            subgraph = G_undirected.subgraph(largest_component)
            
            # Sample for efficiency calculation (large graphs are slow)
            if len(largest_component) > 1000:
                sample_nodes = random.sample(list(largest_component), 1000)
                sample_subgraph = subgraph.subgraph(sample_nodes)
            else:
                sample_subgraph = subgraph
            
            try:
                # Calculate some basic metrics for the sample
                avg_clustering = nx.average_clustering(sample_subgraph)
                if len(sample_subgraph) > 100:
                    # Sample path lengths for large graphs
                    sample_pairs = random.sample(list(sample_subgraph.nodes()), 
                                                min(100, len(sample_subgraph)))
                    path_lengths = []
                    for i in range(0, len(sample_pairs)-1, 2):
                        try:
                            path_len = nx.shortest_path_length(sample_subgraph, 
                                                             sample_pairs[i], sample_pairs[i+1])
                            path_lengths.append(path_len)
                        except nx.NetworkXNoPath:
                            continue
                    
                    if path_lengths:
                        avg_path_length = np.mean(path_lengths)
                    else:
                        avg_path_length = 0
                else:
                    avg_path_length = nx.average_shortest_path_length(sample_subgraph) if nx.is_connected(sample_subgraph) else 0
                
                efficiency_text = (
                    f"Network Efficiency Metrics\n"
                    f"(Sample of {len(sample_subgraph):,} nodes)\n\n"
                    f"Average Clustering: {avg_clustering:.4f}\n"
                    f"Average Path Length: {avg_path_length:.2f}\n"
                    f"Network Diameter: Computing...\n"
                    f"Global Efficiency: {1/avg_path_length if avg_path_length > 0 else 0:.4f}"
                )
                
            except Exception as e:
                efficiency_text = f"Efficiency calculation in progress...\n(Large network: {len(largest_component):,} nodes)"
            
            ax2.text(0.1, 0.9, efficiency_text, transform=ax2.transAxes,
                    verticalalignment='top', fontsize=12, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax2.set_title('Network Efficiency Analysis', fontweight='bold')
        
        # Component coverage visualization
        total_nodes = sum(component_sizes)
        largest_component_pct = max(component_sizes) / total_nodes * 100 if component_sizes else 0
        
        coverage_data = [largest_component_pct, 100 - largest_component_pct]
        coverage_labels = [f'Largest Component\n({largest_component_pct:.1f}%)', 
                          f'Other Components\n({100-largest_component_pct:.1f}%)']
        
        ax3.pie(coverage_data, labels=coverage_labels, autopct='%1.1f%%',
               colors=['lightgreen', 'lightcoral'], startangle=90)
        ax3.set_title('Network Coverage by Components', fontweight='bold')
        
        # Summary statistics
        ax4.axis('off')
        
        component_stats = (
            f"Component Analysis Summary\n"
            f"{'='*35}\n\n"
            f"Total Components: {len(components):,}\n"
            f"Largest Component: {max(component_sizes):,} nodes\n"
            f"Smallest Component: {min(component_sizes):,} nodes\n"
            f"Average Component Size: {np.mean(component_sizes):.1f}\n"
            f"Median Component Size: {np.median(component_sizes):.1f}\n\n"
            f"Coverage Analysis:\n"
            f"• Largest covers {largest_component_pct:.1f}% of network\n"
            f"• {sum(1 for size in component_sizes if size >= 100):,} components ≥100 nodes\n"
            f"• {sum(1 for size in component_sizes if size == 1):,} isolated nodes\n\n"
            f"Network Status: {'Fully Connected' if len(components) == 1 else 'Fragmented'}\n"
            f"Recommendation: {'Proceed with analysis' if largest_component_pct > 90 else 'Consider component selection'}"
        )
        
        ax4.text(0.05, 0.95, component_stats, transform=ax4.transAxes,
                verticalalignment='top', fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        
        plt.tight_layout()
        output_path = viz_dir / "06_component_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"         ✅ Saved: {output_path}")
    
    def _create_runtime_analysis(self, viz_dir):
        """5️⃣ Shortest path runtime analysis."""
        print("      ⏱ Creating shortest path runtime analysis...")
        
        import time
        
        # Sample random node pairs for runtime testing
        nodes = list(self.G_projected.nodes())
        n_tests = min(50, len(nodes) // 100)  # Scale tests based on network size
        
        if len(nodes) < 100:
            print("         ⚠️ Network too small for meaningful runtime analysis")
            return
            
        runtimes = []
        distances = []
        path_lengths = []
        
        print(f"         🔍 Running {n_tests} pathfinding tests...")
        
        for i in range(n_tests):
            # Select random origin and destination
            origin, destination = random.sample(nodes, 2)
            
            start_time = time.perf_counter()
            try:
                path = nx.shortest_path(self.G_projected, origin, destination, weight='length')
                path_distance = nx.shortest_path_length(self.G_projected, origin, destination, weight='length')
                end_time = time.perf_counter()
                
                runtime = (end_time - start_time) * 1000  # Convert to milliseconds
                runtimes.append(runtime)
                distances.append(path_distance)
                path_lengths.append(len(path))
                
            except nx.NetworkXNoPath:
                continue  # Skip disconnected pairs
        
        if not runtimes:
            print("         ⚠️ No valid paths found for runtime analysis")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Shortest Path Runtime Analysis', fontsize=16, fontweight='bold')
        
        # Runtime distribution
        ax1.hist(runtimes, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_xlabel('Runtime (milliseconds)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Runtime Distribution ({len(runtimes)} tests)', fontweight='bold')
        ax1.axvline(np.mean(runtimes), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(runtimes):.2f}ms')
        ax1.axvline(np.median(runtimes), color='orange', linestyle='--', 
                   label=f'Median: {np.median(runtimes):.2f}ms')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distance distribution
        distances_km = [d/1000 for d in distances]
        ax2.hist(distances_km, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Path Distance (km)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Path Distance Distribution', fontweight='bold')
        ax2.axvline(np.mean(distances_km), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(distances_km):.2f}km')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Runtime vs Distance correlation
        ax3.scatter(distances_km, runtimes, alpha=0.6, color='purple')
        
        # Add trend line
        if len(distances_km) > 5:
            z = np.polyfit(distances_km, runtimes, 1)
            p = np.poly1d(z)
            ax3.plot(sorted(distances_km), p(sorted(distances_km)), "r--", alpha=0.8)
            
            # Calculate correlation
            correlation = np.corrcoef(distances_km, runtimes)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax3.transAxes, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Path Distance (km)', fontsize=12)
        ax3.set_ylabel('Runtime (ms)', fontsize=12)
        ax3.set_title('Runtime vs Distance Correlation', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Performance summary
        ax4.axis('off')
        
        performance_text = (
            f"Pathfinding Performance Summary\n"
            f"{'='*40}\n\n"
            f"Test Configuration:\n"
            f"• Tests run: {len(runtimes):,}\n"
            f"• Network size: {len(nodes):,} nodes\n"
            f"• Algorithm: Dijkstra's shortest path\n\n"
            f"Runtime Statistics:\n"
            f"• Mean: {np.mean(runtimes):.2f} ms\n"
            f"• Median: {np.median(runtimes):.2f} ms\n"
            f"• Std Dev: {np.std(runtimes):.2f} ms\n"
            f"• Min: {min(runtimes):.2f} ms\n"
            f"• Max: {max(runtimes):.2f} ms\n"
            f"• 95th %ile: {np.percentile(runtimes, 95):.2f} ms\n\n"
            f"Distance Statistics:\n"
            f"• Mean path: {np.mean(distances_km):.2f} km\n"
            f"• Mean hops: {np.mean(path_lengths):.1f} nodes\n"
            f"• Max distance: {max(distances_km):.2f} km\n\n"
            f"Performance Rating: {'Excellent' if np.mean(runtimes) < 1 else 'Good' if np.mean(runtimes) < 10 else 'Acceptable'}\n"
            f"Scalability: {'Ready for real-time routing' if np.mean(runtimes) < 50 else 'Consider optimization'}"
        )
        
        ax4.text(0.05, 0.95, performance_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.3))
        
        plt.tight_layout()
        output_path = viz_dir / "07_runtime_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"         ✅ Saved: {output_path}")
    
    def _create_spatial_density_analysis(self, viz_dir):
        """6️⃣ Spatial density heatmap analysis."""
        print("      🧠 Creating spatial density analysis...")
        
        # Create grid for density analysis using GeoDataFrame
        x_coords = self.nodes_gdf.geometry.x.values
        y_coords = self.nodes_gdf.geometry.y.values
        
        # Define grid
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Create grid (adjust resolution based on network size)
        grid_size = min(50, int(np.sqrt(len(self.G_projected.nodes())) / 10))
        x_bins = np.linspace(x_min, x_max, grid_size)
        y_bins = np.linspace(y_min, y_max, grid_size)
        
        # Calculate density
        density_grid, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
        
        # Calculate area of each grid cell (in km²)
        cell_area_km2 = ((x_max - x_min) / grid_size) * ((y_max - y_min) / grid_size) / 1e6
        density_per_km2 = density_grid / cell_area_km2
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spatial Density Analysis', fontsize=16, fontweight='bold')
        
        # Node density heatmap
        im1 = ax1.imshow(density_per_km2.T, origin='lower', cmap='YlOrRd', 
                        extent=[x_min, x_max, y_min, y_max], aspect='auto')
        ax1.set_xlabel('UTM Easting (m)', fontsize=12)
        ax1.set_ylabel('UTM Northing (m)', fontsize=12)
        ax1.set_title('Node Density (nodes/km²)', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Density (nodes/km²)')
        
        # Edge density analysis
        edge_coords = []
        for edge in self.edges_gdf.itertuples():
            if hasattr(edge.geometry, 'coords'):
                # Get midpoint of edge
                coords = list(edge.geometry.coords)
                if len(coords) >= 2:
                    mid_x = (coords[0][0] + coords[-1][0]) / 2
                    mid_y = (coords[0][1] + coords[-1][1]) / 2
                    edge_coords.append((mid_x, mid_y))
        
        if edge_coords:
            edge_x, edge_y = zip(*edge_coords)
            edge_density_grid, _, _ = np.histogram2d(edge_x, edge_y, bins=[x_bins, y_bins])
            edge_density_per_km2 = edge_density_grid / cell_area_km2
            
            im2 = ax2.imshow(edge_density_per_km2.T, origin='lower', cmap='Blues',
                            extent=[x_min, x_max, y_min, y_max], aspect='auto')
            ax2.set_xlabel('UTM Easting (m)', fontsize=12)
            ax2.set_ylabel('UTM Northing (m)', fontsize=12)
            ax2.set_title('Edge Density (edges/km²)', fontweight='bold')
            plt.colorbar(im2, ax=ax2, label='Density (edges/km²)')
        
        # Density distribution histogram
        flat_density = density_per_km2.flatten()
        flat_density = flat_density[flat_density > 0]  # Remove empty cells
        
        ax3.hist(flat_density, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Node Density (nodes/km²)', fontsize=12)
        ax3.set_ylabel('Number of Grid Cells', fontsize=12)
        ax3.set_title('Density Distribution Across City', fontweight='bold')
        ax3.axvline(np.mean(flat_density), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(flat_density):.0f} nodes/km²')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Urban analysis summary
        ax4.axis('off')
        
        # Calculate urban metrics
        high_density_threshold = np.percentile(flat_density, 75)
        low_density_threshold = np.percentile(flat_density, 25)
        
        high_density_cells = np.sum(density_per_km2 > high_density_threshold)
        low_density_cells = np.sum((density_per_km2 > 0) & (density_per_km2 < low_density_threshold))
        total_active_cells = np.sum(density_per_km2 > 0)
        
        urban_text = (
            f"Urban Density Analysis\n"
            f"{'='*30}\n\n"
            f"Grid Configuration:\n"
            f"• Grid resolution: {grid_size}×{grid_size}\n"
            f"• Cell area: {cell_area_km2:.3f} km²\n"
            f"• Total area: {((x_max-x_min)*(y_max-y_min))/1e6:.1f} km²\n\n"
            f"Density Statistics:\n"
            f"• Max density: {np.max(flat_density):.0f} nodes/km²\n"
            f"• Mean density: {np.mean(flat_density):.0f} nodes/km²\n"
            f"• Median density: {np.median(flat_density):.0f} nodes/km²\n"
            f"• Std deviation: {np.std(flat_density):.0f} nodes/km²\n\n"
            f"Urban Classification:\n"
            f"• High density areas: {high_density_cells} cells\n"
            f"  (>{high_density_threshold:.0f} nodes/km²)\n"
            f"• Medium density areas: {total_active_cells - high_density_cells - low_density_cells} cells\n"
            f"• Low density areas: {low_density_cells} cells\n"
            f"  (<{low_density_threshold:.0f} nodes/km²)\n\n"
            f"Network Coverage: {total_active_cells}/{grid_size**2} cells ({total_active_cells/grid_size**2*100:.1f}%)"
        )
        
        ax4.text(0.05, 0.95, urban_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.3))
        
        plt.tight_layout()
        output_path = viz_dir / "08_spatial_density.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"         ✅ Saved: {output_path}")
    
    def consolidate_intersections(self):
        """Consolidate overlapping and nearby nodes into single intersection nodes."""
        print("\n🔗 Consolidating intersection nodes for routing accuracy...")
        
        print(f"   📊 Before consolidation: {len(self.G_projected.nodes)} nodes, {len(self.G_projected.edges)} edges")
        
        try:
            # Step 1: First merge nodes that are at exactly the same coordinates (overlapping nodes)
            print("   🎯 Phase 1: Merging overlapping nodes at identical coordinates...")
            self._merge_overlapping_nodes()
            print(f"   📊 After overlapping merge: {len(self.G_projected.nodes)} nodes, {len(self.G_projected.edges)} edges")
            
            # Step 2: Apply distance-based consolidation for nearby intersections
            tolerance = 15.0  # meters - aggressive merging for real intersections
            print(f"   📏 Phase 2: Distance-based consolidation (tolerance: {tolerance}m)")
            
            self.G_projected = ox.consolidate_intersections(
                self.G_projected,
                tolerance=tolerance,
                rebuild_graph=True,
                dead_ends=False,  # Don't consolidate dead ends
                reconnect_edges=True
            )
            
            print(f"   📊 After distance consolidation: {len(self.G_projected.nodes)} nodes, {len(self.G_projected.edges)} edges")
            
            # Step 3: Simplify the graph to remove unnecessary intermediate nodes
            print("   🧹 Phase 3: Topology simplification...")
            
            try:
                # Simplify the graph to remove degree-2 nodes that are just path continuations
                self.G_projected = ox.simplify_graph(
                    self.G_projected,
                    remove_rings=False  # Keep ring roads
                )
                print(f"   📊 After simplification: {len(self.G_projected.nodes)} nodes, {len(self.G_projected.edges)} edges")
            except Exception as e:
                print(f"   ⚠️  Warning: Simplification failed: {e}")
                print("   ➡️  Continuing with current graph structure")
            
            print(f"   📊 After simplification: {len(self.G_projected.nodes)} nodes, {len(self.G_projected.edges)} edges")
            
            # Update GeoDataFrames with consolidated graph
            self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.G_projected)
            
            # Calculate total reduction
            original_nodes = self.metrics.get('num_nodes_before_consolidation', 0)
            if original_nodes > 0:
                total_reduction_pct = (original_nodes - len(self.G_projected.nodes)) / original_nodes * 100
                print(f"   ✅ MULTI-PHASE CONSOLIDATION COMPLETE:")
                print(f"      📉 Total node reduction: {total_reduction_pct:.1f}%")
                print(f"      📊 {original_nodes:,} → {len(self.G_projected.nodes):,} nodes")
                print(f"      🎯 Overlapping nodes merged + intersection consolidation")
                
                # Step 4: Validate that consolidation was successful
                print("   🔎 Phase 4: Post-consolidation validation...")
                self._validate_no_overlapping_nodes()
                
                # Store the consolidation metrics
                self.metrics['nodes_consolidated'] = original_nodes - len(self.G_projected.nodes)
                self.metrics['consolidation_reduction_pct'] = total_reduction_pct
            
            return True
        except Exception as e:
            print(f"   ❌ Error during consolidation: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def _validate_no_overlapping_nodes(self):
        """Validate that no overlapping nodes remain after consolidation."""
        import math
        
        nodes = list(self.G_projected.nodes(data=True))
        if len(nodes) < 2:
            print("      ✅ Graph has less than 2 nodes, no overlaps possible")
            return
        
        overlap_threshold = 1.0  # Very strict 1 meter threshold for validation
        overlapping_pairs = []
        
        # Build positions
        positions = [(node_id, node_data['x'], node_data['y']) for node_id, node_data in nodes]
        
        # Check all pairs for overlaps
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                node1_id, x1, y1 = positions[i]
                node2_id, x2, y2 = positions[j]
                
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance <= overlap_threshold:
                    overlapping_pairs.append((node1_id, node2_id, distance))
        
        if overlapping_pairs:
            print(f"      ⚠️  WARNING: Found {len(overlapping_pairs)} remaining overlapping node pairs (within {overlap_threshold}m):")
            for node1, node2, dist in overlapping_pairs[:5]:  # Show first 5
                print(f"         • Nodes {node1} ↔ {node2}: {dist:.3f}m apart")
            if len(overlapping_pairs) > 5:
                print(f"         ... and {len(overlapping_pairs) - 5} more pairs")
            return False
        else:
            print(f"      ✅ Validation passed: No overlapping nodes found (within {overlap_threshold}m)")
            return True
    
    def _merge_overlapping_nodes(self):
        """Merge nodes that are at the same location (overlapping/very close nodes on two-way streets)."""
        import math
        
        # Get node coordinates
        nodes = list(self.G_projected.nodes(data=True))
        
        if len(nodes) < 2:
            print("      ℹ️  Less than 2 nodes, skipping overlap check")
            return
        
        # Improved spatial clustering using union-find for connected components
        overlap_threshold = 2.0  # 2 meters - very strict for truly overlapping nodes
        
        # Build node positions dictionary
        node_positions = {}
        node_list = []
        
        for node_id, node_data in nodes:
            x, y = node_data['x'], node_data['y']
            node_positions[node_id] = (x, y)
            node_list.append(node_id)
        
        # Build spatial grid for efficient neighbor finding
        grid_size = overlap_threshold  # Grid cells are the size of overlap threshold
        grid = {}
        
        for node_id in node_list:
            x, y = node_positions[node_id]
            
            # Assign to grid cell
            grid_x = int(x // grid_size)
            grid_y = int(y // grid_size)
            
            if grid_x not in grid:
                grid[grid_x] = {}
            if grid_y not in grid[grid_x]:
                grid[grid_x][grid_y] = []
            
            grid[grid_x][grid_y].append(node_id)
        
        # Find all overlapping pairs using Union-Find
        parent = {node_id: node_id for node_id in node_list}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Check each node against neighboring grid cells
        overlapping_pairs = 0
        for grid_x in grid:
            for grid_y in grid[grid_x]:
                cell_nodes = grid[grid_x][grid_y]
                
                # Check all pairs within this cell and adjacent cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_x, neighbor_y = grid_x + dx, grid_y + dy
                        
                        if neighbor_x in grid and neighbor_y in grid[neighbor_x]:
                            neighbor_nodes = grid[neighbor_x][neighbor_y]
                            
                            # Check distances between all pairs
                            for node1 in cell_nodes:
                                for node2 in neighbor_nodes:
                                    if node1 >= node2:  # Avoid duplicate checks
                                        continue
                                    
                                    x1, y1 = node_positions[node1]
                                    x2, y2 = node_positions[node2]
                                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                    
                                    if distance <= overlap_threshold:
                                        union(node1, node2)
                                        overlapping_pairs += 1
        
        # Group nodes by their root parent to form clusters
        clusters = {}
        for node_id in node_list:
            root = find(node_id)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(node_id)
        
        # Filter to only clusters with multiple nodes
        overlapping_clusters = [cluster for cluster in clusters.values() if len(cluster) > 1]
        
        if not overlapping_clusters:
            print("      ℹ️  No overlapping nodes found (within 2m)")
            return
        
        total_overlapping = sum(len(cluster) for cluster in overlapping_clusters)
        print(f"      🔍 Found {len(overlapping_clusters)} overlap clusters with {total_overlapping} overlapping nodes (within {overlap_threshold}m)")
        print(f"      🔗 Detected {overlapping_pairs} overlapping pairs")
        
        # Merge overlapping nodes
        nodes_to_remove = set()
        
        for cluster_nodes in overlapping_clusters:
            if len(cluster_nodes) <= 1:
                continue
                
            # Keep the first node as the master, merge others into it
            master_node = cluster_nodes[0]
            nodes_to_merge = cluster_nodes[1:]
            
            # Collect all edges from nodes to be merged
            edges_to_redirect = []
            
            for merge_node in nodes_to_merge:
                # Get all edges incident to this node
                for neighbor in list(self.G_projected.neighbors(merge_node)):
                    # Store edge data before removing
                    if self.G_projected.has_edge(merge_node, neighbor):
                        edge_data = self.G_projected.get_edge_data(merge_node, neighbor)
                        edges_to_redirect.append((merge_node, neighbor, edge_data))
                
                # Also get incoming edges for directed graphs
                for predecessor in list(self.G_projected.predecessors(merge_node)):
                    if predecessor != merge_node:  # Avoid self-loops we already handled
                        if self.G_projected.has_edge(predecessor, merge_node):
                            edge_data = self.G_projected.get_edge_data(predecessor, merge_node)
                            edges_to_redirect.append((predecessor, merge_node, edge_data))
                
                # Mark node for removal
                nodes_to_remove.add(merge_node)
            
            # Redirect edges to master node
            for source, target, edge_data in edges_to_redirect:
                if source in nodes_to_merge:
                    # Edge starts from a node being merged
                    new_source = master_node
                    new_target = target
                elif target in nodes_to_merge:
                    # Edge ends at a node being merged
                    new_source = source
                    new_target = master_node
                else:
                    continue  # Skip if neither end needs merging
                
                # Avoid self-loops and duplicate edges
                if (new_source != new_target and 
                    not self.G_projected.has_edge(new_source, new_target)):
                    
                    # Add the redirected edge - handle different edge data formats
                    try:
                        if isinstance(edge_data, dict):
                            # Single edge - filter out non-string keys for add_edge
                            filtered_data = {}
                            for k, v in edge_data.items():
                                if isinstance(k, str):
                                    filtered_data[k] = v
                            self.G_projected.add_edge(new_source, new_target, **filtered_data)
                        else:
                            # MultiGraph with multiple edges
                            for key, data in edge_data.items():
                                filtered_data = {}
                                if isinstance(data, dict):
                                    for k, v in data.items():
                                        if isinstance(k, str):
                                            filtered_data[k] = v
                                self.G_projected.add_edge(new_source, new_target, **filtered_data)
                    except Exception as e:
                        # If edge addition fails, just create a basic edge
                        if not self.G_projected.has_edge(new_source, new_target):
                            self.G_projected.add_edge(new_source, new_target)
        
        # Remove the merged nodes
        self.G_projected.remove_nodes_from(nodes_to_remove)
        
        nodes_merged = len(nodes_to_remove)
        print(f"      ✅ Merged {nodes_merged} overlapping nodes into {len(overlapping_clusters)} intersection points")
    
    def construct_complete_graph(self, use_bbox=False):
        """Execute complete graph construction pipeline."""
        print("🚀 STARTING SAN FRANCISCO PEDESTRIAN GRAPH CONSTRUCTION")
        print("=" * 60)
        
        # Step 1: Define scope
        if not self.define_geographic_scope():
            return False
        
        # Step 2: Retrieve network (try place name first, then bbox if that fails)
        if not self.retrieve_pedestrian_network(use_bbox=use_bbox):
            if not use_bbox:
                print("🔄 Place name method failed, trying bounding box method...")
                if not self.retrieve_pedestrian_network(use_bbox=True):
                    return False
            else:
                return False
        
        # Step 3: Project to metric system
        if not self.project_to_metric_system():
            return False
        
        # Step 3.5: Consolidate intersection nodes
        # Store node count before consolidation for metrics
        self.metrics = {'num_nodes_before_consolidation': len(self.G_projected.nodes)}
        self.consolidate_intersections()
        
        # Step 4: Validate structure
        if not self.validate_graph_structure():
            return False
        
        # Step 5: Save data
        if not self.save_graph_data():
            return False
        
        # Step 6: Create visualizations
        self.create_static_visualizations()
        
        print("\n" + "=" * 60)
        print("✅ PEDESTRIAN GRAPH CONSTRUCTION COMPLETE")
        print(f"📊 Final consolidated graph ready for routing engine:")
        
        # Get consolidation info
        original_nodes = self.metrics.get('num_nodes_before_consolidation', 0)
        reduction_pct = self.metrics.get('consolidation_reduction_pct', 0)
        consolidation_info = f" (consolidated from {original_nodes:,}, {reduction_pct:.1f}% reduction)" if original_nodes > 0 else ""
        
        print(f"   • {self.metrics['num_nodes']:,} intersection nodes{consolidation_info}")
        print(f"   • {self.metrics['num_edges']:,} walkable street edges")
        print(f"   • {self.metrics.get('total_length_km', 0):.1f} km total network length")
        print(f"   • Projected to metric coordinates (UTM)")
        print(f"   • Average edge length: {self.metrics.get('avg_edge_length_m', 0):.1f}m")
        print(f"   • 🎯 One node per intersection for accurate routing")
        print(f"📁 Data saved to: {self.output_dir}")
        
        return True


def main():
    """Main execution function."""
    constructor = SFPedestrianGraphConstructor()
    
    # Execute full construction pipeline (try place name first for better reliability)
    success = constructor.construct_complete_graph(use_bbox=False)
    
    if success:
        print("\n🎯 READY FOR NEXT PHASE:")
        print("   • Crime risk weighting can now be applied to edges")
        print("   • Multi-objective routing algorithms can be implemented")
        print("   • Graph structure validated and optimized for routing")
    else:
        print("\n❌ CONSTRUCTION FAILED")
        print("   💡 Try again later when Overpass API servers are less busy")
        print("   💡 Alternative: Use a smaller geographic area or cached data")


if __name__ == "__main__":
    main()
