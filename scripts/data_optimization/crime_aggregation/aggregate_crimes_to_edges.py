"""
Edge-Level Crime Risk Aggregation Module

Maps weighted crime incidents to graph edges and computes normalized danger scores (1-100) 
for each edge in the street network. Implements the specified mathematical model:

Risk = average_severity × log(1 + crime_count)
Normalized to 1-100 range where 1=safest, 100=most dangerous.

Usage:
    python aggregate_crimes_to_edges.py \\
        --graph data/graphs/sf_pedestrian_graph_projected.graphml \\
        --crimes data/processed/scored_crime_data.csv \\
        --output-graph data/graphs/sf_graph_with_crime_risk.graphml \\
        --output-csv data/processed/edge_crime_scores.csv \\
        --max-distance 50
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict
import math

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point
from shapely import wkt
from tqdm import tqdm


class CrimeRiskAggregator:
    """Aggregates crime data to graph edges and computes risk scores."""
    
    def __init__(self, max_distance_m=50.0):
        """
        Initialize the aggregator.
        
        Args:
            max_distance_m: Maximum distance (meters) to assign crime to edge
        """
        self.max_distance_m = max_distance_m
        self.graph = None
        self.graph_crs = None
        self.edges_gdf = None
        self.edge_crime_mapping = defaultdict(list)
        self.edge_stats = {}
        
    def load_graph(self, graph_path):
        """Load NetworkX graph and extract CRS."""
        print("Loading street graph...")
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        
        self.graph = nx.read_graphml(str(graph_path))
        
        # Parse CRS from graph metadata
        crs = self.graph.graph.get("crs")
        if crs is None:
            self.graph_crs = "EPSG:4326"  # Default to WGS84
        elif isinstance(crs, dict):
            self.graph_crs = crs.get("name", "EPSG:4326")
        else:
            self.graph_crs = str(crs)
        
        print(f"   ✅ Loaded graph: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
        print(f"   📍 CRS: {self.graph_crs}")
        
    def build_edges_geodataframe(self):
        """Build edges GeoDataFrame from graph geometry."""
        print("Building edges spatial index...")
        
        rows = []
        edges_list = list(self.graph.edges(keys=True))
        
        for u, v, key in tqdm(edges_list, desc="Processing edges"):
            edge_data = self.graph.edges[u, v, key]
            geom_str = edge_data.get("geometry")
            
            if not geom_str:
                # Skip edges without geometry
                continue
                
            try:
                # Parse WKT geometry
                geom = wkt.loads(geom_str)
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to parse geometry for edge {u}-{v}-{key}: {e}")
                continue
            
            rows.append({
                "u": u,
                "v": v, 
                "key": key,
                "edge_id": f"{u}_{v}_{key}",
                "geometry": geom
            })
        
        if not rows:
            raise ValueError("No valid edge geometries found in graph")
        
        self.edges_gdf = gpd.GeoDataFrame(rows, crs=self.graph_crs)
        print(f"   ✅ Built spatial index for {len(self.edges_gdf):,} edges")
        
    def load_and_process_crimes(self, crimes_path, score_col="Score"):
        """
        Load crime data and map to nearest edges.
        
        Args:
            crimes_path: Path to scored crime CSV
            score_col: Name of the crime score column
        """
        print("Loading crime data...")
        crimes_df = pd.read_csv(crimes_path)
        
        # Validate required columns
        required_cols = ["Latitude", "Longitude", score_col]
        missing_cols = [col for col in required_cols if col not in crimes_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing coordinates or scores
        before_count = len(crimes_df)
        crimes_df = crimes_df.dropna(subset=["Latitude", "Longitude", score_col])
        after_count = len(crimes_df)
        
        if after_count == 0:
            raise ValueError("No valid crime records found (all have missing lat/lon/score)")
        
        print(f"   📊 Loaded {after_count:,} crimes (dropped {before_count - after_count:,} with missing data)")
        
        # Convert to GeoDataFrame
        geometry = [Point(lon, lat) for lon, lat in zip(crimes_df["Longitude"], crimes_df["Latitude"])]
        crimes_gdf = gpd.GeoDataFrame(crimes_df, geometry=geometry, crs="EPSG:4326")
        
        # Reproject to graph CRS if needed
        if crimes_gdf.crs != self.graph_crs:
            print(f"   🔄 Reprojecting crimes from {crimes_gdf.crs} to {self.graph_crs}")
            crimes_gdf = crimes_gdf.to_crs(self.graph_crs)
        
        # Map crimes to edges
        print("Mapping crimes to nearest edges...")
        self._map_crimes_to_edges(crimes_gdf, score_col)
        
    def _map_crimes_to_edges(self, crimes_gdf, score_col):
        """Map each crime to its nearest edge using spatial indexing."""
        
        # Build spatial index for edges
        edge_sindex = self.edges_gdf.sindex
        mapped_count = 0
        
        for idx, crime in tqdm(crimes_gdf.iterrows(), total=len(crimes_gdf), desc="Mapping crimes"):
            crime_point = crime.geometry
            crime_score = crime[score_col]
            
            # Create buffer around crime point
            buffer = crime_point.buffer(self.max_distance_m)
            
            # Find candidate edges using spatial index
            possible_matches = list(edge_sindex.intersection(buffer.bounds))
            
            if not possible_matches:
                continue  # No nearby edges
            
            # Get candidate edges
            candidate_edges = self.edges_gdf.iloc[possible_matches].copy()
            
            # Calculate actual distances
            candidate_edges["distance"] = candidate_edges.geometry.distance(crime_point)
            
            # Filter by max distance
            nearby_edges = candidate_edges[candidate_edges["distance"] <= self.max_distance_m]
            
            if nearby_edges.empty:
                continue  # No edges within max distance
            
            # Find nearest edge
            nearest_edge = nearby_edges.loc[nearby_edges["distance"].idxmin()]
            edge_id = nearest_edge["edge_id"]
            
            # Store crime-edge mapping
            self.edge_crime_mapping[edge_id].append({
                "score": crime_score,
                "distance": nearest_edge["distance"],
                "crime_id": idx
            })
            
            mapped_count += 1
        
        print(f"   ✅ Mapped {mapped_count:,} crimes to {len(self.edge_crime_mapping):,} edges")
        print(f"   📊 {len(crimes_gdf) - mapped_count:,} crimes were beyond {self.max_distance_m}m from any edge")
        
    def compute_edge_risk_scores(self):
        """
        Compute risk scores for all edges using the specified mathematical model:
        Raw Risk = average_severity × log(1 + crime_count)
        Then normalize to 1-100 range.
        """
        print("Computing edge risk scores...")
        
        all_edges = list(self.graph.edges(keys=True))
        raw_risks = []
        
        # Compute raw risk scores for all edges
        for u, v, key in tqdm(all_edges, desc="Computing raw risks"):
            edge_id = f"{u}_{v}_{key}"
            crimes = self.edge_crime_mapping.get(edge_id, [])
            
            if not crimes:
                # No crimes on this edge
                n_e = 0
                s_bar_e = 0.0
                raw_risk = 0.0
            else:
                # Extract crime scores
                scores = [crime["score"] for crime in crimes]
                
                # Compute statistics
                n_e = len(scores)  # Number of crimes
                s_bar_e = np.mean(scores)  # Average severity
                
                # Compute raw risk: R_e = s̄_e × log(1 + n_e)
                raw_risk = s_bar_e * math.log(1 + n_e)
            
            # Store edge statistics
            self.edge_stats[edge_id] = {
                "u": u,
                "v": v, 
                "key": key,
                "crime_count": n_e,
                "avg_severity": s_bar_e,
                "raw_risk": raw_risk,
                "crimes": crimes
            }
            
            raw_risks.append(raw_risk)
        
        # Normalize to 1-100 range
        print("Normalizing risk scores to 1-100 range...")
        
        raw_risks = np.array(raw_risks)
        r_min = np.min(raw_risks)
        r_max = np.max(raw_risks)
        
        if r_max == r_min:
            # All edges have same risk (likely all zero)
            print("   ⚠️  Warning: All edges have identical risk scores. Setting all to 1.")
            normalized_risks = np.ones_like(raw_risks)
        else:
            # Normalize: R_e_norm = 1 + 99 × (R_e - R_min) / (R_max - R_min)
            # This ensures range is exactly [1, 100]
            normalized_risks = 1 + 99 * (raw_risks - r_min) / (r_max - r_min)
        
        # Update edge statistics with normalized scores
        for i, (edge_id, stats) in enumerate(self.edge_stats.items()):
            stats["risk_score"] = float(normalized_risks[i])
        
        print(f"   ✅ Risk score statistics:")
        print(f"      Raw risk range: {r_min:.4f} - {r_max:.4f}")
        print(f"      Normalized range: {np.min(normalized_risks):.4f} - {np.max(normalized_risks):.4f}")
        print(f"      Edges with crimes: {len([s for s in self.edge_stats.values() if s['crime_count'] > 0]):,}")
        print(f"      Edges with no crimes: {len([s for s in self.edge_stats.values() if s['crime_count'] == 0]):,}")
        
    def apply_scores_to_graph(self):
        """Apply computed risk scores to the NetworkX graph edges."""
        print("Applying risk scores to graph edges...")
        
        applied_count = 0
        
        for edge_id, stats in tqdm(self.edge_stats.items(), desc="Updating graph"):
            u, v, key = stats["u"], stats["v"], stats["key"]
            
            # Update edge attributes
            if self.graph.has_edge(u, v, key):
                self.graph.edges[u, v, key]["risk_score"] = stats["risk_score"]
                self.graph.edges[u, v, key]["crime_count"] = stats["crime_count"]  
                self.graph.edges[u, v, key]["avg_severity"] = stats["avg_severity"]
                applied_count += 1
            else:
                print(f"   ⚠️  Warning: Edge {edge_id} not found in graph")
        
        print(f"   ✅ Applied risk scores to {applied_count:,} edges")
        
    def save_results(self, output_graph_path=None, output_csv_path=None):
        """Save results to files."""
        
        if output_graph_path:
            print(f"Saving enhanced graph to {output_graph_path}...")
            nx.write_graphml(self.graph, output_graph_path)
            print(f"   ✅ Saved graph with risk scores")
        
        if output_csv_path:
            print(f"Saving edge statistics to {output_csv_path}...")
            
            # Convert edge statistics to DataFrame
            rows = []
            for edge_id, stats in self.edge_stats.items():
                row = {
                    "edge_id": edge_id,
                    "u": stats["u"],
                    "v": stats["v"],
                    "key": stats["key"],
                    "risk_score": stats["risk_score"],
                    "crime_count": stats["crime_count"],
                    "avg_severity": stats["avg_severity"],
                    "raw_risk": stats["raw_risk"]
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_csv_path, index=False)
            print(f"   ✅ Saved edge statistics: {len(df):,} rows")
            
            # Print summary statistics
            print(f"   📊 Risk Score Summary:")
            print(f"      Mean: {df['risk_score'].mean():.2f}")
            print(f"      Median: {df['risk_score'].median():.2f}")
            print(f"      Std Dev: {df['risk_score'].std():.2f}")
            print(f"      Min: {df['risk_score'].min():.2f}")
            print(f"      Max: {df['risk_score'].max():.2f}")
            
    def run_aggregation(self, graph_path, crimes_path, output_graph_path=None, 
                       output_csv_path=None, score_col="Score"):
        """
        Execute the complete crime aggregation pipeline.
        
        Args:
            graph_path: Path to NetworkX graph (GraphML)
            crimes_path: Path to scored crime CSV
            output_graph_path: Optional path to save enhanced graph
            output_csv_path: Optional path to save edge statistics CSV
            score_col: Name of the crime score column
        """
        start_time = time.time()
        
        try:
            # Step 1: Load graph
            self.load_graph(graph_path)
            
            # Step 2: Build spatial index for edges
            self.build_edges_geodataframe()
            
            # Step 3: Load crimes and map to edges
            self.load_and_process_crimes(crimes_path, score_col)
            
            # Step 4: Compute risk scores
            self.compute_edge_risk_scores()
            
            # Step 5: Apply scores to graph
            self.apply_scores_to_graph()
            
            # Step 6: Save results
            self.save_results(output_graph_path, output_csv_path)
            
            elapsed = time.time() - start_time
            print(f"\n✅ Crime aggregation completed successfully in {elapsed:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during crime aggregation: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Aggregate crime data to graph edges and compute risk scores",
        epilog="Example: python aggregate_crimes_to_edges.py --graph sf_graph.graphml --crimes scored_crimes.csv --output-graph enhanced_graph.graphml --output-csv edge_scores.csv"
    )
    
    parser.add_argument("--graph", required=True,
                       help="Path to NetworkX graph file (GraphML format)")
    parser.add_argument("--crimes", required=True, 
                       help="Path to scored crime data CSV")
    parser.add_argument("--output-graph", default=None,
                       help="Optional path to save enhanced graph with risk scores")
    parser.add_argument("--output-csv", default=None,
                       help="Optional path to save edge statistics CSV")
    parser.add_argument("--score-col", default="Score",
                       help="Name of crime score column in CSV (default: Score)")
    parser.add_argument("--max-distance", type=float, default=50.0,
                       help="Maximum distance (m) to assign crime to edge (default: 50)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.graph).exists():
        print(f"❌ Error: Graph file not found: {args.graph}")
        sys.exit(1)
        
    if not Path(args.crimes).exists():
        print(f"❌ Error: Crimes file not found: {args.crimes}")
        sys.exit(1)
    
    if not args.output_graph and not args.output_csv:
        print("⚠️  Warning: No output files specified. Use --output-graph and/or --output-csv")
    
    # Run aggregation
    print("🚀 STARTING CRIME-TO-EDGE RISK AGGREGATION")
    print("=" * 60)
    
    aggregator = CrimeRiskAggregator(max_distance_m=args.max_distance)
    success = aggregator.run_aggregation(
        graph_path=args.graph,
        crimes_path=args.crimes,
        output_graph_path=args.output_graph,
        output_csv_path=args.output_csv,
        score_col=args.score_col
    )
    
    if success:
        print("🎯 READY FOR CRIME-AWARE ROUTING!")
        print("   • Graph edges now have risk_score (1-100)")
        print("   • Higher scores = more dangerous edges")
        print("   • Use risk_score as edge weight in routing algorithms")
    else:
        print("❌ AGGREGATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
