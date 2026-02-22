"""
Graph-Based Risk Diffusion & Spatial Buffering Algorithm

Implements spatial buffering to prevent artificially safe street segments caused by 
sparse crime reporting. Uses graph-based diffusion to propagate risk from high-crime 
edges to nearby edges, producing a smoother and more realistic safety landscape.

Mathematical Model:
R_e^(t+1) = (1-α)R_e^(t) + α * (1/|N(e)|) * Σ R_e'^(t)

Where:
- R_e^(t) = risk score of edge e at iteration t
- N(e) = neighboring edges of edge e
- α = diffusion coefficient [0,1]
- Higher α = more smoothing

Usage:
    python risk_diffusion_buffer.py \
        --graph data/graphs/sf_pedestrian_graph_risk_weighted.graphml \
        --edge-scores data/processed/edge_risk_scores.csv \
        --output-graph data/graphs/sf_pedestrian_graph_diffused.graphml \
        --output-csv data/processed/edge_risk_scores_diffused.csv \
        --alpha 0.3 \
        --iterations 5
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
import time

import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm


class RiskDiffusionBuffer:
    """Implements graph-based risk diffusion for spatial buffering of edge safety scores."""
    
    def __init__(self, alpha=0.3, iterations=3):
        """
        Initialize the diffusion buffer.
        
        Args:
            alpha: Diffusion coefficient [0,1]. Higher = more smoothing
            iterations: Number of diffusion iterations to apply
        """
        self.alpha = alpha
        self.iterations = iterations
        self.graph = None
        self.edge_adjacency = {}
        self.original_scores = {}
        self.buffered_scores = {}
        
    def load_graph(self, graph_path):
        """Load the graph from GraphML file."""
        print(f"Loading graph from {graph_path}...")
        self.graph = nx.read_graphml(graph_path)
        print(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
    def load_edge_scores(self, scores_csv, score_col="risk_score"):
        """Load edge risk scores from CSV file."""
        print(f"Loading edge scores from {scores_csv}...")
        df = pd.read_csv(scores_csv)
        
        if score_col not in df.columns:
            raise ValueError(f"Score column '{score_col}' not found in CSV. Available: {df.columns.tolist()}")
        
        # Build edge ID to score mapping
        for _, row in df.iterrows():
            # Handle different possible column names
            u = row.get('edge_u', row.get('u'))
            v = row.get('edge_v', row.get('v'))
            key = row.get('edge_key', row.get('key'))
            
            if u is None or v is None or key is None:
                continue
                
            edge_id = f"{u}_{v}_{key}"
            self.original_scores[edge_id] = row[score_col]
        
        print(f"Loaded {len(self.original_scores)} edge scores")
        
        # Initialize buffered scores with original scores
        self.buffered_scores = self.original_scores.copy()
        
    def build_edge_adjacency(self):
        """
        Build edge adjacency structure.
        Two edges are neighbors if they share a common node (intersection).
        """
        print("Building edge adjacency structure...")
        
        # Group edges by nodes they connect to
        node_to_edges = defaultdict(set)
        
        for u, v, key in self.graph.edges(keys=True):
            edge_id = f"{u}_{v}_{key}"
            node_to_edges[u].add(edge_id)
            node_to_edges[v].add(edge_id)
        
        # Build adjacency: edges that share a node are neighbors
        self.edge_adjacency = {}
        
        for edge_id in tqdm(self.original_scores.keys(), desc="Computing edge neighbors"):
            neighbors = set()
            
            # Parse edge ID to get nodes
            parts = edge_id.split('_')
            if len(parts) >= 3:
                u, v = parts[0], parts[1]
                
                # Find all edges connected to either node
                neighbors.update(node_to_edges[u])
                neighbors.update(node_to_edges[v])
                
                # Remove self from neighbors
                neighbors.discard(edge_id)
            
            self.edge_adjacency[edge_id] = list(neighbors)
        
        # Statistics
        neighbor_counts = [len(neighbors) for neighbors in self.edge_adjacency.values()]
        avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
        max_neighbors = max(neighbor_counts) if neighbor_counts else 0
        
        print(f"Built adjacency structure:")
        print(f"  Average neighbors per edge: {avg_neighbors:.1f}")
        print(f"  Maximum neighbors: {max_neighbors}")
        
    def apply_single_diffusion_step(self):
        """
        Apply one step of risk diffusion using the formula:
        R_e^(t+1) = (1-α)R_e^(t) + α * (1/|N(e)|) * Σ R_e'^(t)
        """
        new_scores = {}
        
        for edge_id, current_score in self.buffered_scores.items():
            neighbors = self.edge_adjacency.get(edge_id, [])
            
            if not neighbors:
                # No neighbors - keep original score
                new_scores[edge_id] = current_score
                continue
            
            # Compute average neighbor score
            neighbor_scores = [self.buffered_scores.get(neighbor_id, 1.0) 
                             for neighbor_id in neighbors]
            avg_neighbor_score = np.mean(neighbor_scores)
            
            # Apply diffusion formula
            diffused_score = (1 - self.alpha) * current_score + self.alpha * avg_neighbor_score
            
            # Ensure score stays in valid range [1, 100]
            diffused_score = max(1.0, min(100.0, diffused_score))
            
            new_scores[edge_id] = diffused_score
        
        self.buffered_scores = new_scores
        
    def apply_diffusion(self):
        """Apply multiple iterations of risk diffusion."""
        print(f"Applying {self.iterations} iterations of risk diffusion (α={self.alpha})...")
        
        # Track statistics over iterations
        iteration_stats = []
        
        for iteration in tqdm(range(self.iterations), desc="Diffusion iterations"):
            # Record pre-iteration statistics
            scores_array = np.array(list(self.buffered_scores.values()))
            stats = {
                "iteration": iteration,
                "mean": np.mean(scores_array),
                "std": np.std(scores_array),
                "min": np.min(scores_array),
                "max": np.max(scores_array),
                "isolated_min_count": np.sum(scores_array <= 1.1)  # Nearly minimum scores
            }
            iteration_stats.append(stats)
            
            # Apply diffusion step
            self.apply_single_diffusion_step()
        
        # Final statistics
        final_scores = np.array(list(self.buffered_scores.values()))
        final_stats = {
            "iteration": self.iterations,
            "mean": np.mean(final_scores),
            "std": np.std(final_scores),
            "min": np.min(final_scores),
            "max": np.max(final_scores),
            "isolated_min_count": np.sum(final_scores <= 1.1)
        }
        iteration_stats.append(final_stats)
        
        # Print diffusion summary
        print("\n=== DIFFUSION SUMMARY ===")
        initial = iteration_stats[0]
        final = iteration_stats[-1]
        
        print(f"Risk score statistics:")
        print(f"  Mean: {initial['mean']:.1f} → {final['mean']:.1f}")
        print(f"  Std Dev: {initial['std']:.1f} → {final['std']:.1f}")
        print(f"  Range: [{initial['min']:.1f}, {initial['max']:.1f}] → [{final['min']:.1f}, {final['max']:.1f}]")
        print(f"  Isolated minimum scores: {initial['isolated_min_count']} → {final['isolated_min_count']}")
        
        return iteration_stats
        
    def apply_to_graph(self):
        """Apply buffered scores to the graph edges."""
        print("Applying buffered scores to graph...")
        
        edges_updated = 0
        
        for u, v, key in self.graph.edges(keys=True):
            edge_id = f"{u}_{v}_{key}"
            
            if edge_id in self.buffered_scores:
                # Update with buffered score
                original_score = self.original_scores.get(edge_id, 1.0)
                buffered_score = self.buffered_scores[edge_id]
                
                self.graph.edges[u, v, key]["risk_score_original"] = original_score
                self.graph.edges[u, v, key]["risk_score"] = buffered_score
                self.graph.edges[u, v, key]["risk_score_buffered"] = buffered_score
                
                edges_updated += 1
            else:
                # Assign default minimum risk for edges without scores
                self.graph.edges[u, v, key]["risk_score"] = 1.0
                self.graph.edges[u, v, key]["risk_score_buffered"] = 1.0
        
        print(f"Applied buffered scores to {edges_updated} edges")
        
    def save_results(self, output_graph=None, output_csv=None):
        """Save the buffered results to files."""
        
        if output_graph:
            print(f"Saving buffered graph to {output_graph}")
            nx.write_graphml(self.graph, output_graph)
        
        if output_csv:
            print(f"Saving buffered edge scores to {output_csv}")
            
            # Create output DataFrame
            results = []
            for edge_id, buffered_score in self.buffered_scores.items():
                parts = edge_id.split('_')
                if len(parts) >= 3:
                    original_score = self.original_scores.get(edge_id, 1.0)
                    
                    results.append({
                        "edge_id": edge_id,
                        "u": parts[0],
                        "v": parts[1], 
                        "key": parts[2],
                        "risk_score_original": original_score,
                        "risk_score_buffered": buffered_score,
                        "risk_score_change": buffered_score - original_score
                    })
            
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
        
    def run_diffusion(self, graph_path, edge_scores_csv, output_graph=None, output_csv=None, score_col="risk_score"):
        """Run the complete diffusion pipeline."""
        
        start_time = time.time()
        
        # Load inputs
        self.load_graph(graph_path)
        self.load_edge_scores(edge_scores_csv, score_col)
        
        # Build edge adjacency
        self.build_edge_adjacency()
        
        # Apply diffusion
        iteration_stats = self.apply_diffusion()
        
        # Apply results to graph
        self.apply_to_graph()
        
        # Save outputs
        self.save_results(output_graph, output_csv)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Risk diffusion completed in {total_time:.1f}s")
        print(f"📊 Final statistics:")
        final_scores = np.array(list(self.buffered_scores.values()))
        print(f"   Mean risk: {np.mean(final_scores):.1f}")
        print(f"   Risk range: [{np.min(final_scores):.1f}, {np.max(final_scores):.1f}]") 
        print(f"   Std deviation: {np.std(final_scores):.1f}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply graph-based risk diffusion for spatial buffering of edge safety scores",
        epilog="Example: python risk_diffusion_buffer.py --graph graph.graphml --edge-scores scores.csv --output-graph buffered_graph.graphml --alpha 0.3 --iterations 5"
    )
    
    parser.add_argument("--graph", required=True, help="Path to input GraphML file with risk scores")
    parser.add_argument("--edge-scores", required=True, help="Path to CSV file with edge risk scores")
    parser.add_argument("--output-graph", help="Path to save buffered GraphML file")
    parser.add_argument("--output-csv", help="Path to save buffered edge scores CSV")
    parser.add_argument("--score-col", default="risk_score", help="Column name for risk scores (default: risk_score)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Diffusion coefficient [0,1] (default: 0.3)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of diffusion iterations (default: 3)")
    
    args = parser.parse_args()
    
    # Validate parameters
    if not (0.0 <= args.alpha <= 1.0):
        print("❌ Error: Alpha must be between 0 and 1")
        sys.exit(1)
        
    if args.iterations < 1:
        print("❌ Error: Iterations must be at least 1")
        sys.exit(1)
        
    # Validate input files
    if not Path(args.graph).exists():
        print(f"❌ Error: Graph file not found: {args.graph}")
        sys.exit(1)
        
    if not Path(args.edge_scores).exists():
        print(f"❌ Error: Edge scores file not found: {args.edge_scores}")
        sys.exit(1)
    
    if not args.output_graph and not args.output_csv:
        print("⚠️  Warning: No output files specified. Use --output-graph and/or --output-csv")
    
    try:
        print("🌊 STARTING RISK DIFFUSION & SPATIAL BUFFERING")
        print("=" * 60)
        
        # Initialize and run diffusion
        diffusion_buffer = RiskDiffusionBuffer(alpha=args.alpha, iterations=args.iterations)
        
        success = diffusion_buffer.run_diffusion(
            graph_path=args.graph,
            edge_scores_csv=args.edge_scores,
            output_graph=args.output_graph,
            output_csv=args.output_csv,
            score_col=args.score_col
        )
        
        if success:
            print("\n🎯 SPATIAL BUFFERING COMPLETE!")
            print("   • Risk scores now spatially diffused")
            print("   • Regional patterns preserved")
            print("   • Ready for realistic crime-aware routing")
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Value error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
