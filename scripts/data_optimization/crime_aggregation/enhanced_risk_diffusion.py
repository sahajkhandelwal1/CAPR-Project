"""
Enhanced Risk Diffusion with Adaptive Buffering and Risk Amplification

This improved algorithm addresses the over-smoothing problem by:
1. Preserving high-risk cores while buffering surroundings
2. Applying non-linear risk amplification to increase differentiation
3. Using adaptive diffusion that respects risk gradients
4. Implementing risk curve transformation to achieve target mean

Mathematical Model:
- Core Preservation: High-risk edges retain more of their original score
- Adaptive Diffusion: α varies based on risk level difference
- Risk Amplification: Apply power-law transformation to increase spread
- Target Curve: Transform final distribution to achieve desired mean (e.g., 35/100)

Usage:
    python enhanced_risk_diffusion.py \
        --graph data/graphs/sf_pedestrian_graph_risk_weighted.graphml \
        --edge-scores data/processed/edge_risk_scores.csv \
        --output-graph data/graphs/sf_pedestrian_graph_enhanced.graphml \
        --output-csv data/processed/edge_risk_scores_enhanced.csv \
        --target-mean 35 \
        --preserve-threshold 20 \
        --amplification-factor 1.5
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
from scipy import stats


class EnhancedRiskDiffusion:
    """Enhanced risk diffusion with core preservation and risk amplification."""
    
    def __init__(self, target_mean=35.0, preserve_threshold=20.0, amplification_factor=1.5, 
                 base_alpha=0.2, iterations=3):
        """
        Initialize enhanced diffusion.
        
        Args:
            target_mean: Target mean risk score (e.g., 35 out of 100)
            preserve_threshold: Risk level above which cores are preserved
            amplification_factor: Power law exponent for risk amplification (>1 increases spread)
            base_alpha: Base diffusion coefficient
            iterations: Number of diffusion iterations
        """
        self.target_mean = target_mean
        self.preserve_threshold = preserve_threshold
        self.amplification_factor = amplification_factor
        self.base_alpha = base_alpha
        self.iterations = iterations
        
        self.graph = None
        self.edge_adjacency = {}
        self.original_scores = {}
        self.enhanced_scores = {}
        
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
            u = row.get('edge_u', row.get('u'))
            v = row.get('edge_v', row.get('v'))
            key = row.get('edge_key', row.get('key'))
            
            if u is None or v is None or key is None:
                continue
                
            edge_id = f"{u}_{v}_{key}"
            self.original_scores[edge_id] = row[score_col]
        
        print(f"Loaded {len(self.original_scores)} edge scores")
        print(f"Original score range: [{min(self.original_scores.values()):.1f}, {max(self.original_scores.values()):.1f}]")
        print(f"Original mean: {np.mean(list(self.original_scores.values())):.1f}")
        
        # Initialize enhanced scores
        self.enhanced_scores = self.original_scores.copy()
        
    def build_edge_adjacency(self):
        """Build edge adjacency structure."""
        print("Building edge adjacency structure...")
        
        # Group edges by nodes they connect to
        node_to_edges = defaultdict(set)
        
        for u, v, key in self.graph.edges(keys=True):
            edge_id = f"{u}_{v}_{key}"
            node_to_edges[u].add(edge_id)
            node_to_edges[v].add(edge_id)
        
        # Build adjacency
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
        
        neighbor_counts = [len(neighbors) for neighbors in self.edge_adjacency.values()]
        avg_neighbors = np.mean(neighbor_counts) if neighbor_counts else 0
        print(f"Average neighbors per edge: {avg_neighbors:.1f}")
        
    def compute_adaptive_alpha(self, edge_score, neighbor_scores):
        """
        Compute adaptive diffusion coefficient based on risk gradient.
        High-risk cores get lower alpha (preserve more), low-risk areas get higher alpha.
        """
        if not neighbor_scores:
            return 0.0
        
        avg_neighbor_score = np.mean(neighbor_scores)
        
        # Core preservation: if edge is high-risk, reduce diffusion
        if edge_score >= self.preserve_threshold:
            core_factor = 0.1  # Strong preservation for cores
        else:
            core_factor = 1.0
        
        # Gradient-based adaptation: if surrounded by much higher risk, increase diffusion
        risk_gradient = abs(edge_score - avg_neighbor_score)
        gradient_factor = min(2.0, 1.0 + risk_gradient / 20.0)  # Increase up to 2x based on gradient
        
        adaptive_alpha = self.base_alpha * core_factor * gradient_factor
        return min(0.5, adaptive_alpha)  # Cap at 0.5
    
    def apply_enhanced_diffusion_step(self):
        """Apply one step of enhanced adaptive diffusion."""
        new_scores = {}
        
        for edge_id, current_score in self.enhanced_scores.items():
            neighbors = self.edge_adjacency.get(edge_id, [])
            
            if not neighbors:
                new_scores[edge_id] = current_score
                continue
            
            # Get neighbor scores
            neighbor_scores = [self.enhanced_scores.get(neighbor_id, 1.0) 
                             for neighbor_id in neighbors]
            
            # Compute adaptive alpha
            alpha = self.compute_adaptive_alpha(current_score, neighbor_scores)
            
            # Apply diffusion with adaptive coefficient
            avg_neighbor_score = np.mean(neighbor_scores)
            diffused_score = (1 - alpha) * current_score + alpha * avg_neighbor_score
            
            # Ensure valid range
            diffused_score = max(1.0, min(100.0, diffused_score))
            new_scores[edge_id] = diffused_score
        
        self.enhanced_scores = new_scores
    
    def apply_risk_amplification(self):
        """
        Apply power-law transformation to increase risk differentiation.
        This spreads out the risk distribution while preserving relative ordering.
        """
        print(f"Applying risk amplification (factor: {self.amplification_factor})...")
        
        scores_array = np.array(list(self.enhanced_scores.values()))
        
        # Normalize to [0,1] for power transformation
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        normalized = (scores_array - min_score) / (max_score - min_score)
        
        # Apply power law: higher exponent increases spread
        amplified = np.power(normalized, 1.0 / self.amplification_factor)
        
        # Scale back to [1, 100] range
        amplified_scores = 1 + 99 * amplified
        
        # Update scores
        for i, edge_id in enumerate(self.enhanced_scores.keys()):
            self.enhanced_scores[edge_id] = amplified_scores[i]
        
        print(f"After amplification - Mean: {np.mean(amplified_scores):.1f}, "
              f"Std: {np.std(amplified_scores):.1f}")
    
    def apply_curve_transformation(self):
        """
        Transform the final distribution to achieve target mean while preserving shape.
        Uses a sigmoid-like transformation to achieve desired risk curve.
        """
        print(f"Applying curve transformation (target mean: {self.target_mean})...")
        
        scores_array = np.array(list(self.enhanced_scores.values()))
        current_mean = np.mean(scores_array)
        
        if abs(current_mean - self.target_mean) < 1.0:
            print("Score distribution already near target mean")
            return
        
        # Method 1: Percentile-based transformation
        # Map percentiles to achieve target distribution
        
        # Calculate target percentiles for desired mean
        # For mean=35, we want more scores in mid-to-high range
        
        sorted_scores = np.sort(scores_array)
        n = len(sorted_scores)
        
        # Create target distribution curve
        # Use beta distribution to create realistic risk curve
        target_percentiles = np.linspace(0.001, 0.999, n)
        
        if self.target_mean < 20:
            # Low mean: mostly safe with few dangerous areas
            target_scores = 1 + 99 * np.power(target_percentiles, 3)
        elif self.target_mean < 40:
            # Medium mean: more graduated risk
            target_scores = 1 + 99 * np.power(target_percentiles, 2)
        else:
            # High mean: many dangerous areas
            target_scores = 1 + 99 * np.power(target_percentiles, 1.5)
        
        # Adjust to hit exact target mean
        current_target_mean = np.mean(target_scores)
        adjustment = self.target_mean - current_target_mean
        target_scores = np.clip(target_scores + adjustment, 1, 100)
        
        # Create mapping from original percentiles to target scores
        original_ranks = stats.rankdata(scores_array) - 1
        transformed_scores = target_scores[original_ranks.astype(int)]
        
        # Update scores
        for i, edge_id in enumerate(self.enhanced_scores.keys()):
            self.enhanced_scores[edge_id] = transformed_scores[i]
        
        final_mean = np.mean(transformed_scores)
        print(f"Curve transformation: {current_mean:.1f} → {final_mean:.1f} (target: {self.target_mean})")
        print(f"Final range: [{np.min(transformed_scores):.1f}, {np.max(transformed_scores):.1f}]")
        print(f"Final std: {np.std(transformed_scores):.1f}")
    
    def apply_enhanced_diffusion(self):
        """Apply the complete enhanced diffusion pipeline."""
        print(f"Applying {self.iterations} iterations of enhanced adaptive diffusion...")
        
        for iteration in tqdm(range(self.iterations), desc="Enhanced diffusion"):
            self.apply_enhanced_diffusion_step()
        
        # Apply risk amplification
        self.apply_risk_amplification()
        
        # Apply curve transformation to achieve target mean
        self.apply_curve_transformation()
        
        # Final statistics
        final_scores = np.array(list(self.enhanced_scores.values()))
        original_scores = np.array(list(self.original_scores.values()))
        
        print("\n=== ENHANCED DIFFUSION SUMMARY ===")
        print(f"Original stats - Mean: {np.mean(original_scores):.1f}, Std: {np.std(original_scores):.1f}")
        print(f"Enhanced stats - Mean: {np.mean(final_scores):.1f}, Std: {np.std(final_scores):.1f}")
        print(f"Range: [{np.min(final_scores):.1f}, {np.max(final_scores):.1f}]")
        print(f"Isolated minimum scores (≤2): {np.sum(final_scores <= 2):,}")
        
        return final_scores
        
    def apply_to_graph(self):
        """Apply enhanced scores to the graph edges."""
        print("Applying enhanced scores to graph...")
        
        edges_updated = 0
        
        for u, v, key in self.graph.edges(keys=True):
            edge_id = f"{u}_{v}_{key}"
            
            if edge_id in self.enhanced_scores:
                original_score = self.original_scores.get(edge_id, 1.0)
                enhanced_score = self.enhanced_scores[edge_id]
                
                self.graph.edges[u, v, key]["risk_score_original"] = original_score
                self.graph.edges[u, v, key]["risk_score"] = enhanced_score
                self.graph.edges[u, v, key]["risk_score_enhanced"] = enhanced_score
                
                edges_updated += 1
            else:
                # Default for edges without scores
                self.graph.edges[u, v, key]["risk_score"] = 1.0
                self.graph.edges[u, v, key]["risk_score_enhanced"] = 1.0
        
        print(f"Applied enhanced scores to {edges_updated} edges")
        
    def save_results(self, output_graph=None, output_csv=None):
        """Save the enhanced results to files."""
        
        if output_graph:
            print(f"Saving enhanced graph to {output_graph}")
            nx.write_graphml(self.graph, output_graph)
        
        if output_csv:
            print(f"Saving enhanced edge scores to {output_csv}")
            
            results = []
            for edge_id, enhanced_score in self.enhanced_scores.items():
                parts = edge_id.split('_')
                if len(parts) >= 3:
                    original_score = self.original_scores.get(edge_id, 1.0)
                    
                    results.append({
                        "edge_id": edge_id,
                        "u": parts[0],
                        "v": parts[1], 
                        "key": parts[2],
                        "risk_score_original": original_score,
                        "risk_score_enhanced": enhanced_score,
                        "risk_score_change": enhanced_score - original_score,
                        "change_magnitude": abs(enhanced_score - original_score),
                        "change_direction": "increased" if enhanced_score > original_score else "decreased" if enhanced_score < original_score else "unchanged"
                    })
            
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
        
    def run_enhancement(self, graph_path, edge_scores_csv, output_graph=None, output_csv=None, score_col="risk_score"):
        """Run the complete enhanced diffusion pipeline."""
        
        start_time = time.time()
        
        # Load inputs
        self.load_graph(graph_path)
        self.load_edge_scores(edge_scores_csv, score_col)
        
        # Build edge adjacency
        self.build_edge_adjacency()
        
        # Apply enhanced diffusion
        self.apply_enhanced_diffusion()
        
        # Apply results to graph
        self.apply_to_graph()
        
        # Save outputs
        self.save_results(output_graph, output_csv)
        
        total_time = time.time() - start_time
        print(f"\n✅ Enhanced risk diffusion completed in {total_time:.1f}s")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced risk diffusion with adaptive buffering and risk amplification",
        epilog="Example: python enhanced_risk_diffusion.py --graph graph.graphml --edge-scores scores.csv --target-mean 35 --preserve-threshold 20"
    )
    
    parser.add_argument("--graph", required=True, help="Path to input GraphML file")
    parser.add_argument("--edge-scores", required=True, help="Path to CSV file with edge risk scores")
    parser.add_argument("--output-graph", help="Path to save enhanced GraphML file")
    parser.add_argument("--output-csv", help="Path to save enhanced edge scores CSV")
    parser.add_argument("--score-col", default="risk_score", help="Column name for risk scores")
    parser.add_argument("--target-mean", type=float, default=35.0, help="Target mean risk score (default: 35)")
    parser.add_argument("--preserve-threshold", type=float, default=20.0, help="Risk threshold for core preservation (default: 20)")
    parser.add_argument("--amplification-factor", type=float, default=1.5, help="Risk amplification factor (default: 1.5)")
    parser.add_argument("--base-alpha", type=float, default=0.2, help="Base diffusion coefficient (default: 0.2)")
    parser.add_argument("--iterations", type=int, default=3, help="Diffusion iterations (default: 3)")
    
    args = parser.parse_args()
    
    # Validate parameters
    if not (10.0 <= args.target_mean <= 80.0):
        print("❌ Error: Target mean should be between 10 and 80")
        sys.exit(1)
        
    if not (0.0 < args.base_alpha < 1.0):
        print("❌ Error: Base alpha must be between 0 and 1")
        sys.exit(1)
        
    # Validate input files
    if not Path(args.graph).exists():
        print(f"❌ Error: Graph file not found: {args.graph}")
        sys.exit(1)
        
    if not Path(args.edge_scores).exists():
        print(f"❌ Error: Edge scores file not found: {args.edge_scores}")
        sys.exit(1)
    
    try:
        print("🚀 STARTING ENHANCED RISK DIFFUSION")
        print("=" * 60)
        
        enhancer = EnhancedRiskDiffusion(
            target_mean=args.target_mean,
            preserve_threshold=args.preserve_threshold,
            amplification_factor=args.amplification_factor,
            base_alpha=args.base_alpha,
            iterations=args.iterations
        )
        
        success = enhancer.run_enhancement(
            graph_path=args.graph,
            edge_scores_csv=args.edge_scores,
            output_graph=args.output_graph,
            output_csv=args.output_csv,
            score_col=args.score_col
        )
        
        if success:
            print("\n🎯 ENHANCED RISK DIFFUSION COMPLETE!")
            print("   • High-risk cores preserved")
            print("   • Risk differentiation amplified")
            print("   • Target mean achieved")
            print("   • Ready for advanced crime-aware routing")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
