"""
Run the multi-objective safety-aware routing algorithm.

Loads the projected graph (and optionally attaches edge safety_score from
crime-by-edge data), then computes a single route or the Pareto frontier.

Usage:
  python scripts/run_safety_routing.py --graph data/graphs/sf_pedestrian_graph_projected.graphml --source 0 --target 100 --beta 0.5
  python scripts/run_safety_routing.py --graph data/graphs/sf_pedestrian_graph_projected.graphml --source 0 --target 100 --pareto
"""

import argparse
import json
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from algorithms.routing import pareto_frontier, route


def load_graph(path: str) -> nx.Graph:
    G = nx.read_graphml(path)
    # Ensure numeric node ids if stored as strings
    return G


def ensure_numeric_ids(G: nx.Graph):
    """Convert string node ids to int if they look like integers (for source/target)."""
    try:
        first = next(iter(G.nodes()))
        if isinstance(first, str) and first.isdigit():
            mapping = {n: int(n) for n in G.nodes()}
            return nx.relabel_nodes(G, mapping)
    except StopIteration:
        pass
    return G


def main():
    parser = argparse.ArgumentParser(description="Run safety-aware routing")
    parser.add_argument("--graph", required=True, help="Path to GraphML graph (edges: length; optional: safety_score)")
    parser.add_argument("--source", required=True, type=int, help="Start node id")
    parser.add_argument("--target", required=True, type=int, help="End node id")
    parser.add_argument("--beta", type=float, default=0.5, help="Tradeoff 0=safest, 1=shortest (default 0.5)")
    parser.add_argument("--pareto", action="store_true", help="Sweep beta 0..1 and output Pareto curve")
    parser.add_argument("--output", default=None, help="Optional: write result JSON here")
    args = parser.parse_args()

    path = Path(args.graph)
    if not path.exists():
        print(f"Graph not found: {path}", file=sys.stderr)
        sys.exit(1)

    print("Loading graph...")
    G = load_graph(str(path))
    G = ensure_numeric_ids(G)
    if args.source not in G or args.target not in G:
        print(f"Source {args.source} or target {args.target} not in graph.", file=sys.stderr)
        sys.exit(1)

    if args.pareto:
        print("Computing Pareto frontier (beta = 0, 0.1, ..., 1.0)...")
        results = pareto_frontier(G, args.source, args.target)
        out = [
            {
                "beta_used": r["beta_used"],
                "total_distance_m": r["total_distance_m"],
                "total_risk_score": r["total_risk_score"],
                "combined_cost": r["combined_cost"],
                "num_nodes": len(r["path_nodes"]),
            }
            for r in results
        ]
        print(json.dumps(out, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(out, indent=2))
    else:
        print(f"Computing route (beta={args.beta})...")
        result = route(G, args.source, args.target, beta=args.beta)
        # Minimal print-friendly output (path can be long)
        out = {
            "path_nodes": result["path_nodes"],
            "total_distance_m": result["total_distance_m"],
            "total_risk_score": result["total_risk_score"],
            "combined_cost": result["combined_cost"],
            "beta_used": result["beta_used"],
        }
        print(json.dumps({k: v if k != "path_nodes" else f"<{len(v)} nodes>" for k, v in out.items()}, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
