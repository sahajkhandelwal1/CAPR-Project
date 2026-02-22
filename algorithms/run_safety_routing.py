"""
Run the multi-objective safety-aware routing algorithm.

Loads the projected graph and attaches edge safety_score from
data/processed/edge_risk_scores_enhanced.csv (or --edge-scores), then
computes a single route or the Pareto frontier.

Usage:
  python algorithms/run_safety_routing.py --graph data/graphs/sf_pedestrian_graph_projected.graphml --source 0 --target 100 --beta 0.5
  python algorithms/run_safety_routing.py --graph data/graphs/sf_pedestrian_graph_projected.graphml --source 0 --target 100 --pareto
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from algorithms.routing import pareto_frontier, route

# Default edge risk CSV used for safety_score (1=safest, 100=least safe)
DEFAULT_EDGE_SCORES_PATH = "data/processed/edge_risk_scores_enhanced.csv"


def load_graph(path: str) -> nx.Graph:
    G = nx.read_graphml(path)
    # Ensure numeric node ids if stored as strings
    return G


def load_edge_scores_csv(csv_path: Path) -> dict[tuple, float]:
    """
    Load edge risk scores from CSV with columns u, v, key, risk_score_enhanced.
    Returns a dict (u, v, key) -> safety_score in [1, 100] (1=safest, 100=least safe).
    """
    lookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v, key = int(row["u"]), int(row["v"]), int(row["key"])
            try:
                r = float(row["risk_score_enhanced"])
            except (ValueError, KeyError):
                continue
            safety = max(1.0, min(100.0, r))
            lookup[(u, v, key)] = safety
    return lookup


def attach_edge_scores(G: nx.Graph, edge_scores: dict[tuple, float]) -> None:
    """Set safety_score on each edge from (u, v, key) -> score lookup. In-place."""
    if G.is_multigraph():
        for u, v, key in G.edges(keys=True):
            k = (int(u), int(v), int(key) if key is not None else 0)
            if k in edge_scores:
                G.edges[u, v, key]["safety_score"] = edge_scores[k]
    else:
        for u, v in G.edges():
            k = (int(u), int(v), 0)
            if k in edge_scores:
                G.edges[u, v]["safety_score"] = edge_scores[k]


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
    parser.add_argument("--edge-scores", default=None, help=f"CSV with u,v,key,risk_score_enhanced (default: {DEFAULT_EDGE_SCORES_PATH})")
    parser.add_argument("--output", default=None, help="Optional: write result JSON here")
    args = parser.parse_args()

    path = Path(args.graph)
    if not path.exists():
        print(f"Graph not found: {path}", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    edge_scores_path = Path(args.edge_scores) if args.edge_scores else project_root / DEFAULT_EDGE_SCORES_PATH

    print("Loading graph...")
    G = load_graph(str(path))
    G = ensure_numeric_ids(G)

    if edge_scores_path.exists():
        print("Attaching edge safety scores from", edge_scores_path.name, "...")
        edge_scores = load_edge_scores_csv(edge_scores_path)
        attach_edge_scores(G, edge_scores)
    else:
        print("Edge scores file not found:", edge_scores_path, "(using default safety_score per edge)", file=sys.stderr)
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
