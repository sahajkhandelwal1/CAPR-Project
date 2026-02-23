"""
Run 50 random routes with beta=0, 0.5, and 1, then plot distance vs risk.

Uses the same graph and edge scores as run_safety_routing.py.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.routing import route
from algorithms.run_safety_routing import (
    DEFAULT_EDGE_SCORES_PATH,
    attach_edge_scores,
    ensure_numeric_ids,
    load_edge_scores_csv,
    load_graph,
)


def main():
    project_root = Path(__file__).resolve().parent.parent
    graph_path = project_root / "data/graphs/sf_pedestrian_graph_enhanced.graphml"
    if not graph_path.exists():
        graph_path = project_root / "data/graphs/sf_pedestrian_graph_projected.graphml"
    if not graph_path.exists():
        # use first available graphml
        graphs_dir = project_root / "data/graphs"
        if graphs_dir.exists():
            graphmls = list(graphs_dir.glob("*.graphml"))
            if graphmls:
                graph_path = graphmls[0]
    if not graph_path.exists():
        print("No GraphML file found under data/graphs/. Place a graph there and retry.", file=sys.stderr)
        sys.exit(1)

    edge_scores_path = project_root / DEFAULT_EDGE_SCORES_PATH

    print("Loading graph from", graph_path.name, "...")
    G = load_graph(str(graph_path))
    G = ensure_numeric_ids(G)
    if edge_scores_path.exists():
        print("Attaching edge safety scores...")
        edge_scores = load_edge_scores_csv(edge_scores_path)
        attach_edge_scores(G, edge_scores)

    nodes = list(G.nodes())
    if len(nodes) < 2:
        print("Graph has fewer than 2 nodes.", file=sys.stderr)
        sys.exit(1)

    betas = [0.0, 0.5, 1.0]
    n_routes = 50
    results = []  # list of {beta, total_distance_m, average_risk_score, run_id}

    random.seed(42)
    for run_id in range(n_routes):
        source, target = random.sample(nodes, 2)
        for beta in betas:
            r = route(G, source, target, beta=beta)
            if r["path_nodes"]:
                n_edges = len(r["path_nodes"]) - 1
                avg_risk = r["total_risk_score"] / n_edges if n_edges > 0 else 0.0
                results.append({
                    "beta": beta,
                    "total_distance_m": r["total_distance_m"],
                    "average_risk_score": avg_risk,
                    "run_id": run_id,
                })

    if not results:
        print("No valid routes found.", file=sys.stderr)
        sys.exit(1)

    # Plot
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull
    except ImportError as e:
        print("matplotlib, numpy and scipy required. Install with: pip install matplotlib numpy scipy", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0.0: "#2ecc71", 0.5: "#3498db", 1.0: "#e74c3c"}
    labels = {0.0: "β=0 (safest)", 0.5: "β=0.5 (balanced)", 1.0: "β=1 (shortest)"}

    # Draw cluster hulls first (behind points)
    for beta in betas:
        subset = [x for x in results if x["beta"] == beta]
        if len(subset) < 3:
            continue
        pts = np.column_stack([
            [x["total_distance_m"] for x in subset],
            [x["average_risk_score"] for x in subset],
        ])
        try:
            hull = ConvexHull(pts)
            poly = Polygon(pts[hull.vertices], facecolor=colors[beta], edgecolor=colors[beta], alpha=0.2, linewidth=1.5)
            ax.add_patch(poly)
        except Exception:
            pass

    # Then scatter points on top
    for beta in betas:
        subset = [x for x in results if x["beta"] == beta]
        if not subset:
            continue
        dist = np.array([x["total_distance_m"] for x in subset])
        risk = np.array([x["average_risk_score"] for x in subset])
        ax.scatter(dist, risk, c=colors[beta], label=labels[beta], alpha=0.7, s=40, zorder=2)

    ax.set_xlabel("Total distance (m)")
    ax.set_ylabel("Average risk score (per edge)")
    ax.set_title("50 random routes: distance vs average risk by β")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = project_root / "algorithms/random_routes_results.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved plot to", out_path)


if __name__ == "__main__":
    main()
