"""
Run random routes with beta=0, 0.5, and 1, then plot distance vs risk.

Prompts for: risk type (average/total), line of best fit, cluster highlighting,
and number of routes. Uses the same graph and edge scores as run_safety_routing.py.
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


def _prompt_choice(prompt: str, default: str = "") -> str:
    """Return trimmed user input or default if empty."""
    s = input(prompt).strip().lower() if sys.stdin.isatty() else default
    return s or default


def get_plot_options():
    """Ask user for risk type, line of best fit, cluster highlighting, and n_routes."""
    print("\n--- Plot options ---")
    risk = _prompt_choice("Risk score: (a)verage or (t)otal? [a/t] (default: a): ", "a")
    use_average_risk = risk != "t"
    line_input = _prompt_choice("Show line of best fit? [y/n] (default: n): ", "n")
    show_line = line_input in ("y", "yes")
    cluster_input = _prompt_choice("Show cluster highlighting (convex hulls)? [y/n] (default: y): ", "y")
    show_cluster = cluster_input in ("y", "yes")
    n_input = _prompt_choice("Number of routes to test? (default: 50): ", "50")
    try:
        n_routes = max(1, int(n_input))
    except ValueError:
        n_routes = 50
    return use_average_risk, show_line, show_cluster, n_routes


def main():
    use_average_risk, show_line, show_cluster, n_routes = get_plot_options()
    risk_key = "average_risk_score" if use_average_risk else "total_risk_score"
    risk_label = "Average risk score (per edge)" if use_average_risk else "Total risk score"

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
    results = []  # list of {beta, total_distance_m, average_risk_score, total_risk_score, run_id}

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
                    "total_risk_score": r["total_risk_score"],
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

    # Draw cluster hulls first (behind points), if requested
    if show_cluster:
        for beta in betas:
            subset = [x for x in results if x["beta"] == beta]
            if len(subset) < 3:
                continue
            pts = np.column_stack([
                [x["total_distance_m"] for x in subset],
                [x[risk_key] for x in subset],
            ])
            try:
                hull = ConvexHull(pts)
                poly = Polygon(pts[hull.vertices], facecolor=colors[beta], edgecolor=colors[beta], alpha=0.2, linewidth=1.5)
                ax.add_patch(poly)
            except Exception:
                pass

    # Scatter points (and optionally line of best fit) per beta
    for beta in betas:
        subset = [x for x in results if x["beta"] == beta]
        if not subset:
            continue
        dist = np.array([x["total_distance_m"] for x in subset])
        risk = np.array([x[risk_key] for x in subset])
        ax.scatter(dist, risk, c=colors[beta], label=labels[beta], alpha=0.7, s=40, zorder=2)
        if show_line and len(dist) > 1:
            coefs = np.polyfit(dist, risk, 1)
            x_line = np.linspace(dist.min(), dist.max(), 100)
            ax.plot(x_line, np.polyval(coefs, x_line), c=colors[beta], linestyle="--", linewidth=2)

    ax.set_xlabel("Total distance (m)")
    ax.set_ylabel(risk_label)
    ax.set_title(f"{n_routes} random routes: distance vs risk by β")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = project_root / "algorithms/random_routes_results.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved plot to", out_path)


if __name__ == "__main__":
    main()
