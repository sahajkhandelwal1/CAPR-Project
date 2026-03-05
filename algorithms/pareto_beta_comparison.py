"""
Compare routing performance at β=0 (safest/crime-averse), β=1 (distance-only),
and the Pareto-optimal β per route.

For each trial: randomly pick start/end nodes, compute routes at β=0 and β=1,
sweep β in [0, 0.05, ..., 1] to find the Pareto-optimal tradeoff point, then
record (distance, crime exposure) for all three. Results are visualized in a
scatter plot (distance vs crime) with distinct colors and a progress bar.
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

# Beta sweep step
BETA_STEP = 0.05


def _prompt_int(prompt: str, default: int) -> int:
    """Read integer from stdin or return default if not TTY or invalid."""
    if not sys.stdin.isatty():
        return default
    s = input(prompt).strip()
    if not s:
        return default
    try:
        return max(1, int(s))
    except ValueError:
        return default


def _prompt_yn(prompt: str, default: bool = False) -> bool:
    """Read y/n from stdin; return True for y/yes, False otherwise. Uses default if not TTY."""
    if not sys.stdin.isatty():
        return default
    s = input(prompt).strip().lower()
    if not s:
        return default
    return s in ("y", "yes")


def _sweep_betas():
    """Betas from 0 to 1 inclusive with step BETA_STEP."""
    return [round(i * BETA_STEP, 2) for i in range(int(1.0 / BETA_STEP) + 1)]


def _pareto_optimal_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the subset of (distance, crime) points that are Pareto-optimal (non-dominated)."""
    if not points:
        return []
    pareto = []
    for (d, c) in points:
        dominated = False
        for (d2, c2) in points:
            if (d2, c2) == (d, c):
                continue
            if d2 <= d and c2 <= c and (d2 < d or c2 < c):
                dominated = True
                break
        if not dominated:
            pareto.append((d, c))
    return pareto


def _best_tradeoff_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """
    Among Pareto-optimal points, return the one that minimizes normalized distance + crime
    (best tradeoff). Returns None if points is empty.
    """
    pareto = _pareto_optimal_points(points)
    if not pareto:
        return None
    if len(pareto) == 1:
        return pareto[0]
    d_vals = [p[0] for p in pareto]
    c_vals = [p[1] for p in pareto]
    d_max = max(d_vals)
    c_max = max(c_vals)
    if d_max <= 0:
        d_max = 1.0
    if c_max <= 0:
        c_max = 1.0
    best = min(
        pareto,
        key=lambda p: (p[0] / d_max) + (p[1] / c_max),
    )
    return best


def _safe_pct_change(old_val: float, new_val: float) -> float | None:
    """Return (new - old) / old * 100 if old != 0, else None."""
    if old_val == 0:
        return None
    return ((new_val - old_val) / old_val) * 100.0


def _print_improvement_statistics(results: list[dict], n_trials: int) -> None:
    """Compute and print improvement statistics comparing beta=0, beta=1, and Pareto-optimal routes."""
    b0 = [r["beta_0"] for r in results if r["beta_0"] is not None]
    b1 = [r["beta_1"] for r in results if r["beta_1"] is not None]
    pareto = [r["pareto"] for r in results if r["pareto"] is not None]

    def stats(label: str, points: list[tuple[float, float]]) -> None:
        if not points:
            print(f"  {label}: no data")
            return
        dists = [p[0] for p in points]
        crimes = [p[1] for p in points]
        n = len(points)
        print(f"  {label} (n={n}):")
        print(f"    Distance (m):  mean={sum(dists)/n:.1f}, median={sorted(dists)[n//2]:.1f}, min={min(dists):.1f}, max={max(dists):.1f}")
        print(f"    Crime (score): mean={sum(crimes)/n:.1f}, median={sorted(crimes)[n//2]:.1f}, min={min(crimes):.1f}, max={max(crimes):.1f}")

    print("\n" + "=" * 60)
    print("IMPROVEMENT STATISTICS")
    print("=" * 60)
    print(f"Total trials: {n_trials}")
    print("\n--- Summary by strategy ---")
    stats("beta=0 (safest)", b0)
    stats("beta=1 (distance-only)", b1)
    stats("Pareto-optimal", pareto)

    pairs_p_vs_1 = [(r["pareto"], r["beta_1"]) for r in results if r["pareto"] is not None and r["beta_1"] is not None]
    if pairs_p_vs_1:
        n = len(pairs_p_vs_1)
        crime_reductions_pct = []
        crime_reductions_abs = []
        dist_increases_pct = []
        dist_increases_abs = []
        pareto_lower_crime = 0
        for (dp, cp), (d1, c1) in pairs_p_vs_1:
            if c1 > 0:
                crime_reductions_pct.append(_safe_pct_change(c1, cp))
                crime_reductions_abs.append(c1 - cp)
            if cp < c1:
                pareto_lower_crime += 1
            if d1 > 0:
                dist_increases_pct.append(_safe_pct_change(d1, dp))
                dist_increases_abs.append(dp - d1)
        print("\n--- Pareto vs beta=1 (distance-only) ---")
        print(f"  Trials with both routes: {n}")
        if crime_reductions_pct:
            avg = sum(crime_reductions_pct) / len(crime_reductions_pct)
            print(f"  Average crime change: {avg:+.1f}% (negative = less crime with Pareto)")
        if crime_reductions_abs:
            print(f"  Average crime change (absolute): {sum(crime_reductions_abs)/len(crime_reductions_abs):+.1f} points")
        print(f"  Trials where Pareto had lower crime than beta=1: {pareto_lower_crime} ({100*pareto_lower_crime/n:.0f}%)")
        if dist_increases_pct:
            avg = sum(dist_increases_pct) / len(dist_increases_pct)
            print(f"  Average distance change: {avg:+.1f}% (positive = longer path with Pareto)")
        if dist_increases_abs:
            print(f"  Average distance change (absolute): {sum(dist_increases_abs)/len(dist_increases_abs):+.1f} m")

    pairs_p_vs_0 = [(r["pareto"], r["beta_0"]) for r in results if r["pareto"] is not None and r["beta_0"] is not None]
    if pairs_p_vs_0:
        n = len(pairs_p_vs_0)
        dist_reductions_pct = []
        dist_reductions_abs = []
        crime_increases_pct = []
        crime_increases_abs = []
        for (dp, cp), (d0, c0) in pairs_p_vs_0:
            if d0 > 0:
                dist_reductions_pct.append(_safe_pct_change(d0, dp))
                dist_reductions_abs.append(d0 - dp)
            if c0 > 0:
                crime_increases_pct.append(_safe_pct_change(c0, cp))
                crime_increases_abs.append(cp - c0)
        print("\n--- Pareto vs beta=0 (safest) ---")
        print(f"  Trials with both routes: {n}")
        if dist_reductions_pct:
            avg = sum(dist_reductions_pct) / len(dist_reductions_pct)
            print(f"  Average distance change: {avg:+.1f}% (negative = shorter path with Pareto)")
        if dist_reductions_abs:
            print(f"  Average distance change (absolute): {sum(dist_reductions_abs)/len(dist_reductions_abs):+.1f} m")
        if crime_increases_pct:
            avg = sum(crime_increases_pct) / len(crime_increases_pct)
            print(f"  Average crime change: {avg:+.1f}% (positive = more crime with Pareto)")
        if crime_increases_abs:
            print(f"  Average crime change (absolute): {sum(crime_increases_abs)/len(crime_increases_abs):+.1f} points")

    pairs_0_vs_1 = [(r["beta_0"], r["beta_1"]) for r in results if r["beta_0"] is not None and r["beta_1"] is not None]
    if pairs_0_vs_1:
        n = len(pairs_0_vs_1)
        crime_reductions = []
        dist_increases = []
        for (d0, c0), (d1, c1) in pairs_0_vs_1:
            if c1 > 0:
                crime_reductions.append(_safe_pct_change(c1, c0))
            if d1 > 0:
                dist_increases.append(_safe_pct_change(d1, d0))
        print("\n--- beta=0 (safest) vs beta=1 (distance-only) ---")
        print(f"  Trials with both routes: {n}")
        if crime_reductions:
            print(f"  Average crime decrease (safest vs shortest): {sum(crime_reductions)/len(crime_reductions):.1f}%")
        if dist_increases:
            print(f"  Average distance increase (safest vs shortest): {sum(dist_increases)/len(dist_increases):.1f}%")

    print("=" * 60 + "\n")


def run_trial(G, source, target):
    """
    For one (source, target) pair:
      - Route at β=0 and β=1.
      - Sweep β in 0..1 step 0.05, collect (distance, crime).
      - Compute Pareto-optimal best-tradeoff point.
    Return dict with keys: beta_0, beta_1, pareto (each (distance, crime) or None).
    """
    r0 = route(G, source, target, beta=0.0)
    r1 = route(G, source, target, beta=1.0)

    beta_0 = None
    beta_1 = None
    if r0["path_nodes"]:
        beta_0 = (r0["total_distance_m"], r0["total_risk_score"])
    if r1["path_nodes"]:
        beta_1 = (r1["total_distance_m"], r1["total_risk_score"])

    sweep_points = []
    for b in _sweep_betas():
        r = route(G, source, target, beta=b)
        if r["path_nodes"]:
            sweep_points.append((r["total_distance_m"], r["total_risk_score"]))

    pareto_point = _best_tradeoff_point(sweep_points) if sweep_points else None

    return {"beta_0": beta_0, "beta_1": beta_1, "pareto": pareto_point}


def main():
    print("Pareto-optimal beta comparison: distance vs crime exposure")
    n_trials = _prompt_int("Number of trials? (default: 30): ", 30)
    show_line = _prompt_yn("Show line of best fit? [y/n] (default: n): ", False)
    show_stats = _prompt_yn("Show improvement statistics? [y/n] (default: n): ", False)
    print(f"Using {n_trials} trials, beta sweep 0 to 1 step {BETA_STEP}")

    project_root = Path(__file__).resolve().parent.parent
    graph_path = project_root / "data/graphs/sf_pedestrian_graph_enhanced.graphml"
    if not graph_path.exists():
        graph_path = project_root / "data/graphs/sf_pedestrian_graph_projected.graphml"
    if not graph_path.exists():
        graphs_dir = project_root / "data/graphs"
        if graphs_dir.exists():
            graphmls = list(graphs_dir.glob("*.graphml"))
            if graphmls:
                graph_path = graphmls[0]
    if not graph_path.exists():
        print("No GraphML file found under data/graphs/.", file=sys.stderr)
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

    random.seed(42)
    results = []  # list of run_trial outputs

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable

    for _ in tqdm(range(n_trials), desc="Trials", unit="trial"):
        source, target = random.sample(nodes, 2)
        results.append(run_trial(G, source, target))

    # Collect points for plotting
    dist_b0, crime_b0 = [], []
    dist_b1, crime_b1 = [], []
    dist_p, crime_p = [], []

    for res in results:
        if res["beta_0"] is not None:
            d, c = res["beta_0"]
            dist_b0.append(d)
            crime_b0.append(c)
        if res["beta_1"] is not None:
            d, c = res["beta_1"]
            dist_b1.append(d)
            crime_b1.append(c)
        if res["pareto"] is not None:
            d, c = res["pareto"]
            dist_p.append(d)
            crime_p.append(c)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print("matplotlib and numpy required. pip install matplotlib numpy", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 6))

    if dist_b0:
        ax.scatter(dist_b0, crime_b0, c="C0", label="β = 0 (crime-averse / safest)", alpha=0.8, s=50, zorder=2)
        if show_line and len(dist_b0) > 1:
            coefs = np.polyfit(dist_b0, crime_b0, 1)
            x_line = np.linspace(min(dist_b0), max(dist_b0), 100)
            ax.plot(x_line, np.polyval(coefs, x_line), c="C0", linestyle="--", linewidth=2)
    if dist_b1:
        ax.scatter(dist_b1, crime_b1, c="C1", label="β = 1 (distance-only)", alpha=0.8, s=50, zorder=2)
        if show_line and len(dist_b1) > 1:
            coefs = np.polyfit(dist_b1, crime_b1, 1)
            x_line = np.linspace(min(dist_b1), max(dist_b1), 100)
            ax.plot(x_line, np.polyval(coefs, x_line), c="C1", linestyle="--", linewidth=2)
    if dist_p:
        ax.scatter(dist_p, crime_p, c="C2", label="Pareto-optimal β", alpha=0.8, s=50, marker="s", zorder=3)
        if show_line and len(dist_p) > 1:
            coefs = np.polyfit(dist_p, crime_p, 1)
            x_line = np.linspace(min(dist_p), max(dist_p), 100)
            ax.plot(x_line, np.polyval(coefs, x_line), c="C2", linestyle="--", linewidth=2)

    ax.set_xlabel("Total distance (m)")
    ax.set_ylabel("Crime exposure (cumulative risk score)")
    ax.set_title(f"Routing comparison: β=0, β=1, and Pareto-optimal β ({n_trials} trials)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = project_root / "algorithms/pareto_beta_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved plot to", out_path)

    if show_stats:
        _print_improvement_statistics(results, n_trials)


if __name__ == "__main__":
    main()
