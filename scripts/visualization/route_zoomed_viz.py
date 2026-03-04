#!/usr/bin/env python3
"""
Zoomed route visualization: single route view with safety-colored segments,
crime hotspots, shortest path, and optimal (safest) path.

Uses the existing network and edge risk scores. Colors segments:
  high crime (risk) = red → orange → yellow → green = safe.
Highlights buffers around crime hotspots. Overlays shortest path (β=1)
Optimal (safest) path uses the Pareto-optimal β (best distance–safety tradeoff), not β=0.

Usage:
  python scripts/visualization/route_zoomed_viz.py --source <node_id> --target <node_id>
  python scripts/visualization/route_zoomed_viz.py --source 0 --target 1000 --output data/graphs/visualizations/route_zoomed.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
import networkx as nx
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely import wkt
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.routing import route, pareto_frontier

# Paths
DEFAULT_GRAPH = PROJECT_ROOT / "data/graphs/sf_pedestrian_graph_projected.graphml"
DEFAULT_EDGE_SCORES = PROJECT_ROOT / "data/processed/edge_risk_scores_enhanced.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/graphs/visualizations/route_zoomed.png"

# Safety score 1–100 (1=safest, 100=most dangerous). Used like routing: safety_score = risk.
DEFAULT_SAFETY = 50.0


def _get_edge_length(edge_data: dict) -> float:
    v = edge_data.get("length")
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _get_safety_score(edge_data: dict) -> float:
    v = edge_data.get("safety_score", DEFAULT_SAFETY)
    if v is None:
        return DEFAULT_SAFETY
    try:
        s = float(v)
        return max(1.0, min(100.0, s))
    except (TypeError, ValueError):
        return DEFAULT_SAFETY


def _max_length(G: nx.Graph) -> float:
    out = 0.0
    if G.is_multigraph():
        for u, v, key in G.edges(keys=True):
            L = _get_edge_length(G.edges[u, v, key])
            if L > out:
                out = L
    else:
        for u, v in G.edges():
            L = _get_edge_length(G.edges[u, v])
            if L > out:
                out = L
    return max(out, 1.0)


def _edge_cost(length: float, safety: float, max_length: float, beta: float) -> float:
    d_e = length / max_length
    r_e = (safety - 1.0) / 99.0
    return beta * d_e + (1.0 - beta) * r_e


def path_edge_keys(G: nx.Graph, path_nodes: list, max_length: float, beta: float):
    """Return list of (u, v, key) for the path (same choice as Dijkstra)."""
    edges_used = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        if not G.has_edge(u, v):
            return []
        if G.is_multigraph():
            best_key = min(
                G[u][v].keys(),
                key=lambda k: _edge_cost(
                    _get_edge_length(G[u][v][k]),
                    _get_safety_score(G[u][v][k]),
                    max_length,
                    beta,
                ),
            )
            edges_used.append((u, v, best_key))
        else:
            edges_used.append((u, v, None))
    return edges_used


def pareto_optimal_beta_and_route(G: nx.Graph, source: Any, target: Any, betas: list[float] | None = None):
    """
    Compute Pareto frontier and return the Pareto-optimal route: the one closest to
    the ideal (min distance, min risk) in normalized [0,1] space.
    Returns (result_dict, beta_used); result_dict has path_nodes, total_distance_m, total_risk_score.
    """
    if betas is None:
        betas = [round(i * 0.05, 2) for i in range(21)]  # 0, 0.05, ..., 1.0
    results = pareto_frontier(G, source, target, betas=betas)
    results = [r for r in results if r["path_nodes"]]
    if not results:
        return None, None
    dists = np.array([r["total_distance_m"] for r in results])
    risks = np.array([r["total_risk_score"] for r in results])
    d_min, d_max = dists.min(), dists.max()
    r_min, r_max = risks.min(), risks.max()
    d_span = (d_max - d_min) or 1.0
    r_span = (r_max - r_min) or 1.0
    d_norm = (dists - d_min) / d_span
    r_norm = (risks - r_min) / r_span
    # Closest to ideal (0, 0)
    scores = np.sqrt(d_norm ** 2 + r_norm ** 2)
    idx = int(np.argmin(scores))
    return results[idx], float(results[idx]["beta_used"])


def load_graph(path: Path) -> nx.Graph:
    with tqdm(total=1, desc="Loading graph", unit="file", leave=False) as pbar:
        G = nx.read_graphml(str(path))
        pbar.update(1)
    return G


def ensure_numeric_ids(G: nx.Graph) -> nx.Graph:
    try:
        first = next(iter(G.nodes()))
        if isinstance(first, str) and first.isdigit():
            mapping = {n: int(n) for n in G.nodes()}
            return nx.relabel_nodes(G, mapping)
    except StopIteration:
        pass
    return G


def load_edge_scores_csv(csv_path: Path) -> dict[tuple[int, int, int], float]:
    """Returns (u, v, key) -> safety_score in [1, 100]."""
    lookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in tqdm(rows, desc="Loading edge scores", unit="rows", leave=False):
        u, v = int(row["u"]), int(row["v"])
        key = int(row["key"]) if "key" in row else 0
        try:
            r = float(row["risk_score_enhanced"])
        except (ValueError, KeyError):
            continue
        safety = max(1.0, min(100.0, r))
        lookup[(u, v, key)] = safety
    return lookup


def attach_edge_scores(G: nx.Graph, edge_scores: dict) -> None:
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


def build_edges_gdf(G: nx.Graph) -> gpd.GeoDataFrame:
    """Build edges GeoDataFrame with geometry and risk (safety_score)."""
    rows = []
    multigraph = G.is_multigraph()
    edge_iter = list(G.edges(keys=True) if multigraph else [(a, b, None) for a, b in G.edges()])

    for item in tqdm(edge_iter, desc="Building edge geometries", unit="edges", leave=False):
        if multigraph:
            u, v, key = item
            ed = G.edges[u, v, key]
        else:
            u, v, key = item[0], item[1], None
            ed = G.edges[u, v]
        geom = None
        geom_str = ed.get("geometry")
        if geom_str:
            try:
                geom = wkt.loads(geom_str)
            except Exception:
                pass
        if geom is None:
            try:
                x1, y1 = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
                x2, y2 = float(G.nodes[v]["x"]), float(G.nodes[v]["y"])
                geom = LineString([(x1, y1), (x2, y2)])
            except (KeyError, TypeError, ValueError):
                continue
        risk = _get_safety_score(ed)
        k = key if multigraph else 0
        rows.append({"u": u, "v": v, "key": k, "risk_score": risk, "geometry": geom})

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:32610")
    return gdf


def bbox_from_paths(G: nx.Graph, path_edges_list: list, padding_m: float = 200.0):
    """Compute (minx, miny, maxx, maxy) encompassing all path edges + padding."""
    all_coords = []
    for path_edges in path_edges_list:
        for u, v, key in path_edges:
            ed = G.edges[u, v, key] if G.is_multigraph() else G.edges[u, v]
            geom_str = ed.get("geometry")
            if geom_str:
                try:
                    geom = wkt.loads(geom_str)
                    all_coords.extend(geom.coords)
                except Exception:
                    pass
            try:
                all_coords.append((float(G.nodes[u]["x"]), float(G.nodes[u]["y"])))
                all_coords.append((float(G.nodes[v]["x"]), float(G.nodes[v]["y"])))
            except (KeyError, TypeError, ValueError):
                pass
    if not all_coords:
        return 0, 0, 500, 500
    xs, ys = zip(*all_coords)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return (
        minx - padding_m,
        miny - padding_m,
        maxx + padding_m,
        maxy + padding_m,
    )


def plot_route_zoomed(
    G: nx.Graph,
    edges_gdf: gpd.GeoDataFrame,
    shortest_path_edges: list,
    optimal_path_edges: list,
    bbox: tuple,
    output_path: Path,
    source_node: int,
    target_node: int,
    hotspot_risk_threshold: float = 75.0,
    hotspot_buffer_m: float = 25.0,
):
    """Draw zoomed map: safety-colored segments, hotspots, shortest and optimal routes."""
    minx, miny, maxx, maxy = bbox

    # Clip edges to bbox (by bounds)
    bounds = edges_gdf.bounds
    in_bbox = (
        (bounds["maxx"] >= minx)
        & (bounds["minx"] <= maxx)
        & (bounds["maxy"] >= miny)
        & (bounds["miny"] <= maxy)
    )
    plot_edges = edges_gdf.loc[in_bbox].copy()

    # Safety colormap: high risk = red → orange → yellow → green = safe (risk 1)
    # risk_score 1–100 -> color
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "safety", ["#2E8B57", "#90EE90", "#FFFF00", "#FFA500", "#FF4500", "#8B0000"], N=256
    )
    vmin, vmax = 1.0, 100.0

    fig, ax = plt.subplots(1, 1, figsize=(14, 12), facecolor="white")
    # Reserve extra space at bottom so colorbar and its numbers are not cut off
    fig.subplots_adjust(bottom=0.14)

    # 1) Hotspots: buffer high-risk segments
    high_risk = plot_edges[plot_edges["risk_score"] >= hotspot_risk_threshold]
    if len(high_risk) > 0:
        hotspots_geom = high_risk.geometry.buffer(hotspot_buffer_m).union_all()
        if hotspots_geom is not None and not hotspots_geom.is_empty:
            try:
                if hotspots_geom.geom_type == "Polygon":
                    patch = MplPolygon(
                        list(hotspots_geom.exterior.coords),
                        facecolor="red",
                        edgecolor="darkred",
                        alpha=0.18,
                        zorder=1,
                    )
                    ax.add_patch(patch)
                else:
                    for g in (hotspots_geom.geoms if hasattr(hotspots_geom, "geoms") else [hotspots_geom]):
                        if hasattr(g, "exterior") and g.exterior is not None:
                            patch = MplPolygon(
                                list(g.exterior.coords),
                                facecolor="red",
                                edgecolor="darkred",
                                alpha=0.18,
                                zorder=1,
                            )
                            ax.add_patch(patch)
            except Exception:
                pass

    # 2) Street segments colored by safety (red = high crime, green = safe) — lower opacity so routes stand out
    plot_edges.plot(
        ax=ax,
        column="risk_score",
        cmap=cmap,
        linewidth=2.0,
        alpha=0.55,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        legend=True,
        legend_kwds={
            "shrink": 0.7,
            "label": "Crime risk (1=safest, 100=highest)",
            "orientation": "horizontal",
            "pad": 0.06,
        },
    )

    # 3) Shortest path (β=1)
    for i, (u, v, key) in enumerate(shortest_path_edges):
        ed = G.edges[u, v, key] if G.is_multigraph() else G.edges[u, v]
        geom_str = ed.get("geometry")
        if geom_str:
            try:
                geom = wkt.loads(geom_str)
            except Exception:
                geom = LineString([
                    (float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
                    (float(G.nodes[v]["x"]), float(G.nodes[v]["y"])),
                ])
        else:
            geom = LineString([
                (float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
                (float(G.nodes[v]["x"]), float(G.nodes[v]["y"])),
            ])
        xs, ys = geom.xy
        ax.plot(xs, ys, color="blue", linewidth=4, linestyle="--", alpha=0.95, zorder=4, label="Shortest path" if i == 0 else None)

    # 4) Optimal (safest) path (β=0)
    for i, (u, v, key) in enumerate(optimal_path_edges):
        ed = G.edges[u, v, key] if G.is_multigraph() else G.edges[u, v]
        geom_str = ed.get("geometry")
        if geom_str:
            try:
                geom = wkt.loads(geom_str)
            except Exception:
                geom = LineString([
                    (float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
                    (float(G.nodes[v]["x"]), float(G.nodes[v]["y"])),
                ])
        else:
            geom = LineString([
                (float(G.nodes[u]["x"]), float(G.nodes[u]["y"])),
                (float(G.nodes[v]["x"]), float(G.nodes[v]["y"])),
            ])
        xs, ys = geom.xy
        ax.plot(xs, ys, color="purple", linewidth=4, linestyle="-", alpha=0.95, zorder=5, label="Optimal (Pareto) path" if i == 0 else None)

    # Start/end markers (high zorder so they stay on top; legend placed so it won't cover them)
    try:
        sx, sy = float(G.nodes[source_node]["x"]), float(G.nodes[source_node]["y"])
        tx, ty = float(G.nodes[target_node]["x"]), float(G.nodes[target_node]["y"])
        ax.scatter([sx], [sy], s=200, c="lime", edgecolors="black", linewidths=2, zorder=8, label="Start")
        ax.scatter([tx], [ty], s=200, c="fuchsia", edgecolors="black", linewidths=2, zorder=8, label="End")
    except (KeyError, TypeError, ValueError):
        pass

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    # Remove the "1e6" offset text near the top left (scientific notation for large UTM coordinates)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.set_xlabel("Easting (m, UTM 10N)")
    ax.set_ylabel("Northing (m, UTM 10N)")
    ax.set_title("Shortest vs optimal path", fontsize=14, fontweight="bold")
    # Place legend outside plot area (right side); single column so Start and End dots don't overlap
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.95, ncol=1, markerscale=0.6, handletextpad=0.8)
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    tqdm.write(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Zoomed route visualization: shortest + optimal path on safety-colored network with hotspots")
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH), help="GraphML file")
    parser.add_argument("--edge-scores", default=str(DEFAULT_EDGE_SCORES), help="CSV with u,v,key,risk_score_enhanced")
    parser.add_argument("--source", type=int, default=None, help="Start node id (default: auto-pick)")
    parser.add_argument("--target", type=int, default=None, help="End node id (default: auto-pick)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path")
    parser.add_argument("--hotspot-threshold", type=float, default=75.0, help="Risk >= this for hotspot buffer")
    parser.add_argument("--hotspot-buffer", type=float, default=25.0, help="Hotspot buffer radius (m)")
    parser.add_argument("--padding", type=float, default=200.0, help="Bbox padding around routes (m)")
    args = parser.parse_args()

    graph_path = Path(args.graph)
    edge_scores_path = Path(args.edge_scores)
    if not graph_path.exists():
        print(f"Graph not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    tqdm.write("Loading graph...")
    G = load_graph(graph_path)
    G = ensure_numeric_ids(G)

    if edge_scores_path.exists():
        tqdm.write("Loading edge scores...")
        edge_scores = load_edge_scores_csv(edge_scores_path)
        attach_edge_scores(G, edge_scores)
    else:
        tqdm.write("Edge scores not found; using default risk per edge.", file=sys.stderr)

    # Auto-pick source/target if not provided: use two nodes that are connected and give a reasonable route
    nodes = list(G.nodes())
    if args.source is None or args.target is None:
        # Pick first node and a node ~15% through the list so route is not trivial
        args.source = int(nodes[0]) if nodes else 0
        idx_target = min(len(nodes) // 6, len(nodes) - 1) if len(nodes) > 1 else 0
        args.target = int(nodes[idx_target]) if nodes else 0
        tqdm.write(f"Using nodes: source={args.source}, target={args.target}")

    if args.source not in G or args.target not in G:
        print(f"Source {args.source} or target {args.target} not in graph.", file=sys.stderr)
        sys.exit(1)

    tqdm.write("Computing shortest path (β=1)...")
    res_shortest = route(G, args.source, args.target, beta=1.0)
    tqdm.write("Computing Pareto frontier and selecting Pareto-optimal path...")
    res_optimal, beta_optimal = pareto_optimal_beta_and_route(G, args.source, args.target)
    if res_optimal is None or beta_optimal is None:
        tqdm.write("No path found for optimal; falling back to β=0.", file=sys.stderr)
        res_optimal = route(G, args.source, args.target, beta=0.0)
        beta_optimal = 0.0

    if not res_shortest["path_nodes"] or not res_optimal["path_nodes"]:
        print("No path found between source and target.", file=sys.stderr)
        sys.exit(1)

    max_length = _max_length(G)
    shortest_edges = path_edge_keys(G, res_shortest["path_nodes"], max_length, 1.0)
    optimal_edges = path_edge_keys(G, res_optimal["path_nodes"], max_length, beta_optimal)

    bbox = bbox_from_paths(G, [shortest_edges, optimal_edges], padding_m=args.padding)
    tqdm.write("Building edges GeoDataFrame...")
    edges_gdf = build_edges_gdf(G)

    tqdm.write("Drawing map...")
    plot_route_zoomed(
        G,
        edges_gdf,
        shortest_edges,
        optimal_edges,
        bbox,
        Path(args.output),
        args.source,
        args.target,
        hotspot_risk_threshold=args.hotspot_threshold,
        hotspot_buffer_m=args.hotspot_buffer,
    )


if __name__ == "__main__":
    main()
