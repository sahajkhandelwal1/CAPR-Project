"""
Map datapoints (e.g. crime incidents) to graph nodes or edges.

Reusable CLI script: run with different graphs and point CSVs as needed.
Handles coordinate systems automatically: if the graph is in UTM (or any CRS),
points are reprojected to the graph CRS before nearest-node/nearest-edge lookup.

Example:
  python scripts/data_optimization/map_points_to_graph.py \\
    --graph data/graphs/sf_pedestrian_graph_projected.graphml \\
    --points data/processed/filtered_police_data.csv \\
    --lat-col Latitude --lon-col Longitude \\
    --output-csv data/processed/points_to_graph.csv \\
    --mode both
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from shapely import wkt
from tqdm import tqdm


def _parse_crs(crs_value):
    """Parse CRS from graph attribute (may be string like 'EPSG:32610' or dict)."""
    if crs_value is None:
        return "EPSG:4326"
    if isinstance(crs_value, dict):
        return crs_value.get("name", "EPSG:4326")
    return str(crs_value)


def load_graph_and_crs(graph_path):
    """Load NetworkX graph from GraphML and return (G, graph_crs)."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    G = nx.read_graphml(str(path))
    crs = G.graph.get("crs")
    graph_crs = _parse_crs(crs)
    return G, graph_crs


def load_edges_geodataframe(edges_path, graph_crs):
    """Load edges GeoDataFrame from GeoJSON; ensure CRS matches graph."""
    path = Path(edges_path)
    if not path.exists():
        raise FileNotFoundError(f"Edges file not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(graph_crs, inplace=True)
    elif gdf.crs != graph_crs:
        gdf = gdf.to_crs(graph_crs)
    return gdf


def build_edges_from_graph(G, graph_crs):
    """Build edges GeoDataFrame from graph using edge geometry WKT."""
    rows = []
    edges_list = list(G.edges(keys=True))
    for u, v, key in tqdm(edges_list, desc="Building edges from graph"):
        data = G.edges[u, v, key]
        geom_str = data.get("geometry")
        if not geom_str:
            continue
        try:
            geom = wkt.loads(geom_str)
        except Exception:
            continue
        rows.append({"u": u, "v": v, "key": key, "geometry": geom})
    if not rows:
        raise ValueError("No edge geometries found in graph")
    gdf = gpd.GeoDataFrame(rows, crs=graph_crs)
    return gdf


def load_points_gdf(csv_path, lat_col=None, lon_col=None, x_col=None, y_col=None, points_crs="EPSG:4326"):
    """Load points from CSV into GeoDataFrame. Use either (lat, lon) or (x, y) columns."""
    df = pd.read_csv(csv_path)
    if lat_col and lon_col:
        df = df.dropna(subset=[lat_col, lon_col])
        geometry = [Point(x, y) for x, y in zip(df[lon_col], df[lat_col])]
    elif x_col and y_col:
        df = df.dropna(subset=[x_col, y_col])
        geometry = [Point(x, y) for x, y in zip(df[x_col], df[y_col])]
    else:
        raise ValueError("Provide either --lat-col/--lon-col or --x-col/--y-col")
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=points_crs)
    return gdf


def build_nodes_gdf(G, graph_crs):
    """Build nodes GeoDataFrame from graph node x, y."""
    rows = []
    nodes_list = list(G.nodes(data=True))
    for nid, data in tqdm(nodes_list, desc="Building nodes GeoDataFrame"):
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            continue
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            continue
        rows.append({"node_id": nid, "geometry": Point(x, y)})
    gdf = gpd.GeoDataFrame(rows, crs=graph_crs)
    return gdf


def map_points_to_nodes_fast(points_gdf, nodes_gdf, max_distance_m):
    """Use spatial index for nearest node within max_distance_m."""
    tree = nodes_gdf.sindex
    results = []
    for idx, row in tqdm(points_gdf.iterrows(), total=len(points_gdf), desc="Mapping to nodes"):
        pt = row.geometry
        buffered = pt.buffer(max_distance_m)
        possible = list(tree.intersection(buffered.bounds))
        if not possible:
            results.append({"point_id": idx, "node_id": None, "distance_to_node_m": None})
            continue
        sub = nodes_gdf.iloc[possible]
        sub = sub.copy()
        sub["_dist"] = sub.geometry.distance(pt)
        sub = sub[sub["_dist"] <= max_distance_m]
        if sub.empty:
            results.append({"point_id": idx, "node_id": None, "distance_to_node_m": None})
            continue
        best = sub.loc[sub["_dist"].idxmin()]
        results.append({"point_id": idx, "node_id": best["node_id"], "distance_to_node_m": best["_dist"]})
    return pd.DataFrame(results)


def map_points_to_edges_fast(points_gdf, edges_gdf, max_distance_m):
    """Use spatial index for nearest edge within max_distance_m."""
    tree = edges_gdf.sindex
    results = []
    for idx, row in tqdm(points_gdf.iterrows(), total=len(points_gdf), desc="Mapping to edges"):
        pt = row.geometry
        buffered = pt.buffer(max_distance_m)
        possible = list(tree.intersection(buffered.bounds))
        if not possible:
            results.append({"point_id": idx, "edge_u": None, "edge_v": None, "edge_key": None, "distance_to_edge_m": None})
            continue
        sub = edges_gdf.iloc[possible].copy()
        sub["_dist"] = sub.geometry.distance(pt)
        sub = sub[sub["_dist"] <= max_distance_m]
        if sub.empty:
            results.append({"point_id": idx, "edge_u": None, "edge_v": None, "edge_key": None, "distance_to_edge_m": None})
            continue
        best = sub.loc[sub["_dist"].idxmin()]
        u = best.get("u")
        v = best.get("v")
        k = best.get("key", 0)
        if u is None and "u" in best.index:
            u = best["u"]
        if v is None and "v" in best.index:
            v = best["v"]
        results.append({"point_id": idx, "edge_u": u, "edge_v": v, "edge_key": k, "distance_to_edge_m": best["_dist"]})
    return pd.DataFrame(results)


def run(
    graph_path,
    points_path,
    output_csv,
    *,
    lat_col=None,
    lon_col=None,
    x_col=None,
    y_col=None,
    points_crs="EPSG:4326",
    edges_geojson=None,
    max_distance=100.0,
    mode="both",
    output_graph=None,
):
    """Load graph and points, map points to nodes/edges, write CSV and optionally enriched graph."""
    graph_path = Path(graph_path)
    points_path = Path(points_path)
    output_csv = Path(output_csv)

    print("Loading graph...")
    G, graph_crs = load_graph_and_crs(graph_path)
    print(f"  Graph CRS: {graph_crs}, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    if edges_geojson:
        print("Loading edges GeoDataFrame from GeoJSON...")
        edges_gdf = load_edges_geodataframe(edges_geojson, graph_crs)
    else:
        print("Building edges from graph (geometry WKT)...")
        edges_gdf = build_edges_from_graph(G, graph_crs)
    print(f"  Edges: {len(edges_gdf)}")

    print("Building nodes GeoDataFrame...")
    nodes_gdf = build_nodes_gdf(G, graph_crs)
    print(f"  Nodes: {len(nodes_gdf)}")

    print("Loading points CSV...")
    points_gdf = load_points_gdf(
        points_path,
        lat_col=lat_col,
        lon_col=lon_col,
        x_col=x_col,
        y_col=y_col,
        points_crs=points_crs,
    )
    n_before = len(pd.read_csv(points_path))
    points_gdf = points_gdf.to_crs(graph_crs).reset_index(drop=True)
    print(f"  Points with valid coordinates: {len(points_gdf)} (dropped {n_before - len(points_gdf)} with missing coords)")

    out_df = pd.DataFrame({"point_id": range(len(points_gdf))})

    if mode in ("node", "both"):
        print("Mapping points to nearest nodes...")
        node_df = map_points_to_nodes_fast(points_gdf, nodes_gdf, max_distance)
        out_df = out_df.merge(node_df, on="point_id", how="left")
    if mode in ("edge", "both"):
        print("Mapping points to nearest edges...")
        edge_df = map_points_to_edges_fast(points_gdf, edges_gdf, max_distance)
        out_df = out_df.merge(edge_df, on="point_id", how="left")
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote mapping table: {output_csv} ({len(out_df)} rows)")

    if output_graph:
        # Optional: add point_count to nodes/edges and save graph
        if mode in ("node", "both") and "node_id" in out_df.columns:
            from collections import Counter
            node_counts = Counter(out_df["node_id"].dropna().astype(int))
            for nid in G.nodes():
                G.nodes[nid]["point_count"] = node_counts.get(nid, 0)
        if mode in ("edge", "both") and "edge_u" in out_df.columns:
            from collections import Counter
            edge_tuples = list(zip(out_df["edge_u"].dropna(), out_df["edge_v"].dropna(), out_df["edge_key"].fillna(0)))
            edge_counts = Counter((str(a), str(b), int(k)) for a, b, k in edge_tuples)
            for u, v, k in G.edges(keys=True):
                key = int(k) if (k is not None and (not isinstance(k, float) or k == k)) else 0
                G.edges[u, v, k]["point_count"] = edge_counts.get((str(u), str(v), key), 0)
        nx.write_graphml(G, output_graph)
        print(f"Wrote enriched graph: {output_graph}")

    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Map datapoints (e.g. crime) to graph nodes or edges. Handles CRS automatically.",
        epilog="Example: python map_points_to_graph.py --graph data/graphs/sf_pedestrian_graph_projected.graphml --points data/processed/filtered_police_data.csv --lat-col Latitude --lon-col Longitude --output-csv out.csv --mode both",
    )
    parser.add_argument("--graph", required=True, help="Path to GraphML file")
    parser.add_argument("--points", required=True, help="Path to points CSV")
    parser.add_argument("--output-csv", required=True, help="Path to output mapping CSV")
    parser.add_argument("--lat-col", default=None, help="Latitude column (use with --lon-col)")
    parser.add_argument("--lon-col", default=None, help="Longitude column (use with --lat-col)")
    parser.add_argument("--x-col", default=None, help="X column in graph CRS (use with --y-col)")
    parser.add_argument("--y-col", default=None, help="Y column in graph CRS (use with --x-col)")
    parser.add_argument("--points-crs", default="EPSG:4326", help="CRS of point coordinates (default WGS84)")
    parser.add_argument("--edges-geojson", default=None, help="Optional path to edges GeoJSON (faster than parsing graph)")
    parser.add_argument("--max-distance", type=float, default=100.0, help="Max distance (m) to assign point to node/edge (default 100)")
    parser.add_argument("--mode", choices=["node", "edge", "both"], default="both", help="Map to nodes, edges, or both")
    parser.add_argument("--output-graph", default=None, help="Optional path to write graph with point_count on nodes/edges")
    args = parser.parse_args()

    if not args.lat_col and not args.x_col:
        parser.error("Provide either --lat-col/--lon-col or --x-col/--y-col")
    if args.lat_col and not args.lon_col:
        parser.error("Provide both --lat-col and --lon-col")
    if args.lon_col and not args.lat_col:
        parser.error("Provide both --lat-col and --lon-col")
    if args.x_col and not args.y_col:
        parser.error("Provide both --x-col and --y-col")
    if args.y_col and not args.x_col:
        parser.error("Provide both --x-col and --y-col")

    try:
        run(
            args.graph,
            args.points,
            args.output_csv,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            x_col=args.x_col,
            y_col=args.y_col,
            points_crs=args.points_crs,
            edges_geojson=args.edges_geojson,
            max_distance=args.max_distance,
            mode=args.mode,
            output_graph=args.output_graph,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
