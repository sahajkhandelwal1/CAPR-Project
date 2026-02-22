"""
Multi-objective safety-aware routing algorithm.

Balances distance and safety via a scalar combined cost:
  c_e = β * d_e + (1-β) * r_e
where d_e = length_e / max(length), r_e = (safety_score_e - 1) / 99.
β=1 → shortest path; β=0 → safest path.

Expects graph edges to have:
  - length (float, meters)
  - safety_score (float, 1–100; 1=safest). If missing, defaults to 50.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import networkx as nx

# Default safety score when edge has none (mid of 1–100)
_DEFAULT_SAFETY_SCORE = 50.0


def _get_edge_length(edge_data: dict) -> float:
    v = edge_data.get("length")
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _get_edge_safety_score(edge_data: dict) -> float:
    v = edge_data.get("safety_score", _DEFAULT_SAFETY_SCORE)
    if v is None:
        return _DEFAULT_SAFETY_SCORE
    try:
        s = float(v)
        return max(1.0, min(100.0, s))
    except (TypeError, ValueError):
        return _DEFAULT_SAFETY_SCORE


def _max_length(G: nx.Graph) -> float:
    """Max edge length in the graph (for normalization)."""
    out = 0.0
    if G.is_multigraph():
        for _u, _v, key in G.edges(keys=True):
            d = G.edges[_u, _v, key]
            L = _get_edge_length(d)
            if L > out:
                out = L
    else:
        for _u, _v in G.edges():
            L = _get_edge_length(G.edges[_u, _v])
            if L > out:
                out = L
    return max(out, 1.0)  # avoid div by zero


def _edge_cost(length: float, safety_score: float, max_length: float, beta: float) -> float:
    """
    Combined edge cost: c_e = β * d_e + (1-β) * r_e.
    d_e = length / max_length, r_e = (safety_score - 1) / 99.
    """
    d_e = length / max_length
    r_e = (safety_score - 1.0) / 99.0
    return beta * d_e + (1.0 - beta) * r_e


def _graph_with_cost_weights(
    G: nx.Graph,
    max_length: float,
    beta: float,
) -> nx.Graph:
    """Return a copy of G with each edge having a 'cost' attribute (scalar weight)."""
    H = copy.deepcopy(G)
    if H.is_multigraph():
        for u, v, key in list(H.edges(keys=True)):
            d = H.edges[u, v, key]
            L = _get_edge_length(d)
            S = _get_edge_safety_score(d)
            H.edges[u, v, key]["cost"] = _edge_cost(L, S, max_length, beta)
    else:
        for u, v in H.edges():
            d = H.edges[u, v]
            L = _get_edge_length(d)
            S = _get_edge_safety_score(d)
            H.edges[u, v]["cost"] = _edge_cost(L, S, max_length, beta)
    return H


def _path_edge_keys(
    G: nx.Graph,
    path_nodes: list,
    max_length: float,
    beta: float,
) -> list[tuple[Any, Any, Any]]:
    """
    For a node path, return the sequence of edges (u, v, key) that minimize
    combined cost c_e for this beta (same choice as Dijkstra).
    For simple graphs, key is None.
    """
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
                    _get_edge_safety_score(G[u][v][k]),
                    max_length,
                    beta,
                ),
            )
            edges_used.append((u, v, best_key))
        else:
            edges_used.append((u, v, None))
    return edges_used


def _path_metrics(
    G: nx.Graph,
    path_nodes: list,
    max_length: float,
    beta: float,
) -> tuple[float, float, float]:
    """Return (total_distance_m, total_risk_score, combined_cost)."""
    edges_used = _path_edge_keys(G, path_nodes, max_length, beta)
    if len(edges_used) != max(0, len(path_nodes) - 1):
        return 0.0, 0.0, 0.0
    total_length = 0.0
    total_risk = 0.0
    combined_cost = 0.0
    for u, v, key in edges_used:
        d = G.edges[u, v, key] if key is not None else G.edges[u, v]
        L = _get_edge_length(d)
        S = _get_edge_safety_score(d)
        total_length += L
        total_risk += S
        combined_cost += _edge_cost(L, S, max_length, beta)
    return total_length, total_risk, combined_cost


def route(
    G: nx.Graph,
    source: Any,
    target: Any,
    beta: float = 0.5,
) -> dict[str, Any]:
    """
    Compute a single route minimizing combined cost c_e = β*d_e + (1-β)*r_e.

    Parameters
    ----------
    G : NetworkX graph
        Must have edge attributes 'length' (meters) and optionally 'safety_score' (1–100).
    source : node
        Start node.
    target : node
        End node.
    beta : float in [0, 1]
        1.0 = shortest path (distance only), 0.0 = safest path (risk only).

    Returns
    -------
    dict with keys:
        path_nodes : list of nodes
        total_distance_m : float
        total_risk_score : float (sum of safety_score along path)
        combined_cost : float (sum of c_e along path)
        beta_used : float
    """
    beta = max(0.0, min(1.0, float(beta)))
    if source not in G or target not in G:
        return {
            "path_nodes": [],
            "total_distance_m": 0.0,
            "total_risk_score": 0.0,
            "combined_cost": 0.0,
            "beta_used": beta,
        }
    max_length = _max_length(G)
    H = _graph_with_cost_weights(G, max_length, beta)
    try:
        path_nodes = nx.dijkstra_path(H, source, target, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {
            "path_nodes": [],
            "total_distance_m": 0.0,
            "total_risk_score": 0.0,
            "combined_cost": 0.0,
            "beta_used": beta,
        }
    total_distance_m, total_risk_score, combined_cost = _path_metrics(
        G, path_nodes, max_length, beta
    )
    return {
        "path_nodes": path_nodes,
        "total_distance_m": total_distance_m,
        "total_risk_score": total_risk_score,
        "combined_cost": combined_cost,
        "beta_used": beta,
    }


def pareto_frontier(
    G: nx.Graph,
    source: Any,
    target: Any,
    betas: Optional[list[float]] = None,
) -> list[dict[str, Any]]:
    """
    Sweep β and return the distance vs risk tradeoff (approximate Pareto frontier).

    Parameters
    ----------
    G, source, target : as in route()
    betas : list of float in [0, 1], optional
        Default: [0, 0.1, 0.2, ..., 1.0]

    Returns
    -------
    list of dicts, each with keys:
        beta_used, path_nodes, total_distance_m, total_risk_score, combined_cost
    """
    if betas is None:
        betas = [round(i * 0.1, 1) for i in range(11)]
    results = []
    for b in betas:
        r = route(G, source, target, beta=b)
        results.append(r)
    return results
