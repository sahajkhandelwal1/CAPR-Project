"""
Multi-objective safety-aware routing: public API.

Use:
  from algorithms.algorithm import route, pareto_frontier
  result = route(G, source, target, beta=0.5)
  results = pareto_frontier(G, source, target)
"""

from .routing import pareto_frontier, route

__all__ = ["route", "pareto_frontier"]
