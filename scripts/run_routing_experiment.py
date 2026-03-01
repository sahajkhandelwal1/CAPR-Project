#!/usr/bin/env python3
"""
Run experimental routing analysis with real San Francisco crime data.
This script loads the actual graph with crime-enhanced safety scores and runs the experiment.
"""

import os
import sys
import pickle
import networkx as nx
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from algorithms.experimental_routing_analysis import RoutingExperiment


def load_sf_graph() -> nx.Graph:
    """
    Load the San Francisco graph with crime-enhanced safety scores.
    This function should load your actual processed graph.
    """
    
    # Check for processed graph files
    graph_paths = [
        "data/processed/sf_graph_with_crime_scores.pickle",
        "data/processed/sf_graph_with_safety.gpickle",
        "data/processed/enhanced_risk_graph.pickle"
    ]
    
    for graph_path in graph_paths:
        if os.path.exists(graph_path):
            print(f"Loading graph from: {graph_path}")
            if graph_path.endswith('.pickle'):
                with open(graph_path, 'rb') as f:
                    return pickle.load(f)
            elif graph_path.endswith('.gpickle'):
                return nx.read_gpickle(graph_path)
    
    print("No processed graph found. You may need to:")
    print("1. Run the crime aggregation and risk diffusion scripts first")
    print("2. Ensure the graph has 'length' and 'safety_score' edge attributes")
    print("3. Update the graph_paths list with your actual file paths")
    
    return None


def prepare_graph_for_routing(G: nx.Graph) -> nx.Graph:
    """
    Prepare the graph for routing by ensuring required attributes exist.
    """
    
    if G is None:
        return None
    
    print("Preparing graph for routing analysis...")
    
    # Check for required edge attributes
    sample_edge = list(G.edges(data=True))[0][2] if G.edges() else {}
    
    has_length = 'length' in sample_edge
    has_safety = 'safety_score' in sample_edge
    
    print(f"Graph has length attribute: {has_length}")
    print(f"Graph has safety_score attribute: {has_safety}")
    
    # If missing attributes, you might need to process the graph
    if not has_length:
        print("Warning: Graph missing 'length' attribute. Adding default lengths...")
        for u, v, data in G.edges(data=True):
            # Use a default length if not present
            data['length'] = data.get('length', 100.0)
    
    if not has_safety:
        print("Warning: Graph missing 'safety_score' attribute. Adding default scores...")
        for u, v, data in G.edges(data=True):
            # Use default safety score if not present
            data['safety_score'] = data.get('safety_score', 50.0)
    
    # Ensure the graph is connected (at least the largest component)
    if not nx.is_connected(G):
        print("Graph is not connected. Using largest connected component...")
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def run_sf_routing_experiment():
    """Run the routing experiment on San Francisco data."""
    
    print("Starting San Francisco Crime-Aware Routing Experiment")
    print("="*60)
    
    # Load the graph
    sf_graph = load_sf_graph()
    
    if sf_graph is None:
        print("\nCreating sample graph for demonstration...")
        from algorithms.experimental_routing_analysis import create_sample_graph
        sf_graph = create_sample_graph()
        print("Using sample graph for analysis.")
    
    # Prepare graph
    sf_graph = prepare_graph_for_routing(sf_graph)
    
    if sf_graph is None:
        print("Failed to prepare graph for analysis.")
        return
    
    # Run experiment with 150 trials
    print(f"\nRunning experimental analysis...")
    experiment = RoutingExperiment(sf_graph, num_trials=150)
    
    # Execute the experiment
    experiment.run_experiment()
    
    # Generate comprehensive outputs
    print("\nGenerating results and visualizations...")
    experiment.print_summary()
    experiment.save_results("data/processed/routing_experiments")
    experiment.generate_visualizations("visualization/routing_experiments")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    
    print("\nOutput files generated:")
    print("- data/processed/routing_experiments/routing_experiment_results.csv")
    print("- data/processed/routing_experiments/pareto_experiment_results.json") 
    print("- data/processed/routing_experiments/routing_experiment_summary.json")
    print("- visualization/routing_experiments/*.png")


if __name__ == "__main__":
    run_sf_routing_experiment()
