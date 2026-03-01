#!/usr/bin/env python3
"""
Experimental analysis of CAPR routing algorithms.
Conducts ~150 trials comparing different routing strategies:
- Safest route (beta=0)
- Shortest route (beta=1) 
- Pareto optimal routes (various beta values)

Generates statistics on safety improvements, distance trade-offs, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import random
import json
from pathlib import Path
import networkx as nx
from algorithms.routing import route, pareto_frontier
import time

class RoutingExperiment:
    """Experimental framework for routing algorithm analysis."""
    
    def __init__(self, graph: nx.Graph, num_trials: int = 150):
        """
        Initialize the experiment.
        
        Parameters:
        - graph: NetworkX graph with safety scores and lengths
        - num_trials: Number of random origin-destination pairs to test
        """
        self.graph = graph
        self.num_trials = num_trials
        self.results = []
        self.pareto_results = []
        self.summary_stats = {}
        
        # Get all nodes for random sampling
        self.nodes = list(graph.nodes())
        
        # Beta values for Pareto analysis
        self.beta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
    def generate_random_od_pairs(self) -> List[Tuple[Any, Any]]:
        """Generate random origin-destination pairs for testing."""
        od_pairs = []
        for i in range(self.num_trials):
            # Ensure origin != destination and both are connected
            while True:
                origin = random.choice(self.nodes)
                destination = random.choice(self.nodes)
                if origin != destination:
                    # Quick connectivity check
                    try:
                        nx.shortest_path(self.graph, origin, destination)
                        od_pairs.append((origin, destination))
                        break
                    except nx.NetworkXNoPath:
                        continue
        return od_pairs
    
    def run_single_trial(self, origin: Any, destination: Any) -> Dict[str, Any]:
        """Run a single trial comparing different routing strategies."""
        
        trial_result = {
            'trial_id': len(self.results),
            'origin': origin,
            'destination': destination,
        }
        
        # 1. Safest route (beta = 0)
        safest = route(self.graph, origin, destination, beta=0.0)
        trial_result['safest_distance'] = safest['total_distance_m']
        trial_result['safest_risk'] = safest['total_risk_score']
        trial_result['safest_nodes'] = len(safest['path_nodes'])
        
        # 2. Shortest route (beta = 1)
        shortest = route(self.graph, origin, destination, beta=1.0)
        trial_result['shortest_distance'] = shortest['total_distance_m']
        trial_result['shortest_risk'] = shortest['total_risk_score']
        trial_result['shortest_nodes'] = len(shortest['path_nodes'])
        
        # 3. Balanced route (beta = 0.5)
        balanced = route(self.graph, origin, destination, beta=0.5)
        trial_result['balanced_distance'] = balanced['total_distance_m']
        trial_result['balanced_risk'] = balanced['total_risk_score']
        trial_result['balanced_nodes'] = len(balanced['path_nodes'])
        
        # 4. Calculate improvements and trade-offs
        if shortest['total_distance_m'] > 0:
            trial_result['distance_increase_safest'] = (
                (safest['total_distance_m'] - shortest['total_distance_m']) / 
                shortest['total_distance_m'] * 100
            )
        else:
            trial_result['distance_increase_safest'] = 0
            
        if shortest['total_risk_score'] > 0:
            trial_result['risk_reduction_safest'] = (
                (shortest['total_risk_score'] - safest['total_risk_score']) / 
                shortest['total_risk_score'] * 100
            )
        else:
            trial_result['risk_reduction_safest'] = 0
            
        # Balanced route comparisons
        if shortest['total_distance_m'] > 0:
            trial_result['distance_increase_balanced'] = (
                (balanced['total_distance_m'] - shortest['total_distance_m']) / 
                shortest['total_distance_m'] * 100
            )
        else:
            trial_result['distance_increase_balanced'] = 0
            
        if shortest['total_risk_score'] > 0:
            trial_result['risk_reduction_balanced'] = (
                (shortest['total_risk_score'] - balanced['total_risk_score']) / 
                shortest['total_risk_score'] * 100
            )
        else:
            trial_result['risk_reduction_balanced'] = 0
            
        # 5. Efficiency metrics
        trial_result['safest_efficiency'] = (
            safest['total_distance_m'] / safest['total_risk_score'] 
            if safest['total_risk_score'] > 0 else 0
        )
        trial_result['shortest_efficiency'] = (
            shortest['total_distance_m'] / shortest['total_risk_score'] 
            if shortest['total_risk_score'] > 0 else 0
        )
        trial_result['balanced_efficiency'] = (
            balanced['total_distance_m'] / balanced['total_risk_score'] 
            if balanced['total_risk_score'] > 0 else 0
        )
        
        return trial_result
    
    def run_pareto_analysis(self, origin: Any, destination: Any) -> Dict[str, Any]:
        """Run Pareto frontier analysis for a single OD pair."""
        
        pareto_result = {
            'trial_id': len(self.pareto_results),
            'origin': origin,
            'destination': destination,
            'pareto_routes': []
        }
        
        # Get Pareto frontier
        frontier = pareto_frontier(self.graph, origin, destination, self.beta_values)
        
        for route_result in frontier:
            if route_result['path_nodes']:  # Valid route
                pareto_route = {
                    'beta': route_result['beta_used'],
                    'distance': route_result['total_distance_m'],
                    'risk': route_result['total_risk_score'],
                    'cost': route_result['combined_cost'],
                    'nodes': len(route_result['path_nodes'])
                }
                pareto_result['pareto_routes'].append(pareto_route)
        
        return pareto_result
    
    def run_experiment(self):
        """Run the complete experiment."""
        print(f"Starting routing experiment with {self.num_trials} trials...")
        
        # Generate random OD pairs
        od_pairs = self.generate_random_od_pairs()
        
        # Run trials
        for i, (origin, destination) in enumerate(od_pairs):
            if i % 25 == 0:
                print(f"Progress: {i}/{self.num_trials} trials completed")
            
            # Single trial analysis
            trial_result = self.run_single_trial(origin, destination)
            self.results.append(trial_result)
            
            # Pareto analysis for every 5th trial (to manage computation)
            if i % 5 == 0:
                pareto_result = self.run_pareto_analysis(origin, destination)
                self.pareto_results.append(pareto_result)
        
        print(f"Experiment completed: {len(self.results)} trials, {len(self.pareto_results)} Pareto analyses")
        
        # Calculate summary statistics
        self.calculate_summary_stats()
    
    def calculate_summary_stats(self):
        """Calculate comprehensive summary statistics."""
        df = pd.DataFrame(self.results)
        
        self.summary_stats = {
            'total_trials': len(self.results),
            'successful_trials': len([r for r in self.results if r['safest_distance'] > 0]),
            
            # Distance statistics
            'avg_distance_increase_safest': df['distance_increase_safest'].mean(),
            'median_distance_increase_safest': df['distance_increase_safest'].median(),
            'std_distance_increase_safest': df['distance_increase_safest'].std(),
            'max_distance_increase_safest': df['distance_increase_safest'].max(),
            
            'avg_distance_increase_balanced': df['distance_increase_balanced'].mean(),
            'median_distance_increase_balanced': df['distance_increase_balanced'].median(),
            
            # Risk reduction statistics
            'avg_risk_reduction_safest': df['risk_reduction_safest'].mean(),
            'median_risk_reduction_safest': df['risk_reduction_safest'].median(),
            'std_risk_reduction_safest': df['risk_reduction_safest'].std(),
            'max_risk_reduction_safest': df['risk_reduction_safest'].max(),
            
            'avg_risk_reduction_balanced': df['risk_reduction_balanced'].mean(),
            'median_risk_reduction_balanced': df['risk_reduction_balanced'].median(),
            
            # Categorical analysis
            'trials_with_risk_reduction': len(df[df['risk_reduction_safest'] > 0]),
            'trials_with_significant_risk_reduction': len(df[df['risk_reduction_safest'] > 10]),
            'trials_with_minor_distance_increase': len(df[df['distance_increase_safest'] < 20]),
            'trials_with_balanced_advantage': len(df[
                (df['risk_reduction_balanced'] > 5) & (df['distance_increase_balanced'] < 15)
            ]),
            
            # Efficiency analysis
            'avg_safest_efficiency': df['safest_efficiency'].mean(),
            'avg_shortest_efficiency': df['shortest_efficiency'].mean(),
            'avg_balanced_efficiency': df['balanced_efficiency'].mean(),
        }
        
        # Calculate percentages
        total = self.summary_stats['successful_trials']
        if total > 0:
            self.summary_stats['pct_with_risk_reduction'] = (
                self.summary_stats['trials_with_risk_reduction'] / total * 100
            )
            self.summary_stats['pct_with_significant_risk_reduction'] = (
                self.summary_stats['trials_with_significant_risk_reduction'] / total * 100
            )
            self.summary_stats['pct_with_minor_distance_increase'] = (
                self.summary_stats['trials_with_minor_distance_increase'] / total * 100
            )
            self.summary_stats['pct_with_balanced_advantage'] = (
                self.summary_stats['trials_with_balanced_advantage'] / total * 100
            )
    
    def generate_visualizations(self, output_dir: str = "visualization/routing_experiments"):
        """Generate comprehensive visualizations of the experimental results."""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distance vs Risk Trade-off Scatter Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot different route types
        ax.scatter(df['shortest_distance'], df['shortest_risk'], 
                  alpha=0.6, label='Shortest Route (β=1)', s=50, color='red')
        ax.scatter(df['safest_distance'], df['safest_risk'], 
                  alpha=0.6, label='Safest Route (β=0)', s=50, color='green')
        ax.scatter(df['balanced_distance'], df['balanced_risk'], 
                  alpha=0.6, label='Balanced Route (β=0.5)', s=50, color='blue')
        
        ax.set_xlabel('Total Distance (meters)')
        ax.set_ylabel('Total Risk Score')
        ax.set_title('Distance vs Risk Trade-off Across All Trials')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distance_vs_risk_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Risk Reduction Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(df['risk_reduction_safest'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax1.set_xlabel('Risk Reduction (%)')
        ax1.set_ylabel('Number of Trials')
        ax1.set_title('Risk Reduction: Safest vs Shortest Route')
        ax1.axvline(df['risk_reduction_safest'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["risk_reduction_safest"].mean():.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(df['risk_reduction_balanced'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Risk Reduction (%)')
        ax2.set_ylabel('Number of Trials')
        ax2.set_title('Risk Reduction: Balanced vs Shortest Route')
        ax2.axvline(df['risk_reduction_balanced'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["risk_reduction_balanced"].mean():.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'risk_reduction_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Distance Increase Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(df['distance_increase_safest'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_xlabel('Distance Increase (%)')
        ax1.set_ylabel('Number of Trials')
        ax1.set_title('Distance Increase: Safest vs Shortest Route')
        ax1.axvline(df['distance_increase_safest'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["distance_increase_safest"].mean():.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(df['distance_increase_balanced'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax2.set_xlabel('Distance Increase (%)')
        ax2.set_ylabel('Number of Trials')
        ax2.set_title('Distance Increase: Balanced vs Shortest Route')
        ax2.axvline(df['distance_increase_balanced'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["distance_increase_balanced"].mean():.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'distance_increase_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Pareto Frontier Analysis (if available)
        if self.pareto_results:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Plot several Pareto frontiers
            for i, pareto_result in enumerate(self.pareto_results[:10]):  # Show first 10
                routes = pareto_result['pareto_routes']
                if routes:
                    distances = [r['distance'] for r in routes]
                    risks = [r['risk'] for r in routes]
                    ax.plot(distances, risks, 'o-', alpha=0.6, label=f'Trial {i+1}')
            
            ax.set_xlabel('Total Distance (meters)')
            ax.set_ylabel('Total Risk Score')
            ax.set_title('Pareto Frontiers: Sample Trials')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'pareto_frontiers_sample.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Summary Statistics Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Box plot of key metrics
        metrics_data = [
            df['risk_reduction_safest'],
            df['risk_reduction_balanced'],
            df['distance_increase_safest'],
            df['distance_increase_balanced']
        ]
        metrics_labels = ['Risk Reduction\n(Safest)', 'Risk Reduction\n(Balanced)', 
                         'Distance Increase\n(Safest)', 'Distance Increase\n(Balanced)']
        
        ax1.boxplot(metrics_data, labels=metrics_labels)
        ax1.set_title('Distribution of Key Metrics (%)')
        ax1.grid(True, alpha=0.3)
        
        # Efficiency comparison
        efficiency_data = [df['shortest_efficiency'], df['balanced_efficiency'], df['safest_efficiency']]
        efficiency_labels = ['Shortest', 'Balanced', 'Safest']
        ax2.boxplot(efficiency_data, labels=efficiency_labels)
        ax2.set_title('Route Efficiency (Distance/Risk)')
        ax2.grid(True, alpha=0.3)
        
        # Categorical analysis
        categories = ['Risk Reduction', 'Significant Risk\nReduction (>10%)', 
                     'Minor Distance\nIncrease (<20%)', 'Balanced\nAdvantage']
        percentages = [
            self.summary_stats['pct_with_risk_reduction'],
            self.summary_stats['pct_with_significant_risk_reduction'],
            self.summary_stats['pct_with_minor_distance_increase'],
            self.summary_stats['pct_with_balanced_advantage']
        ]
        
        bars = ax3.bar(categories, percentages, color=['green', 'darkgreen', 'orange', 'blue'])
        ax3.set_ylabel('Percentage of Trials (%)')
        ax3.set_title('Success Rates by Category')
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        # Route length comparison
        ax4.scatter(df['shortest_nodes'], df['safest_nodes'], alpha=0.6)
        ax4.plot([df['shortest_nodes'].min(), df['shortest_nodes'].max()], 
                [df['shortest_nodes'].min(), df['shortest_nodes'].max()], 'r--', alpha=0.7)
        ax4.set_xlabel('Shortest Route Length (nodes)')
        ax4.set_ylabel('Safest Route Length (nodes)')
        ax4.set_title('Route Length Comparison')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_path}")
    
    def save_results(self, output_dir: str = "data/processed"):
        """Save experimental results and summary statistics."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        df = pd.DataFrame(self.results)
        df.to_csv(output_path / 'routing_experiment_results.csv', index=False)
        
        # Save Pareto results
        with open(output_path / 'pareto_experiment_results.json', 'w') as f:
            json.dump(self.pareto_results, f, indent=2, default=str)
        
        # Save summary statistics
        with open(output_path / 'routing_experiment_summary.json', 'w') as f:
            json.dump(self.summary_stats, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def print_summary(self):
        """Print a comprehensive summary of experimental results."""
        
        print("\n" + "="*80)
        print("CAPR ROUTING ALGORITHM EXPERIMENTAL ANALYSIS")
        print("="*80)
        
        print(f"\nEXPERIMENT OVERVIEW:")
        print(f"  Total trials conducted: {self.summary_stats['total_trials']}")
        print(f"  Successful trials: {self.summary_stats['successful_trials']}")
        print(f"  Pareto analyses: {len(self.pareto_results)}")
        
        print(f"\nSAFETY IMPROVEMENTS (Safest vs Shortest Route):")
        print(f"  Average risk reduction: {self.summary_stats['avg_risk_reduction_safest']:.1f}%")
        print(f"  Median risk reduction: {self.summary_stats['median_risk_reduction_safest']:.1f}%")
        print(f"  Maximum risk reduction: {self.summary_stats['max_risk_reduction_safest']:.1f}%")
        print(f"  Trials with any risk reduction: {self.summary_stats['pct_with_risk_reduction']:.1f}%")
        print(f"  Trials with significant risk reduction (>10%): {self.summary_stats['pct_with_significant_risk_reduction']:.1f}%")
        
        print(f"\nDISTANCE TRADE-OFFS (Safest vs Shortest Route):")
        print(f"  Average distance increase: {self.summary_stats['avg_distance_increase_safest']:.1f}%")
        print(f"  Median distance increase: {self.summary_stats['median_distance_increase_safest']:.1f}%")
        print(f"  Maximum distance increase: {self.summary_stats['max_distance_increase_safest']:.1f}%")
        print(f"  Trials with minor distance increase (<20%): {self.summary_stats['pct_with_minor_distance_increase']:.1f}%")
        
        print(f"\nBALANCED ROUTING (β=0.5):")
        print(f"  Average risk reduction: {self.summary_stats['avg_risk_reduction_balanced']:.1f}%")
        print(f"  Average distance increase: {self.summary_stats['avg_distance_increase_balanced']:.1f}%")
        print(f"  Trials with balanced advantage: {self.summary_stats['pct_with_balanced_advantage']:.1f}%")
        
        print(f"\nROUTE EFFICIENCY:")
        print(f"  Shortest route efficiency: {self.summary_stats['avg_shortest_efficiency']:.3f}")
        print(f"  Balanced route efficiency: {self.summary_stats['avg_balanced_efficiency']:.3f}")
        print(f"  Safest route efficiency: {self.summary_stats['avg_safest_efficiency']:.3f}")
        
        print(f"\nKEY FINDINGS:")
        
        # Generate insights based on data
        if self.summary_stats['pct_with_risk_reduction'] > 80:
            print(f"  ✓ CAPR consistently improves safety in {self.summary_stats['pct_with_risk_reduction']:.0f}% of routes")
        
        if self.summary_stats['avg_distance_increase_safest'] < 30:
            print(f"  ✓ Safety improvements come with modest distance increases (avg: {self.summary_stats['avg_distance_increase_safest']:.0f}%)")
        
        if self.summary_stats['pct_with_balanced_advantage'] > 60:
            print(f"  ✓ Balanced routing (β=0.5) offers good trade-offs in {self.summary_stats['pct_with_balanced_advantage']:.0f}% of cases")
        
        if self.summary_stats['avg_safest_efficiency'] > self.summary_stats['avg_shortest_efficiency']:
            print(f"  ✓ Safest routes are more efficient overall (better distance/risk ratio)")
        
        print("\n" + "="*80)


def create_sample_graph() -> nx.Graph:
    """Create a sample graph for testing (when real graph not available)."""
    
    # Create a grid graph as base
    G = nx.grid_2d_graph(20, 20)
    
    # Convert to regular graph with string node names
    G = nx.convert_node_labels_to_integers(G)
    
    # Add random lengths and safety scores
    np.random.seed(42)
    for u, v in G.edges():
        # Random length between 50-500 meters
        G.edges[u, v]['length'] = np.random.uniform(50, 500)
        
        # Safety scores: mostly safe (20-40) with some dangerous areas (60-90)
        if np.random.random() < 0.8:  # 80% safe areas
            G.edges[u, v]['safety_score'] = np.random.uniform(10, 40)
        else:  # 20% higher risk areas
            G.edges[u, v]['safety_score'] = np.random.uniform(50, 90)
    
    return G


if __name__ == "__main__":
    # Create or load graph
    print("Loading graph for experimental analysis...")
    
    # For demo purposes, create a sample graph
    # In practice, you would load your real San Francisco graph with crime data
    graph = create_sample_graph()
    
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Run experiment
    experiment = RoutingExperiment(graph, num_trials=150)
    experiment.run_experiment()
    
    # Generate outputs
    experiment.print_summary()
    experiment.save_results()
    experiment.generate_visualizations()
    
    print("\nExperimental analysis complete!")
