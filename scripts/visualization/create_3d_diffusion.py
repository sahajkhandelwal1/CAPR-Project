#!/usr/bin/env python3
"""
3D Representation of Crime Risk Diffusion Mathematics
Shows how risk spreads across network surface over time
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from matplotlib import cm

def gaussian_kernel(distance, sigma=1.0):
    """Gaussian kernel for risk diffusion"""
    return np.exp(-distance**2 / (2 * sigma**2))

def create_2d_network_grid(size=20):
    """Create a 2D grid representing street network"""
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    X, Y = np.meshgrid(x, y)
    return X, Y

def simulate_3d_diffusion(X, Y, crime_locations, num_steps=30, alpha=0.15):
    """Simulate additive crime risk diffusion in 3D"""
    
    # Initialize risk surface
    risk = np.zeros_like(X)
    
    # Place initial crimes
    for crime_x, crime_y, intensity in crime_locations:
        # Find nearest grid points
        i = int(round(crime_y))
        j = int(round(crime_x))
        if 0 <= i < risk.shape[0] and 0 <= j < risk.shape[1]:
            risk[i, j] = intensity
    
    # Store diffusion steps
    diffusion_steps = [risk.copy()]
    
    # Simulate diffusion over time
    for step in range(num_steps):
        # Create new risk surface for additive diffusion
        new_risk = risk.copy() * 0.98  # Slight decay
        
        # Add diffused risk from each cell
        for i in range(1, risk.shape[0]-1):
            for j in range(1, risk.shape[1]-1):
                if risk[i, j] > 0.01:
                    # Spread to 8 neighbors
                    spread_amount = alpha * risk[i, j] / 8
                    
                    # Add to neighbors with distance weighting
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < risk.shape[0] and 0 <= nj < risk.shape[1]:
                                distance = np.sqrt(di**2 + dj**2)
                                weight = gaussian_kernel(distance, 0.8)
                                new_risk[ni, nj] += spread_amount * weight
        
        # Update risk surface
        risk = new_risk
        
        # Maintain crime origins
        for crime_x, crime_y, intensity in crime_locations:
            i = int(round(crime_y))
            j = int(round(crime_x))
            if 0 <= i < risk.shape[0] and 0 <= j < risk.shape[1]:
                risk[i, j] = max(risk[i, j], intensity * 0.7)
        
        # Store step
        if step % 3 == 0 or step < 10:
            diffusion_steps.append(risk.copy())
    
    return diffusion_steps

def create_3d_diffusion_surface():
    """Create main 3D diffusion surface visualization"""
    
    # Create network grid
    X, Y = create_2d_network_grid(20)
    
    # Define crime locations (x, y, intensity)
    crime_locations = [(5, 8, 2.0), (12, 6, 1.8), (8, 14, 1.5)]
    
    # Simulate diffusion
    diffusion_steps = simulate_3d_diffusion(X, Y, crime_locations)
    
    # Create figure with multiple 3D subplots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('3D Crime Risk Diffusion Mathematics', fontsize=20, fontweight='bold')
    
    # Time steps to visualize
    time_indices = [0, 3, 6, 9, -1]  # Initial, early, mid, late, final
    time_labels = ['Initial\n(t=0)', 'Early Spread\n(t=9)', 'Growing\n(t=18)', 
                   'Advanced\n(t=27)', 'Final State\n(t=30)']
    
    # Custom colormap for risk
    colors = ['#000080', '#0066CC', '#00CCFF', '#66FF66', '#FFFF00', '#FF6600', '#FF0000']
    risk_cmap = LinearSegmentedColormap.from_list('risk3d', colors, N=256)
    
    # Create subplots
    for idx, (time_idx, label) in enumerate(zip(time_indices, time_labels)):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        if time_idx == -1:
            time_idx = len(diffusion_steps) - 1
        
        risk_surface = diffusion_steps[time_idx]
        max_risk = np.max([np.max(step) for step in diffusion_steps])
        
        # Create 3D surface plot
        surf = ax.plot_surface(X, Y, risk_surface, 
                              cmap=risk_cmap, 
                              vmin=0, vmax=max_risk,
                              alpha=0.8, 
                              linewidth=0.5, 
                              edgecolors='black',
                              antialiased=True)
        
        # Mark crime locations with red spheres
        for crime_x, crime_y, intensity in crime_locations:
            ax.scatter(crime_x, crime_y, intensity, 
                      c='darkred', s=100, marker='o', alpha=1.0, zorder=10)
        
        # Customize plot
        ax.set_title(label, fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Risk Level')
        
        # Set consistent view angle
        ax.view_init(elev=25, azim=45)
        
        # Set consistent z-limits
        ax.set_zlim(0, max_risk)
    
    # Add mathematical equation subplot
    ax_eq = fig.add_subplot(2, 3, 6)
    ax_eq.text(0.5, 0.8, '3D Diffusion Mathematics', 
              ha='center', fontsize=14, fontweight='bold', transform=ax_eq.transAxes)
    
    equations = [
        r'Additive Diffusion PDE:',
        r'$\frac{\partial R}{\partial t} = \alpha \nabla^2 R + S(x,y,t)$',
        '',
        r'Where:',
        r'$R(x,y,t)$ = Risk at position $(x,y)$ and time $t$',
        r'$\alpha$ = Diffusion coefficient',
        r'$S(x,y,t)$ = Crime source terms',
        r'$\nabla^2$ = 2D Laplacian operator',
        '',
        r'Discrete Implementation:',
        r'$R_{i,j}^{n+1} = R_{i,j}^n + \alpha \sum_{neighbors} w_{ij} R_{neighbor}^n$',
        r'$R_{source} = \max(R_{source}, R_{min})$'
    ]
    
    y_positions = np.linspace(0.7, 0.1, len(equations))
    for eq, y_pos in zip(equations, y_positions):
        if eq:
            if 'Additive' in eq or 'Where:' in eq or 'Discrete' in eq:
                ax_eq.text(0.1, y_pos, eq, fontsize=10, fontweight='bold', transform=ax_eq.transAxes)
            else:
                ax_eq.text(0.15, y_pos, eq, fontsize=9, transform=ax_eq.transAxes)
    
    ax_eq.axis('off')
    
    # Add single colorbar for all subplots
    cbar = fig.colorbar(surf, ax=fig.get_axes()[:-1], shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Crime Risk Level', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_3d_cross_section():
    """Create 3D cross-section view showing diffusion profile"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # Create 1D cross-section through crime center
    x = np.linspace(-10, 10, 100)
    times = [0, 5, 10, 15, 20]
    
    ax1 = fig.add_subplot(2, 2, (1, 2), projection='3d')
    
    # Generate diffusion profiles over time
    for i, t in enumerate(times):
        sigma = 0.5 + 0.3 * t  # Increasing spread over time
        amplitude = 2.0 * np.exp(-0.05 * t)  # Decreasing amplitude but stays high
        
        # Additive diffusion profile
        y = amplitude * np.exp(-x**2 / (2 * sigma**2))
        
        # Add slight background diffusion
        y += 0.1 * t * np.exp(-x**2 / (2 * (sigma + 2)**2))
        
        # Plot 3D curve
        time_array = np.full_like(x, t)
        color = plt.cm.plasma(i / len(times))
        ax1.plot(x, time_array, y, color=color, linewidth=3, alpha=0.8, label=f't={t}')
    
    ax1.set_xlabel('Distance from Crime Center')
    ax1.set_ylabel('Time')
    ax1.set_zlabel('Risk Level')
    ax1.set_title('3D Risk Profile Evolution', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # 2D heatmap view
    ax2 = fig.add_subplot(2, 2, 3)
    
    # Create time-distance heatmap
    X_heat, T_heat = np.meshgrid(x, times)
    Z_heat = np.zeros_like(X_heat)
    
    for i, t in enumerate(times):
        sigma = 0.5 + 0.3 * t
        amplitude = 2.0 * np.exp(-0.05 * t)
        Z_heat[i, :] = amplitude * np.exp(-x**2 / (2 * sigma**2)) + 0.1 * t * np.exp(-x**2 / (2 * (sigma + 2)**2))
    
    im = ax2.imshow(Z_heat, extent=[x.min(), x.max(), times[0], times[-1]], 
                   aspect='auto', origin='lower', cmap='plasma')
    ax2.set_xlabel('Distance from Crime Center')
    ax2.set_ylabel('Time')
    ax2.set_title('Risk Diffusion Heatmap', fontsize=14, fontweight='bold')
    
    # Add contour lines
    contours = ax2.contour(x, times, Z_heat, colors='white', alpha=0.5, linewidths=0.8)
    ax2.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    # Mathematical explanation
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.text(0.5, 0.9, '3D Diffusion Analysis', ha='center', fontsize=14, fontweight='bold', transform=ax3.transAxes)
    
    analysis_text = [
        'Key Mathematical Properties:',
        '',
        '1. Origin Preservation:',
        '   Crime source maintains high risk',
        '',
        '2. Gaussian Spreading:',
        '   Risk follows Gaussian distribution',
        '',
        '3. Time Evolution:',
        '   σ(t) = σ₀ + αt (spreading)',
        '   A(t) = A₀ × decay (amplitude)',
        '',
        '4. Additive Nature:',
        '   Total risk increases over time',
        '   ∫∫ R(x,y,t) dx dy ↗',
        '',
        '5. Network Constraints:',
        '   Diffusion follows street topology'
    ]
    
    y_positions = np.linspace(0.8, 0.05, len(analysis_text))
    for text, y_pos in zip(analysis_text, y_positions):
        if text and text.endswith(':'):
            ax3.text(0.1, y_pos, text, fontsize=11, fontweight='bold', transform=ax3.transAxes)
        elif text.startswith('   '):
            ax3.text(0.15, y_pos, text, fontsize=9, transform=ax3.transAxes, family='monospace')
        elif text:
            ax3.text(0.1, y_pos, text, fontsize=10, transform=ax3.transAxes)
    
    ax3.axis('off')
    
    # Add colorbar for heatmap
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Risk Level', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_3d_network_diffusion():
    """Create 3D network-based diffusion showing graph structure"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create a small 3D network for visualization
    import networkx as nx
    
    # Create 3D grid network
    G = nx.grid_2d_graph(8, 8)
    
    # Convert to 3D positions
    pos_3d = {}
    risk_values = {}
    
    # Initialize positions and risks
    for node in G.nodes():
        x, y = node
        pos_3d[node] = (x, y, 0)  # Start at z=0
        risk_values[node] = 0.0
    
    # Place crimes
    crime_nodes = [(2, 3), (5, 5)]
    for node in crime_nodes:
        risk_values[node] = 2.0
    
    # Simulate network diffusion steps
    diffusion_steps = [risk_values.copy()]
    
    for step in range(15):
        new_risks = risk_values.copy()
        
        # Diffuse along network edges
        for node in G.nodes():
            if risk_values[node] > 0.01:
                neighbors = list(G.neighbors(node))
                if neighbors:
                    spread_per_neighbor = 0.2 * risk_values[node] / len(neighbors)
                    for neighbor in neighbors:
                        new_risks[neighbor] += spread_per_neighbor
        
        # Apply decay and maintain sources
        for node in G.nodes():
            new_risks[node] *= 0.95
            if node in crime_nodes:
                new_risks[node] = max(new_risks[node], 1.5)
        
        risk_values = new_risks
        if step % 3 == 0:
            diffusion_steps.append(risk_values.copy())
    
    # Create 3D visualization
    steps_to_show = [0, 1, 2, 3, 4]
    for idx, step_idx in enumerate(steps_to_show):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        step_risks = diffusion_steps[step_idx]
        max_risk = max(max(step.values()) for step in diffusion_steps)
        
        # Draw network edges
        for edge in G.edges():
            node1, node2 = edge
            x1, y1, z1 = pos_3d[node1]
            x2, y2, z2 = pos_3d[node2]
            ax.plot([x1, x2], [y1, y2], [0, 0], 'k-', alpha=0.3, linewidth=1)
        
        # Draw nodes with risk-based height and color
        for node in G.nodes():
            x, y, _ = pos_3d[node]
            risk = step_risks[node]
            
            # Height represents risk
            height = risk
            
            # Color represents risk intensity
            color_intensity = risk / max_risk if max_risk > 0 else 0
            color = plt.cm.Reds(color_intensity)
            
            # Size represents risk
            size = 50 + risk * 200
            
            ax.scatter(x, y, height, c=[color], s=size, alpha=0.8, edgecolors='black')
            
            # Draw vertical line from base to node
            if height > 0.01:
                ax.plot([x, x], [y, y], [0, height], 'k-', alpha=0.5, linewidth=1)
        
        ax.set_title(f'Network Diffusion\nStep {step_idx * 3}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Risk Level')
        ax.set_zlim(0, max_risk)
    
    # Add mathematical formulation
    ax_math = fig.add_subplot(2, 3, 6)
    ax_math.text(0.5, 0.9, 'Network Diffusion Mathematics', 
                ha='center', fontsize=14, fontweight='bold', transform=ax_math.transAxes)
    
    math_text = [
        'Graph-Based Diffusion:',
        '',
        'Node Update Rule:',
        r'$r_i^{(t+1)} = r_i^{(t)} + \alpha \sum_{j \in N(i)} \frac{r_j^{(t)}}{deg(j)}$',
        '',
        'Where:',
        '• N(i) = neighbors of node i',
        '• deg(j) = degree of node j',
        '• α = diffusion rate',
        '',
        'Network Properties:',
        '• Diffusion follows graph topology',
        '• Risk spreads only along edges',
        '• Maintains graph connectivity',
        '',
        'Crime Source Preservation:',
        r'$r_{source}^{(t+1)} = \max(r_{source}^{(t+1)}, r_{min})$'
    ]
    
    y_positions = np.linspace(0.8, 0.05, len(math_text))
    for text, y_pos in zip(math_text, y_positions):
        if text and text.endswith(':'):
            ax_math.text(0.05, y_pos, text, fontsize=11, fontweight='bold', transform=ax_math.transAxes)
        elif text.startswith('•'):
            ax_math.text(0.1, y_pos, text, fontsize=9, transform=ax_math.transAxes)
        elif text:
            ax_math.text(0.1, y_pos, text, fontsize=10, transform=ax_math.transAxes)
    
    ax_math.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all 3D diffusion visualizations"""
    print("Creating 3D representations of crime diffusion mathematics...")
    
    # Create 3D surface diffusion
    fig1 = create_3d_diffusion_surface()
    surface_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/3d_diffusion_surface.png'
    fig1.savefig(surface_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # Create 3D cross-section analysis
    fig2 = create_3d_cross_section()
    cross_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/3d_diffusion_analysis.png'
    fig2.savefig(cross_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    # Create 3D network diffusion
    fig3 = create_3d_network_diffusion()
    network_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/3d_network_diffusion.png'
    fig3.savefig(network_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    
    print(f"3D diffusion visualizations saved:")
    print(f"  3D surface evolution: {surface_path}")
    print(f"  Cross-section analysis: {cross_path}")
    print(f"  Network-based diffusion: {network_path}")
    print("\nThese visualizations demonstrate:")
    print("  ✓ 3D risk surface evolution over time")
    print("  ✓ Mathematical diffusion profiles")
    print("  ✓ Network topology constraints")
    print("  ✓ Additive diffusion mathematics")
    print("  ✓ Judge-friendly 3D presentation")

if __name__ == "__main__":
    main()
