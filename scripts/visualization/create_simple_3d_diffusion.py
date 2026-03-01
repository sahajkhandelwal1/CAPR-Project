#!/usr/bin/env python3
"""
Simple 3D Crime Risk Diffusion Surface
Shows just initial and final states, clean and minimal
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def gaussian_kernel(distance, sigma=1.0):
    """Gaussian kernel for risk diffusion"""
    return np.exp(-distance**2 / (2 * sigma**2))

def create_2d_network_grid(size=20):
    """Create a 2D grid representing street network"""
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    X, Y = np.meshgrid(x, y)
    return X, Y

def simulate_3d_diffusion_simple(X, Y, crime_locations, num_steps=30, alpha=0.15):
    """Simulate additive crime risk diffusion in 3D - simple version"""
    
    # Initialize risk surface
    risk = np.zeros_like(X)
    
    # Place initial crimes
    for crime_x, crime_y, intensity in crime_locations:
        i = int(round(crime_y))
        j = int(round(crime_x))
        if 0 <= i < risk.shape[0] and 0 <= j < risk.shape[1]:
            risk[i, j] = intensity
    
    # Store initial state
    initial_risk = risk.copy()
    
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
    
    return initial_risk, risk

def create_simple_3d_diffusion():
    """Create simple 3D diffusion surface with just initial and final"""
    
    # Create network grid
    X, Y = create_2d_network_grid(20)
    
    # Define crime locations (x, y, intensity)
    crime_locations = [(5, 8, 2.0), (12, 6, 1.8), (8, 14, 1.5)]
    
    # Simulate diffusion
    initial_risk, final_risk = simulate_3d_diffusion_simple(X, Y, crime_locations)
    
    # Create figure with just two 3D subplots
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle('Crime Risk Diffusion', fontsize=18, fontweight='bold', y=0.95)
    
    # Custom colormap for risk
    colors = ['#000080', '#0066CC', '#00CCFF', '#66FF66', '#FFFF00', '#FF6600', '#FF0000']
    risk_cmap = LinearSegmentedColormap.from_list('risk3d', colors, N=256)
    
    # Find max risk for consistent scaling
    max_risk = max(np.max(initial_risk), np.max(final_risk))
    
    # Initial state
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, initial_risk, 
                            cmap=risk_cmap, 
                            vmin=0, vmax=max_risk,
                            alpha=0.9, 
                            linewidth=0, 
                            antialiased=True)
    
    # Mark crime locations
    for crime_x, crime_y, intensity in crime_locations:
        ax1.scatter(crime_x, crime_y, intensity, 
                   c='darkred', s=150, marker='o', alpha=1.0, zorder=10)
    
    ax1.set_title('Initial State', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Risk Level')
    ax1.view_init(elev=30, azim=45)
    ax1.set_zlim(0, max_risk)
    
    # Final state
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, final_risk, 
                            cmap=risk_cmap, 
                            vmin=0, vmax=max_risk,
                            alpha=0.9, 
                            linewidth=0, 
                            antialiased=True)
    
    # Mark crime locations
    for crime_x, crime_y, intensity in crime_locations:
        ax2.scatter(crime_x, crime_y, intensity * 0.7, 
                   c='darkred', s=150, marker='o', alpha=1.0, zorder=10)
    
    ax2.set_title('Final Diffused State', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('Risk Level')
    ax2.view_init(elev=30, azim=45)
    ax2.set_zlim(0, max_risk)
    
    # Clean layout without colorbar or equations
    plt.tight_layout()
    
    return fig

def main():
    """Generate simple 3D diffusion visualization"""
    print("Creating simple 3D crime risk diffusion surface...")
    
    # Create simple 3D surface diffusion
    fig = create_simple_3d_diffusion()
    simple_path = '/Users/Sahaj/CAPR Project/CAPR-Project/visualization/simple_3d_diffusion_surface.png'
    fig.savefig(simple_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Simple 3D diffusion visualization saved:")
    print(f"  Simple surface comparison: {simple_path}")
    print("\nThis visualization shows:")
    print("  ✓ Clean initial vs final comparison")
    print("  ✓ No mathematical equations")
    print("  ✓ No color key or legend")
    print("  ✓ Focus purely on visual diffusion")
    print("  ✓ Judge-friendly simplicity")

if __name__ == "__main__":
    main()
