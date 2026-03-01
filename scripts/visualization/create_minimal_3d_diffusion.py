#!/usr/bin/env python3
"""
Minimal 3D Crime Risk Diffusion Surface Visualization
Shows only initial and final states without math or color key
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_minimal_diffusion():
    """Create a minimal 3D diffusion visualization with only initial and final states"""
    
    # Create grid
    x = np.linspace(0, 20, 21)
    y = np.linspace(0, 20, 21)
    X, Y = np.meshgrid(x, y)
    
    # Initial state: three crime hotspots
    R_initial = np.zeros_like(X)
    
    # Crime source locations and intensities
    sources = [(5, 15, 2.0), (15, 5, 1.5), (10, 10, 1.8)]
    
    for sx, sy, intensity in sources:
        R_initial[int(sy), int(sx)] = intensity
    
    # Final state: after diffusion (simplified)
    R_final = np.zeros_like(X)
    
    # Apply Gaussian diffusion pattern
    for sx, sy, intensity in sources:
        for i in range(21):
            for j in range(21):
                dist = np.sqrt((i - sy)**2 + (j - sx)**2)
                # Simple exponential decay
                R_final[i, j] += intensity * np.exp(-dist/3.0)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(12, 5))
    
    # Initial state
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, R_initial, cmap='Reds', alpha=0.8, 
                            vmin=0, vmax=2.5, linewidth=0, antialiased=True)
    
    # Reduce grid opacity
    ax1.grid(True, alpha=0.3)
    
    # Add crime source markers
    for sx, sy, intensity in sources:
        ax1.scatter([sx], [sy], [intensity], color='darkred', s=100, alpha=1.0)
    
    ax1.set_title('Initial', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 20)
    ax1.set_zlim(0, 2.5)
    ax1.view_init(elev=25, azim=45)
    
    # Keep axis labels and ticks, with better spacing
    ax1.set_xlabel('X Coordinate', fontsize=10)
    ax1.set_ylabel('Y Coordinate', fontsize=10)
    ax1.set_zlabel('Risk Level', fontsize=10)
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.set_yticks([0, 5, 10, 15, 20])
    
    # Final state
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, R_final, cmap='Reds', alpha=0.8,
                            vmin=0, vmax=2.5, linewidth=0, antialiased=True)
    
    # Reduce grid opacity
    ax2.grid(True, alpha=0.3)
    
    ax2.set_title('Final', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 20)
    ax2.set_zlim(0, 2.5)
    ax2.view_init(elev=25, azim=45)
    
    # Keep axis labels and ticks, with better spacing
    ax2.set_xlabel('X Coordinate', fontsize=10)
    ax2.set_ylabel('Y Coordinate', fontsize=10)
    ax2.set_zlabel('Risk Level', fontsize=10)
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_yticks([0, 5, 10, 15, 20])
    
    # Overall title
    fig.suptitle('3D Crime Risk Diffusion', fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save
    plt.savefig('visualization/minimal_3d_diffusion_surface.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('visualization/minimal_3d_diffusion_surface.pdf', 
                bbox_inches='tight', facecolor='white')
    
    print("✓ Minimal 3D diffusion surface visualization saved")
    plt.show()

if __name__ == "__main__":
    create_minimal_diffusion()
