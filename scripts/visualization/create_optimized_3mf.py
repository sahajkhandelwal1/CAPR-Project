"""
Quick 3MF Export for San Francisco Crime Risk Visualization

Creates an optimized 3D model specifically for 3MF export and 3D printing.
"""

import pandas as pd
import numpy as np
import networkx as nx
import trimesh
from pathlib import Path
from shapely import wkt
from scipy.interpolate import griddata
from tqdm import tqdm


def create_optimized_3mf():
    """Create an optimized 3MF file for 3D printing."""
    
    print("🔧 Creating Optimized 3MF for 3D Printing...")
    
    # Load enhanced edge scores
    print("Loading enhanced edge risk scores...")
    df = pd.read_csv('data/processed/edge_risk_scores_enhanced.csv')
    
    # Load graph for geometry
    print("Loading graph...")
    graph = nx.read_graphml('data/graphs/sf_pedestrian_graph_enhanced.graphml')
    
    # Extract risk points with coordinates
    print("Extracting spatial coordinates...")
    risk_points = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing edges"):
        u, v, key = str(row['u']), str(row['v']), int(row['key'])
        risk = row['risk_score_enhanced']
        
        if graph.has_edge(u, v, key):
            edge_data = graph.edges[u, v, key]
            geom_str = edge_data.get('geometry')
            
            if geom_str:
                try:
                    geom = wkt.loads(geom_str)
                    if hasattr(geom, 'coords'):
                        # Sample points along the edge
                        coords = list(geom.coords)
                        for x, y in coords:
                            risk_points.append([float(x), float(y), float(risk)])
                except:
                    continue
    
    points_array = np.array(risk_points)
    print(f"Extracted {len(points_array)} risk points")
    
    # Create grid (lower resolution for 3D printing)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1] 
    risk_values = points_array[:, 2]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Smaller grid for 3D printing efficiency
    grid_size = 100  # Reduced from 200
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate risk to grid
    print("Interpolating risk values...")
    points = np.column_stack([x_coords, y_coords])
    Z_linear = griddata(points, risk_values, (X, Y), method='linear', fill_value=2.0)
    Z_nearest = griddata(points, risk_values, (X, Y), method='nearest')
    Z = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)
    
    # Scale height for 3D printing (more dramatic)
    height_scale = 5.0  # Scale for realistic 3D print
    base_height = 1.0
    risk_normalized = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
    heights = base_height + risk_normalized * height_scale
    
    print(f"Height range: {np.min(heights):.1f} to {np.max(heights):.1f}")
    
    # Create mesh
    print("Creating 3D mesh...")
    vertices = []
    faces = []
    
    # Scale coordinates for reasonable 3D printing size (200mm x 200mm)
    x_scale = 200.0 / (x_max - x_min)  # Scale to 200mm width
    y_scale = 200.0 / (y_max - y_min)  # Scale to 200mm height
    
    # Create vertices
    for i in range(grid_size):
        for j in range(grid_size):
            x_scaled = (X[i, j] - x_min) * x_scale
            y_scaled = (Y[i, j] - y_min) * y_scale
            z_scaled = heights[i, j]
            vertices.append([x_scaled, y_scaled, z_scaled])
    
    # Create faces
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            v1 = i * grid_size + j
            v2 = i * grid_size + (j + 1)
            v3 = (i + 1) * grid_size + j
            v4 = (i + 1) * grid_size + (j + 1)
            
            # Two triangles per grid cell
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    # Create trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Add base plate for stability
    print("Adding base plate...")
    base_vertices = [
        [0, 0, 0], [200, 0, 0], [200, 200, 0], [0, 200, 0],  # Bottom
        [0, 0, base_height], [200, 0, base_height], [200, 200, base_height], [0, 200, base_height]  # Top
    ]
    
    base_faces = [
        [0, 2, 1], [0, 3, 2],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top  
        [0, 1, 5], [0, 5, 4],  # Sides
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], 
        [3, 0, 4], [3, 4, 7]
    ]
    
    base_mesh = trimesh.Trimesh(vertices=base_vertices, faces=base_faces)
    combined_mesh = trimesh.util.concatenate([mesh, base_mesh])
    
    # Ensure output directory exists
    output_dir = Path('visualization/3d_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export 3MF
    print("Exporting 3MF file...")
    mf_path = output_dir / 'sf_crime_risk_3d_optimized.3mf'
    
    try:
        combined_mesh.export(str(mf_path))
        print(f"✅ Successfully exported 3MF: {mf_path}")
        
        file_size = mf_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size:.1f} MB")
        print(f"   Vertices: {len(combined_mesh.vertices):,}")
        print(f"   Faces: {len(combined_mesh.faces):,}")
        
    except Exception as e:
        print(f"❌ 3MF export failed: {e}")
        
        # Try STL as fallback
        stl_path = output_dir / 'sf_crime_risk_3d_optimized.stl'
        combined_mesh.export(str(stl_path))
        print(f"✅ Exported STL instead: {stl_path}")
    
    # Also export additional formats
    print("Exporting additional formats...")
    
    # STL for 3D printing
    stl_path = output_dir / 'sf_crime_risk_3d_optimized.stl'
    combined_mesh.export(str(stl_path))
    
    # OBJ for visualization
    obj_path = output_dir / 'sf_crime_risk_3d_optimized.obj'
    combined_mesh.export(str(obj_path))
    
    print(f"✅ Additional exports:")
    print(f"   STL: {stl_path}")
    print(f"   OBJ: {obj_path}")
    
    return combined_mesh, mf_path


if __name__ == "__main__":
    try:
        mesh, path = create_optimized_3mf()
        
        print("\n🎯 3D MODEL READY FOR 3D PRINTING!")
        print("=" * 50)
        print("📐 Model Specifications:")
        print("   • Base size: 200mm × 200mm")
        print("   • Height: 1-6mm (scaled to crime risk)")
        print("   • Material: PLA/PETG recommended")
        print("   • Layer height: 0.1-0.2mm")
        print("   • Supports: Not required")
        print("   • Print time: ~2-4 hours")
        
        print("\n🗺️  Visualization Guide:")
        print("   • Flat areas: Low crime risk")
        print("   • Raised areas: High crime risk")
        print("   • Peak height: Most dangerous zones")
        print("   • Smooth gradients show risk diffusion")
        
        print(f"\n📁 File ready: {path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
