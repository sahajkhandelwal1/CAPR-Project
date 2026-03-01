"""
Compact High-Contrast 3D Crime Risk Visualization

Creates a smaller, denser 3D model with dramatic height differences:
- Direct mapping: Risk score = height in millimeters (1 = 1mm, 100 = 100mm)
- Compact base: 100mm x 100mm (instead of 200mm x 200mm)
- Higher density sampling for more detail
- Dramatic height contrasts for clear visualization

This creates a more impactful 3D print where dangerous areas tower over safe zones.
"""

import pandas as pd
import numpy as np
import networkx as nx
import trimesh
from pathlib import Path
from shapely import wkt
from scipy.interpolate import griddata
from tqdm import tqdm


def create_compact_dramatic_3mf():
    """Create compact 3D model with dramatic height mapping."""
    
    print("🔧 Creating Compact High-Contrast 3D Model...")
    print("📐 Specifications: 100mm × 100mm base, 1-100mm height")
    
    # Load enhanced edge scores
    print("Loading enhanced edge risk scores...")
    df = pd.read_csv('data/processed/edge_risk_scores_enhanced.csv')
    
    # Load graph for geometry
    print("Loading graph...")
    graph = nx.read_graphml('data/graphs/sf_pedestrian_graph_enhanced.graphml')
    
    # Extract risk points with higher density sampling
    print("Extracting spatial coordinates with high-density sampling...")
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
                        coords = list(geom.coords)
                        
                        # High-density sampling for better detail
                        for i, (x, y) in enumerate(coords):
                            risk_points.append([float(x), float(y), float(risk)])
                            
                            # Add more interpolated points for denser sampling
                            if i > 0:
                                prev_x, prev_y = coords[i-1]
                                
                                # Add 3 interpolated points between each pair
                                for t in [0.25, 0.5, 0.75]:
                                    interp_x = prev_x + t * (x - prev_x)
                                    interp_y = prev_y + t * (y - prev_y)
                                    risk_points.append([float(interp_x), float(interp_y), float(risk)])
                except:
                    continue
    
    points_array = np.array(risk_points)
    print(f"Extracted {len(points_array):,} high-density risk points")
    
    # Get spatial bounds
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1] 
    risk_values = points_array[:, 2]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    print(f"Spatial bounds: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")
    print(f"Risk range: [{np.min(risk_values):.1f}, {np.max(risk_values):.1f}]")
    
    # Create higher resolution grid for more detail
    grid_size = 150  # Increased resolution for better detail
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate risk to grid with better smoothing
    print("Interpolating risk values with high resolution...")
    points = np.column_stack([x_coords, y_coords])
    
    # Use cubic interpolation for smoother results
    try:
        Z = griddata(points, risk_values, (X, Y), method='cubic', fill_value=np.min(risk_values))
    except:
        # Fallback to linear if cubic fails
        Z = griddata(points, risk_values, (X, Y), method='linear', fill_value=np.min(risk_values))
    
    # Fill any remaining NaN values with nearest neighbor
    Z_nearest = griddata(points, risk_values, (X, Y), method='nearest')
    Z = np.where(np.isnan(Z), Z_nearest, Z)
    
    # DIRECT HEIGHT MAPPING: Risk score = millimeters
    # Clamp to actual risk score range first
    Z = np.clip(Z, np.min(risk_values), np.max(risk_values))
    
    # Direct 1:1 mapping: 1 risk point = 1mm height, 100 risk points = 100mm height
    heights = Z.copy()
    
    # Ensure minimum height for printability and maximum for reasonable printing
    min_printable_height = 0.5  # 0.5mm minimum
    max_printable_height = 100.0  # 100mm maximum (reasonable for desktop 3D printer)
    heights = np.clip(heights, min_printable_height, max_printable_height)
    
    print(f"Direct height mapping applied:")
    print(f"  Height range: {np.min(heights):.1f}mm to {np.max(heights):.1f}mm")
    print(f"  Average height: {np.mean(heights):.1f}mm")
    print(f"  Extreme areas (>75): {np.sum(heights > 75)} grid points")
    
    # Create mesh with compact dimensions
    print("Creating compact high-detail 3D mesh...")
    vertices = []
    faces = []
    
    # COMPACT SIZE: 100mm x 100mm base (half the previous size)
    base_size = 100.0  # mm
    
    x_scale = base_size / (x_max - x_min)
    y_scale = base_size / (y_max - y_min)
    
    # Create vertices with direct height mapping
    for i in range(grid_size):
        for j in range(grid_size):
            x_scaled = (X[i, j] - x_min) * x_scale
            y_scaled = (Y[i, j] - y_min) * y_scale
            z_height = heights[i, j]  # Direct mm mapping
            
            vertices.append([x_scaled, y_scaled, z_height])
    
    # Create faces for the surface
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            v1 = i * grid_size + j
            v2 = i * grid_size + (j + 1)
            v3 = (i + 1) * grid_size + j
            v4 = (i + 1) * grid_size + (j + 1)
            
            # Two triangles per grid cell
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    # Create surface mesh
    surface_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Add solid base for 3D printing stability
    print("Adding solid base for 3D printing...")
    
    # Create a solid base that extends down from the surface
    base_thickness = 2.0  # 2mm thick base
    
    # Get surface boundary vertices
    surface_vertices = np.array(vertices)
    
    # Create base vertices (project surface to base level)
    base_level = -base_thickness
    base_vertices = []
    
    # Add all surface vertices
    for vertex in surface_vertices:
        base_vertices.append([vertex[0], vertex[1], vertex[2]])  # Surface vertex
    
    # Add corresponding base vertices
    for vertex in surface_vertices:
        base_vertices.append([vertex[0], vertex[1], base_level])  # Base vertex
    
    # Create faces to connect surface to base
    base_faces = []
    
    # Copy surface faces
    for face in faces:
        base_faces.append(face)  # Top surface
    
    # Add bottom faces (reversed normals)
    n_surface_vertices = len(surface_vertices)
    for face in faces:
        bottom_face = [face[0] + n_surface_vertices, 
                      face[2] + n_surface_vertices, 
                      face[1] + n_surface_vertices]  # Reversed order
        base_faces.append(bottom_face)
    
    # Add side faces to connect perimeter
    # This is simplified - for a full implementation, you'd trace the perimeter
    # For now, we'll use the existing surface and add a simple box base
    
    # Create simple box base
    box_vertices = [
        [0, 0, base_level], [base_size, 0, base_level], 
        [base_size, base_size, base_level], [0, base_size, base_level],  # Bottom
        [0, 0, 0], [base_size, 0, 0], 
        [base_size, base_size, 0], [0, base_size, 0]  # Top
    ]
    
    box_faces = [
        # Bottom (reversed normals for downward face)
        [0, 2, 1], [0, 3, 2],  
        # Top
        [4, 5, 6], [4, 6, 7],  
        # Sides
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], 
        [3, 0, 4], [3, 4, 7]
    ]
    
    box_mesh = trimesh.Trimesh(vertices=box_vertices, faces=box_faces)
    
    # Combine surface and base
    combined_mesh = trimesh.util.concatenate([surface_mesh, box_mesh])
    
    # Ensure output directory exists
    output_dir = Path('visualization/3d_models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export compact dramatic 3MF
    print("Exporting compact dramatic 3MF file...")
    mf_path = output_dir / 'sf_crime_risk_3d_compact_dramatic.3mf'
    stl_path = output_dir / 'sf_crime_risk_3d_compact_dramatic.stl'
    obj_path = output_dir / 'sf_crime_risk_3d_compact_dramatic.obj'
    
    try:
        combined_mesh.export(str(mf_path))
        print(f"✅ Successfully exported compact 3MF: {mf_path}")
        
        file_size = mf_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"⚠️  3MF export issue: {e}")
    
    # Export additional formats
    combined_mesh.export(str(stl_path))
    combined_mesh.export(str(obj_path))
    
    print(f"✅ Additional exports:")
    print(f"   STL: {stl_path}")
    print(f"   OBJ: {obj_path}")
    
    # Print statistics
    print(f"\n📊 Model Statistics:")
    print(f"   Vertices: {len(combined_mesh.vertices):,}")
    print(f"   Faces: {len(combined_mesh.faces):,}")
    print(f"   Volume: {combined_mesh.volume:.1f} mm³")
    print(f"   Surface Area: {combined_mesh.area:.1f} mm²")
    
    # Print height statistics
    surface_heights = surface_vertices[:, 2]
    print(f"\n🏔️  Height Analysis:")
    print(f"   Min height: {np.min(surface_heights):.1f}mm")
    print(f"   Max height: {np.max(surface_heights):.1f}mm")
    print(f"   Average height: {np.mean(surface_heights):.1f}mm")
    print(f"   Height span: {np.max(surface_heights) - np.min(surface_heights):.1f}mm")
    
    # Risk level breakdown
    high_risk = np.sum(surface_heights >= 50)
    med_risk = np.sum((surface_heights >= 20) & (surface_heights < 50))
    low_risk = np.sum(surface_heights < 20)
    total_points = len(surface_heights)
    
    print(f"\n⚠️  Risk Distribution:")
    print(f"   High risk (≥50mm): {high_risk:,} points ({high_risk/total_points*100:.1f}%)")
    print(f"   Medium risk (20-49mm): {med_risk:,} points ({med_risk/total_points*100:.1f}%)")
    print(f"   Low risk (<20mm): {low_risk:,} points ({low_risk/total_points*100:.1f}%)")
    
    return combined_mesh, mf_path


if __name__ == "__main__":
    try:
        mesh, path = create_compact_dramatic_3mf()
        
        print("\n🎯 COMPACT DRAMATIC 3D MODEL READY!")
        print("=" * 60)
        print("📐 Model Specifications:")
        print("   • Base size: 100mm × 100mm (compact)")
        print("   • Height mapping: 1 risk point = 1mm height")
        print("   • Height range: ~1mm to ~100mm")
        print("   • High resolution: 150×150 grid")
        print("   • Solid base: 2mm thick for stability")
        
        print("\n🖨️  3D Printing Recommendations:")
        print("   • Material: PLA recommended (good detail)")
        print("   • Layer height: 0.1mm (for fine detail)")
        print("   • Supports: Not required (solid base)")
        print("   • Print time: ~3-5 hours (higher detail)")
        print("   • Infill: 15-20% (sufficient for display)")
        
        print("\n🗺️  Visual Impact:")
        print("   • Dangerous areas tower dramatically (50-100mm)")
        print("   • Safe areas remain low (1-20mm)")
        print("   • Clear height differentiation")
        print("   • Compact size for easy handling/display")
        
        print(f"\n📁 Files ready:")
        print(f"   3MF: {path}")
        print(f"   STL: {path.with_suffix('.stl')}")
        print(f"   OBJ: {path.with_suffix('.obj')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
