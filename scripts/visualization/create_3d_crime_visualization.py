"""
3D Crime Risk Visualization Generator

Creates a 3D heightmap where crime risk levels are represented as vertical amplitude.
Higher risk areas will have greater height, creating a 3D landscape of danger.

Outputs:
- 3MF file for 3D printing/viewing
- PLY file for visualization software
- STL file for CAD applications

Usage:
    python create_3d_crime_visualization.py
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import trimesh
from shapely import wkt
from shapely.geometry import Point, LineString
import geopandas as gpd
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Crime3DVisualizer:
    """Creates 3D visualizations of crime risk data."""
    
    def __init__(self, output_dir="visualization/3d_models"):
        """Initialize the 3D visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.graph = None
        self.edge_data = None
        self.risk_points = []
        self.grid_size = 200  # Resolution of 3D grid
        
    def load_data(self, graph_path, edge_scores_path):
        """Load graph and edge risk scores."""
        print("Loading graph and edge risk data...")
        
        # Load graph
        self.graph = nx.read_graphml(graph_path)
        print(f"Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Load edge scores
        self.edge_data = pd.read_csv(edge_scores_path)
        print(f"Loaded {len(self.edge_data)} edge risk scores")
        
    def extract_spatial_risk_points(self):
        """Extract spatial points with risk values from edge geometries."""
        print("Extracting spatial risk points from edge geometries...")
        
        self.risk_points = []
        
        for _, row in tqdm(self.edge_data.iterrows(), total=len(self.edge_data), desc="Processing edges"):
            u, v, key = str(row['u']), str(row['v']), int(row['key'])
            risk_score = row['risk_score_enhanced']
            
            # Get edge geometry from graph
            if self.graph.has_edge(u, v, key):
                edge_data = self.graph.edges[u, v, key]
                geom_str = edge_data.get('geometry')
                
                if geom_str:
                    try:
                        geom = wkt.loads(geom_str)
                        
                        if hasattr(geom, 'coords'):
                            # Sample points along the edge geometry
                            coords = list(geom.coords)
                            
                            # Add multiple points along each edge for better density
                            for i in range(len(coords)):
                                x, y = coords[i]
                                self.risk_points.append({
                                    'x': float(x),
                                    'y': float(y), 
                                    'risk': float(risk_score)
                                })
                                
                                # Add interpolated points for longer edges
                                if i > 0:
                                    prev_x, prev_y = coords[i-1]
                                    # Add midpoint
                                    mid_x = (x + prev_x) / 2
                                    mid_y = (y + prev_y) / 2
                                    self.risk_points.append({
                                        'x': float(mid_x),
                                        'y': float(mid_y),
                                        'risk': float(risk_score)
                                    })
                    
                    except Exception as e:
                        continue
        
        print(f"Extracted {len(self.risk_points)} spatial risk points")
        
        # Convert to arrays for easier processing
        self.points_df = pd.DataFrame(self.risk_points)
        
    def create_3d_heightmap(self, height_scale=50.0, base_height=2.0):
        """
        Create a 3D heightmap where risk level determines height.
        
        Args:
            height_scale: Multiplier for risk to height conversion
            base_height: Minimum height for all areas
        """
        print("Creating 3D heightmap...")
        
        if not self.risk_points:
            raise ValueError("No risk points available. Run extract_spatial_risk_points first.")
        
        # Get bounds
        x_coords = self.points_df['x'].values
        y_coords = self.points_df['y'].values
        risk_values = self.points_df['risk'].values
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        print(f"Spatial bounds: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")
        print(f"Risk range: [{np.min(risk_values):.1f}, {np.max(risk_values):.1f}]")
        
        # Create regular grid
        x_grid = np.linspace(x_min, x_max, self.grid_size)
        y_grid = np.linspace(y_min, y_max, self.grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate risk values to grid
        print("Interpolating risk values to regular grid...")
        
        # Use multiple interpolation methods and combine
        points = np.column_stack([x_coords, y_coords])
        
        # Method 1: Linear interpolation
        Z_linear = griddata(points, risk_values, (X, Y), method='linear', fill_value=1.0)
        
        # Method 2: Nearest neighbor for filling gaps
        Z_nearest = griddata(points, risk_values, (X, Y), method='nearest')
        
        # Combine: use linear where available, nearest for gaps
        Z = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)
        
        # Apply height scaling
        # Normalize risk to [0, 1] then scale
        risk_normalized = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
        heights = base_height + risk_normalized * height_scale
        
        print(f"Height range: [{np.min(heights):.1f}, {np.max(heights):.1f}]")
        
        return X, Y, heights
        
    def create_mesh_from_heightmap(self, X, Y, Z, simplify_factor=0.1):
        """Create a 3D mesh from the heightmap data."""
        print("Creating 3D mesh from heightmap...")
        
        # Convert to vertices and faces
        vertices = []
        faces = []
        
        rows, cols = X.shape
        
        # Create vertices
        for i in range(rows):
            for j in range(cols):
                vertices.append([X[i, j], Y[i, j], Z[i, j]])
        
        # Create faces (triangles)
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Current vertex indices
                v1 = i * cols + j
                v2 = i * cols + (j + 1)
                v3 = (i + 1) * cols + j
                v4 = (i + 1) * cols + (j + 1)
                
                # Create two triangles per grid cell
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        print(f"Created mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Simplify mesh if requested
        if simplify_factor < 1.0:
            target_faces = int(len(faces) * simplify_factor)
            print(f"Simplifying mesh to {target_faces} faces...")
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                print(f"Simplified mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            except Exception as e:
                print(f"⚠️  Mesh simplification failed: {e}")
                print("Using original mesh without simplification")
        
        return mesh
        
    def add_base_plate(self, mesh, plate_thickness=1.0):
        """Add a base plate under the heightmap for 3D printing stability."""
        print("Adding base plate for 3D printing...")
        
        # Get mesh bounds
        bounds = mesh.bounds
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]
        
        # Create base plate slightly larger than mesh
        margin = (x_max - x_min) * 0.02  # 2% margin
        
        base_vertices = [
            [x_min - margin, y_min - margin, z_min - plate_thickness],
            [x_max + margin, y_min - margin, z_min - plate_thickness],
            [x_max + margin, y_max + margin, z_min - plate_thickness],
            [x_min - margin, y_max + margin, z_min - plate_thickness],
            [x_min - margin, y_min - margin, z_min],
            [x_max + margin, y_min - margin, z_min],
            [x_max + margin, y_max + margin, z_min],
            [x_min - margin, y_max + margin, z_min]
        ]
        
        base_faces = [
            # Bottom
            [0, 1, 2], [0, 2, 3],
            # Top  
            [4, 6, 5], [4, 7, 6],
            # Sides
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0]
        ]
        
        base_mesh = trimesh.Trimesh(vertices=base_vertices, faces=base_faces)
        
        # Combine with main mesh
        combined_mesh = trimesh.util.concatenate([mesh, base_mesh])
        
        print(f"Added base plate: {len(base_mesh.vertices)} vertices, {len(base_mesh.faces)} faces")
        
        return combined_mesh
        
    def export_3d_models(self, mesh, filename_prefix="sf_crime_risk_3d"):
        """Export 3D models in multiple formats."""
        print("Exporting 3D models...")
        
        # Export 3MF (main format for 3D printing)
        mf_path = self.output_dir / f"{filename_prefix}.3mf"
        try:
            mesh.export(str(mf_path))
            print(f"✅ Exported 3MF: {mf_path}")
        except Exception as e:
            print(f"⚠️  3MF export failed: {e}")
        
        # Export STL (standard 3D printing format)
        stl_path = self.output_dir / f"{filename_prefix}.stl"
        try:
            mesh.export(str(stl_path))
            print(f"✅ Exported STL: {stl_path}")
        except Exception as e:
            print(f"⚠️  STL export failed: {e}")
        
        # Export PLY (for visualization software)
        ply_path = self.output_dir / f"{filename_prefix}.ply"
        try:
            mesh.export(str(ply_path))
            print(f"✅ Exported PLY: {ply_path}")
        except Exception as e:
            print(f"⚠️  PLY export failed: {e}")
        
        # Export OBJ (widely compatible)
        obj_path = self.output_dir / f"{filename_prefix}.obj"
        try:
            mesh.export(str(obj_path))
            print(f"✅ Exported OBJ: {obj_path}")
        except Exception as e:
            print(f"⚠️  OBJ export failed: {e}")
        
        return {
            '3mf': mf_path if mf_path.exists() else None,
            'stl': stl_path if stl_path.exists() else None,
            'ply': ply_path if ply_path.exists() else None,
            'obj': obj_path if obj_path.exists() else None
        }
    
    def create_preview_visualization(self, X, Y, Z, filename="3d_crime_preview.png"):
        """Create a 2D preview of the 3D visualization."""
        print("Creating 3D preview visualization...")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D surface plot
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='Reds', alpha=0.8, linewidth=0, antialiased=True)
        ax1.set_title('3D Crime Risk Surface', fontweight='bold')
        ax1.set_xlabel('Longitude (UTM)')
        ax1.set_ylabel('Latitude (UTM)')
        ax1.set_zlabel('Risk Height')
        
        # Contour plot (top view)
        ax2 = fig.add_subplot(222)
        contour = ax2.contourf(X, Y, Z, levels=20, cmap='Reds')
        ax2.set_title('Risk Contours (Top View)', fontweight='bold')
        ax2.set_xlabel('Longitude (UTM)')
        ax2.set_ylabel('Latitude (UTM)')
        fig.colorbar(contour, ax=ax2, label='Risk Level')
        
        # Height distribution
        ax3 = fig.add_subplot(223)
        ax3.hist(Z.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax3.set_title('Height Distribution', fontweight='bold')
        ax3.set_xlabel('Risk Height')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Statistics
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        stats_text = f"""
3D Model Statistics:

Grid Resolution: {self.grid_size} × {self.grid_size}
Total Points: {X.size:,}

Height Statistics:
  Min Height: {np.min(Z):.1f}
  Max Height: {np.max(Z):.1f}
  Mean Height: {np.mean(Z):.1f}
  Std Height: {np.std(Z):.1f}

Risk Statistics:
  Min Risk: {np.min(self.points_df['risk']):.1f}
  Max Risk: {np.max(self.points_df['risk']):.1f}
  Mean Risk: {np.mean(self.points_df['risk']):.1f}

Model Info:
  Suitable for 3D printing
  Base plate included
  Multiple export formats
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('San Francisco Crime Risk 3D Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        preview_path = self.output_dir / filename
        plt.savefig(preview_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved preview: {preview_path}")
        return preview_path
        
    def run_full_pipeline(self, graph_path, edge_scores_path, height_scale=100.0, with_base=True):
        """Run the complete 3D visualization pipeline."""
        print("🚀 STARTING 3D CRIME RISK VISUALIZATION PIPELINE")
        print("=" * 60)
        
        # Load data
        self.load_data(graph_path, edge_scores_path)
        
        # Extract spatial points
        self.extract_spatial_risk_points()
        
        # Create heightmap
        X, Y, Z = self.create_3d_heightmap(height_scale=height_scale)
        
        # Create mesh
        mesh = self.create_mesh_from_heightmap(X, Y, Z, simplify_factor=0.3)
        
        # Add base plate if requested
        if with_base:
            mesh = self.add_base_plate(mesh)
        
        # Export models
        exported_files = self.export_3d_models(mesh)
        
        # Create preview
        preview_path = self.create_preview_visualization(X, Y, Z)
        
        # Summary
        print("\n🎯 3D VISUALIZATION COMPLETE!")
        print("=" * 40)
        print(f"📊 Model Statistics:")
        print(f"   Vertices: {len(mesh.vertices):,}")
        print(f"   Faces: {len(mesh.faces):,}")
        print(f"   Volume: {mesh.volume:.1f} cubic units")
        print(f"   Surface Area: {mesh.area:.1f} square units")
        
        print(f"\n📁 Generated Files:")
        for format_name, file_path in exported_files.items():
            if file_path and file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                print(f"   {format_name.upper()}: {file_path} ({file_size:.1f} MB)")
        
        print(f"   Preview: {preview_path}")
        
        print(f"\n🎨 Usage Instructions:")
        print(f"   • 3MF file: Use with 3D printing software (PrusaSlicer, Cura)")
        print(f"   • STL file: Standard 3D printing format")
        print(f"   • PLY file: For visualization in MeshLab, CloudCompare")
        print(f"   • OBJ file: Import into Blender, Maya, or CAD software")
        
        print(f"\n🏗️  3D Printing Notes:")
        print(f"   • Higher areas = more dangerous crime zones")
        print(f"   • Base plate included for stability")
        print(f"   • Recommended print scale: 200mm × 200mm base")
        print(f"   • Layer height: 0.2mm or finer for detail")
        
        return mesh, exported_files


def main():
    """Run the 3D crime visualization generator."""
    
    # Paths
    graph_path = "data/graphs/sf_pedestrian_graph_enhanced.graphml"
    edge_scores_path = "data/processed/edge_risk_scores_enhanced.csv"
    
    # Create visualizer
    visualizer = Crime3DVisualizer()
    
    try:
        # Run pipeline with high detail
        mesh, files = visualizer.run_full_pipeline(
            graph_path=graph_path,
            edge_scores_path=edge_scores_path,
            height_scale=150.0,  # Exaggerate height for visual impact
            with_base=True
        )
        
        print("\n✅ 3D Crime Risk Visualization Successfully Generated!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure the enhanced graph and edge scores exist.")
    except Exception as e:
        print(f"❌ Error creating 3D visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
