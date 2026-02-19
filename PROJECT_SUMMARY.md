# 🏗️ San Francisco Crime-Aware Pedestrian Routing Project

## 🎯 Project Objective
Build a multi-objective pedestrian routing system that balances **distance efficiency** with **crime safety** for San Francisco's walkable street network.

---

## ✅ COMPLETED: Foundation & Technical Analysis

### 1. Data Pipeline & Optimization
- **Crime Dataset**: SF Police Department Incident Reports (2018-Present)
  - Filtered to post-2020 incidents only
  - Reduced from 148MB to 88MB (GitHub-compatible)
  - 11 essential columns selected from 29 total
- **File Management**: Proper `.gitignore` setup for large datasets

### 2. Pedestrian Network Graph Construction
- **Data Source**: OpenStreetMap via OSMnx (pedestrian-only paths)
- **Network Scope**: Complete San Francisco walkable infrastructure
- **Graph Statistics**:
  - 24,728 nodes (intersections) - *consolidated from 50,945*
  - 93,696 edges (walkable segments) - *optimized from 153,214*
  - 6,766.8 km total network length
  - 44.2m average segment length
  - **51.5% node reduction** through intersection consolidation
- **Projection**: UTM Zone 10N (EPSG:32610) for metric accuracy
- **Validation**: Fully connected, optimized graph structure
- **Routing Optimization**: Each intersection has exactly one node

### 3. Technical Analysis & Validation
Generated comprehensive static analysis visualizations:
- **Network Overview**: Complete pedestrian network structure
- **Edge Length Distribution**: Color-coded segment analysis
- **Statistical Analysis**: Edge length histogram with metrics
- **Connectivity Analysis**: Node degree distribution
- **Network Summary**: Professional dashboard with key metrics
- **Component Analysis**: Graph connectivity validation
- **Performance Analysis**: Routing algorithm runtime benchmarks
- **Spatial Analysis**: Geographic density distribution

### 4. Data Export & Format Support
- **NetworkX GraphML**: `sf_pedestrian_graph_projected.graphml`
- **GeoJSON**: Separate node and edge geographic files
- **JSON Metrics**: Complete validation statistics
- **PNG Analysis**: 8 technical visualization charts

---

## 📁 Project Structure
```
CAPR-Project/
├── data/
│   ├── processed/
│   │   └── filtered_police_data.csv        # 88MB crime dataset
│   └── graphs/
│       ├── sf_pedestrian_graph_projected.graphml
│       ├── sf_pedestrian_nodes_projected.geojson
│       ├── sf_pedestrian_edges_projected.geojson
│       ├── validation_metrics.json
│       ├── network_stats.txt
│       └── visualizations/                 # 8 technical analysis charts
├── scripts/
│   ├── data_optimization/
│   │   └── cleaning/filter_data.py         # Data processing pipeline
│   └── graph_construction/
│       └── sf_pedestrian_graph.py          # Network builder & validator
└── requirements.txt                        # Dependencies
```

---

## 🔬 Technical Achievements

### Graph Theory Implementation
- Directed graph with 50K+ nodes and 150K+ edges
- Proper spatial projection for accurate distance calculations
- Network connectivity validation and optimization
- Performance benchmarking for routing algorithms

### Data Engineering
- Efficient processing of 850K+ crime records
- Smart filtering strategies (temporal, spatial, columnar)
- Memory-optimized large dataset handling
- Robust error handling and caching systems

### Spatial Analysis
- Coordinate system projection (WGS84 → UTM)
- Spatial density and coverage analysis
- Geographic validation and boundary checking
- Street type classification and validation

---

## 🎯 Current Status: FOUNDATION COMPLETE

**Ready for Routing Engine Development**:
- ✅ **Validated Graph Structure**: Projected, connected pedestrian network
- ✅ **Crime Dataset**: Cleaned and optimized for spatial analysis
- ✅ **Technical Validation**: Comprehensive analysis and benchmarking
- ✅ **Performance Baseline**: Runtime analysis for routing algorithms

**Next Development Phase**:
- 🎯 **Crime Risk Integration**: Spatial buffering around crime locations
- 🎯 **Multi-Objective Algorithms**: Distance vs safety optimization
- 🎯 **Pathfinding Implementation**: Shortest path with risk weighting
- 🎯 **Route Analysis**: Generate and evaluate optimal paths

---

## 📊 Key Network Statistics
- **Coverage**: 6,766.8 km of walkable infrastructure
- **Resolution**: 44.2m average segment length
- **Connectivity**: 6.01 average streets per intersection
- **Validation**: Fully connected graph (no isolated components)
- **Performance**: <50ms average pathfinding runtime (50-node paths)

The project foundation is **complete and validated**, ready for advanced routing algorithm implementation.
