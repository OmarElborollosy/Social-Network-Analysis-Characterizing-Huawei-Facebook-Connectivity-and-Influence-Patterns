import pandas as pd
import networkx as nx
import os

def load_and_verify_data(file_path):
    print(f"Loading data from {file_path}...")
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # First column is names, the rest is the adjacency matrix
    names = df.iloc[:, 0].tolist()
    adj_matrix = df.iloc[:, 1:].values
    
    print(f"Data shape: {df.shape}")
    print(f"Number of nodes: {len(names)}")
    
    # Check for symmetry
    import numpy as np
    is_symmetric = np.allclose(adj_matrix, adj_matrix.T)
    print(f"Is matrix symmetric? {is_symmetric}")
    
    # Construct NetworkX graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Map node IDs to names
    mapping = {i: name for i, name in enumerate(names)}
    G = nx.relabel_nodes(G, mapping)
    
    return G

def compute_basic_stats(G):
    print("\nComputing Basic Statistics...")
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)
    
    # Diameter (requires a connected graph, or we do it per component)
    if nx.is_connected(G):
        diameter = nx.diameter(G)
    else:
        # For disconnected graphs, report diameter of the largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        print(f"Graph is disconnected. Diameter of the largest component: {diameter}")
    
    avg_clustering = nx.average_clustering(G)
    
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    print(f"Density: {density:.6f}")
    print(f"Diameter: {diameter}")
    print(f"Average Clustering Coefficient: {avg_clustering:.6f}")
    
    # Degree Distribution
    degrees = [d for n, d in G.degree()]
    print(f"Max Degree: {max(degrees)}")
    print(f"Min Degree: {min(degrees)}")
    print(f"Avg Degree: {sum(degrees)/len(degrees):.2f}")
    
    # Connected Components
    num_cc = nx.number_connected_components(G)
    print(f"Number of Connected Components: {num_cc}")
    
    return {
        "nodes": nodes,
        "edges": edges,
        "density": density,
        "diameter": diameter,
        "avg_clustering": avg_clustering,
        "num_cc": num_cc
    }

if __name__ == "__main__":
    file_path = os.path.join("Huawei Social Data", "Facebook_Data.xlsx")
    if os.path.exists(file_path):
        G = load_and_verify_data(file_path)
        stats = compute_basic_stats(G)
        
        # Save a small summary for later
        import json
        with open("basic_stats.json", "w") as f:
            json.dump(stats, f)
    else:
        print(f"Error: {file_path} not found.")
