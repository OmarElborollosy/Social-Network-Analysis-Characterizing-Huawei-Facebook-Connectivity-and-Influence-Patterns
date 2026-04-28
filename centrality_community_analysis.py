import pandas as pd
import networkx as nx
import os
import json
import numpy as np

def load_graph(file_path):
    df = pd.read_excel(file_path)
    names = df.iloc[:, 0].tolist()
    adj_matrix = df.iloc[:, 1:].values
    G = nx.from_numpy_array(adj_matrix)
    mapping = {i: name for i, name in enumerate(names)}
    G = nx.relabel_nodes(G, mapping)
    return G

def run_centrality_analysis(G):
    print("Computing Centrality Measures...")
    
    # Degree Centrality
    degree_cent = nx.degree_centrality(G)
    
    # Closeness Centrality
    closeness_cent = nx.closeness_centrality(G)
    
    # Betweenness Centrality
    print("Computing Betweenness (this may take a moment)...")
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Eigenvector Centrality
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
    
    centrality_data = {
        "Degree": degree_cent,
        "Closeness": closeness_cent,
        "Betweenness": betweenness_cent,
        "Eigenvector": eigenvector_cent
    }
    
    # Get Top-5 for each
    top_5 = {}
    for measure, data in centrality_data.items():
        sorted_nodes = sorted(data.items(), key=lambda item: item[1], reverse=True)
        top_5[measure] = [{"node": n, "score": s} for n, s in sorted_nodes[:5]]
        
    return centrality_data, top_5

def run_community_detection(G):
    print("Running Louvain Community Detection...")
    
    communities = nx.community.louvain_communities(G, seed=42)
    
    # Map nodes to community IDs
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i
            
    community_stats = {
        "num_communities": len(communities),
        "community_sizes": [len(c) for c in communities]
    }
    
    print(f"Found {len(communities)} communities.")
    return node_to_community, community_stats

if __name__ == "__main__":
    file_path = os.path.join("Huawei Social Data", "Facebook_Data.xlsx")
    if os.path.exists(file_path):
        G = load_graph(file_path)
        
        centrality_data, top_5 = run_centrality_analysis(G)
        node_to_community, community_stats = run_community_detection(G)
        
        # Consolidate results for saving
        # Note: Centrality data is a dictionary of dictionaries, we'll simplify for JSON
        results = {
            "top_5": top_5,
            "community_stats": community_stats,
            "node_metadata": {}
        }
        
        # Add centrality and community info per node
        for node in G.nodes():
            results["node_metadata"][node] = {
                "community": node_to_community[node],
                "centrality": {
                    "Degree": centrality_data["Degree"][node],
                    "Closeness": centrality_data["Closeness"][node],
                    "Betweenness": centrality_data["Betweenness"][node],
                    "Eigenvector": centrality_data["Eigenvector"][node]
                }
            }
            
        with open("analysis_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        print("Analysis completed and saved to analysis_results.json")
    else:
        print(f"Error: {file_path} not found.")
