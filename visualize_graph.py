import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def load_graph(file_path):
    df = pd.read_excel(file_path)
    names = df.iloc[:, 0].tolist()
    adj_matrix = df.iloc[:, 1:].values
    G = nx.from_numpy_array(adj_matrix)
    mapping = {i: name for i, name in enumerate(names)}
    G = nx.relabel_nodes(G, mapping)
    return G

def plot_degree_distribution(G):
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(10, 6))
    sns.histplot(degrees, kde=True, bins=20, color='skyblue')
    plt.title("Degree Distribution - Facebook Network")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig("degree_distribution.png", dpi=300)
    print("Saved degree_distribution.png")

def plot_communities(G, node_metadata):
    plt.figure(figsize=(12, 12))
    
    # Get community IDs
    communities = [node_metadata[node]["community"] for node in G.nodes()]
    
    # Use a spring layout
    print("Computing layout (this might take a moment)...")
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=communities, cmap=plt.cm.tab10, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')
    
    plt.title("Facebook Network Community Visualization")
    plt.axis('off')
    plt.savefig("community_visualization.png", dpi=300, bbox_inches='tight')
    print("Saved community_visualization.png")

if __name__ == "__main__":
    file_path = os.path.join("Huawei Social Data", "Facebook_Data.xlsx")
    with open("analysis_results.json", "r") as f:
        analysis_results = json.load(f)
        
    G = load_graph(file_path)
    
    plot_degree_distribution(G)
    plot_communities(G, analysis_results["node_metadata"])
