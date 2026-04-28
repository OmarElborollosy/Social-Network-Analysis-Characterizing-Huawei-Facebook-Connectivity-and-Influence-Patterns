import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

def plot_lorenz_curve(G):
    print("Generating Lorenz Curve...")
    degrees = sorted([d for n, d in G.degree()])
    n = len(degrees)
    lorenz = np.cumsum(degrees) / sum(degrees)
    lorenz = np.insert(lorenz, 0, 0)
    
    # Gini Coefficient
    gini = (n + 1 - 2 * (np.sum(np.arange(1, n + 1) * degrees) / np.sum(degrees))) / n
    
    plt.figure(figsize=(8, 8))
    plt.plot(np.linspace(0, 1, len(lorenz)), lorenz, label=f'Degree Lorenz Curve (Gini={gini:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Equality')
    plt.fill_between(np.linspace(0, 1, len(lorenz)), np.linspace(0, 1, len(lorenz)), lorenz, color='orange', alpha=0.1)
    plt.title("Connectivity Inequality (Lorenz Curve)")
    plt.xlabel("Cumulative Proportion of Nodes")
    plt.ylabel("Cumulative Proportion of Degrees")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig("degree_lorenz_curve.png", dpi=300, bbox_inches='tight')
    return gini

def plot_kcore_shells(G):
    print("Computing K-Core decomposition...")
    core_numbers = nx.core_number(G)
    shells = list(core_numbers.values())
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x=shells, palette="viridis")
    plt.title("K-Core Shell Distribution")
    plt.xlabel("Coreness (Shell Number)")
    plt.ylabel("Number of Nodes")
    plt.savefig("kcore_distribution.png", dpi=300, bbox_inches='tight')
    
    return max(shells)

def plot_community_degree_comparison(G, analysis_results):
    print("Generating Community Degree Comparison...")
    data = []
    for node in G.nodes():
        data.append({
            "Node": node,
            "Degree": G.degree(node),
            "Community": analysis_results["node_metadata"][node]["community"]
        })
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Community", y="Degree", data=df, palette="tab10")
    plt.title("Degree Distribution by Community")
    plt.savefig("community_degree_boxplot.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    file_path = os.path.join("Huawei Social Data", "Facebook_Data.xlsx")
    with open("analysis_results.json", "r") as f:
        analysis_results = json.load(f)
        
    G = load_graph(file_path)
    
    gini = plot_lorenz_curve(G)
    max_k = plot_kcore_shells(G)
    plot_community_degree_comparison(G, analysis_results)
    
    stats = {
        "gini_coefficient": gini,
        "max_k_core": max_k
    }
    with open("advanced_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print("Advanced analytics completed.")
