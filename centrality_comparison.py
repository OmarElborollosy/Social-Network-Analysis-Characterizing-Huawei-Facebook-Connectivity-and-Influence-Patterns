import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np

def generate_centrality_correlation():
    print("Loading analysis results...")
    with open("analysis_results.json", "r") as f:
        data = json.load(f)
    
    # Extract centrality scores for each node
    nodes = list(data["node_metadata"].keys())
    measures = ["Degree", "Closeness", "Betweenness", "Eigenvector"]
    
    centrality_df = pd.DataFrame(index=nodes, columns=measures)
    
    for node in nodes:
        for measure in measures:
            centrality_df.loc[node, measure] = data["node_metadata"][node]["centrality"][measure]
    
    # Convert to numeric
    centrality_df = centrality_df.apply(pd.to_numeric)
    
    # Compute Spearman correlation
    corr_matrix, p_values = spearmanr(centrality_df)
    corr_df = pd.DataFrame(corr_matrix, index=measures, columns=measures)
    
    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=0.5)
    plt.title("Spearman Rank Correlation - Centrality Measures")
    plt.savefig("centrality_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print("Saved centrality_correlation_heatmap.png")
    
    # Plot 2: Pairwise Scatter Plots
    plt.figure(figsize=(12, 10))
    g = sns.pairplot(centrality_df, diag_kind="kde", plot_kws={'alpha':0.5, 's':10, 'color':'#2c3e50'})
    g.fig.suptitle("Pairwise Distribution of Centrality Scores", y=1.02)
    plt.savefig("centrality_pairwise_distribution.png", dpi=300, bbox_inches='tight')
    print("Saved centrality_pairwise_distribution.png")
    
    # Compute some stats for the report
    stats = {
        "correlation": corr_df.to_dict(),
        "description": "High correlation between Degree and Betweenness suggests that the most connected nodes are also the primary gatekeepers."
    }
    with open("centrality_comparison_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    generate_centrality_correlation()
