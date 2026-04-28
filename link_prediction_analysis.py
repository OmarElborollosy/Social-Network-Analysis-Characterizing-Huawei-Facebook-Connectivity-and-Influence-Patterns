import pandas as pd
import networkx as nx
import numpy as np
import os
import random
from sklearn.metrics import roc_auc_score
import json

def load_graph(file_path):
    df = pd.read_excel(file_path)
    names = df.iloc[:, 0].tolist()
    adj_matrix = df.iloc[:, 1:].values
    G = nx.from_numpy_array(adj_matrix)
    mapping = {i: name for i, name in enumerate(names)}
    G = nx.relabel_nodes(G, mapping)
    return G

def split_graph(G, test_ratio=0.1):
    all_edges = list(G.edges())
    random.seed(42)
    random.shuffle(all_edges)
    
    num_test = int(len(all_edges) * test_ratio)
    test_edges = all_edges[:num_test]
    train_edges = all_edges[num_test:]
    
    # Create training graph
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)
    
    return G_train, test_edges

def compute_katz_similarity(G_train, beta=0.005):
    # S = (I - beta * A)^-1 - I
    A = nx.adjacency_matrix(G_train).todense()
    I = np.identity(len(G_train))
    S = np.linalg.inv(I - beta * A) - I
    return S

def get_link_prediction_scores(G_train, test_edges, katz_matrix, nodes_list):
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    
    # Candidates: all non-edges in G_train
    # To keep it efficient, we'll evaluate all true test edges and a sample of negative edges
    # OR if manageable, all non-edges. 1000 nodes -> ~500k pairs.
    
    # We need to distinguish between:
    # 1. Positive edges (test_edges)
    # 2. Negative edges (pairs that are NOT in G at all)
    
    existing_edges = set(G_train.edges()) | set(test_edges)
    
    # Build a set of all non-edges
    all_nodes = list(G_train.nodes())
    neg_candidates = []
    
    # We want to score:
    # - The test edges (positive)
    # - All other possible links that don't exist in G (negative)
    
    # To make evaluation robust and manageable:
    # Let's take all test edges and an equal number of negative edges for AUC
    # AND rank everything for Precision@k
    
    print("Computing scores for candidates...")
    
    # Methods: CN, Adamic-Adar, Katz
    cn_results = list(nx.common_neighbor_centrality(G_train)) # This returns (u, v, score) for all non-edges? 
    # Actually nx.common_neighbors(G, u, v) is better.
    
    # Let's define a scorer function
    def get_scores(u, v):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        
        # Common Neighbors
        cn = len(list(nx.common_neighbors(G_train, u, v)))
        
        # Adamic-Adar
        try:
            aa = list(nx.adamic_adar_index(G_train, [(u, v)]))[0][2]
        except (ZeroDivisionError, IndexError):
            aa = 0
            
        # Katz
        katz = katz_matrix[u_idx, v_idx]
        
        return cn, aa, katz

    # For ranking, we really need to score all non-edges of G_train
    # Non-edges of G_train = test_edges + true negatives
    
    results = []
    
    # All possible pairs
    print("Iterating over all pairs (this might take a while for 1000 nodes)...")
    import itertools
    all_pairs = list(itertools.combinations(all_nodes, 2))
    
    y_true = []
    scores_cn = []
    scores_aa = []
    scores_katz = []
    
    # To avoid 500k loop in Python being too slow, we can optimize
    # But let's see if it's okay.
    
    test_edges_set = set(tuple(sorted(e)) for e in test_edges)
    train_edges_set = set(tuple(sorted(e)) for e in G_train.edges())
    
    count = 0
    for u, v in all_pairs:
        pair = tuple(sorted((u, v)))
        if pair in train_edges_set:
            continue
            
        count += 1
        is_positive = 1 if pair in test_edges_set else 0
        y_true.append(is_positive)
        
        cn, aa, katz = get_scores(u, v)
        scores_cn.append(cn)
        scores_aa.append(aa)
        scores_katz.append(katz)
        
        if count % 50000 == 0:
            print(f"Processed {count} pairs...")

    return y_true, scores_cn, scores_aa, scores_katz

def evaluate(y_true, scores, name, k=100):
    auc = roc_auc_score(y_true, scores)
    
    # Precision@k
    # Rank indices by score
    sorted_indices = np.argsort(scores)[::-1]
    top_k_indices = sorted_indices[:k]
    
    precision_at_k = sum(y_true[i] for i in top_k_indices) / k
    
    print(f"--- {name} ---")
    print(f"AUC: {auc:.4f}")
    print(f"Precision@{k}: {precision_at_k:.4f}")
    
    return {"auc": auc, "precision_at_k": precision_at_k}

if __name__ == "__main__":
    file_path = os.path.join("Huawei Social Data", "Facebook_Data.xlsx")
    G = load_graph(file_path)
    
    G_train, test_edges = split_graph(G, test_ratio=0.1)
    nodes_list = list(G_train.nodes())
    
    print(f"Train edges: {G_train.number_of_edges()}")
    print(f"Test edges: {len(test_edges)}")
    
    katz_matrix = compute_katz_similarity(G_train)
    
    y_true, scores_cn, scores_aa, scores_katz = get_link_prediction_scores(G_train, test_edges, katz_matrix, nodes_list)
    
    metrics = {}
    metrics["Common Neighbors"] = evaluate(y_true, scores_cn, "Common Neighbors")
    metrics["Adamic-Adar"] = evaluate(y_true, scores_aa, "Adamic-Adar")
    metrics["Katz"] = evaluate(y_true, scores_katz, "Katz")
    
    with open("link_prediction_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Link prediction analysis completed.")
