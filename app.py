import streamlit as st
import pandas as pd
import networkx as nx
import json
import os
from pyvis.network import Network
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(page_title="Huawei Social Network Analysis", layout="wide")

st.title("Huawei Social Network Analysis Dashboard")
st.markdown("Analyzing Facebook connectivity patterns, influence, and communities.")

# Load data helper
@st.cache_data
def load_data():
    with open("basic_stats.json", "r") as f:
        basic_stats = json.load(f)
    with open("analysis_results.json", "r") as f:
        analysis_results = json.load(f)
    with open("link_prediction_results.json", "r") as f:
        link_prediction_results = json.load(f)
    return basic_stats, analysis_results, link_prediction_results

basic_stats, analysis_results, link_prediction_results = load_data()

# Sidebar
st.sidebar.header("Analysis Options")
view_mode = st.sidebar.selectbox("Select View", ["Overview", "Centrality", "Communities", "Link Prediction", "Interactive Graph"])

if view_mode == "Overview":
    st.header("Network Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", basic_stats["nodes"])
    col2.metric("Edges", basic_stats["edges"])
    col3.metric("Density", f"{basic_stats['density']:.4f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Diameter", basic_stats["diameter"])
    col5.metric("Avg Clustering", f"{basic_stats['avg_clustering']:.4f}")
    col6.metric("Connected Components", basic_stats["num_cc"])

elif view_mode == "Centrality":
    st.header("Centrality Analysis")
    st.markdown("Identifying the most influential accounts across different measures.")
    
    measure = st.selectbox("Select Centrality Measure", ["Degree", "Closeness", "Betweenness", "Eigenvector"])
    
    top_5 = analysis_results["top_5"][measure]
    df_top5 = pd.DataFrame(top_5)
    
    st.subheader(f"Top 5 Nodes: {measure} Centrality")
    st.table(df_top5)
    
    st.info("Centrality measure explanations are available in the project report.")

elif view_mode == "Communities":
    st.header("Community Detection")
    st.markdown(f"Found **{analysis_results['community_stats']['num_communities']}** communities using the Louvain algorithm.")
    
    sizes = analysis_results["community_stats"]["community_sizes"]
    st.bar_chart(sizes)
    st.caption("Community sizes (number of nodes per community)")

elif view_mode == "Link Prediction":
    st.header("Advanced Analysis: Link Prediction")
    st.markdown("Predicting potential future connections using structural similarity.")
    
    df_lp = pd.DataFrame(link_prediction_results).T
    st.table(df_lp)
    
    st.warning("Note: The random-like AUC scores suggest this network may be modeled closely by a random graph structure, lacking strong triadic closure for these specific predictors.")

elif view_mode == "Interactive Graph":
    st.header("Interactive Network Visualization")
    st.markdown("Visualizing the top connected nodes to maintain performance.")
    
    # Due to size (1000 nodes/50k edges), we visualize a subgraph of top degree nodes
    # or a specific community.
    
    sample_nodes = st.sidebar.slider("Number of top nodes to show", 50, 500, 100)
    
    # Generate the graph for Pyvis
    # We need the actual G for this. 
    # For performance in Streamlit, we'll build a small subgraph.
    
    @st.cache_resource
    def generate_subgraph_html(num_nodes):
        # We need to rebuild G briefly or load from file
        df = pd.read_excel(os.path.join("Huawei Social Data", "Facebook_Data.xlsx"))
        names = df.iloc[:, 0].tolist()
        adj_matrix = df.iloc[:, 1:].values
        G = nx.from_numpy_array(adj_matrix)
        mapping = {i: name for i, name in enumerate(names)}
        G = nx.relabel_nodes(G, mapping)
        
        # Get top nodes by degree
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:num_nodes]
        subgraph_nodes = [n[0] for n in top_nodes]
        sub_G = G.subgraph(subgraph_nodes)
        
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        
        # Add nodes with community colors
        for node in sub_G.nodes():
            comm_id = analysis_results["node_metadata"][node]["community"]
            net.add_node(node, label=node, title=f"Community {comm_id}", group=comm_id)
            
        for u, v in sub_G.edges():
            net.add_edge(u, v)
            
        net.toggle_physics(False) # Better performance for larger samples
        return net.generate_html()

    html_content = generate_subgraph_html(sample_nodes)
    components.html(html_content, height=650)
