# Social Network Analysis: Huawei Facebook Dataset

This project performs an end-to-end Social Network Analysis (SNA) on the Huawei Facebook dataset.

## Features
- **Exploratory Data Analysis**: Basic network statistics (Density, Diameter, Clustering).
- **Centrality Analysis**: Identification of influential nodes using Degree, Closeness, Betweenness, and Eigenvector centralities.
- **Community Detection**: Natural user grouping using the Louvain algorithm.
- **Advanced Analysis**: Link Prediction using Common Neighbors, Adamic-Adar, and Katz similarity with rigorous hold-out evaluation.
- **Interactive Dashboard**: A Streamlit application for dynamic exploration.

## Project Structure
- `initial_analysis.py`: Performs initial EDA and graph construction.
- `centrality_community_analysis.py`: Computes centralities and detects communities.
- `link_prediction_analysis.py`: Implements the link prediction workflow.
- `visualize_graph.py`: Generates high-quality static visualizations for the report.
- `app.py`: The Streamlit dashboard.
- `Huawei Social Data/`: Contains the raw Excel datasets.
- `basic_stats.json`, `analysis_results.json`, `link_prediction_results.json`: Cached analysis data for the dashboard.

## Setup Instructions

### 1. Requirements
Ensure you have Python 3.10+ installed. Install the required libraries:
```bash
pip install pandas networkx matplotlib seaborn scikit-learn openpyxl pyvis streamlit
```

### 2. Running the Analysis
The analysis results are already pre-computed in the `.json` files. If you want to re-run the full pipeline:
```bash
python initial_analysis.py
python centrality_community_analysis.py
python link_prediction_analysis.py
python visualize_graph.py
```

### 3. Launching the Dashboard
To launch the interactive dashboard:
```bash
streamlit run app.py
```

## Authors
- Individual Project by [Omar Elborollosy]
- Course: C-DE422 – Big Data Engineering II: Social Network Analytics
- Spring 2026
