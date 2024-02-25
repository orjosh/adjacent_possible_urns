import sys
sys.path.append("scripts/")
from analysisfuncs import csv_to_graph
import networkx as nx
import pandas as pd


bridge_data_path = "adjpos_islands_bridge_run1.csv"
nobridge_data_path = "adjpos_islands_nobridge_run1.csv"

G_bridge = csv_to_graph(bridge_data_path)
G_nobridge = csv_to_graph(nobridge_data_path)

n_islands_bridge = nx.number_connected_components(G_bridge)
largest_cc_bridge = max(nx.connected_components(G_bridge), key=len)

n_islands_nobridge = nx.number_connected_components(G_nobridge)
largest_cc_nobridge = max(nx.connected_components(G_nobridge), key=len)

print(f"Bridge graph has {G_bridge.number_of_nodes()} nodes and {n_islands_bridge} connected components")
print(f"Largest component is of size {len(largest_cc_bridge)}")

print(f"\nBridgeless graph has {G_nobridge.number_of_nodes()} nodes and {n_islands_nobridge} connected components")
print(f"Largest component is of size {len(largest_cc_nobridge)}")