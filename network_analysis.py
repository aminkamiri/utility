import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle_manager

def draw_network(T):
    circle_pos = nx.circular_layout(T)
    nx.draw_networkx(T,alpha=0.6,pos=circle_pos)
    plt.axis('off'); plt.show()

def draw_hist_degree_centrality(T):
    # Plot the degree distribution of the GitHub collaboration network
    plt.hist(list(nx.degree_centrality(T).values()))
    plt.show()

def draw_hist_betweenness_centrality(T):
    plt.hist(list(nx.betweenness_centrality(T).values()))
    plt.show()

def top_nodes(T, mode):
    if mode=='in_degree_centrality':
        bc=nx.in_degree_centrality(T)
    elif mode=='out_degree_centrality':
        bc=nx.out_degree_centrality(T)
    elif mode=='betweenness_centrality':
        bc=nx.betweenness_centrality(T)
    # bc = nx.betweenness_centrality(T)
    df_cent = pd.DataFrame(list(bc.items()),
                    columns = ['username', 'weight'])
    
    return df_cent.sort_values('weight', ascending = False)
    
# Define find_nodes_with_highest_deg_cent()
def find_nodes_with_highest_deg_cent(G):

    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)

    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values())) #nx.betweenness_centrality(G)

    nodes = set()

    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():

        # Check if the current value has the maximum degree centrality
        if v == max_dc:

            # Add the current node to the set of nodes
            nodes.add(k)

    return nodes

# # Find the node(s) that has the highest degree centrality in T: top_dc
# top_dc = find_nodes_with_highest_deg_cent(T)
# print(top_dc)

# # Write the assertion statement
# for node in top_dc:
#     assert nx.degree_centrality(T)[node] == max(nx.degree_centrality(T).values())

def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = []
    # Iterate over the nodes of interest
    for n in nodes_of_interest:

        # Append the nodes of interest to nodes_to_draw
        nodes_to_draw.append(n)
        # Iterate over all the neighbors of node n
        for nbr in G.neighbors(n):

            # Append the neighbors of n to nodes_to_draw
            nodes_to_draw.append(nbr)

    return G.subgraph(nodes_to_draw)

def get_nodes_and_nbrs2(T, nodes_of_interest):
    # Create the set of nodes: nodeset
    nodeset = set(nodes_of_interest)

    # Iterate over nodes
    for n in nodes_of_interest:

        # Compute the neighbors of n: nbrs
        nbrs = T.neighbors(n)

        # Compute the union of nodeset and nbrs: nodeset
        nodeset = nodeset.union(nbrs)

    # Compute the subgraph using nodeset: T_sub
    T_sub = T.subgraph(nodeset)
    return T_sub

from itertools import combinations
from collections import defaultdict
def return_recommended_connections(G):
    # Initialize the defaultdict: recommended
    recommended = defaultdict(int)

    # Iterate over all the nodes in G
    for n in G.nodes():

        # Iterate over all possible triangle relationship combinations
        for n1, n2 in combinations(G.neighbors(n), 2):

            # Check whether n1 and n2 do not have an edge
            if not G.has_edge(n1, n2):

                # Increment recommended
                recommended[(n1, n2)] += 1

    # Identify the top 10 pairs of users
    all_counts = sorted(recommended.values())
    top10_pairs = [pair for pair, count in recommended.items() if count > all_counts[-10]]
    return top10_pairs

def return_all_connected_component(T):
    # Calculate the largest connected component subgraph: largest_ccs
    ccs = sorted(nx.connected_components(T.to_undirected()), key=lambda x: len(x))
    return ccs

def return_largest_connected_component(T):
    # Calculate the largest connected component subgraph: largest_ccs
    ccs = sorted(nx.connected_components(T.to_undirected()), key=lambda x: len(x))
    largest_ccs=ccs[-1]
    return largest_ccs

