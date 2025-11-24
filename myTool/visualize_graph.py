
import igraph as ig
import os
import networkx as nx
from pyvis.network import Network

# Path to the pickle file
pickle_path = 'outputs/demo_llama/meta-llama_Llama-3.1-8B-Instruct_facebook_contriever/graph.pickle'

if not os.path.exists(pickle_path):
    print(f"Error: File not found at {pickle_path}")
    exit(1)

print(f"Loading graph from {pickle_path}...")
g = ig.Graph.Read_Pickle(pickle_path)

print(f"Graph loaded.")
print(f"Nodes: {g.vcount()}")
print(f"Edges: {g.ecount()}")

# Convert to NetworkX for Pyvis
print("Converting to NetworkX...")
# igraph to networkx conversion
# We can iterate edges and add them to nx graph
nx_graph = nx.Graph() if not g.is_directed() else nx.DiGraph()

# Add nodes with attributes
for v in g.vs:
    # Pyvis expects 'label' for node text
    attrs = v.attributes()
    
    # Use 'content' for label if available, otherwise 'name', otherwise index
    # 'content' usually contains the human-readable text (e.g. "Cinderella", "prince")
    raw_label = attrs.get('content', attrs.get('name', str(v.index)))
    label = str(raw_label)
    
    # Truncate label if too long for visualization to keep it readable
    if len(label) > 30:
        label = label[:27] + "..."

    # If there is a 'content' attribute or similar, maybe use that as title (hover text)
    title = str(attrs)
    nx_graph.add_node(v.index, label=label, title=title, **attrs)

# Add edges
for e in g.es:
    attrs = e.attributes()
    # Transfer edge attributes
    nx_graph.add_edge(e.source, e.target, **attrs)

print(f"NetworkX graph created with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges.")

# Visualize with Pyvis
print("Generating visualization...")
net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=False)
net.from_nx(nx_graph)

# Set options for better physics/layout if needed, or use default
# net.show_buttons(filter_=['physics']) 

output_file = 'graph_visualization.html'
net.save_graph(output_file)
print(f"Visualization saved to {output_file}")
