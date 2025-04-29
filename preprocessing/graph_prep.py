import pandas as pd
import pickle

def get_spatial_data():
    with open("data/subway_network.pkl", "rb") as f:
        graph = pickle.load(f)
    
    node_86 = graph.nodes['86']
    node_193 = graph.nodes['193']

    node_86["station_complex"] = "Cypress Hills (J)"
    node_86["station_name"] = "Cypress Hills"
    node_86["lines"] = ['J']
    node_86["borough"] = "Brooklyn"

    node_193["station_complex"] = "104 St (A)"
    node_193["station_name"] = "104 St"
    node_193["lines"] = ['A']
    node_193["borough"] = "Queens"

    with open("data/subway_network.pkl", "wb") as f:
        pickle.dump(graph, f)

    graph_data = []
    for node, attrs in graph.nodes(data=True):
        graph_data.append({
            'node': node,
            'borough': attrs.get('borough'),
            'lines': attrs.get('lines')
        })

    graph_df = pd.DataFrame(graph_data)
    graph_df = graph_df.set_index('node')

    # Gathering all unique lines
    unique_lines = set()
    for entry in graph_df['lines']:
        for line in entry:
            unique_lines.add(line)

    # Final data
    borough_onehot = pd.get_dummies(graph_df['borough'])
    line_multi_hot = pd.DataFrame({
        f'{line}': graph_df['lines'].apply(lambda x: int(line in x))
        for line in unique_lines
    })

    return pd.concat([borough_onehot, line_multi_hot], axis=1)