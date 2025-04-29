import networkx as nx
import pandas as pd
import pickle

# import helpers
from parse_lines import lines_from_station_complex
from parse_edges import load_gtfs_data, edge_by_trip_unique_lines
from add_edges import add_edges

def populate_nodes_from_parquet(parquet_path):
    """
    loads data from a Parquet file, initializes a NetworkX graph, and adds nodes
    based on unique station_complex_id
    extract station name and lines using
    the imported lines_from_station_complex function.

    inputs:
        parquet_path (str): path to the Parquet file containing station data.
                            expected columns: "station_complex_id", "station_complex",
                            "borough", "latitude", "longitude".
                            Other columns like "transit_timestamp", "ridership", "transfers" are ignored
                            for node creation but exist in the source data.

    return:
        nx.Graph: networkx graph with nodes added, keyed by station_complex_id.
                   None if the file cannot be loaded or required columns
                  are missing.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"ERRRRRROR: no Parquet file not found at {parquet_path}")
        return None
    except Exception as e:
        print(f"ERRRRRROR loading Parquet file: {e}")
        return None

    required_cols = ["station_complex_id", "station_complex", "borough", "latitude", "longitude"]
    if not all(col in df.columns for col in required_cols):
        print(f"ERRRRRROR: Parquet file missing one or more required columns: {required_cols}")
        return None

    G = nx.Graph()
    nodes_added = set() # track added station_complex_id

    for _, row in df.iterrows():
        complex_id = row['station_complex_id']

        # skip if complex_id is missing or already added
        if pd.isna(complex_id) or complex_id in nodes_added:
            continue

        # extract station name and lines using imported helper functions
        station_name, lines = lines_from_station_complex(row['station_complex'])

        # add node with relevant attributes
        G.add_node(
            str(complex_id),
            station_complex=row['station_complex'],
            station_name=station_name,
            lines=lines, 
            borough=row['borough'],
            latitude=row['latitude'],
            longitude=row['longitude'],
            coordinates=(row['latitude'], row['longitude']) # nice to have as tuple
        )
        nodes_added.add(complex_id)

    print(f"graph initialized with {G.number_of_nodes()} unique nodes")
    return G

if __name__ == "__main__":
    PARQUET_FILE = 'data/turnstile_data/2023_turnstile_data.parquet'
    graph = populate_nodes_from_parquet(PARQUET_FILE)
    
    df_stops = pd.read_csv('data/gtfs_June_2023/stops.txt')
    df_stops = df_stops[df_stops['location_type'] == 1].reset_index(drop=True)

    # if graph and graph.number_of_nodes() > 0:
    #     # print info for the first node
    #     first_node_id = list(graph.nodes())[0]
    #     print(f"\test attr. for node '{first_node_id}':")
    #     print(graph.nodes[first_node_id])

    #     # ex: find node with multiple lines if possible
    #     node_with_lines = None
    #     for node_id, data in graph.nodes(data=True):
    #         if data.get('lines') and len(data['lines']) > 1:
    #             node_with_lines = node_id
    #             break
    #     if node_with_lines:
    #          print(f"\nAttributes for node '{node_with_lines}' (example with lines):")
    #          print(graph.nodes[node_with_lines])

    STOPS_FILE = 'data/gtfs_June_2023/stops.txt'
    ROUTES_FILE = 'data/gtfs_June_2023/routes.txt'
    TRIPS_FILE = 'data/gtfs_June_2023/trips.txt'
    STOP_TIMES_FILE = 'data/gtfs_June_2023/stop_times.txt'
    # OUTPUT_GRAPH_FILE = 'subway_graph_parent_nodes_by_id.pkl' # Example filename

    # load data
    stops, routes, trips, stop_times, id_to_name_map = load_gtfs_data(
        STOPS_FILE, ROUTES_FILE, TRIPS_FILE, STOP_TIMES_FILE
    )

    # make edge pairs with unique line counts
    consecutive_edges_with_line_counts = edge_by_trip_unique_lines(stop_times, trips, routes)
    
    stop_id_to_sc_id = {}
    add_edges(graph, consecutive_edges_with_line_counts, stop_id_to_sc_id, df_stops)
    
    
    with open("subway_network.pkl", "wb") as f:
        pickle.dump(graph, f)
    
    # if already saved graph:
    # with open("subway_network.pkl", "rb") as f:
    #     graph = pickle.load(f)

    # test edges 
    # node1 = list(graph.nodes())[0]
    
    # for neighbor in graph.neighbors(node1):
    #     print(f"Edge from {node1} to {neighbor}")
    
    # node2 = '611'
    
    # for neighbor in graph.neighbors(node2):
    #     print(f"Edge from {node2} to {neighbor}")