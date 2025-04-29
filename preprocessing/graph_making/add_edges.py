from geopy.distance import geodesic
import pandas as pd
import networkx as nx
import numpy as np

def calculate_distance(coord1, coord2):
    """Calculate geodesic distance between two (lat, lon) tuples."""
    if pd.isna(coord1[0]) or pd.isna(coord1[1]) or pd.isna(coord2[0]) or pd.isna(coord2[1]):
        return float('inf') # Treat missing coords as infinitely far
    try:
        return geodesic(coord1, coord2).km
    except ValueError:
        return float('inf') # Handle potential errors in geopy

def find_station_id(graph: nx.Graph, stop_id: str, stop_id_to_sc_id: dict, stop_df: pd.DataFrame):
    # print(stop_df)
    stop_row = stop_df[stop_df["stop_id"] == str(stop_id)]
    stop_coords = (stop_row["stop_lat"].iloc[0], stop_row["stop_lon"].iloc[0])
    # stop_name = stop_row["stop_name"]
    min_distance = np.inf
    node_match = None
    
    for node in graph.nodes:
        node_coords = graph.nodes[node]['coordinates']
        
        curr_distance = calculate_distance(stop_coords, node_coords)
        
        if curr_distance < min_distance:
            min_distance = curr_distance
            node_match = node
    
    stop_id_to_sc_id[stop_id] = node_match
    
    return node_match

def add_edge(graph: nx.Graph, stop_ids: tuple[str, str, int], stop_id_to_sc_id: dict, stop_df: pd.DataFrame):
    stop1_id = stop_ids[0]
    stop2_id = stop_ids[1]
    num_connections = stop_ids[2]
    
    try:
        station_id1 = stop_id_to_sc_id[stop1_id]
    except KeyError:
        # Find station_id1 and add it to the dictionary
        station_id1 = find_station_id(graph, stop1_id, stop_id_to_sc_id, stop_df)
    
    try:
        station_id2 = stop_id_to_sc_id[stop2_id]
    except KeyError:
        # Find station_id2 and add it to the dictionary
        station_id2 = find_station_id(graph, stop2_id, stop_id_to_sc_id, stop_df)
    
    if not(graph.has_edge(station_id1, station_id2)):
        # Calculate distance between stations by getting latitude and longitude
        coord1 = graph.nodes[station_id1]['coordinates']
        coord2 = graph.nodes[station_id2]['coordinates']
        distance = calculate_distance(coord1, coord2)
        graph.add_edge(station_id1, station_id2, distance=distance, num_connections=num_connections)
        
def add_edges(graph, stop_pairs, stop_id_to_sc_id, stop_df):
    for pair in stop_pairs:
        add_edge(graph, pair, stop_id_to_sc_id, stop_df)
    
 
    
        