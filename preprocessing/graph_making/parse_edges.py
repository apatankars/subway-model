import pandas as pd
from collections import defaultdict

def load_gtfs_data(stops_path, routes_path, trips_path, stop_times_path):
    """
    load GTFS data from inputted file paths into pandas dfs

    return:
        tuple: stops_df, routes_df, trips_df, stop_times_df, stop_id_to_name_map
    """
    stops_df = pd.read_csv(stops_path)
    routes_df = pd.read_csv(routes_path)
    trips_df = pd.read_csv(trips_path)
    stop_times_df = pd.read_csv(stop_times_path)

    # Create a mapping from stop_id to stop_name for easy lookup
    stop_id_to_name_map = stops_df.set_index('stop_id')['stop_name'].to_dict()

    print("done loding GTFS data")
    return stops_df, routes_df, trips_df, stop_times_df, stop_id_to_name_map

def edge_by_trip_unique_lines(stop_times_df, trips_df, routes_df):
    '''
    using stop_times df iterate through each unique trip_id,
    find adjacent stop_ids based on stop_sequence, 
    count the amount of distinct lines (route_short_name) connecting each pair of stops

    inputs:
        stop_times_df (pd.df): df containing stop time information
        trips_df (pd.df): df containing trip information (trip_id to route_id)
        routes_df (pd.df): df containing route information (route_id to route_short_name)

    return:
        list: list of tuples, where each tuple is
              (smaller_stop_id, larger_stop_id, unique_line_count)
    '''
    edge_lines = defaultdict(set)

    # merge dataframes to get route_short_name for each stop_time
    try:
        # ensure necessary columns exist before merge
        if 'trip_id' not in stop_times_df.columns or 'trip_id' not in trips_df.columns:
            raise ValueError("no 'trip_id' column in stop_times_df or trips_df")
        if 'route_id' not in trips_df.columns or 'route_id' not in routes_df.columns:
             raise ValueError("no 'route_id' column in trips_df or routes_df")
        if 'route_short_name' not in routes_df.columns:
             raise ValueError("no 'route_short_name' column in routes_df")

        merged_df = stop_times_df.merge(trips_df[['trip_id', 'route_id']], on='trip_id', how='left')
        merged_df = merged_df.merge(routes_df[['route_id', 'route_short_name']], on='route_id', how='left')
    except Exception as e:
        print(f"error during merge: {e}")
        return []


    # group by trip_id
    grouped_trips = merged_df.groupby('trip_id')

    for trip_id, group in grouped_trips:
        # sort stops within the trip by sequence
        sorted_stops = group.sort_values('stop_sequence')

        # list of stop_ids in order
        stop_ids_in_trip = sorted_stops['stop_id'].tolist()
        # line name for trip (should be consistent within the group)
        line_name = sorted_stops['route_short_name'].iloc[0] # Get from first row

        # iter thru consecutive pairs of stop_ids
        for i in range(len(stop_ids_in_trip) - 1):
            u = stop_ids_in_trip[i]
            v = stop_ids_in_trip[i+1]
            if u != v: # no self-edfes
                #(smaller_id, larger_id) order for the tuple key
                # remove S or N from end
                if u[-1] == 'N' or u[-1] == 'S':
                    u = u[:-1]
                if v[-1] == 'N' or v[-1] == 'S':
                    v = v[:-1]
                edge = tuple(sorted((u, v)))
                # add line name to the set for this edge
                edge_lines[edge].add(line_name)

    # convert the edge_lines dictionary to list format
    edge_list_with_line_counts = [(edge[0], edge[1], len(lines)) for edge, lines in edge_lines.items()]

    print(f"{len(edge_list_with_line_counts)} stop pairs")
    return edge_list_with_line_counts


# if __name__ == "__main__":
#     STOPS_FILE = 'data/gtfs_subway/stops.txt'
#     ROUTES_FILE = 'data/gtfs_subway/routes.txt'
#     TRIPS_FILE = 'data/gtfs_subway/trips.txt'
#     STOP_TIMES_FILE = 'data/gtfs_subway/stop_times.txt'
#     OUTPUT_GRAPH_FILE = 'subway_graph_parent_nodes_by_id.pkl' # Example filename

#     # load data
#     stops, routes, trips, stop_times, id_to_name_map = load_gtfs_data(
#         STOPS_FILE, ROUTES_FILE, TRIPS_FILE, STOP_TIMES_FILE
#     )

#     # generate edge pairs with unique line counts
#     consecutive_edges_with_line_counts = edge_by_trip_unique_lines(stop_times, trips, routes)

#     # TESTING!!!!
#     # some example edges with line counts (sorted by count descending)
#     print("ex edges with unique line counts (top 10 by count):")
    
    
#     sorted_edges = sorted(consecutive_edges_with_line_counts, key=lambda x: x[2], reverse=True)
#     for i, edge_data in enumerate(sorted_edges):
#         if i >= 10:
#             break
#         print(f"edge: ({edge_data[0]}, {edge_data[1]}), unique lines: {edge_data[2]}")

#     print("10 Random Example edges with unique line counts:")
    
#     if consecutive_edges_with_line_counts: # Check if the list is not empty
#         # number of samples
#         num_samples = min(10, len(consecutive_edges_with_line_counts))
#         random_samples = random.sample(consecutive_edges_with_line_counts, num_samples)
#         for edge_data in random_samples:
#             print(f" edge: ({edge_data[0]}, {edge_data[1]}), unique lines: {edge_data[2]}")
#     else:
#         print("no edges generated.")