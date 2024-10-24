import pandas as pd
import numpy as np
from datetime import time


def calculate_distance_matrix(file_path: str) -> pd.DataFrame:
    """
    Calculate a distance matrix from the dataset provided in a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing distance data.

    Returns:
        pd.DataFrame: A symmetric distance matrix with distances between IDs.
    """
    df = pd.read_csv(file_path)
    
    distance_matrix = pd.DataFrame(columns=df['id_start'].unique(), index=df['id_start'].unique()).fillna(0)

    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance 

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if distance_matrix.at[id_start, id_end] > 0:
                for intermediate in distance_matrix.columns:
                    if distance_matrix.at[id_end, intermediate] > 0:
                        new_distance = distance_matrix.at[id_start, id_end] + distance_matrix.at[id_end, intermediate]
                        distance_matrix.at[id_start, intermediate] = min(
                            distance_matrix.at[id_start, intermediate],
                            new_distance) if distance_matrix.at[id_start, intermediate] > 0 else new_distance

    for id in distance_matrix.index:
        distance_matrix.at[id, id] = 0

    return distance_matrix

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_df = df.reset_index()
    
    unrolled_df = unrolled_df.melt(id_vars='index', var_name='id_end', value_name='distance')
    
    unrolled_df.rename(columns={'index': 'id_start'}, inplace=True)
    
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    
    return unrolled_df

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    average_distances = df.groupby('id_start')['distance'].mean().reset_index()
    average_distances.columns = ['id', 'average_distance']
    
    # Get the average distance for the reference ID
    reference_distance = average_distances[average_distances['id'] == reference_id]
    
    if reference_distance.empty:
        return pd.DataFrame()  # Return an empty DataFrame if reference_id is not found

    reference_avg = reference_distance['average_distance'].values[0]
    
    # Define the percentage threshold
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1
    
    # Find IDs within the threshold
    result_df = average_distances[(average_distances['average_distance'] >= lower_bound) &
                                   (average_distances['average_distance'] <= upper_bound)]
    
    return result_df

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
