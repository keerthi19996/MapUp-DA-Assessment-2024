from typing import Dict, List
import pandas as pd
from typing import Dict, Any
import re
import polyline
from geopy.distance import geodesic


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    for i in range(0, len(lst), n):
        start = i
        end = i+(n-1)
        if(end >= len(lst)):
            end-=1
            while (start<=end):
                temp = lst[start]
                lst[start] = lst[end]
                lst[end] = temp
                start+=1
                end-=1
        else:
            while (start<=end):
                temp = lst[start]
                lst[start] = lst[end]
                lst[end] = temp
                start+=1
                end-=1
    return lst

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    output = {}
    for item in lst:
        itemLength = len(item)
        if(output.get(itemLength) != None):
            s:list = output.get(itemLength)
            s.append(item)
        else:
            output[itemLength] = [item]
        
    keys = list(output.keys())
    keys.sort()

    # Sorted Dictionary
    sortedDictionary = {i: output[i] for i in keys}
    return sortedDictionary

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten(nested_dict: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        items = {}
        for k, v in nested_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.update(flatten(item, f"{new_key}[{i}]"))
                    else:
                        items[f"{new_key}[{i}]"] = item
            else:
                items[new_key] = v
        return items
    
    return flatten(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()
        for i in range(start, len(nums)):
            if nums[i] not in seen: 
                seen.add(nums[i])
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]

    nums.sort()
    result = []
    backtrack(0)
    return result

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    regExPattern = re.compile(r'(?:\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b)')
    return re.findall(regExPattern, text)

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """

    decoded_points = polyline.decode(polyline_str)


    latitudes = []
    longitudes = []
    distances = []

    for i in range(len(decoded_points)):
        lat, lon = decoded_points[i]
        latitudes.append(lat)
        longitudes.append(lon)

        if i > 0:
            distance = geodesic(decoded_points[i - 1], decoded_points[i]).meters
            distances.append(distance)
        else:
            distances.append(0) 

    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance_meters': distances
    })

    return df

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    # Step 2: Create a new matrix to store the final sums
    final_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Calculate the sum excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7-day period.

    Args:
        df (pandas.DataFrame): Input DataFrame with at least the columns 'id', 'id_2', and 'timestamp'.

    Returns:
        pd.Series: A boolean series indicating whether each unique (`id`, `id_2`) pair has complete coverage.
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    grouped = df.groupby(['id', 'id_2'])

    def check_time_coverage(group):
        start_time = group['timestamp'].min()
        end_time = group['timestamp'].max()
        
        if (end_time - start_time).days < 7:
            return False
        
        return (end_time - start_time).total_seconds() >= 24 * 3600

    result = grouped.apply(check_time_coverage)

    return result
