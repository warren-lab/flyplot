"""
In order to utilize the following methods a preliminary analysis is necessary analyzing the 
preliminary plots of the fly.

This is useful for getting explicitly defined sections if using static data. 
Not good for loop led data

We will be able to achieve the extraction of multiple 90 second chunks 
as well as an additional 30s dark period for both male and female


This is the `BASIC` Method as it is not as robust as the `EXACT` method in the other module in this directory
"""

import numpy as np

def find_index(idx,data,disp = 30, findstart= False):
    """
    parameters:
        idx: this is the known start or ending index
        data: this is the data that we are wanting to index
        disp: this is the total length of time we are wanting to capture
        findstart: whether or not the wanting to determine start or end index, default is set to False
    return the closest index and row
    """
    if findstart:
        # this means that need find the starting index
        target_value = data[idx] - disp
    else:
        # find end
        target_value = data[idx]+ disp

    # Calculate absolute differences between target value and the values in the DataFrame
    abs_differences = np.abs(data - target_value)
    # Find the index of the closest value
    closest_index = abs_differences.idxmin()

    # Extract the row with the closest value
    closest_row = data.loc[closest_index]
    return closest_index,closest_index


def find_index_by_value(target_value,data):
    """
    parameters:
        target_value: the value we are looking for the index of
        data: the data array we are search in
    returns the index for the particular value in that array
    """
    # Calculate absolute differences between target value and the values in the DataFrame
    abs_differences = np.abs(data - target_value)
    # Find the index of the closest value
    closest_index = abs_differences.idxmin()

    return closest_index

def find_dark_end(data):
    """
    Parameters:
        data: dataframe associated with fly
    returns the index that is end of dark and start of LED
    """
    cutoff_index = len(data['led position'])//2
    cut_data = data.loc[:cutoff_index] 
    filtered_data = cut_data[(cut_data['led position'] == -1) | (cut_data['led position'] == 150)]
    return filtered_data.index.max(),  filtered_data.index.max()+1
def get_led_position(idx, data):
    """
    Parameters:
        idx: the idx we want to classify as the start of experiment after dark period
        data: dataframe associated with fly

    returns the led position based on the idx we want to consider as the start of the experiment and the 
    data related to the fly dataframe  
    """
    
    return data['led position'].loc[idx]

def get_start_end_from_led(data):
    """
    Parameters:
        - data: dataframe associated with fly
    
    First will determine the first change in led, and based on that we will get the starting index.
    Also determined the ending index based on when it changed back to 150 aka dark mode

    Returns:
        - starting led num
        - the index for start of cycle
        - the index for end of cycle
    """
    return