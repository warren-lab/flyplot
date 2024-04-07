"""
Methods to extract the proper data range
What this section provides is the necessary functions in order to access based on preliminary visualizations 
which of the different sections you want to start at and end at

you can select a loop, dark or static section to start or end...Then speifiy which loop, dark or staic section it corresponds to.. 
For loop sections this is still in progress to develop a better method to address accessing this. However, for the static sections 
this is working very well as we can use the saved indexes 
and then use the below methods to ignore the wrongly classified static sections and only focus on the sections that have a large set of static points.
"""


# Method to get the range that we want... basically will extract the specific values we want....
def find_startdata(start_led_type, start_led_num, data, data_dict):
    """
    parameters:
        start_led_type: this is the type "dark","stat", or "loop". That is a description of the section where you are wanting to start you data access from
        start_led_num: this corresponds to visually on the plot which number of section it is ... ex: 1st static section "stat" and 1 ..... or 2nd loop section "loop" and 2.
    
    This will return the starting key for the where data will be accessed from. 
    
    """
    # look at the folloowing dictionary that is going to hold all the data that relatesx to the experimental runs 
    ## find the start...
    count = 1
    starting_key = ''
    for k in data_dict:
        if k[:4] == start_led_type and len(data['delta-t'][data_dict[k][0]:data_dict[k][1]]) > 2:
            if count == start_led_num:
                starting_key = k
                return starting_key
            else:
                count +=1
    return starting_key

def find_enddata(end_led_type, end_led_num):
    """
    parameters:
        end_led_type: this is the type "dark","stat", or "loop". That is a description of the section where you are wanting to end you data access from
        end_led_num: this corresponds to visually on the plot which number of section it is ... ex: 1st static section "stat" and 1 ..... or 2nd loop section "loop" and 2.
    
    This will return the end key to which point the data will be accessed until. 
    """
    return find_startdata(end_led_type, end_led_num)