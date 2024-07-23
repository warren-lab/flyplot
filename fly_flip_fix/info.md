# Fly Flip Fix
Within this module different algorithms were tested to deal with the issue of the 180 degree flip occuring within Magnotether Experiments

def dark_led_positions(data):
    """
    Parameters:
        - data -> the data for the fly.
    return start time for experiment and the dark mode
    """
    ## Start of dark period
    start_time_idx = data[(data['led position'] ==150) | (data['led position'] ==149)].index[0]
    start_time = data['time'][start_time_idx]
    ## find the point in which there is change in led_position
    end_time_idx = 0
    led_pos = data['led position'][start_time_idx:]
    for pnt in range(start_time_idx , len(data['led position'])): #checking values until we find a match
        if data['led position'][pnt] != 150 and data['led position'][pnt] != 149:
            end_time_idx = pnt # the index for the dark mode ending
            break
    end_time = data['time'][end_time_idx]
    


    ## get the indicies for the the last dark period

    return start_time, start_time_idx,end_time, end_time_idx

def cali_led_pos(data,df):
    """
    Parameters:
        - data -> the data for the fly (time, led position...etc)
    return the starting and ending index for the calibration
    """
    
    ## Start of dark period
    start_time_idx = data[(data['led position'] ==-1)].index[0]
    start_time = data['time'][start_time_idx]
    ## find the point in which there is change in led_position
    end_time_idx = 0
    led_pos = data['led position'][start_time_idx:]
    for pnt in range(start_time_idx , len(data['led position'])): #checking values until we find a match
        if data['led position'][pnt] != -1:
            end_time_idx = pnt # the index for the dark mode ending
            break
    end_time = data['time'][end_time_idx]
    df['cali'] = (start_time_idx,end_time_idx)
    # return df

def all_dark_led_pos(data,df):
    """
    Parameters:
        - data -> the data for the fly.
    return all dark led positions
    """    
    dark_led_lst=[]
    start_run = True
    led_pair = [None,None]
    count_dark = 0
    for led in range(1,len(data['led position'])):
        diff = abs(data['led position'][led]-data['led position'][led-1])
        if ((diff == 0) | (diff == 1)) and ((data['led position'][led-1] ==150) | (data['led position'][led-1] ==149)):
        # if (diff == 0 or diff == 1) and (data['led position'][led] == 150 or data['led position'][led] == 149):
            if start_run:
                start_run = False
                led_pair[0] = led-1
                # print("led pair",led_pair,start_run)
        elif diff > 1 and ((data['led position'][led] !=150) | (data['led position'][led] !=149)) and (start_run == False):
            start_run = True
            led_pair[1] = led
            dark_led_lst.append(tuple(led_pair))
            count_dark +=1
            df["dark_"+str(count_dark)] = tuple(led_pair)
            led_pair = [None,None]
            
            
        if ( led == len(data['led position'])-1) and (start_run == False) and ((data['led position'][led] == 150) | (data['led position'][led] == 149)):
            start_run = True
            led_pair[1] = led
            dark_led_lst.append(tuple(led_pair))
            count_dark +=1
            df["dark_"+str(count_dark)] = tuple(led_pair)
            led_pair = [None,None]
    # return dark_led_lst
def all_static_led_pos(data,df):
    """
    Parameters:
        - data -> the data for the fly.
    return the ranges of the different static led experiments that are not dark...
    """    
    static_led_lst=[]
    start_run = True
    led_pair = [None,None]
    count_static=0
    for led in range(1,len(data['led position'])):
        diff = abs(data['led position'][led]-data['led position'][led-1])
        if (diff == 0) and ((data['led position'][led-1] !=150) and (data['led position'][led-1] !=149) and (data['led position'][led-1] !=-1)):
            if start_run:
                start_run = False
                led_pair[0] = led-1
        elif (diff > 1) and (led>2) and (data['led position'][led-1] == data['led position'][led-2]) and (start_run == False):
            start_run = True
            led_pair[1] = led
            static_led_lst.append(tuple(led_pair))
            count_static +=1
            df["static_"+str(count_static)] = tuple(led_pair)
            led_pair = [None,None]

    return static_led_lst

def all_loop_led_pos(data,df):
    """
    Parameters:
        - data -> the data for the fly.
    return the ranges of the different looping led experiments
    """    
    loop_led_lst=[]
    start_run = True
    led_pair = [None,None]
    count_loop = 0
    for led in range(1,len(data['led position'])):
        diff = abs(data['led position'][led]-data['led position'][led-1])
        if (diff == 1) and ((data['led position'][led-1] !=150) and (data['led position'][led-1] !=149) and (data['led position'][led-1] !=-1)):
            if start_run:
                start_run = False
                led_pair[0] = led
                print(led_pair)
        elif (diff == 0) and (led>2) and (data['led position'][led-1]== data['led position'][led-2])and(data['led position'][led-1] != data['led position'][led]) and (start_run == False):
            start_run = True
            led_pair[1] = led
            loop_led_lst.append(tuple(led_pair))
            count_loop +=1
            df["loop_"+str(count_loop)] = tuple(led_pair)
            led_pair = [None,None]

    return loop_led_lst

def sub_exp_groupv2(data):
    """

    REVISED FOR ALL CONDITIONS!


    creates dictionary that holds the ranges of time in seconds
    for the different periods which occur within the experiment

    Ex:
    {
    'cali':[0,9],
    'dark1':[9,946],
    'loop1':[946,1075],
    'loop2':[1074,1202]
    }

    Where the first 10 seconds is calibration followed by a dark period, and then some experimental looping periods.
    """

    # get from end of dark period/start of first sub experiment to the start of the the last dark period...
    ## empty list for groups... will add start and end sub indexes as sub lists...
    fly_dict = {} 
    cali_led_pos(data,fly_dict)
    all_dark_led_pos(data,fly_dict)
    all_static_led_pos(data,fly_dict)
    all_loop_led_pos(data, fly_dict)

    fly_dict = {w: fly_dict[w] for w in sorted(fly_dict, key=fly_dict.get, reverse=False)}

    return fly_dict