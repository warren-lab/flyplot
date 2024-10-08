"""
Below is a more robust data parsing methodology. This essentially will extract each of the individual experimental
groups static, loop, dark and calibration. 

It will also extract each of the loop interations



FUTURE:
- Work to provude the loop iterations but also just general bounds for loop period
- work on the accuracy of the sections between the transition zones

"""


# Methods Specific for Looping Experiments -> WORKING VERSION ALMOST!

## Ignore the -1 start...-> Include into dark? Right now we take start of dark as start of exp...

## Break up seections between constant LED and Looping LED

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
# print(dark_led_positions())
# print(data['time'][946])
### For each section get the start and the end to create information for this grouping of data

####create the index groups...
#### check if i == i+1 if it does check when there is change from that value of i to different value...
#### if ths is not true and all values are different then will determine when there is a change to a constant value....
def sub_exp_group(data):
    """
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
    fem_dict_sec = {}


    count_loop_led = 0
    count_dark = 0
    count_static_led = 0
    count_calibrate = 0
    curr_exp = ''
    sub_exp = [None,None] # start index, end index
    for pnt in range(1,len(data['led position'])): ## ACCOUNT FOR 149 AS DARK CURRENTLY SCRIPT IS BREAKING!!! v
        ## check difference:
        diff_led = abs( data['led position'][pnt] -  data['led position'][pnt-1])
        ## using this difference the following can be found
        ##  1. if difference is zero then we know that the points are equal to eachother. This indicates either dark mode,single LED model or,starting calibration mode (not considered currently)
            ## 1.a First check if the value is 150 and then assign the correct values...
        # print("ENTER",curr_exp, diff_led,data['led position'][pnt-1],data['led position'][pnt])
        if diff_led == 0:
            if  sub_exp[1] is None and curr_exp == 'loop' and data['led position'][pnt-1]== 138:
                # make sure that it is still loop...
                count_loop_led+=1
                sub_exp[1] = pnt+1
                fem_dict_sec[curr_exp+str(count_loop_led)] = [sub_exp[0],sub_exp[1]]
                print(data['led position'][sub_exp[0]],data['led position'][sub_exp[1]-1], fem_dict_sec[curr_exp+str(count_loop_led)])
                sub_exp = [None,None]
                curr_exp =''
            if data['led position'][pnt-1] == -1 and curr_exp == '':
                if sub_exp[0] is None:
                    curr_exp ='cali'
                    sub_exp[0]=pnt-1
            elif data['led position'][pnt-1] == 150:
                if sub_exp[0] is None:
                    sub_exp[0] = pnt-1
                    curr_exp = 'dark'
                
                elif curr_exp == 'dark' and (sub_exp[1] is None) and (pnt == len(data['led position'])-1) and (data['led position'][pnt]==150): 
                    # print("ENTER",pnt)
                    sub_exp[1] = pnt
                    count_dark+=1
                    fem_dict_sec["dark_"+str(count_dark)] = [sub_exp[0],sub_exp[1]]
                    print(data['led position'][sub_exp[0]],data['led position'][sub_exp[1]-1], fem_dict_sec["dark_"+str(count_dark)])
                    sub_exp = [None,None]
                    curr_exp =''

            elif data['led position'][pnt] == data['led position'][pnt-1] and data['led position'][pnt-1] == data['led position'][pnt+1]and curr_exp == '': # static LED
                if sub_exp[0] is None:
                    curr_exp ='stat'
                    sub_exp[0] = pnt-1
                


        elif diff_led > 0:
            
            # the case when curr_exp is dark, and when there is a different LED signal after the dark period
            # There is a case where between last 1-3 values could be 149...so look for end or the change from 150 to other value..
            if sub_exp[1] is None and curr_exp == 'dark' and ((pnt == len(data['led position'])-1) or (data['led position'][pnt]!=150)):
                sub_exp[1] = pnt
                count_dark+=1
                fem_dict_sec["dark"+str(count_dark)] = [sub_exp[0],sub_exp[1]]
                print(data['led position'][sub_exp[0]],data['led position'][sub_exp[1]-1], fem_dict_sec["dark"+str(count_dark)])
                sub_exp = [None,None]
                curr_exp =''
            
            elif curr_exp == 'cali' and sub_exp[1] is None and data['led position'][pnt-1] == -1 and data['led position'][pnt] ==150: # calibrate to dark mode
                sub_exp[1] = pnt
                count_calibrate+=1
                fem_dict_sec["cali"+str(count_calibrate)] = [sub_exp[0],sub_exp[1]]
                print(data['led position'][sub_exp[0]],data['led position'][sub_exp[1]-1], fem_dict_sec["cali"+str(count_calibrate)])
                sub_exp = [None,None]
                curr_exp =''
            
            elif (sub_exp[1] is None) and (curr_exp == 'stat') and (data['led position'][pnt] != data['led position'][pnt-1]):
                sub_exp[1]= pnt
                count_static_led+=1
                fem_dict_sec["stat"+str(count_static_led)] = [sub_exp[0],sub_exp[1]]
                print(data['led position'][sub_exp[0]],data['led position'][sub_exp[1]-1], fem_dict_sec["stat"+str(count_static_led)])
                sub_exp = [None,None]
                curr_exp =''


            # Primary Case: Led Loop, after ruling out the edge case of the dark mode...
            elif ((data['led position'][pnt] != data['led position'][pnt-1]) and (diff_led>=1)):
                # print(data['led position'][pnt-1],data['led position'][pnt])
                if sub_exp[0] is None and curr_exp == '':
                    curr_exp = 'loop'
                    sub_exp[0] = pnt-1
                elif curr_exp == 'loop' and sub_exp[1] is None and (diff_led>5):
                    count_loop_led+=1
                    sub_exp[1] = pnt+1
                    fem_dict_sec[curr_exp+str(count_loop_led)] = [sub_exp[0],sub_exp[1]]
                    print(data['led position'][sub_exp[0]],data['led position'][sub_exp[1]-1], fem_dict_sec["loop"+str(count_loop_led)])
                    sub_exp = [None,None]
                    curr_exp =''
        if pnt == 945:
            print(curr_exp)
            print(sub_exp)
        if pnt ==946:
            print(curr_exp)
            print(diff_led )

                
                    
    
    return fem_dict_sec

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
# Testing
# fem_dict_sec = sub_exp_group(data)
# print(fem_dict_sec)
# print(len(fem_dict_sec))

