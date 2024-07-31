"""
Methods for dealing with fruit fly wrapping
"""
from flyplot.plotting import polarplot as polarplt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def wrapping_fix_single(data):
    """
    Parameters:
        - data:
            Corresponds to a dataframe containing heading related data
    NEED TO CHANGE Adjusted HEADING TO fly heading
    """
def wrapping_fix(exp_date, df_orig,df_revised):
    """
    Parameters:
        - df_orig:
            Corresponds to the dataframe `df_fly_org`, or any other dataframe that contains 
            the original data from Magnotether experiment.
        
        - df_revised:
            Corresponds to the dataframe `df_fly_revised`, or any other dataframe taht contains 
            revised Magnotether data after going through MSE Heading Adjustment algorithm.

    This function will plot the original and revised fly heading along with the led angle.
    """
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(36,18))
    split_idx = []
    split_idx = (np.where(np.abs(np.diff(df_revised['Adjusted Angle']))>179)[0]+1)
    print(split_idx)
    sub_idx_groups = []
    if len(split_idx>0):
        for count,i in enumerate(split_idx):
            if count == 0:
                sub_idx_groups.append([0,i])
            elif count == len(split_idx) -1:
                sub_idx_groups.append([i,len(df_revised['Adjusted Angle'])])
            else:
                sub_idx_groups.append([i,split_idx[count+1]])
        for group_idx in range(1,len(sub_idx_groups)):
            print(split_idx[group_idx-1],split_idx[group_idx])
        # Now what will be done is to rest this data so that it can just be directly plotted by filling in the proper interpolated value..
        # f = interpolate.interp1d(df_male_dat['delta-t'], df_male_dat['fly heading'], kind = 'cubic')
        ## basically doubled the number of points...\n",
        # xnew = np.linspace(df_male_dat['delta-t'].loc[0],df_male_dat['delta-t'].loc[len(df_male_dat['delta-t'])-1],20000)
        # ynew = f(xnew)   # use interpolation function returned by `interp1d`\n",
    else:
        # f = interpolate.interp1d(df_fem_loop['delta-t'],df_fem_loop['fly heading'], kind='cubic')
        sub_idx_groups = [[0,len(df_revised['Adjusted Angle'])]]

    for k_sub in range(len(sub_idx_groups)):
        start_k = sub_idx_groups[k_sub][0]
        end_k = sub_idx_groups[k_sub][1]
        # Original Plot
        ax1.plot(df_orig['delta-t'][start_k:end_k],df_orig['fly heading'][start_k:end_k],color = 'k',linewidth = .5, label = 'Original Trace')
        ax1.plot(df_orig['delta-t'][start_k:end_k],df_orig['led angle'][start_k:end_k],color = 'goldenrod', linewidth = 1, label = 'LED')
        # Revised Plot
        ax2.plot(df_revised['delta-t'][start_k:end_k],df_revised['Adjusted Angle'][start_k:end_k],color = 'k', linewidth = .8, label = 'Adjusted Heading')
        ax2.plot(df_revised['delta-t'][start_k:end_k],df_revised['led angle'][start_k:end_k],color = 'goldenrod', linewidth = 1, label = 'LED')
    ax1.set_title("Fly Trace Original")
    ax1.set_xlabel("time")
    ax1.set_ylabel("Heading/LED Angle")
    ax1.set_yticks(np.arange(-180,181,90))
    ax2.set_title("Fly Trace Adjusted")
    ax2.set_xlabel("time")
    ax2.set_ylabel("Heading/LED Angle")
    ax2.set_yticks(np.arange(-180,181,90))
    if k_sub ==  0:
        ax1.legend(loc='lower right')
        ax2.legend(loc = "lower right")
    plt.savefig(f'{exp_date}_wrap_Orig_AdjHeading.png',facecolor = 'white')
    plt.savefig(f'{exp_date}_wrap_Orig_AdjHeading.svg',facecolor = 'white')

def wrapping_fix_old(data,subgroups,plot=True):
    """
    Parameters:
        - data:
            In current methods data refers to `df_fem_loop` or df_fem_static`. This corresponds to
            a dataframe
        - subgroups:
            dictionary holding keys of the subgroup name and values with the corresponding time ranges.

        - plot:
            default set to True, indicating that this function will fix the data wrapping
        - 
    Returns:
    This method generates and saves a plot that addresses issue with continuous point connections in plotting which do not represent realistic movement. 

    """
    data_df_plot = ['dark1', 'stat1','stat2']
    labels_final = ['fly heading','LED heading','mean fly heading',]
    colors = ['black','goldenrod','limegreen']
    fig = plt.figure(figsize=(24,12))
    ax1 = fig.add_subplot()
    # iterate over the sections
    for count_main, k in enumerate(data_df_plot):
        start = subgroups[k][0]
        end = subgroups[k][1]
        # calculate the mean heading
        vec_str , mean_drxn = polarplt.calc_mean_and_vector_strength(data['fly heading'][start:end]) # Subtract LED ANGLE TO? THEN ALREADY RELATIVE TO THE FIRST LED THO RIGHT?
        # plot the mean heading
        heading = polarplt.deg360to180(np.rad2deg(mean_drxn))
        print(f"{k}\n Vector Strength:{round(vec_str,3)}, Mean Heading: {round(heading,3)}")
        ax1.plot((data['delta-t'].loc[start],data['delta-t'].loc[end]),(heading,heading),color = 'limegreen', linewidth = 2, label = 'Mean Heading')
        # subsplit
        split_idx = []
        split_idx = (np.where(np.abs(np.diff(data['fly heading'][start:end]))>179)[0]+1)
        print(split_idx)
        sub_idx_groups = []
        if len(split_idx)>0:
            for count,i in enumerate(split_idx):
                if count == 0:
                    sub_idx_groups.append([start,i])
                elif count == len(split_idx) -1:
                    sub_idx_groups.append([i,end])
                else:
                    sub_idx_groups.append([i,split_idx[count+1]])
        else:
            sub_idx_groups = [[start,end]]
        
        # using sub index split groups now plot...
        for k_sub in range(len(sub_idx_groups)):

            start_k = sub_idx_groups[k_sub][0]
            end_k = sub_idx_groups[k_sub][1]
            ax1.plot(data['delta-t'][start_k:end_k],data['fly heading'][start_k:end_k],color = 'k', linewidth = .5, label = 'Heading')
            ax1.plot(data['delta-t'][start_k:end_k],data['led angle'][start_k:end_k],color = 'goldenrod', linewidth = 1, label = 'LED')


    legend_elements = [Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(colors, labels_final)]
    ax1.legend(handles = legend_elements, fontsize='x-large')
    # ax1.set_title("Female Fly 20231121124749")
    ax1.set_xlabel("Time (s)", fontsize = 15)
    ax1.set_ylabel("Heading / LED Angle (\N{DEGREE SIGN})", fontsize = 15)
    ax1.set_yticks(np.arange(-180,181,90))
    ax1.tick_params(axis='both',which='major', labelsize = 15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # if k_sub ==  0:
    #     ax1.legend(loc='lower right')
    plt.savefig('figs/main_fig_with_subsplit.svg',facecolor = 'white')