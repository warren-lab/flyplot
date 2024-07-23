"""
This script is to be utilized in the analysis of Magnotether experiments for 
fly orientation adjustments, development of figures and generation of informatics. 

What is necessary to utilize this script?
- The path to the directory that contains raw images for an experiment
- The folder/path to the folder that contains the data file for the same experiment.

What will this script do?
- This script will require you to input the path to the folder that you will be looking at.
I addition this script will assume that you are working on a linux machine, but if not attach an additional flag of M or W. Corresponding 
to MacOS or Windows to allow for the access of the relevant files. 
    - M: MacOS
    - W: Windows
- With the proper path obtained the flyflip script will be copied into the directory containing the datafile.
- With this script duplicated you can now perform the installation of the fly_plot_warren package.
"""