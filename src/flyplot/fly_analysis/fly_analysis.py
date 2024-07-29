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
import os
import shutil
import json
import sys
import configparser
import importlib.resources as pkg_resources

def flyplot_setup():
    """
    generates config file within current directory to use for data paths
    """
    config = configparser.ConfigParser()
    
    print("\nEstablish Image Directory Path:")
    imgs_dir = "'"+input("")+"'" # "'/home/flyranch/image_data/20240621/fly1/20240621102556'"
    date_exp = imgs_dir.strip("'").split('/')[-1]
    print("Set Location for Fly Flip Notebook Copy") # same as textfile location
    path_dir = input("Target Directory: ") # "/media/loganrower/D5E2-7968/20240502125110_data/" 
    path_dir_dict = "'"+path_dir+"'"
    print("Textfile Data Path:")
    txtfile = "'"+path_dir+"/"+date_exp+".txt"+"'" # "'/media/loganrower/D5E2-7968/20240502125110_data/20240502125110.txt'" 

    config['Data Paths'] ={'image_data':imgs_dir,
                           'raw_data_dir':path_dir_dict,
                           'txtfile':txtfile }
    # Write the configuration to a file
    print(path_dir.strip("'"))
    config_path = os.path.join(path_dir.strip("'"),'config.ini')
    with open(config_path, 'w') as configfile:
	    config.write(configfile)
    
def flyplot_copy():
    """
    Generates notebook copy of template using the config.ini that was set up 
    to populate the directory paths. This allows the user to create multiple different 
    jupyter notebooks based on their analysis. 

    """
    # Set Directory Path for Text file data and the Notebook copy
    print("Set Location for Fly Flip Notebook Copy") # same as textfile location
    path_dir = input("Target Directory: ") # "/media/loganrower/D5E2-7968/20240502125110_data"
    # Set Image Directory 
    print("\nEstablish Image Directory Path:")
    imgs_dir = "'"+input("")+"'" # "'/home/flyranch/image_data/20240621/fly1/20240621102556'" 
    date_exp = imgs_dir.strip("'").split('/')[-1]
    print("Experiment Date: ",date_exp,"\n")

    # Set Text file
    print("Textfile Name:")
    txtfile = "'"+path_dir+input("")+"'" # "'/media/loganrower/D5E2-7968/20240502125110_data/20240502125110.txt'" 
    date_exp = imgs_dir.strip("'").split('/')[-1]

    print("EXP Date",date_exp)
    # Copy Notebook
    print("Copied Notebook Location")
    target_dir = os.path.join(path_dir,'flyflip_{date_exp}.ipynb')
    print("Does notebook exists?",os.path.exists('flyflip.ipynb'))
    print("Does target path exists?",os.path.exists(path_dir))
    with pkg_resources.open_binary('flyplot.fly_analysis', 'flyflip.ipynb') as nb_file:
        with open(target_dir,'wb') as target:
            notebook_loc = shutil.copyfile(nb_file,target)
            print(notebook_loc,"\n")
    print(os.path.exists(target_dir))

    img_source_new = ""
    txt_source_new =""
    fly_book_content ={}

    # read data and add to dictionary
    with open(target_dir, 'r', encoding='utf-8') as fly_book:
        fly_book_content = json.load(fly_book)
        img_source = fly_book_content['cells'][6]['source'][0].split("=")
        print(img_source)
        img_source_new = img_source[0] + " = " + str(imgs_dir) 
        print(img_source_new)
        txt_source = fly_book_content['cells'][7]['source'][0].split("=")
        print(txt_source)
        txt_source_new = txt_source[0] + " = " + str(txtfile) 
        fly_book_content['cells'][7]['source'][0] = txt_source_new

    # Write paths
    with open(target_dir, 'w', encoding='utf-8') as fly_book:
        json.dump(fly_book_content,fly_book)

    # Check the changed paths
    with open(target_dir, 'r', encoding='utf-8') as fly_book:
        fly_book_content = json.load(fly_book)
        source_img = fly_book_content['cells'][6]['source'][0].split("=")
        print(source_img)
        source_txt = fly_book_content['cells'][7]['source'][0].split("=")
        print(source_txt)
if __name__ == "__main__":
     flyplot_setup()