import argparse
import os
import shutil
import json
import sys
import configparser
import importlib.resources as pkg_resources

"""
How this works:

activate your virtual environment

go to your target directory/project directory or make note of the path

If you are in the target directory for your notebook then add `nb`

If you are not in target directory for notebook then add `nb-path`

Next if you dont want it to automatically look for the textfile and you want it to source it from else where include the following:
`txt-path`

img dir path is mandatory...

"""

def get_input():
    """
    gets user input based on arguments
    """
def copy_nb(proj_dir,date_exp,imgs_dir,txtfile):
    """
    Generates notebook copy of template using the config.ini that was set up 
    to populate the directory paths. This allows the user to create multiple different 
    jupyter notebooks based on their analysis. 

    """
    # Copy Notebook
    print("Copied Notebook Location")
    target_dir = os.path.join(proj_dir,f'flyflip_{date_exp}.ipynb')
    print("Does notebook exists?",os.path.exists('flyflip.ipynb'))
    print("Does image path exist?",os.path.exists(imgs_dir.strip("'").strip('/*.png')))
    print("Does target path exists?",os.path.exists(proj_dir))
    with pkg_resources.open_binary('flyplot.fly_analysis', 'flyflip.ipynb') as nb_file:
        with open(target_dir,'wb') as target:
            notebook_loc = shutil.copyfileobj(nb_file,target)
            print(notebook_loc,"\n")
    print(os.path.exists(target_dir))

    img_source_new = ""
    txt_source_new =""
    date_exp_new =""
    fly_book_content ={}

    # read data and add to dictionary
    with open(target_dir, 'r', encoding='utf-8') as fly_book:
        fly_book_content = json.load(fly_book)
        img_source = fly_book_content['cells'][6]['source'][0].split("=")
        print(img_source)
        img_source_new = img_source[0] + " = " + str(imgs_dir) 
        fly_book_content['cells'][6]['source'][0] = img_source_new # set new img source
        print('NEW IMAGE SOURCE->',img_source_new)
        
        
        txt_source = fly_book_content['cells'][7]['source'][0].split("=")
        print(txt_source)
        txt_source_new = txt_source[0] + " = " + str(txtfile) 
        print('NEW TXT SOURCE',txt_source_new)
        fly_book_content['cells'][7]['source'][0] = txt_source_new # set new txt source

        date_exp_source = fly_book_content['cells'][8]['source'][0].split("=")
        print(date_exp_source)
        date_exp_new = date_exp_source [0] + " = " + str("'"+str(date_exp)+"'") 
        print('NEW date_exp SOURCE',date_exp_new)
        fly_book_content['cells'][8]['source'][0] = date_exp_new # set new date exp source


        # set the exp_date source

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
        source_txt = fly_book_content['cells'][8]['source'][0].split("=")
        print(source_txt)
def setup():
    """
    Setup a new project --new
    """
    # Argparse V1
    parser = argparse.ArgumentParser(prog = "project_setup",
                                     description="setup in project_setup results in the complete initialization of a new fly analysis project, or the ability to add to a previously made project.")
    parser.add_argument("--target",action='store_true',
                        help="If you are in target directory add this argument to provide the appropriate path to make the project")
    parser.add_argument("--txt",action='store_true',help="To include path to text file")

    # determine what arguments were used 
    args = parser.parse_args()

    # Image Directory:
    print("\nEstablish Image Directory Path:")
    imgs_dir = "'"+input("")+"/*.png"+"'" # "'/home/flyranch/image_data/20240621/fly1/20240621102556'"
    date_exp = imgs_dir.strip("'").split('/')[-2]
    path_dir = ''
    proj_num = 1 # new directory
    # Check 1: Create Directory through user input or via current directory
    print("Set Location for FlyPlot Project") 
    if args.target:
        path_dir = os.getcwd() # current path

    else:
        # user defined path:
        # # same as textfile location
        path_dir = input("Target Directory: ") # Ex:"/media/username/D5E2-7968/20240502125110_data" 

    # check to see if directory already exists
    check_dir_1 = [d for d in os.listdir(path_dir) if os.path.isdir(d)]
    print(check_dir_1)
    if check_dir_1 == None:
        directory = f'analysis_{proj_num}'
        proj_path = os.path.join(path_dir,directory)
    else:
        check_dir_2 = [d for d in check_dir_1 if d[:8]==('analysis')]
        print(check_dir_2)
        check_dir_2.sort(reverse=True)
        print(check_dir_2)
        if len(check_dir_2)>0:
            if len(check_dir_2) ==1:
                proj_num+=int(check_dir_2[0][-1])
            else:
                print(check_dir_2[0][-1])
                proj_num+=int(check_dir_2[0][-1])
    directory = f'analysis_{proj_num}'
    proj_path = os.path.join(path_dir,directory)
    os.mkdir(proj_path) # make the project directory
    print(os.path.exists(proj_path))
    # create figures sub directory in project directory
    fig_path = os.path.join(proj_path,'figs')
    os.mkdir(fig_path)
    # Check 2: Set Path for text file:
    if args.txt:
        print("Set Location for textfile") 
        txtfile = "'"+input("")+"'" # location of textfile 
    else:
        print("Textfile Name:") # location of textfile that is within project directory
        txtfile = "'"+path_dir+"/"+input("")+"'" # "'/media/loganrower/D5E2-7968/data/20240620/fly4m/20240502125110.txt'" 
    
    # Check 3: Copy Jupyter Notebook
    print("Copied Notebook Location")
    target_dir = os.path.join(proj_path,f'flyflip_{date_exp}.ipynb')
    with pkg_resources.open_binary('flyplot.fly_analysis', 'flyflip.ipynb') as nb_file:
            with open(target_dir,'wb') as target:
                notebook_loc = shutil.copyfileobj(nb_file,target)
                print(notebook_loc,"\n")
    print(os.path.exists(target_dir))
    
    copy_nb(proj_path,date_exp,imgs_dir,txtfile)

    # Add to configuration file in project directory:
    config = configparser.ConfigParser()
    print("Generate Config in project")
    path_dir_dict = "'"+path_dir+"'"
    proj_path_dict = "'"+proj_path+"'"
    config['Data Paths'] ={'image_data':imgs_dir,
                           'raw_data_dir':path_dir_dict,
                           'project_path':proj_path_dict,
                           'txtfile':txtfile }
    # Write the configuration to a file
    print(path_dir.strip("'"))
    config_path = os.path.join(proj_path,'config.ini')
    with open(config_path, 'w') as configfile:
	    config.write(configfile)

if __name__ == "__main__":
    setup()