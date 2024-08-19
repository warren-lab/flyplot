#!/bin/bash
# update lock file
poetry lock
# install dependencies
poetry install
# build package/library
poetry build
# install the new version of the library
pip install dist/*whl
# remove the dist
dir_remove="dist"
echo -e "\n"
if [ -d "$dir_remove" ]; then
    echo "Are you sure that you want to remove '$dir_remove'?"  
    read -p "(y/n):" value

    if [ "$value" == "y" ]; then
        rm -rf "$dir_remove"
        echo "dist directory has been removed"
    else
        echo "'$dir_remove' deletion cancelled"
        exit 1
    fi
else
    echo "No '$dir_remove' directory exists"
fi
echo -e "\n---------------------"
echo "Updating GitHub repo"
echo -e "---------------------\n"
# add all changes
git add .
# commit changes
read -p "Enter commit message: " message
git commit -m "$message"
# pull changes
git pull origin
# push changes
git push origin