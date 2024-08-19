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
rm -rf dist
# add all changes
git add .
# commit changes
read -p "Enter commit message: " message
git commit -m"$message"

# pull changes
git pull origin
# push changes
git push origin