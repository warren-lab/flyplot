# USER GUIDE

## Purpose

## Installation
#### Virtual Environment Setup
- install python 3.9 or later...
    ```
    sudo apt install python3.9
    ```

- install venv or later...
    ```
    sudo apt install python3.9-venv
    ```
- create virtual environment
    ```
    mkdir flyplot_venv
    ```
    ```
    python3 -m venv flyplot_venv OR python3.9 -m venv flyplot_venv
    ```
- activate
    ```
    source flyplot_venv/bin/activate
    ```
#### Install Poetry
```
pip install poetry
```
#### Clone Repository
Clone repository and navigate to flyplot root
```
git clone https://github.com/warren-lab/flyplot
```
```
cd flyplot
```

#### Install Library
There are two methods with which to install the library
1. Run the [installation script](tools/install.sh):
    ```
    ./tools/install.sh
    ```
2. Step-wise method (same as in install.sh but ) 
    - Execute installation of dependencies
        ```
        poetry install
        ```
    - Build the dist directory which will contain the wheel file that is used for installation of the package
        ```
        poetry build
        ```

    - Go to dist directory and install package locally
        ```
        cd dist
        pip install flyplot-0.1.0-py3-none-any.whl
        ```

## Notebooks
These are example notebooks of how to use some of the various modules included within this package

## Using Poetry:

- Install poetry
    ```
    pip install poetry
    ```
- (ADDITONAL INFO) If project already exists use the following command to initialize poetry, and follow the steps it provides.
    ```
    poetry init
    ```
#### Process to make changes to Library: 
If changes are made to this repository the following process must be performed:

1. Update the .lock file   
    ```
    poetry lock
    ```

2. Follow the steps in **Install Library** to build the package 

    Using the [installation script](tools/install.sh) is the recommended approach
        
    ```
    ./tools/install.sh
    ```

## Using Jupyter Notebook Local Host...
- Activate your virtual environment

- Install jupyter kernel for virtual environment:
    ```
    ipython kernel install --user --name=flyplot_venv
    ```

- Select installed kernel

- If wanting to uninstall the kernel
    ```
    jupyter-kernelspec uninstall venv
    ```



