# USER GUIDE

## Purpose

## Installation

## Dependencies

## Starting a New Project

## Notebooks
These are example notebooks of how to use some of the various modules included within this package

## Using Poetry:

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
activate
```
source flyplot_venv/bin/activate
```

- install poetry

- If repo already exists execute the following and follow the steps within the repo:
```
poetry init
```

- After this point use the following steps
    1. (Optional) If have made changes since last update to the package then run following:
    ```
    poetry lock
    ```
    2. execute installation of dependencies
    ```
    poetry install
    ```
    3. Build the dist directory which will contain the wheel file that is used for installation of the package
    ```
    poetry build
    ```

    4. Go to dist directory and install package locally
    ```
    cd dist
    pip install flyplot-0.1.0-py3-none-any.whl
    ```

- At the moment this library is still undergoing revisions and will not be published to PyPi.


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
