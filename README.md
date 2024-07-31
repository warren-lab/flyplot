# flyplot
This repository contains the methods for conducting the necessary analysis for drosophila-related experiments, by developing informative plots and statistics.

The possible goal of this repository being developed is as a full-fledged library for internal or external usage for drosophila-related analyses.


# Directions for Install:
- install python 3.9

- install venv

- create virtual environment

- install poetry

- build and then install wheel
```
poetry build
```
```
cd dist
pip install flyplot-0.1.0-py3-none-any.whl
```

# Using this package:
- ability to import various modules for fly analys that utilize custome modules and some that are inspired by the find_fly_angle module (will link those in future)
- run the module fly_analysis.py to create copy of flyflip.ipynb where your data lies for specific experiment

## Using Poetry:

- install python 3.9

- install venv

- create virtual environment

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

- Publishing -> TBD


## Running:

- Creating project directory in current directory and requires custom textfile and image full paths
```
proj-setup --target --txt
```

- Creating project directory in current directory and requires textfile name in current directory with custom image paths 
```
proj-setup --target 
```
