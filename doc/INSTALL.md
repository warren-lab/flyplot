# Installation Documentation
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
----------------

- Clone the repository:
```
git clone https://github.com/warren-lab/flyplot.git
```
- Navigate to the root of the repository, and ensure that proper virtual environment is active:
```
cd flyplot
```
- Install poetry
```
pip install poetry
```
- Run the [installation script](tools/install.sh) using the following methods
```
./tools/install.sh
```
- See more at [Install Guide](doc/INSTALL.md)