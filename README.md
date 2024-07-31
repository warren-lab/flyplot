# flyplot
Flyplot is a Python library that can be used to streamline the analysis pipeline for Drosophila-based experiments. This library contains modules for analyzing drosophila-related experiments, by developing informative plots and statistics.

## Quick Installation Guide:
- Clone the repository:
```
git clone https://github.com/warren-lab/flyplot.git
```
- Navigate to the root of the repository, and ensure that proper virtual environment is active:
```
cd flyplot
```
- Run the [installation script](tools/install.sh)
```
./tools/install.sh
```
- See more at [Install Guide](doc/INSTALL.md)

## Setting Up New Project:
The primary use case of this library is to aid the user in their management of analysis files. This project comes with a pre-made template notebook for an analysis workflow which is copied into a pre-defined directory by the user. The project directory allows for separation between different analyses and datasets.

A copy of the notebook can be editted by the user for their own purposed. The example notebooks can be examined to see the various different methods that can be used in examining Drosophila orientation.

- Creating project directory in current directory and requires full paths for the text file and image directory
```
proj-setup --target --txt
```

- Creating project directory in current directory and requires the full path for the image directory. However, only the filename of the text file is required, as it is assumed that it is located in the same location as the project directory.
```
proj-setup --target 
```

## Library Utility:
- Automates file access and management
- Ensures that all analyses are maintained to the same standard by starting with a template notebook.
- Ability to utilize various modules that allow for a greater accuracy in correcting the computed fly orientation. 
- Incorporates aspects of the [find_fly_angle](ACKNOWLEDGEMENTS.md) module.
