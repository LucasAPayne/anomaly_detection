# Anomaly Detection
This repository provides an end-to-end system for anomaly detection using machine learning on knowledge graphs. Two sets of log files are used to create knowledge graph datasets. One set of log files should contain only normal system behavior, and the other set should include both normal and malicious behavior. These datasets are transformed into knowledge graphs using [a custom fork of SLOGERT](https://github.com/LucasAPayne/slogert). The dataset containing normal behavior is used to train a knowledge graph completion model what baseline behavior of the system looks like. The dataset containing both normal and malicious behavior is used to test this model on its ability to recognize deviations from this baseline behavior. The models used here are from [CyberML](https://github.com/LucasAPayne/cyberML).

## Getting Started
First, note that this repository has only been tested on Ubuntu 21.10, so no guarantees can currently be made that it will work on other operating systems.

Navigate to the folder where you would like to store this repository, then enter this command to recursively clone this repository and its submodules:

    git clone --recursive https://github.com/LucasAPayne/anomaly_detection.git

It is first recommended to use [Anaconda](https://www.anaconda.com/) to manage virtual environments and Python packages. Use Anaconda to create and activate a virtual environment, then install the additional dependencies of this repository:

    conda create --name anomaly_detection
    conda activate anomaly_detection
    cd path/to/repo
    pip3 install -r requirements.txt

## Running Demo
This repository includes a demo using the AIT log dataset. To run this demo, it is required to [download the dataset](https://zenodo.org/record/4264796). Once the download is complete, ensure you are in the proper Anaconda environment, then run the `preprocess.sh` script:

    bash preprocess.sh

The script will ask for the path to the AIT log dataset. It will extract a subset of the AIT dataset as training and testing datasets and use SLOGERT to transform these into knowledge graphs. When this script is finished running, open JupyterLab:
    
    jupyter-lab

Inside JupyterLab, open `AIT.ipynb` and run it. This will train and test the three knowledge graph completion models supplied by CyberML and output graphical results using matplotlib.
