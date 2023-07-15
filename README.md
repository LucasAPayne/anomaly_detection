# Anomaly Detection
This repository provides an end-to-end system for log file anomaly detection using machine learning on knowledge graphs. To accomplish this, a dataset is constructed by transforming log lines into sets of knowledge graph triples using a custom method built with [Drain3](https://github.com/logpai/Drain3). The training dataset helps the model learn what normal behavior of the system looks like. The testing and validation sets contain both normal and malicious behavior and test this model on its ability to recognize deviations from this baseline behavior. The models used here are from [Graph4NLP](https://github.com/graph4ai/graph4nlp_demo).

## Getting Started
First, note that this repository has only been tested on Ubuntu 21.10, so no guarantees can currently be made that it will work on other operating systems.

Clone the repository with

    git clone https://github.com/LucasAPayne/anomaly_detection.git

Use Anaconda to create and activate a virtual environment, then install the additional dependencies of this repository:

    conda create --name anomaly_detection
    conda activate anomaly_detection
    cd path/to/repo
    pip3 install -r requirements.txt
    ./setup

## Running the Demo
This repository includes a demo using the AIT log dataset. To run this demo, it is required to download the [AIT](https://zenodo.org/record/4264796) and [cyberML](https://github.com/dodo47/cyberML/tree/main/data) datasets and place them into the `data` directory. Next, run

    python3 demo.py

For the AIT dataset, which consists of raw log data, the demo extracts a training, testing, and validation set and transforms them into knowledge graph triples using a custom method, then trains a knowledge graph completion model with the data. The cyberML dataset consists of log data that has already been preprocessed with a different method. In this case, the demo only creates a validation set and trains a knowledge graph completion model with the data.
