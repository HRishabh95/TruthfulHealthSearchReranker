# Truthful Health Search Reranker

This repository contains the code implementation for the paper titled "A Passage Retrieval Transformer-based Re-ranking Model for Truthful Consumer Health Search".

## Contents
The repository includes the following files:

* `first_stage_retrieval.py`: Python script for the first stage retrieval process using pyterrier.
* `requirements.txt`: File specifying the required dependencies for running the code.
* `sent_retrieval.py`: Python script for the sentence retrieval process.
* `utils.py`: Utility functions used in the code.

## Usage
To use the code, follow these steps:

* Install the required dependencies by running: 

``` bash
  pip install -r requirements.txt.
```
* Clone the dataset from the following github: [Health Misinformation Dataset](https://github.com/ikr3-lab/TREC-CLEF-HealthMisinfoSubdatasets)
* Execute `first_stage_retrieval.py` to perform the first stage retrieval process.
```bash
python first_stage_retrieval.py TREC True <dataset_path>
```
* Run `sent_retrieval.py` to perform the sentence retrieval process.
```bash
python sent_retrieval.py TREC True <dataset_path> <top relevant sentences>
```
Example:
For top 20 % sentences.
```bash
python sent_retrieval.py TREC True /tmp/dataset 0.20
```