# Code examples
![example workflow](https://github.com/GlobalMin/code_examples/actions/workflows/tests.yml/badge.svg)

This repo gathers a number of benchmark datasets for binary classification and runs various algorithms on them.


## Installation and notes
It's recommended that you create a fresh virtual environment first. Then run these commands in a terminal.
    
```bash
pip install -e .
pip install -r requirements.txt
 ```

In order to grab the datasets from Kaggle you'll need to [create an API token there](https://www.kaggle.com/docs/api) and load it as an environment variable. Then you'll need to create a .env file in this directory with the following content:

```
KAGGLE_USERNAME="your_username"
KAGGLE_KEY="your_key"
```
