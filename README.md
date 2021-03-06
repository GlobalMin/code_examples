# Code examples
![example workflow](https://github.com/GlobalMin/code_examples/actions/workflows/tests.yml/badge.svg)

This repo gathers a number of benchmark datasets for binary classification and runs various algorithms on them.


## Installation and notes
You will need an account with [Kaggle](https://www.kaggle.com/). It's free.

It's recommended that you create a fresh virtual environment first. Then run these commands in a terminal.
    
```bash
pip install -e .
pip install -r requirements.txt
 ```

In order to grab the datasets from Kaggle you'll need to [create an API token there](https://www.kaggle.com/docs/api) and load it as an environment variable. Modify the file in the repo ".env.sample" and add in your credentials as shown below, then rename it to just ".env".

```
KAGGLE_USERNAME="your_username"
KAGGLE_KEY="your_key"
```
## Running the benchmarks
Run the file `app.py` to download all the benchmarks datasets and the various pipelines. The model outputs will be saved in the `model_objects` folder.