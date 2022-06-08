import io
import os
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv

from code_examples.pipelines.xgboost_pipeline import XGBoostPipeline
from code_examples.utils import get_logger, stratified_sample

load_dotenv()
logger = get_logger(__name__)


def get_australian_credit_data():
    url = "https://raw.githubusercontent.com/shraddha-an/cleansed-datasets/master/credit_approval.csv"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))

    return df


AUTRALIAN_CREDIT_DATA = get_australian_credit_data()


def get_cc_fraud_data():
    os.system("kaggle datasets download -d mlg-ulb/creditcardfraud")
    with zipfile.ZipFile("creditcardfraud.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    df = pd.read_csv("creditcard.csv")

    df_sampled = stratified_sample(df, "Class", 0.3)

    os.remove("creditcard.csv")
    os.remove("creditcardfraud.zip")

    return df_sampled


CC_FRAUD_DATA = get_cc_fraud_data()

DATASETS_INFO = [
    [AUTRALIAN_CREDIT_DATA, "Target", "Australian Credit"],
    [CC_FRAUD_DATA, "Class", "CC Fraud Data"],
]

xgb_params = {
    "eta": [0.01, 0.025],
    "max_depth": [5, 6, 7],
    "n_estimators": [1000],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "objective": ["binary:logistic"],
    "eval_metric": ["auc"],
    "early_stopping_rounds": [300],
}


def train(X, y):
    pipeline = XGBoostPipeline(X, y, xgb_params)
    results = (
        pipeline.define_preprocess_pipeline()
        .apply_train_test_split(X, y)
        .fit_and_tune_xboost()
    )


for dataset in DATASETS_INFO:
    df = dataset[0]
    target = dataset[1]
    desc = dataset[2]

    y = df.pop(target)
    X = df

    logger.info(f"Training model on {desc}")
    train(X, y)
