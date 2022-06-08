import io
import os
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv

from code_examples.utils import get_logger, stratified_sample

load_dotenv()
logger = get_logger(__name__)


def get_australian_credit_data():
    url = "https://raw.githubusercontent.com/shraddha-an/cleansed-datasets/master/credit_approval.csv"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))

    return df


def get_cc_fraud_data():
    os.system("kaggle datasets download -d mlg-ulb/creditcardfraud")
    with zipfile.ZipFile("creditcardfraud.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    df = pd.read_csv("creditcard.csv")

    df_sampled = stratified_sample(df, "Class", 0.3)

    os.remove("creditcard.csv")
    os.remove("creditcardfraud.zip")

    return df_sampled


def get_cc_approval_data():
    os.system("kaggle datasets download -d rikdifos/credit-card-approval-prediction")
    with zipfile.ZipFile("credit-card-approval-prediction.zip", "r") as zip_ref:
        zip_ref.extractall(".")

    credit_record = pd.read_csv("credit_record.csv")

    application_record = pd.read_csv("application_record.csv")

    # Construct target
    credit_record["target"] = credit_record["STATUS"]
    credit_record["target"].replace("X", 0, inplace=True)
    credit_record["target"].replace("C", 0, inplace=True)
    credit_record["target"] = credit_record["target"].astype(int)
    credit_record.loc[credit_record["target"] >= 1, "target"] = 1

    target_df = pd.DataFrame(
        credit_record.groupby(["ID"])["target"].agg(max)
    ).reset_index()

    new_df = pd.merge(application_record, target_df, how="inner", on=["ID"])

    os.remove("credit_record.csv")
    os.remove("application_record.csv")
    os.remove("credit-card-approval-prediction.zip")

    return new_df


()

CC_FRAUD_DATA = get_cc_fraud_data()
AUTRALIAN_CREDIT_DATA = get_australian_credit_data()
CC_APPROVAL_DATA = get_cc_approval_data()

DATASETS_INFO = [
    [CC_APPROVAL_DATA, "target", "CC Approval Data"],
    [AUTRALIAN_CREDIT_DATA, "Target", "Australian Credit"],
    [CC_FRAUD_DATA, "Class", "CC Fraud Data"],
]
