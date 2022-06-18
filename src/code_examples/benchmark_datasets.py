import io
import os
import zipfile

import numpy as np
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

    # Select class column from df and sample it

    df_positive_class = pd.DataFrame(df[df["Class"] == 1])
    df_positive_class.reset_index(drop=True, inplace=True)
    df_positive_class.dropna(inplace=True)
    df_negative_class = pd.DataFrame(df[df["Class"] == 0])
    df_negative_class.reset_index(drop=True, inplace=True)
    df_negative_class.dropna(inplace=True)

    df_downsampled_negative_class = df_negative_class.sample(
        frac=0.025, random_state=42
    )

    # Combine y_downsample and x_downsample
    df_downsample = pd.concat(
        [df_positive_class, df_downsampled_negative_class], axis=0
    )
    df_downsample.sample(frac=1, random_state=42).reset_index(drop=True, inplace=True)

    print(df_downsample["Class"].value_counts())

    os.remove("creditcard.csv")
    os.remove("creditcardfraud.zip")

    return df_downsample


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

    new_df_downsampled = stratified_sample(
        df=new_df, stratify_column="target", proportion=0.25
    )

    os.remove("credit_record.csv")
    os.remove("application_record.csv")
    os.remove("credit-card-approval-prediction.zip")

    return new_df_downsampled


CC_FRAUD_DATA = get_cc_fraud_data()
AUTRALIAN_CREDIT_DATA = get_australian_credit_data()
CC_APPROVAL_DATA = get_cc_approval_data()

DATASETS_INFO = [
    [AUTRALIAN_CREDIT_DATA, "Target", "Australian Credit"],
    [CC_FRAUD_DATA, "Class", "CC Fraud Data"],
    [CC_APPROVAL_DATA, "target", "CC Approval Data"],
]
