import time

import yaml
from dotenv import load_dotenv

from code_examples.benchmark_datasets import DATASETS_INFO
from code_examples.pipelines.elastic_net_pipeline import ElasticNetPipeline
from code_examples.pipelines.lightgbm_pipeline import LightGBMPipeline
from code_examples.pipelines.random_forest_pipeline import RandomForestPipeline
from code_examples.pipelines.xgboost_pipeline import XGBoostPipeline
from code_examples.utils import get_logger

load_dotenv()
logger = get_logger(__name__)


# Load params.yaml
with open("config.yaml", "r") as f:
    params = yaml.safe_load(f)


def train(X, y, dataset_name):
    """Train a model on a dataset."""
    for p in params:
        for key, value in p.items():
            if key == "XGBoost":
                xgb_params = value
            elif key == "Random Forest":
                rf_params = value
            elif key == "LightGBM":
                lgbm_params = value

    light_gbm_pipeline = LightGBMPipeline(
        X, y, params=lgbm_params, dataset_name=dataset_name
    )
    light_gbm_pipeline.fit()

    rf_pipeline = RandomForestPipeline(X, y, rf_params, dataset_name, n_jobs=-1)
    rf_pipeline.fit()

    enet_pipeline = ElasticNetPipeline(X, y, xgb_params, dataset_name)
    enet_pipeline.fit()

    pipeline = XGBoostPipeline(X, y, xgb_params, dataset_name)
    pipeline.fit()

    # Format f string to 2 decimals 
    logger.info("")    
    logger.info(f"LightGBM AUC: {light_gbm_pipeline.best_auc:.2f}")
    logger.info(f"RF AUC:{rf_pipeline.best_auc:.2f}")
    logger.info(f"ENET AUC:{enet_pipeline.best_auc:.2f}")
    logger.info(f"XGBoost AUC:{pipeline.best_auc:.2f}")
    logger.info("-----------------------------------------------------")

tic = time.time()
for dataset in DATASETS_INFO:
    logger.info("-----------------------------------------------------")
    df, target, desc = dataset

    y = df.pop(target)
    X = df

    logger.info(f"Training model on {desc} with {df.shape[0]} rows")
    train(X, y, dataset_name=desc)

toc = time.time()

logger.info(f"Total time: {(toc - tic) / 60:.2f} minutes")
