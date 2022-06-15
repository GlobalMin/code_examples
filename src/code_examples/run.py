import yaml
from dotenv import load_dotenv

from code_examples.benchmark_datasets import DATASETS_INFO
from code_examples.pipelines.elastic_net_pipeline import ElasticNetPipeline
from code_examples.pipelines.random_forest_pipeline import RandomForestPipeline
from code_examples.pipelines.xgboost_pipeline import XGBoostPipeline
from code_examples.utils import get_logger

load_dotenv()
logger = get_logger(__name__)


# Load params.yaml
with open("config.yaml", "r") as f:
    params = yaml.safe_load(f)


def train(X, y, dataset_name):

    for p in params:
        for key, value in p.items():
            if key == "XGBoost":
                xgb_params = value
            elif key == "Random Forest":
                rf_params = value

    rf_pipeline = RandomForestPipeline(X, y, rf_params, dataset_name)
    rf_pipeline.fit()

    enet_pipeline = ElasticNetPipeline(X, y, xgb_params, dataset_name)
    enet_pipeline.fit()

    pipeline = XGBoostPipeline(X, y, xgb_params, dataset_name)

    pipeline.define_preprocess_pipeline().apply_train_test_split(
        X, y
    ).fit_and_tune_xboost()


for dataset in DATASETS_INFO:
    df, target, desc = dataset

    y = df.pop(target)
    X = df

    logger.info(f"Training model on {desc}")
    train(X, y, dataset_name=desc)
