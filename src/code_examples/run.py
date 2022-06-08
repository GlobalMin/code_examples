from dotenv import load_dotenv

from code_examples.benchmark_datasets import DATASETS_INFO
from code_examples.pipelines.xgboost_pipeline import XGBoostPipeline
from code_examples.utils import get_logger

load_dotenv()
logger = get_logger(__name__)


xgb_params = {
    "eta": [0.01, 0.005],
    "max_depth": [5, 6, 7],
    "n_estimators": [1000],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "objective": ["binary:logistic"],
    "eval_metric": ["auc"],
    "early_stopping_rounds": [200],
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
