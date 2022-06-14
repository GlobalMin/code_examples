import os
import pickle

import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from code_examples.utils import (find_categorical_features,
                                 find_numeric_features, get_logger,
                                 make_list_all_param_combinations)

logger = get_logger(__name__)


class RandomForestPipeline:
    """Fit RandomForest classifier on an arbitrary dataset."""

    def __init__(self, X, y, params, dataset_name):
        self.X = X
        self.y = y
        self.params = params
        self.dataset_name = dataset_name
        self.categorical_cols = find_categorical_features(self.X)
        self.numeric_cols = find_numeric_features(self.X)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_pipeline = None
        self.results = []

    def define_pipeline(self):
        """Set up sklearn Pipeline for imputing missings and encoding categoricals"""
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant")),
                ("onehot", OrdinalEncoder(encoded_missing_value=-1)),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )

        self.rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(),
                ),
            ]
        )

        return self

    def make_train_test_split(self):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        return self

    def fit(self):
        """Fit the pipeline to the training data"""

        self.make_train_test_split()
        self.define_pipeline()
        # Loop over all combinations of hyperparameters
        all_params = make_list_all_param_combinations(self.params)

        for d in tqdm.tqdm(all_params):
            full_pipeline = self.rf_pipeline
            full_pipeline.named_steps["clf"].set_params(**d)
            full_pipeline.fit(self.X_train, self.y_train)
            y_pred = full_pipeline.predict(self.X_test)
            auc = roc_auc_score(self.y_test, y_pred)

            logger.info("Hyperparameters: {}".format(d))
            logger.info("AUC: {}".format(auc))

            self.results.append(
                {"AUC": auc, "hyperparameters": d, "model": full_pipeline}
            )

        # Sort the results by AUC
        self.results.sort(key=lambda x: x["AUC"], reverse=True)

        # Pickle dump best pipeline
        with open(
            os.path.join(
                os.getcwd(),
                "model_objects",
                f"{self.dataset_name}_rf_pipeline.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.results[0]["model"], f)
