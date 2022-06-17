import os
import pickle
import statistics

import numpy as np
import tqdm
from numpy import mean
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
                ("onehot", OrdinalEncoder()),
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
                    RandomForestClassifier(oob_score=True, random_state=42, n_jobs=3),
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

        logger.info("Fitting RF pipeline...")
        for d in tqdm.tqdm(all_params):

            oob_scores = []
            current_best = 10000
            # Loop over n_estimators from 50 to 1000, checking if accuracy is improving over last 5 rounds.
            # If not, then stop loop.

            n_estimators_range = np.arange(50, 1000, 25)
            for i, n_estimators in enumerate(n_estimators_range):
                # Combine d with n_estimators
                full_pipeline = self.rf_pipeline
                d["n_estimators"] = n_estimators
                full_pipeline.named_steps["clf"].set_params(**d)

                full_pipeline.fit(self.X_train, self.y_train)
                # get oob scores
                oob_scores.append(full_pipeline.named_steps["clf"].oob_score_)

                # calculate rolling mean of oob scores from last 5 iterations
                def rolling_mean(oob_scores, n):
                    return statistics.mean(oob_scores[-n:])

                if i >= 5:
                    oob_scores_mean = rolling_mean(oob_scores, 5)

                else:
                    oob_scores_mean = rolling_mean(oob_scores, i + 1)
                if oob_scores_mean < current_best:
                    current_best = oob_scores_mean
                elif i >= 5:
                    full_pipeline.named_steps["clf"].set_params(
                        n_estimators=n_estimators
                    )
                    d["n_estimators"] = n_estimators
                    break
                else:
                    continue

            y_pred = full_pipeline.predict(self.X_test)
            auc = roc_auc_score(self.y_test, y_pred)

            self.results.append(
                {"AUC": auc, "hyperparameters": d, "model": full_pipeline}
            )

        # Sort the results by AUC
        self.results.sort(key=lambda x: x["AUC"], reverse=True)

        logger.info(f'Best results: AUC = {self.results[0]["AUC"]}')
        logger.info(f'Best hyperparameters: {self.results[0]["hyperparameters"]}')

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
