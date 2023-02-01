from __future__ import annotations

import copy
import os
import pickle

import numpy as np
import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from code_examples.utils import (
    find_categorical_features,
    find_numeric_features,
    get_logger,
    make_list_all_param_combinations,
)

logger = get_logger(__name__)


class RandomForestPipeline:
    """Fit RandomForest classifier on an arbitrary dataset."""

    def __init__(self, input_features, target, params, dataset_name, n_jobs=3):
        self.input_features = input_features
        self.target = target
        self.params = params
        self.dataset_name = dataset_name
        self.categorical_cols = find_categorical_features(self.input_features)
        self.numeric_cols = find_numeric_features(self.input_features)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        # Initialize rf_pipeline as a Pipeline type object
        # This will be set in the define_pipeline method
        self.results = []
        self.n_jobs = n_jobs
        self.full_pipeline = None

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

        rf_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        oob_score=True, random_state=42, n_jobs=self.n_jobs
                    ),
                ),
            ]
        )

        return rf_pipeline

    def make_train_test_split(self):
        """Split data into train and test sets"""
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_features, self.target, test_size=0.2, random_state=42
        )

        return self

    def fit(self):
        """Fit the pipeline to the training data"""

        self.make_train_test_split()
        rf_pipeline = self.define_pipeline()
        # Loop over all combinations of hyperparameters
        all_params = make_list_all_param_combinations(self.params)

        logger.info("Fitting RF pipeline...")

        for param in tqdm.tqdm(all_params):

            oob_scores = []
            rounds_without_improvement = 0
            # Loop over n_estimators from 50 to 1000, checking if accuracy is improving over last 5 rounds.
            # If not, then stop loop.

            full_pipeline = rf_pipeline
            n_estimators_range = np.arange(50, 500, 25)
            for i, n_estimators in enumerate(n_estimators_range):
                # Combine d with n_estimators
                if i == 0:
                    full_pipeline: Pipeline = copy.deepcopy(
                        rf_pipeline)

                    param["warm_start"] = True

                else:
                    assert isinstance(full_pipeline, Pipeline)

                param["n_estimators"] = n_estimators

                full_pipeline.named_steps["clf"].set_params(**param)

                full_pipeline.fit(self.x_train, self.y_train)
                # get oob scores
                current_oob_score = full_pipeline.named_steps["clf"].oob_score_
                oob_scores.append(full_pipeline.named_steps["clf"].oob_score_)

                if i == 0:
                    best_last_two_rounds = current_oob_score
                    continue
                elif i <= 3:
                    best_last_two_rounds = min(oob_scores)
                    continue
                else:
                    best_last_two_rounds = min(oob_scores[-2:])

                if current_oob_score > best_last_two_rounds:
                    rounds_without_improvement += 1
                else:
                    rounds_without_improvement = 0

                if rounds_without_improvement == 2:
                    full_pipeline.named_steps["clf"].set_params(
                        n_estimators=n_estimators
                    )
                    param["n_estimators"] = n_estimators
                    break

            y_pred = full_pipeline.predict(self.x_test)
            auc = roc_auc_score(self.y_test, y_pred)

            self.results.append(
                {"AUC": auc, "hyperparameters": param, "model": full_pipeline}
            )

        # Sort the results by AUC
        self.results.sort(key=lambda x: x["AUC"], reverse=True)

        logger.info(f'Best results: AUC = {self.results[0]["AUC"]}')
        logger.info(
            f'Best hyperparameters: {self.results[0]["hyperparameters"]}')

        # Pickle dump best pipeline
        with open(
            os.path.join(
                os.getcwd(),
                "model_objects",
                f"{self.dataset_name}_rf_pipeline.pkl",
            ),
            "wb",
        ) as dump_ubject:
            pickle.dump(self.results[0]["model"], dump_ubject)

    @ property
    def best_auc(self):
        """Return the AUC of the best model"""
        return self.results[0]["AUC"]

    @ property
    def best_hyperparameters(self):
        """Return the hyperparameters of the best model"""
        return self.results[0]["hyperparameters"]
