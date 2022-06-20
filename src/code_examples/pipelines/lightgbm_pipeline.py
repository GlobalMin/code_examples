import os
import pickle

import lightgbm
import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from code_examples.utils import (find_categorical_features,
                                 find_numeric_features, get_logger,
                                 make_drop_cols_list,
                                 make_list_all_param_combinations,
                                 stratified_sample)

logger = get_logger(__name__)


class LightGBMPipeline:
    """Fit LighGBM classifier on an arbitrary dataset."""

    def __init__(self, X, y, params, dataset_name):
        self.X = X
        self.y = y
        self.params = params
        self.dataset_name = dataset_name
        self.categorical_cols = find_categorical_features(self.X)
        self.numeric_cols = find_numeric_features(self.X)
        self.preprocess_pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_train = None
        self.X_train_valid = None
        self.y_train_train = None
        self.y_train_valid = None
        self.X_train_preprocessed = None
        self.X_test_preprocessed = None
        self.X_train_train_preprocessed = None
        self.X_train_valid_preprocessed = None
        self.results = []

    def define_preprocess_pipeline(self):
        """Set up sklearn Pipeline for imputing missings and encoding categoricals"""
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[("ordinal_encoding", OrdinalEncoder())]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )

        # Data preprocessing pipeline
        self.preprocess_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

        return self

    def apply_train_test_split(self, X, y, test_size=0.3):
        """Apply partitiioning scheme. Important point is that we have the usual
        train/test split of the base data as is common, but in order to find the
        optimal number of trees to use, we need to split the train partition AGAIN.
        We can't use the test set to tune anything, it should be blind to the data
        until we use our model to predict."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        (
            self.X_train_train,
            self.X_train_valid,
            self.y_train_train,
            self.y_train_valid,
        ) = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        return self

    def apply_preprocess_pipline_to_all_partitions(self):
        """Apply preprocessing pipeline to all partitions"""
        self.X_train_train_preprocessed = self.preprocess_pipeline.fit_transform(
            self.X_train_train
        )
        self.X_train_valid_preprocessed = self.preprocess_pipeline.transform(
            self.X_train_valid
        )
        self.X_test_preprocessed = self.preprocess_pipeline.transform(self.X_test)

        self.X_train_preprocessed = self.preprocess_pipeline.fit_transform(self.X_train)

        return self

    def fit(self):
        """Loop over hyperparameters and select best combination based on out of sample performance"""

        # Loop over all combinations of hyperparameters
        all_params = make_list_all_param_combinations(self.params)

        self.define_preprocess_pipeline()
        self.apply_train_test_split(self.X, self.y)
        self.apply_preprocess_pipline_to_all_partitions()

        logger.info("Fitting LGBM pipeline...")

        results = []

        for d in tqdm.tqdm(all_params):
            eval_set = [(self.X_train_valid_preprocessed, self.y_train_valid)]
            # Train the model on the current set of hyperparameters

            callbacks = [
                lightgbm.early_stopping(10, verbose=0),
                lightgbm.log_evaluation(period=0),
            ]
            model = lightgbm.LGBMClassifier(**d, force_row_wise=True)
            model.fit(
                self.X_train_train_preprocessed,
                self.y_train_train,
                eval_set=eval_set,
                callbacks=callbacks,
            )

            # Overwrite n_estimators with best_iteration
            d["n_estimators"] = model.best_iteration_

            # Refit on the whole training set using optimal n_estimators
            del d["early_stopping_round"]

            # Setting up whole flow as a Pipeline object using proper n_estimators
            model = lightgbm.LGBMClassifier(**d, force_row_wise=True)

            full_pipeline = Pipeline(
                [("preprocess", self.preprocess_pipeline), ("LGBM", model)]
            )

            # full_pipeline.named_steps["LGBM"]['verbose_eval'] = False

            full_pipeline.fit(self.X_train, self.y_train)

            # Evaluate the model on the test set and calculate the AUC
            y_pred = full_pipeline.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_pred)

            # Store AUC and hyperparameters
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
                f"{self.dataset_name}_lightgbm_pipeline.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.results[0]["model"], f)

        return self.results

    @property
    def best_auc(self):
        """Return the AUC of the best model"""
        return self.results[0]["AUC"]

    @property
    def best_hyperparameters(self):
        """Return the hyperparameters of the best model"""
        return self.results[0]["hyperparameters"]
