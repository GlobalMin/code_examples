from operator import ne
import os
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from code_examples.utils import (
    find_categorical_features,
    find_numeric_features,
    get_logger,
)

logger = get_logger(__name__)


class ElasticNetPipeline:
    """Fit ElasticNet classifier on an arbitrary dataset."""

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
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ]
        )

        new_var = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "clf",
                    LogisticRegressionCV(
                        cv=3,
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratios=[0.5],
                        max_iter=3000,
                    ),
                ),
            ]
        )

        return new_var

    def make_train_test_split(self):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        return self

    def fit(self):
        """Fit the pipeline to the training data"""

        self.make_train_test_split()
        enet_pipeline = self.define_pipeline()

        logger.info("Fitting ENET pipeline...")
        enet_pipeline.fit(self.X_train, self.y_train)

        y_pred = enet_pipeline.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_pred)

        # Read parameters from classifier in the pipeline

        l1_ratios = enet_pipeline.named_steps["clf"].l1_ratios_
        coefficients = enet_pipeline.named_steps["clf"].coef_

        self.results.append(
            {"AUC": auc, "l1_ratio": l1_ratios, "model": enet_pipeline}
        )

        # Sort the results by AUC
        self.results.sort(key=lambda x: x["AUC"], reverse=True)
        logger.info(f'Best results: AUC = {self.results[0]["AUC"]}')

        # Pickle dump best pipeline
        with open(
            os.path.join(
                os.getcwd(),
                "model_objects",
                f"{self.dataset_name}_enet_pipeline.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.results[0]["model"], f)

    @property
    def best_auc(self):
        """Return the AUC of the best model"""
        return self.results[0]["AUC"]

    @property
    def best_hyperparameters(self):
        """Return the hyperparameters of the best model"""
        return self.results[0]["hyperparameters"]
