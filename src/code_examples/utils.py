import logging
from itertools import product

import pandas as pd

logger = logging.getLogger(__name__)


def get_logger(name):
    """Set up logger level and formatting"""
    logging.getLogger().handlers = []
    logger = logging.getLogger(name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logger


def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same


def stratified_sample(df, stratify_column, proportion, seed=42):
    """Calculate a stratified sample of the dataframe, keeping proportion of each class"""
    df_stratified = df.groupby(stratify_column).apply(
        lambda x: x.sample(frac=proportion, replace=False, random_state=seed)
    )

    return df_stratified


def find_unique_cols(df):
    """Find columns with 1 unique value"""
    unique_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            unique_cols.append(col)
    logger.info("Found {} columns with 1 unique value".format(len(unique_cols)))
    return unique_cols


def find_high_counts_cols(df, threshold=0.99):
    """Find columns with value counts percent greater than threshold"""
    high_counts_cols = []
    for col in df.columns:
        if df[col].value_counts(normalize=True).max() > threshold:
            high_counts_cols.append(col)
    logger.info(
        "Found {} columns with values {}% the same".format(
            len(high_counts_cols), round(threshold * 100, 2)
        )
    )
    return high_counts_cols


def make_drop_cols_list(df):
    """Make a list of columns to drop from the dataframe"""
    drop_cols = find_unique_cols(df)
    drop_cols += find_high_counts_cols(df)
    return list(set(drop_cols))


def find_categorical_features(df):
    """Find categorical features based on two heuristics:
    1. If a column has a type of 'object'
    2. If a column has 2 unique values

    Currently this will have an issue detecting text.

    TODO: Extend to handle text"""

    cat_cols = []
    for col in df.columns:
        if df[col].dtype == "object" or len(df[col].unique()) == 2:
            cat_cols.append(col)
    logger.info("Found {} categorical features".format(len(cat_cols)))
    return cat_cols


def find_numeric_features(df):
    """Find numeric features"""
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_cols = []
    for col in df.columns:
        if df[col].dtype in numeric_types:
            num_cols.append(col)
    logger.info("Found {} numeric features".format(len(num_cols)))
    return num_cols


def make_list_all_param_combinations(params):
    """Make a list of dictionaries of all combinations of parameters"""
    keys, values = zip(*params.items())
    return [dict(zip(keys, v)) for v in product(*values)]
