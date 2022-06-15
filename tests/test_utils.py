import random

import numpy as np
import pandas as pd

from code_examples.utils import (find_high_counts_cols, find_unique_cols,
                                 stratified_sample)


# Test stratified_sample
def test_stratified_sample():
    """Create dummy df and assert proportion of each
    group is the same before and after sampling"""

    random.seed(42)
    groups = ["Group 1", "Group 2", "Group 3"]
    col_a = random.choices(groups, k=1000)
    col_b = [random.randint(0, 100) for _ in range(1000)]
    df = pd.DataFrame({"Column A": col_a, "Column B": col_b})

    df_stratified = stratified_sample(
        df=df, stratify_column="Column A", proportion=0.25, seed=42
    )
    df_stratified.reset_index(drop=True, inplace=True)

    # Calculate proportion of each group
    group_proportions_before = df.groupby("Column A").size() / len(df)
    group_proportions_after = df_stratified.groupby("Column A").size() / len(
        df_stratified
    )

    # Assert proportion of each group is the same within a small tolerance window
    assert np.allclose(group_proportions_before, group_proportions_after, atol=0.01)


# Test find_unique_cols
def test_find_unique_cols():
    """Create dummy df and assert columns with 1 unique value are found"""

    random.seed(42)
    col_a = [random.randint(0, 100) for _ in range(1000)]
    col_b = [random.randint(0, 100) for _ in range(1000)]
    col_c = np.repeat(1, 1000)
    col_d = col_c.copy()
    col_d[1] = 0

    df = pd.DataFrame(
        {"Column A": col_a, "Column B": col_b, "Column C": col_c, "Column D": col_d}
    )

    unique_cols = find_unique_cols(df)

    assert "Column C" in unique_cols
    # Column D is 99% unique but it has 1 value changed manually, so it should not be in list
    assert "Column D" not in unique_cols


# Test find_high_counts_cols
def test_find_high_counts_cols():
    """Create dummy df and assert columns with high counts are found"""

    random.seed(42)
    col_a = [random.randint(0, 100) for _ in range(1000)]
    col_b = [random.randint(0, 100) for _ in range(1000)]
    col_c = np.repeat(1, 1000)
    col_d = col_c.copy()
    col_d[1] = 0

    df = pd.DataFrame(
        {"Column A": col_a, "Column B": col_b, "Column C": col_c, "Column D": col_d}
    )

    high_counts_cols = find_high_counts_cols(df, threshold=0.9)

    assert "Column C" in high_counts_cols
    assert "Column D" in high_counts_cols

    # Replace Column D index 1:101 with 0
    df.loc[0:101, "Column D"] = 0

    high_counts_cols = find_high_counts_cols(df, threshold=0.9)

    assert "Column D" not in high_counts_cols
