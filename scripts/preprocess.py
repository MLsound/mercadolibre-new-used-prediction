# scripts/preprocess.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from new_or_used import build_dataset

# Import original provided dataset
X_train, y_train, X_test, y_test = build_dataset()
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train_df = pd.Series(y_train)
y_test_df = pd.Series(y_test)

# 0. Remove target value from train (avoids data leakage)
if X_train_df['condition']:
    del X_train_df['condition']

# 1. Remove empty values
empty_columns = ['differential_pricing', 'subtitle', 'listing_source', 'coverage_areas','international_delivery_mode']
X_train_df.drop(columns=empty_columns, inplace=True)
y_train_df.drop(columns=empty_columns, inplace=True)

# Apply the same transformation to the test set
X_test_df.drop(columns=empty_columns, inplace=True)
y_test_df.drop(columns=empty_columns, inplace=True)