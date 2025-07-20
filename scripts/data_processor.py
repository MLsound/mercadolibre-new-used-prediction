# scripts/preprocess.py
"""
This module handles reading raw JSONLines data, performing necessary feature
engineering and scaling, and splitting the dataset into training and testing
sets.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# This goes up one level from 'scripts/' to the project root './'
import sys
import os
script_dir = os.path.dirname(__file__) # Get the directory of the current script
sys.path.append(os.path.join(script_dir, '..')) # Add the parent directory to sys.path

from new_or_used import build_dataset

def remove_outliers_iqr(data: list[pd.DataFrame], column: str, iqr_factor: float = 1.5) -> list[pd.DataFrame]:
    """
    Removes outliers from a specified column in a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the numerical column to remove outliers from.
        iqr_factor (float): The multiplier for the IQR to define outlier bounds (default is 1.5).

    Returns:
        list[pd.DataFrame]: A list with new DataFrames for every split with outliers removed from the specified column.
    """
    df_train, df_test = data

    if column not in df_train.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    Q1 = df_train[column].quantile(0.25)
    Q3 = df_train[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR

    # Filter out the outliers
    df_train_cleaned = df_train[(df_train[column] >= lower_bound) & (df_train[column] <= upper_bound)]
    df_test_cleaned = df_test[(df_test[column] >= lower_bound) & (df_test[column] <= upper_bound)]

    return [df_train_cleaned, df_test_cleaned]

def load_processed_data():
    """
    Load unprocessed data (.jsonlines) and transform data for training & evaluation.

    Returns:
        pd.DataFrame: features_train - Transformed features for training.
        pd.DataFrame: target_train - Corresponding target variable for training.
        pd.DataFrame: features_test - Transformed features for evaluation.
        pd.DataFrame: target_test - Corresponding target variable for evaluation.
    """

    print("Importing data from 'new_or_used.py'…")
    # Import original provided dataset
    X_train, y_train, X_test, y_test = build_dataset()
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_df = pd.Series(y_train)
    y_test_df = pd.Series(y_test)

    print("Starting preprocessing…")

    # 0. Remove target value from train (avoids data leakage) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if 'condition' in X_train_df.columns:
        del X_train_df['condition']

    # 1. Remove empty values (features) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    empty_columns = ['differential_pricing', 'subtitle', 'listing_source', 'coverage_areas','international_delivery_mode']
    X_train_df.drop(columns=empty_columns, inplace=True)
    y_train_df.drop(columns=empty_columns, inplace=True)
    X_test_df.drop(columns=empty_columns, inplace=True)
    y_test_df.drop(columns=empty_columns, inplace=True)

    # 2. Remove irrelevant data (items) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # items with inactive posts ('sub_status'!=[])
    mask = X_train_df['sub_status'].astype(str) == '[]'
    X_train_df = X_train_df[mask]
    y_train_df = y_train_df[mask]
    X_train_df.drop(columns=['sub_status'], inplace=True)
    X_train_df.reset_index(drop=True, inplace=True)

    mask = X_test_df['sub_status'].astype(str) == '[]'
    X_test_df = X_test_df[mask]
    y_test_df = y_test_df[mask]
    X_test_df.drop(columns=['sub_status'], inplace=True)
    X_test_df.reset_index(drop=True, inplace=True)

    # 3. Outliers handling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    continuous_features = ['price'] # 'initial_quantity' removed for 6️⃣

    # Initialize the DataFrame that will store the cumulatively filtered data.
    X_train_outlier_df = X_train_df.copy()
    X_test_outlier_df = X_test_df.copy()

    for feature in continuous_features:
        X_train_outlier_df, X_test_outlier_df = remove_outliers_iqr([X_train_outlier_df,X_test_outlier_df], feature, iqr_factor=1.5)

    # 4. Normalization (Z-score) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X_train_scaled_df = X_train_outlier_df.copy().dropna()
    X_test_scaled_df = X_test_outlier_df.copy().dropna()

    scaler = MinMaxScaler() # Initialize the scaler
    scaler.fit(X_train_df[continuous_features]) # Fit the scaler to continuous data

    # Transform the continuous features
    X_train_scaled_df[continuous_features] = scaler.transform(X_train_df[continuous_features])
    X_test_scaled_df[continuous_features] = scaler.transform(X_test_df[continuous_features])

    # 5. Concatenate normalized data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    non_continuous_features = [col for col in X_train_df.columns if col not in continuous_features]
    # df_train_non_continuous = X_train_df[non_continuous_features] # Select these non-continuous features
    # df_test_non_continuous = X_test_df[non_continuous_features]
    df_train_non_continuous = X_train_df.copy()
    df_test_non_continuous = X_test_df.copy()

    X_train_scaled_df.drop(columns=non_continuous_features, inplace=True)  # Drop original continuous features
    X_test_scaled_df.drop(columns=non_continuous_features, inplace=True)  

    X_train_scaled_df_renamed = X_train_scaled_df.add_suffix('_scaled')  # Rename columns to add '_scaled' suffix
    X_test_scaled_df_renamed = X_test_scaled_df.add_suffix('_scaled')

    X_train_df = pd.concat([df_train_non_continuous, X_train_scaled_df_renamed], axis=1) # Joining scaled features with non-continuous
    X_test_df = pd.concat([df_test_non_continuous, X_test_scaled_df_renamed], axis=1)

    # 6. Binary Encoding (boolean variables) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Encode string binary features
    X_train_df['is_USD'] = (X_train_df['currency_id'] == 'USD').astype('uint8') # as numeric type (1/0)
    X_test_df['is_USD'] = (X_test_df['currency_id'] == 'USD').astype('uint8')
    X_train_df.drop(columns=['currency_id'], inplace=True)
    X_test_df.drop(columns=['currency_id'], inplace=True)

    # Convert booleans into integers
    for col in ['accepts_mercadopago', 'automatic_relist']:
        X_train_df[col] = X_train_df[col].dropna().astype('uint8')
        X_test_df[col] = X_test_df[col].dropna().astype('uint8')

    # 5. One-hot Encoding (dummy variables) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    categorical_features = ['listing_type_id', 'buying_mode'] # 'status' no longer used
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first') # Initialize OneHotEncoder
    encoded_train_data = encoder.fit_transform(X_train_df[categorical_features]) # Fit and transform for train split
    encoded_feature_names = encoder.get_feature_names_out(categorical_features) # Get features names
    encoded_test_data = encoder.transform(X_test_df[categorical_features]) # Transform for test split
    ecoded_features = pd.DataFrame(encoded_train_data, columns=encoded_feature_names)
    encoded_features_test = pd.DataFrame(encoded_test_data, columns=encoded_feature_names)
    X_train_transformed_df = pd.concat([X_train_df.drop(columns=categorical_features), ecoded_features], axis=1) # Concatenate with the original DataFrame 
    X_test_transformed_df = pd.concat([X_test_df.drop(columns=categorical_features), encoded_features_test], axis=1)

    # 6. Synthetic data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2️⃣ Second model trained
    X_train_transformed_df['free_tier'] = X_train_transformed_df['listing_type_id_free'] == True
    X_test_transformed_df['free_tier'] = X_test_transformed_df['listing_type_id_free'] == True

    # 3️⃣ Third model trained
    # TIME FEATURES
    # Convert 'date_created' to datetime objects
    X_train_transformed_df['date_created_dt'] = pd.to_datetime(X_train_transformed_df['date_created'])
    X_test_transformed_df['date_created_dt'] = pd.to_datetime(X_test_transformed_df['date_created'])
    # Create the 'time_created' (hour of the day) & 'day_of_week' (0=Monday, 6=Sunday) features
    X_train_transformed_df['time_created'] = X_train_transformed_df['date_created_dt'].dt.hour
    X_test_transformed_df['time_created'] = X_test_transformed_df['date_created_dt'].dt.hour
    X_train_transformed_df['day_of_week'] = X_train_transformed_df['date_created_dt'].dt.dayofweek
    X_test_transformed_df['day_of_week'] = X_test_transformed_df['date_created_dt'].dt.dayofweek
    # Drop unnecesary columns
    X_train_transformed_df = X_train_transformed_df.drop(columns=['date_created', 'date_created_dt'])
    X_test_transformed_df = X_test_transformed_df.drop(columns=['date_created', 'date_created_dt'])
    # Create 'is_weekend' & 'is_working_hours' features
    X_train_transformed_df['is_weekend'] = X_train_transformed_df['day_of_week'].isin([5, 6]).astype(int)
    X_test_transformed_df['is_weekend'] = X_test_transformed_df['day_of_week'].isin([5, 6]).astype(int)
    X_train_transformed_df['is_working_hours'] = ((X_train_transformed_df['time_created'] >= 6) & (X_train_transformed_df['time_created'] <= 20)).astype(int)
    X_test_transformed_df['is_working_hours'] = ((X_test_transformed_df['time_created'] >= 6) & (X_test_transformed_df['time_created'] <= 20)).astype(int)

    # 6️⃣ Sixth model trained
    # CATEGORICAL QUANTITY
    def categorize_quantity(quantity_value):
        if quantity_value == 1:
            return 'single_unit'
        elif 1 < quantity_value < 10:
            return 'small'
        else: # value >= 10
            return 'large'

    # Encode categories
    X_train_transformed_df['quantity_category'] = X_train_transformed_df['initial_quantity'].apply(categorize_quantity)
    X_test_transformed_df['quantity_category'] = X_test_transformed_df['initial_quantity'].apply(categorize_quantity)
    # Apply one-hot encoding to the 'quantity_category' column
    X_train_transformed_df = pd.get_dummies(X_train_transformed_df, columns=['quantity_category'], prefix='quant', drop_first=True)
    X_test_transformed_df = pd.get_dummies(X_test_transformed_df, columns=['quantity_category'], prefix='quant', drop_first=True)
    cat_quant = [x for x in X_train_transformed_df.columns.to_list() if x.startswith('quant')]
    # Optional: for applying just 1 vs many categorization ('is_single_unit')
    # X_train_transformed_df['is_single_unit'] = X_train_transformed_df['initial_quantity'] == 1
    # X_test_transformed_df['is_single_unit'] = X_test_transformed_df['initial_quantity'] == 1
    
    # 7️⃣ Seventh model trained
    # 'parent_item_id' -> 'has_paren_item'
    X_train_transformed_df['has_parent_item'] = ~X_train_transformed_df['parent_item_id'].isnull()
    X_test_transformed_df['has_parent_item'] = ~X_test_transformed_df['parent_item_id'].isnull()
    # 'official_store_id' -> 'has_store'
    X_train_transformed_df['has_store'] = ~X_train_transformed_df['official_store_id'].isnull()
    X_test_transformed_df['has_store'] = ~X_test_transformed_df['official_store_id'].isnull()
    # 'price' -> 'high_ticket'
    q3_price = X_train_df['price'].quantile(0.75) # Use the 75th percentile (Q3) as threshold
    X_train_transformed_df['high_ticket'] = X_train_transformed_df['price'] > q3_price
    X_test_transformed_df['high_ticket'] = X_test_transformed_df['price'] > q3_price

    # 7. Feature selection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1️⃣ First model trained
    # features = ['accepts_mercadopago', 'automatic_relist', 'price_scaled',
    #     'initial_quantity_scaled', 'is_USD', 'listing_type_id_free',
    #     'listing_type_id_gold', 'listing_type_id_gold_premium',
    #     'listing_type_id_gold_pro', 'listing_type_id_gold_special',
    #     'listing_type_id_silver', 'buying_mode_buy_it_now',
    #     'buying_mode_classified', 'status_not_yet_active', 'status_paused']

    # 2️⃣ Second model trained (-50% features)
    # features = ['accepts_mercadopago', 'automatic_relist', 'price_scaled',
    #     'initial_quantity_scaled', 'is_USD', 'free_tier', 'buying_mode_buy_it_now',
    #     'buying_mode_classified']
    
    # 3️⃣ Third model trained (datetime features)
    # features = ['accepts_mercadopago', 'automatic_relist', 'price_scaled',
    #     'initial_quantity_scaled', 'is_USD', 'free_tier', 'buying_mode_buy_it_now',
    #     'buying_mode_classified','is_weekend','is_working_hours']

    # 4️⃣ Fourth & 5️⃣ Fifth models trained (removing 'initial_quantity_scaled')
    # features = ['accepts_mercadopago', 'automatic_relist', 'price_scaled',
    #     'is_USD', 'free_tier', 'buying_mode_buy_it_now',
    #     'buying_mode_classified','is_weekend','is_working_hours']

    # 6️⃣ Sixth model trained (categorical 'initial_quantity')
    # features = ['accepts_mercadopago', 'automatic_relist', 'price_scaled',
    #     'is_USD', 'free_tier', 'buying_mode_buy_it_now',
    #     'buying_mode_classified','is_weekend','is_working_hours'] + cat_quant  # 'is_single_unit'
    
    # 7️⃣ Seventh model trained (synth features)
    features = ['accepts_mercadopago', 'automatic_relist', 'price_scaled',
        'is_USD', 'free_tier', 'buying_mode_buy_it_now',
        'buying_mode_classified','is_weekend','is_working_hours',
        'has_parent_item','has_store','high_ticket'] + cat_quant  # 'is_single_unit'

    features_train = X_train_transformed_df[features].copy()
    target_train = y_train_df.copy()
    features_test = X_test_transformed_df[features].copy()
    target_test = y_test_df.copy()

    # 8. Encoded target ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # String classes will be replaced by integers as a good performance practice
    # classes = {1: 'new', 0: 'used'} # tags for human readabilty
    target_train_encoded = (target_train == 'new').astype(int)
    target_test_encoded = (target_test == 'new').astype(int)

    print("Succesfully finished.\n")

    #return features_train, target_train, features_test, target_test
    return features_train, target_train, features_test, target_test



if __name__=='__main__':

    features_train, target_train, features_test, target_test = load_processed_data()

    # Processed dataframe output
    print("TRAIN SPLIT")
    print(features_train.info(),'\n')
    print("Target:")
    print(target_train.value_counts(),'\n')
    print("TEST SPLIT")
    print(features_test.info(),'\n')
    print("Target:")
    print(target_test.value_counts(),'\n')