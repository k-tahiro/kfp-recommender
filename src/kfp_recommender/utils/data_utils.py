"""Data preprocessing utilities for irspack components."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


def load_interaction_data(
    data_path: str,
    user_col: str = "user_id", 
    item_col: str = "item_id",
    rating_col: str = "rating"
) -> pd.DataFrame:
    """Load interaction data from CSV or Parquet file.
    
    Args:
        data_path: Path to the data file
        user_col: Name of the user ID column
        item_col: Name of the item ID column  
        rating_col: Name of the rating column
        
    Returns:
        DataFrame with interaction data
    """
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Validate required columns
    required_cols = [user_col, item_col]
    if rating_col in df.columns:
        required_cols.append(rating_col)
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df[[user_col, item_col, rating_col]].rename(columns={
        user_col: 'user_id',
        item_col: 'item_id', 
        rating_col: 'rating'
    })


def preprocess_for_irspack(
    df: pd.DataFrame,
    test_split_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    """Preprocess interaction data for irspack training.
    
    Args:
        df: DataFrame with columns ['user_id', 'item_id', 'rating']
        test_split_ratio: Ratio of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_matrix, test_matrix, user_ids, item_ids)
    """
    # Create user and item mappings
    unique_users = df['user_id'].unique()
    unique_items = df['item_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    # Map to indices
    df_indexed = df.copy()
    df_indexed['user_idx'] = df_indexed['user_id'].map(user_to_idx)
    df_indexed['item_idx'] = df_indexed['item_id'].map(item_to_idx)
    
    # Split data by user to ensure users appear in both train/test
    if test_split_ratio > 0:
        users_train, users_test = train_test_split(
            unique_users, test_size=test_split_ratio, random_state=random_seed
        )
        
        train_df = df_indexed[df_indexed['user_id'].isin(users_train)]
        test_df = df_indexed[df_indexed['user_id'].isin(users_test)]
    else:
        train_df = df_indexed
        test_df = pd.DataFrame(columns=df_indexed.columns)
    
    # Create sparse matrices
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    def create_matrix(data_df):
        if len(data_df) == 0:
            return csr_matrix((n_users, n_items))
        
        return csr_matrix(
            (data_df['rating'].values, 
             (data_df['user_idx'].values, data_df['item_idx'].values)),
            shape=(n_users, n_items)
        )
    
    train_matrix = create_matrix(train_df)
    test_matrix = create_matrix(test_df)
    
    return train_matrix, test_matrix, unique_users, unique_items


def split_by_time(
    df: pd.DataFrame,
    time_col: str,
    test_split_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time for temporal evaluation.
    
    Args:
        df: DataFrame with interaction data
        time_col: Name of timestamp column
        test_split_ratio: Ratio of recent data for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")
    
    df_sorted = df.sort_values(time_col)
    split_idx = int(len(df_sorted) * (1 - test_split_ratio))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    return train_df, test_df