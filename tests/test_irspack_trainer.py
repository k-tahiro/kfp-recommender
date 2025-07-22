"""Tests for irspack training component."""

import pytest
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import tempfile
import os

from kfp_recommender.utils.data_utils import load_interaction_data, preprocess_for_irspack


def create_test_data():
    """Create sample interaction data for testing."""
    np.random.seed(42)
    
    n_users = 100
    n_items = 50
    n_interactions = 1000
    
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.rand(n_interactions) * 4 + 1  # ratings 1-5
    
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings
    })
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    
    return df


def test_load_interaction_data():
    """Test loading interaction data from CSV."""
    df = create_test_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        loaded_df = load_interaction_data(temp_path)
        
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == ['user_id', 'item_id', 'rating']
        assert loaded_df['user_id'].nunique() == df['user_id'].nunique()
        assert loaded_df['item_id'].nunique() == df['item_id'].nunique()
        
    finally:
        os.unlink(temp_path)


def test_preprocess_for_irspack():
    """Test preprocessing data for irspack."""
    df = create_test_data()
    
    train_matrix, test_matrix, user_ids, item_ids = preprocess_for_irspack(
        df, test_split_ratio=0.2, random_seed=42
    )
    
    assert isinstance(train_matrix, csr_matrix)
    assert isinstance(test_matrix, csr_matrix)
    assert len(user_ids) == df['user_id'].nunique()
    assert len(item_ids) == df['item_id'].nunique()
    
    # Check matrix shapes
    assert train_matrix.shape == (len(user_ids), len(item_ids))
    assert test_matrix.shape == (len(user_ids), len(item_ids))
    
    # Check that we have some data in both splits
    assert train_matrix.nnz > 0
    assert test_matrix.nnz > 0


def test_preprocess_no_test_split():
    """Test preprocessing with no test split."""
    df = create_test_data()
    
    train_matrix, test_matrix, user_ids, item_ids = preprocess_for_irspack(
        df, test_split_ratio=0.0
    )
    
    assert train_matrix.nnz > 0
    assert test_matrix.nnz == 0


if __name__ == "__main__":
    # Simple test runner
    test_load_interaction_data()
    test_preprocess_for_irspack()
    test_preprocess_no_test_split()
    print("All tests passed!")