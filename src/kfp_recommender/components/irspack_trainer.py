"""KFP Component for irspack model training."""

import json
import os
from typing import List

from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, Artifact


@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "irspack>=0.5.0",
        "pandas>=1.5.0", 
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0"
    ]
)
def irspack_model_trainer(
    data_path: Input[Dataset],
    algorithm: str = "ials",
    hyperparameters: str = "{}",
    test_split_ratio: float = 0.2,
    evaluation_metrics: List[str] = None,
    user_col: str = "user_id",
    item_col: str = "item_id", 
    rating_col: str = "rating",
    random_seed: int = 42,
    model_path: Output[Model],
    metrics_path: Output[Metrics]
) -> None:
    """Train an irspack recommendation model.
    
    Args:
        data_path: Input dataset with user-item interactions
        algorithm: Algorithm to use (ials, bpr, p3alpha, itemknn, userknn, random)
        hyperparameters: JSON string with model hyperparameters
        test_split_ratio: Fraction of data to use for testing (0.0 to 1.0)
        evaluation_metrics: List of metrics to compute during evaluation
        user_col: Name of user ID column in data
        item_col: Name of item ID column in data
        rating_col: Name of rating column in data
        random_seed: Random seed for reproducibility
        model_path: Output path for trained model
        metrics_path: Output path for evaluation metrics
    """
    import pandas as pd
    import numpy as np
    import json
    import pickle
    from scipy.sparse import csr_matrix
    from sklearn.model_selection import train_test_split
    
    # Import irspack components
    from irspack.recommenders import (
        IALSRecommender, 
        BPRFMRecommender,
        P3alphaRecommender,
        ItemKNNRecommender,
        UserKNNRecommender,
        RandomRecommender
    )
    from irspack.evaluator import Evaluator
    
    # Set defaults
    if evaluation_metrics is None:
        evaluation_metrics = ['ndcg', 'recall', 'precision']
    
    print(f"Starting irspack model training with algorithm: {algorithm}")
    print(f"Test split ratio: {test_split_ratio}")
    print(f"Evaluation metrics: {evaluation_metrics}")
    
    # Load and validate data
    print(f"Loading data from: {data_path.path}")
    if data_path.path.endswith('.parquet'):
        df = pd.read_parquet(data_path.path)
    elif data_path.path.endswith('.csv'):
        df = pd.read_csv(data_path.path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.path}")
    
    print(f"Loaded {len(df)} interactions")
    
    # Validate columns
    required_cols = [user_col, item_col]
    if rating_col in df.columns:
        required_cols.append(rating_col)
    else:
        # Create implicit ratings if no rating column
        df[rating_col] = 1.0
        print("No rating column found, using implicit feedback (rating=1.0)")
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Standardize column names
    interaction_df = df[[user_col, item_col, rating_col]].rename(columns={
        user_col: 'user_id',
        item_col: 'item_id', 
        rating_col: 'rating'
    })
    
    # Create user and item mappings
    unique_users = interaction_df['user_id'].unique()
    unique_items = interaction_df['item_id'].unique()
    
    print(f"Dataset: {len(unique_users)} users, {len(unique_items)} items")
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    # Map to indices
    interaction_df['user_idx'] = interaction_df['user_id'].map(user_to_idx)
    interaction_df['item_idx'] = interaction_df['item_id'].map(item_to_idx)
    
    # Split data
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    if test_split_ratio > 0:
        print(f"Splitting data: {1-test_split_ratio:.1%} train, {test_split_ratio:.1%} test")
        users_train, users_test = train_test_split(
            unique_users, test_size=test_split_ratio, random_state=random_seed
        )
        
        train_df = interaction_df[interaction_df['user_id'].isin(users_train)]
        test_df = interaction_df[interaction_df['user_id'].isin(users_test)]
    else:
        print("No test split - using all data for training")
        train_df = interaction_df
        test_df = pd.DataFrame(columns=interaction_df.columns)
    
    # Create sparse matrices
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
    
    print(f"Train matrix shape: {train_matrix.shape}, density: {train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]):.6f}")
    if test_split_ratio > 0:
        print(f"Test matrix shape: {test_matrix.shape}, density: {test_matrix.nnz / (test_matrix.shape[0] * test_matrix.shape[1]):.6f}")
    
    # Parse hyperparameters
    try:
        hyperparams = json.loads(hyperparameters)
        print(f"Using hyperparameters: {hyperparams}")
    except json.JSONDecodeError:
        print(f"Invalid JSON in hyperparameters: {hyperparameters}")
        hyperparams = {}
    
    # Get default hyperparameters for algorithm
    default_hyperparams = {
        'ials': {
            'n_components': 64,
            'reg': 1e-3,
            'alpha': 1.0,
            'n_epochs': 20
        },
        'bpr': {
            'n_components': 64,
            'reg': 1e-4,
            'learning_rate': 1e-3,
            'n_epochs': 100
        },
        'p3alpha': {
            'alpha': 1.0,
            'normalize_weight': False
        },
        'itemknn': {
            'top_k': 20,
            'normalize': True,
            'similarity': 'cosine'
        },
        'userknn': {
            'top_k': 20,
            'normalize': True,
            'similarity': 'cosine'
        },
        'random': {}
    }
    
    # Merge with defaults
    final_hyperparams = default_hyperparams.get(algorithm, {})
    final_hyperparams.update(hyperparams)
    
    # Train model
    algorithm_classes = {
        'ials': IALSRecommender,
        'bpr': BPRFMRecommender,
        'p3alpha': P3alphaRecommender,
        'itemknn': ItemKNNRecommender,
        'userknn': UserKNNRecommender,
        'random': RandomRecommender
    }
    
    if algorithm not in algorithm_classes:
        raise ValueError(f"Unsupported algorithm: {algorithm}. "
                        f"Supported: {list(algorithm_classes.keys())}")
    
    print(f"Training {algorithm} model...")
    recommender_class = algorithm_classes[algorithm]
    model = recommender_class(train_matrix, **final_hyperparams)
    print(f"Training completed: {type(model).__name__}")
    
    # Evaluate model
    results = {}
    if test_split_ratio > 0 and test_matrix.nnz > 0:
        print("Evaluating model...")
        
        evaluator = Evaluator(test_matrix, cutoff=20)
        scores = model.get_score_remove_seen(test_matrix)
        
        for cutoff in [10, 20]:
            cutoff_results = evaluator.get_scores(scores, cutoff=cutoff)
            
            for metric in evaluation_metrics:
                if metric in cutoff_results:
                    key = f"{metric}@{cutoff}"
                    results[key] = float(cutoff_results[metric])
                    print(f"{key}: {results[key]:.4f}")
    else:
        print("Skipping evaluation (no test data)")
        for metric in evaluation_metrics:
            for cutoff in [10, 20]:
                results[f"{metric}@{cutoff}"] = 0.0
    
    # Save model
    model_data = {
        'algorithm': algorithm,
        'model': model,
        'hyperparameters': final_hyperparams,
        'user_mapping': user_to_idx,
        'item_mapping': item_to_idx,
        'n_users': n_users,
        'n_items': n_items
    }
    
    with open(model_path.path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {model_path.path}")
    
    # Save metrics
    with open(metrics_path.path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path.path}")
    print("Training completed successfully!")