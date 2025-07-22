"""Core training logic for irspack models."""

import json
import pickle
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix

try:
    from irspack import Recommenders
    from irspack.recommenders import (
        IALSRecommender, 
        BPRFMRecommender,
        P3alphaRecommender,
        ItemKNNRecommender,
        UserKNNRecommender,
        RandomRecommender
    )
    from irspack.evaluator import Evaluator
except ImportError:
    raise ImportError("irspack is required. Install with: pip install irspack")


class IrspackTrainer:
    """Handles irspack model training and evaluation."""
    
    SUPPORTED_ALGORITHMS = {
        'ials': IALSRecommender,
        'bpr': BPRFMRecommender,
        'p3alpha': P3alphaRecommender,
        'itemknn': ItemKNNRecommender,
        'userknn': UserKNNRecommender,
        'random': RandomRecommender
    }
    
    def __init__(self, algorithm: str = 'ials'):
        """Initialize trainer with specified algorithm.
        
        Args:
            algorithm: Algorithm name (ials, bpr, p3alpha, itemknn, userknn, random)
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Supported: {list(self.SUPPORTED_ALGORITHMS.keys())}")
        
        self.algorithm = algorithm
        self.recommender_class = self.SUPPORTED_ALGORITHMS[algorithm]
        self.model = None
        
    def train(
        self, 
        train_matrix: csr_matrix,
        hyperparameters: Dict[str, Any] = None
    ) -> None:
        """Train the irspack model.
        
        Args:
            train_matrix: Training interaction matrix (users x items)
            hyperparameters: Model hyperparameters
        """
        if hyperparameters is None:
            hyperparameters = self._get_default_hyperparameters()
        
        print(f"Training {self.algorithm} model with parameters: {hyperparameters}")
        
        self.model = self.recommender_class(train_matrix, **hyperparameters)
        
        print(f"Training completed. Model: {type(self.model).__name__}")
        
    def evaluate(
        self,
        test_matrix: csr_matrix,
        metrics: List[str] = None,
        cutoffs: List[int] = None
    ) -> Dict[str, float]:
        """Evaluate the trained model.
        
        Args:
            test_matrix: Test interaction matrix
            metrics: List of metrics to compute
            cutoffs: List of cutoff values for ranking metrics
            
        Returns:
            Dictionary of metric names to values
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        if metrics is None:
            metrics = ['ndcg', 'recall', 'precision']
            
        if cutoffs is None:
            cutoffs = [10, 20]
            
        # Create evaluator
        evaluator = Evaluator(test_matrix, cutoff=max(cutoffs))
        
        # Compute scores for all users
        scores = self.model.get_score_remove_seen(test_matrix)
        
        results = {}
        for cutoff in cutoffs:
            cutoff_results = evaluator.get_scores(scores, cutoff=cutoff)
            
            for metric in metrics:
                if metric in cutoff_results:
                    key = f"{metric}@{cutoff}"
                    results[key] = float(cutoff_results[metric])
                    
        return results
        
    def save_model(self, model_path: str) -> None:
        """Save the trained model to file.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        model_data = {
            'algorithm': self.algorithm,
            'model': self.model,
            'model_class': self.recommender_class.__name__
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to: {model_path}")
        
    def load_model(self, model_path: str) -> None:
        """Load a trained model from file.
        
        Args:
            model_path: Path to the saved model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.algorithm = model_data['algorithm']
        self.model = model_data['model']
        self.recommender_class = self.SUPPORTED_ALGORITHMS[self.algorithm]
        
        print(f"Model loaded from: {model_path}")
        
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for each algorithm."""
        defaults = {
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
        
        return defaults.get(self.algorithm, {})
        
    def get_recommendations(
        self,
        user_ids: np.ndarray,
        n_recommendations: int = 10,
        remove_seen: bool = True
    ) -> Dict[int, List[int]]:
        """Get recommendations for specified users.
        
        Args:
            user_ids: Array of user indices
            n_recommendations: Number of items to recommend per user
            remove_seen: Whether to remove items the user has already seen
            
        Returns:
            Dictionary mapping user_id to list of recommended item_ids
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting recommendations")
            
        recommendations = {}
        
        for user_id in user_ids:
            if remove_seen:
                scores = self.model.get_score_remove_seen([user_id])
                user_scores = scores[0]
            else:
                user_scores = self.model.get_score([user_id])[0]
                
            # Get top items
            top_items = np.argsort(user_scores)[::-1][:n_recommendations]
            recommendations[int(user_id)] = top_items.tolist()
            
        return recommendations