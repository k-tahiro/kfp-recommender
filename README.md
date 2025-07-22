# kfp-recommender

KFP Components for Recommender Systems using irspack

## Overview

This package provides Kubeflow Pipelines (KFP) components for building recommendation systems using the [irspack](https://github.com/tohtsky/irspack) library. Currently includes a model training component with support for multiple recommendation algorithms.

## Features

- **irspack Model Trainer**: Train recommendation models using various algorithms
- **Supported Algorithms**: iALS, BPR, P3alpha, ItemKNN, UserKNN, Random
- **Flexible Input**: CSV and Parquet data formats
- **Evaluation Metrics**: NDCG, Recall, Precision at multiple cutoffs
- **Easy Integration**: Standard KFP component interface

## Installation

```bash
pip install kfp-recommender
```

## Quick Start

### Basic Usage

```python
from kfp import dsl
from kfp_recommender import irspack_model_trainer

@dsl.pipeline(name="recommendation-training")
def recommendation_pipeline(data_path: str):
    training_task = irspack_model_trainer(
        data_path=data_path,
        algorithm="ials",
        hyperparameters='{"n_components": 64, "reg": 0.001}',
        test_split_ratio=0.2
    )
    
    return training_task.outputs

# Compile pipeline
from kfp import compiler
compiler.Compiler().compile(recommendation_pipeline, "pipeline.yaml")
```

### Data Format

Your input data should be a CSV or Parquet file with user-item interactions:

```csv
user_id,item_id,rating
1,101,4.5
1,102,3.0
2,101,5.0
2,103,2.5
```

### Supported Algorithms

- **ials**: Implicit Alternating Least Squares
- **bpr**: Bayesian Personalized Ranking
- **p3alpha**: P3-alpha neighborhood method
- **itemknn**: Item-based k-Nearest Neighbors
- **userknn**: User-based k-Nearest Neighbors  
- **random**: Random baseline

### Hyperparameters

Pass algorithm-specific hyperparameters as JSON string:

```python
# iALS example
hyperparameters = '{"n_components": 128, "reg": 0.005, "n_epochs": 30}'

# BPR example
hyperparameters = '{"n_components": 64, "learning_rate": 0.01, "n_epochs": 100}'
```

## Component Reference

### irspack_model_trainer

Trains recommendation models using irspack algorithms.

**Parameters:**
- `data_path`: Input dataset with user-item interactions
- `algorithm`: Algorithm to use (default: "ials")
- `hyperparameters`: JSON string with model parameters (default: "{}")
- `test_split_ratio`: Fraction for testing (default: 0.2)
- `evaluation_metrics`: Metrics to compute (default: ["ndcg", "recall", "precision"])
- `user_col`: User ID column name (default: "user_id")
- `item_col`: Item ID column name (default: "item_id")
- `rating_col`: Rating column name (default: "rating")

**Outputs:**
- `model_path`: Trained model file
- `metrics_path`: Evaluation metrics JSON

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Run example
python src/kfp_recommender/examples/training_pipeline.py
```

## License

This project is licensed under the MIT License.
