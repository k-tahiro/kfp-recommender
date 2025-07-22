"""Example pipeline for irspack model training."""

from kfp import dsl
from kfp.client import Client
from kfp_recommender.components import irspack_model_trainer


@dsl.pipeline(
    name="irspack-training-pipeline",
    description="Train recommendation models using irspack"
)
def irspack_training_pipeline(
    data_path: str,
    algorithm: str = "ials",
    hyperparameters: str = '{"n_components": 64, "reg": 0.001}',
    test_split_ratio: float = 0.2
):
    """Pipeline for training irspack recommendation models.
    
    Args:
        data_path: Path to training data (CSV/Parquet)
        algorithm: Algorithm to use (ials, bpr, p3alpha, itemknn, userknn, random)
        hyperparameters: JSON string with model hyperparameters
        test_split_ratio: Fraction of data for testing
    """
    
    training_task = irspack_model_trainer(
        data_path=data_path,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        test_split_ratio=test_split_ratio,
        evaluation_metrics=["ndcg", "recall", "precision"]
    )
    
    return {
        "model": training_task.outputs["model_path"],
        "metrics": training_task.outputs["metrics_path"]
    }


# Example usage
if __name__ == "__main__":
    # Example of how to compile and run the pipeline
    
    # Compile pipeline
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=irspack_training_pipeline,
        package_path="irspack_training_pipeline.yaml"
    )
    
    print("Pipeline compiled to: irspack_training_pipeline.yaml")
    
    # Example run configuration
    run_config = {
        "data_path": "/path/to/your/interaction_data.csv",
        "algorithm": "ials",
        "hyperparameters": '{"n_components": 128, "reg": 0.005, "n_epochs": 30}',
        "test_split_ratio": 0.2
    }
    
    print(f"Example run configuration: {run_config}")
    
    # To run on Kubeflow:
    # client = Client()
    # run = client.run_pipeline(
    #     experiment_id="your-experiment-id",
    #     job_name="irspack-training-run",
    #     pipeline_package_path="irspack_training_pipeline.yaml",
    #     params=run_config
    # )