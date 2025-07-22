"""KFP Components for Recommender Systems."""

from .components import irspack_model_trainer

__version__ = "0.1.0"
__all__ = ["irspack_model_trainer"]


def hello() -> str:
    return "Hello from kfp-recommender!"
