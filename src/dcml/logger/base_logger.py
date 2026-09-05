import abc
from typing import Iterable, List
from ..training.metrics import EvaluationMetrics

import torch


class BaseLogger(abc.ABC):
    """
    Abstract base class for logging various types of data during ML training.

    This class defines methods for logging scalar values, artifacts, and images
    with or without target data.
    """

    # @abc.abstractmethod
    # def log_scalar_without_target(self,
    #                               predictions: Iterable,
    #                               prediction_scores: Iterable,
    #                               stage: str,
    #                               epoch: int,):
    #     """
    #     Log scalar values without target data.
    #
    #     Parameters
    #     ----------
    #     predictions : Iterable
    #         The predictions made by the model.
    #     prediction_scores : Iterable
    #         Scores associated with each prediction.
    #     stage: str
    #         Either `train` or `val`
    #     epoch: int
    #         Current epoch of the training process
    #     """
    #     pass

    # @abc.abstractmethod
    # def log_artifacts_without_target(self,
    #                                  predictions: Iterable,
    #                                  prediction_scores: Iterable,
    #                                  stage: str,
    #                                  epoch: int):
    #     """
    #     Log artifacts without target data.
    #
    #     Parameters
    #     ----------
    #     predictions : Iterable
    #         The predictions made by the model.
    #     prediction_scores : Iterable
    #         Scores associated with each prediction.
    #     stage: str
    #         Either `train` or `val`
    #     epoch: int
    #         Current epoch of the training process
    #     """
    #     pass

    @abc.abstractmethod
    def log_scalar(self, loss_values: Iterable,
                   metrics: EvaluationMetrics,
                   stage: str,
                   epoch: int,
                   individual_classes: List[str]):
        """
        Log scalar values with target data.

        Parameters
        ----------
        predictions : Iterable
            The predictions made by the model.
        prediction_scores : Iterable
            Scores associated with each prediction.
        targets : Iterable
            Ground truth values.
        loss_values : Iterable
            Loss values for each prediction.
        stage: str
            Either `train` or `val`
        epoch: int
            Current epoch of the training process
        """
        pass

    @abc.abstractmethod
    def log_artifacts(self, metrics: EvaluationMetrics, epoch: int):
        """
        Log artifacts with target data.

        Parameters
        ----------
        metrics:
        epoch: int
            Current epoch of the training process
        """
        pass

    @abc.abstractmethod
    def log_images(self,
                   torch_dataset: torch.utils.data.Dataset,
                   predictions: Iterable,
                   prediction_scores: Iterable,
                   targets: Iterable,
                   ):
        """
        Log images.

        Parameters
        ----------
        torch_dataset : torch.utils.data.Dataset
            The dataset containing the images to log.
        predictions : Iterable
            The predictions made by the model.
        prediction_scores : Iterable
            Scores associated with each prediction.
        targets : Iterable
            Ground truth values.
        """
        pass

    # def log_with_target(self,
    #                     predictions: Iterable,
    #                     prediction_scores: Iterable,
    #                     targets: Iterable,
    #                     loss_values: Iterable,
    #                     scores: Iterable,
    #                     stage: str,
    #                     epoch: int):
    #     """
    #     Combines logging of scalar and artifacts that require target data.
    #
    #     Parameters
    #     ----------
    #     predictions : Iterable
    #         The predictions made by the model.
    #     prediction_scores : Iterable
    #         Scores associated with each prediction.
    #     targets : Iterable
    #         Ground truth values.
    #     loss_values : Iterable
    #         Loss values for each prediction.
    #     stage: str
    #         Either `train` or `val`
    #     epoch: int
    #         Current epoch of the training process
    #     """
    #     print(f"Log scalar {time.ctime()}", flush=True)
    #     self.log_scalar(predictions=predictions,
    #                     prediction_scores=prediction_scores,
    #                     targets=targets,
    #                     loss_values=loss_values,
    #                     scores=scores,
    #                     stage=stage,
    #                     epoch=epoch)
    #     # if stage != 'train':
    #     #     print(f"Log artifacts {time.ctime()}", flush=True)
    #     #     self.log_artifacts(predictions=predictions,
    #     #                        prediction_scores=prediction_scores,
    #     #                        targets=targets,
    #     #                        loss_values=loss_values,
    #     #                        scores=scores,
    #     #                        stage=stage,
    #     #                        epoch=epoch)

    # def log_without_target(self,
    #                        predictions: Iterable,
    #                        prediction_scores: Iterable,
    #                        stage: str,
    #                        epoch: int):
    #     """
    #     Combines logging of scalar and artifacts that don't require target data
    #
    #     Parameters
    #     ----------
    #     predictions : Iterable
    #         The predictions made by the model.
    #     prediction_scores : Iterable
    #         Scores associated with each prediction.
    #     stage: str
    #         Either `train` or `val`
    #     epoch: int
    #         Current epoch of the training process
    #     """
    #     self.log_scalar_without_target(predictions=predictions,
    #                                    prediction_scores=prediction_scores,
    #                                    stage=stage,
    #                                    epoch=epoch)
    #     self.log_artifacts_without_target(predictions=predictions,
    #                                       prediction_scores=prediction_scores,
    #                                       stage=stage,
    #                                       epoch=epoch)

    @abc.abstractmethod
    def log_params(self, params: dict, filename: str) -> None:
        """Log parameter file

        Parameters
        ----------
        params: dict
            Parameter dictionary containing the training metadata
        filename: str
            Filename used for saving the model
        """
        pass

    @abc.abstractmethod
    def save_model(self,
                   model: torch.nn.Module,
                   model_name: str,
                   transform: torch.nn.Module,
                   params: dict,
                   epoch: int):
        """ Saves the model as jit-file

        Parameters:
        -----------
        model: torch.nn.Module
            Model-file that will be saved
        model_name: str
            Name under which the model will be saved
        transform: torch.nn.Module
                Transformations that need to be applied before the model
        params: dict
            Dictionary of yaml file containing the training parameters.
            This should be stored with the model in some way to make sure that
            we can trace back which parameters have been used for training.
        epoch: int
            Epoch of model
        """
        pass

    @abc.abstractmethod
    def close(self):
        """Make sure the everthing is closed properly"""
        pass

    # @abc.abstractmethod
    # def __del__(self):
    #     """Make sure the everthing is closed properly"""
    #     pass
