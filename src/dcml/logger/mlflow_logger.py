import os
import pathlib
import tempfile
import warnings
from typing import Iterable, List

import numpy as np
import torch
import mlflow
from mlflow.tracking.client import MlflowClient
import yaml

from .base_logger import BaseLogger
from ._utils import flatten_dict
from ..evaluation.artifacts import (
    artifact_confusion_matrix,
    artifact_classification_report,
    artifact_sample_images_with_targets,
)
# from ..evaluation.helpers import retrieve_bio_relevance_weights, retrieve_f1_score_weights
# from ..evaluation.scalars import compute_certainty_scores
from ..utils.models import combine_to_jit  # save_jit_model
from ..training.metrics import EvaluationMetrics

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

ENV_VARS = ["MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
            "MLFLOW_TRACKING_URI"]


class MLflowLogger(BaseLogger):
    def __init__(self, class_label_dict, log_dir=None, **kwargs):

        if log_dir is None:
            self._log_dir_tmp = tempfile.TemporaryDirectory()
            self._log_dir = pathlib.Path(self._log_dir_tmp.name)
        else:
            self._log_dir_tmp = None
            self._log_dir = pathlib.Path(log_dir)

        self.class_label_dict = class_label_dict
        # self.ml_score_features = ml_score_features

        # Check if all necessary environment variables are set:
        for env_var in ENV_VARS:
            var_name = env_var.split("_")[-1].upper()
            if os.environ.get(env_var) is None:
                warnings.warn(f"MLflow-{var_name} is not set via environment "
                              f"variables. Please Set '{env_var}' "
                              "accordingly!")

        # Set experiment-name and run-name
        experiment_name = kwargs.get("experiment_name")
        experiment_description = kwargs.get("experiment_description")
        if experiment_name is None:
            raise ValueError("Experiment name was not defined in params-file!"
                             "Please make sure to set it at "
                             "'params['create_trainer']['logger']['experiment_name']'.")  # noqa: E501

        mlflow.set_experiment(experiment_name)
        #curr_exp = mlflow.set_experiment(experiment_name)

        # if experiment description was given in config file and current experiment does not have it (new experiment)
        # if experiment_description is not None and not curr_exp.tags:
        #     mlflow.set_experiment_tag("mlflow.note.content", experiment_description)

        # Set run-name
        # Check that the defined run-name is not already used
        existing_runs = mlflow.search_runs(output_format="list")
        existing_names = [run.info.run_name for run in existing_runs]

        run_name = kwargs.get("run_name")
        if run_name in existing_names:
            warnings.warn("Run-name already exists.")

        if run_name is None:
            mlflow.start_run(run_name=run_name, description=experiment_description)
            rid = mlflow.active_run().info.run_id
            custom_run_name = f"{"id"}-{rid[:7]}-{rid[-7:]}"
            mlflow.set_tag("mlflow.runName", custom_run_name)

        else:
            mlflow.start_run(run_name=run_name)

        self.active_run = mlflow.active_run()
        self.run_id = self.active_run.info.run_uuid

        if run_name is None:
            run_name = self.active_run.info.run_name
            warnings.warn("No unique run-name defined. "
                          f"Assigning random run_name: {run_name}")

        # self.bio_relevance_weights = retrieve_bio_relevance_weights(
        #     [*class_label_dict.values()], ml_score_features, all_bio_relevance_weights=bio_relevance_weights)
        # self.bio_relevance_weights = retrieve_bio_relevance_weights(class_label_dict, ml_score_features,
        #                                                             all_bio_relevance_weights=bio_relevance_weights)

        # self.f1_score_weights = retrieve_f1_score_weights(
        #     [*class_label_dict.values()], ml_score_features, all_f1_score_weights=f1_score_weights)
        # self.f1_score_weights = retrieve_f1_score_weights(class_label_dict, ml_score_features,
        #                                                   all_f1_score_weights=f1_score_weights)

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
    #     ###
    #     # Logging Certainty Statistics (Median)
    #     ###
    #
    #     certainty_scores: Iterable = compute_certainty_scores(
    #                                                         prediction_scores,
    #                                                         np.median)
    #     for idx, class_label in self.class_label_dict.items():
    #         mlflow.log_metric(key=f"{stage.upper()}-{class_label} "
    #                               "Median Score of positive predictions",
    #                           value=certainty_scores[idx],
    #                           step=epoch)
    #
    # # def log_artifacts_without_target(self,
    # #                                  predictions: Iterable,
    # #                                  prediction_scores: Iterable,
    # #                                  stage: str,
    # #                                  epoch: int):
    # #     """
    # #     Log artifacts without target data.
    # #
    # #     Parameters
    # #     ----------
    # #     predictions : Iterable
    # #         The predictions made by the model.
    # #     prediction_scores : Iterable
    # #         Scores associated with each prediction.
    # #     stage: str
    # #         Either `train` or `val`
    # #     epoch: int
    # #         Current epoch of the training process
    # #     """
    # #     pass

    def log_scalar(self, loss_values: Iterable,
                   metrics: EvaluationMetrics,
                   stage: str,
                   epoch: int,
                   individual_classes: List[str] = None):

        """
        Log scalar values with target data.

        Parameters
        ----------
        loss_values: Iterable
            Loss values for each prediction.
        metrics: instance of a class that defines which metrics are used
        individual_classes: defines for which individual classes (in addition to summaries) logging of metrics is
        requested.

        stage: str
            Either `train` or `val`
        epoch: int
            Current epoch of the training process
        """

        ###
        # Logging Loss
        ###
        loss_mean = np.mean(np.array(loss_values))
        mlflow.log_metric(key=f"{stage.upper()} - Loss",
                          value=loss_mean,
                          step=epoch, run_id=self.run_id)

        if stage.upper() != "TRAIN":
            if individual_classes:

                for metric_name in metrics.requested_columns:
                    current_metric_values = getattr(metrics, metric_name + "_per_class", None)
                    if current_metric_values is None:
                        continue

                    for idx, class_label in self.class_label_dict.items():
                        if class_label in individual_classes:
                            mlflow.log_metric(key=f"{stage.upper()} - {class_label} - {metric_name}",
                                              value=current_metric_values[idx],
                                              step=epoch, run_id=self.run_id)

            for metric_name in metrics.requested_metric_names:
                current_metric_value = getattr(metrics, metric_name)
                mlflow.log_metric(key=f"{stage.upper()} - {metric_name}",
                                  value=current_metric_value,
                                  step=epoch, run_id=self.run_id)

    def log_artifacts(self, metrics: EvaluationMetrics, epoch: int):
        """
        Log artifacts with target data.

        Parameters
        ----------
        confusion_mat: confusion matrix
        report: report with computed metrics
        epoch: int
            Current epoch of the training process
        """
        artifact_path_dict = {}

        ###
        # Add Confusion Matrix Artifact
        ###
        artifact_path_dict.update(artifact_confusion_matrix(
            confusion_mat=metrics.confusion_matrix,
            labels=metrics.confusion_matrix_labels,
            artifacts_dir=self._log_dir,
            epoch=epoch))

        ###
        # Add Classification Report Artifact
        ###
        artifact_path_dict.update(artifact_classification_report(artifacts_dir=self._log_dir, report=metrics.report,
                                                                 epoch=epoch))

        ###
        # Log Artifacts to MLflow Tracking Server
        ###
        for art_path, local_path in artifact_path_dict.items():
            mlflow.log_artifact(local_path=local_path, artifact_path=art_path, run_id=self.run_id)

    def log_images(self,
                   torch_dataset: torch.utils.data.Dataset,
                   predictions: Iterable,
                   prediction_scores: Iterable,
                   targets: Iterable):
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
        images_path_dict = {}
        targets = targets.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        pred_probs = prediction_scores.cpu().detach().numpy()
        images_path_dict.update(
            artifact_sample_images_with_targets(torch_dataset, targets,
                                                predictions, pred_probs,
                                                self.class_label_dict,
                                                10, self._log_dir,
                                                ))
        ###
        # Logs artifacts to MLflow tracking Server
        ###

        for img_path, local_path in images_path_dict.items():
            mlflow.log_artifact(local_path=local_path,
                                artifact_path=img_path, run_id=self.run_id)

    def save_model(self,
                   model: torch.nn.Module,
                   model_name: str,
                   transform: torch.nn.Module,
                   params: dict,
                   epoch: int):
        """
        To do: Find another solution for saving jit model
        This currently saves the model twice.
        Logs model to mlflow and saves it as jit-file.
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

        Note:
        -----
        Using `mlflow.pytorch.log_model()` does not allow to store
        `_extra_files` like for `torch.jit.save` because it uses `torch.save`
        instead.
        Therefore MLflowLogger also stores the training parameters additionally
        in corresponding MLflow-run (see issue 106)
        """
        # Log model
        combined_jit_model = combine_to_jit(model=model, transform=transform)
        mlflow.pytorch.log_model(pytorch_model=combined_jit_model, artifact_path=model_name)

        mlflow_model_path = str(self.active_run.info.artifact_uri)[7:] + "/" + model_name + "/data/model.pth"
        print("Epoch[", epoch, "]: Model saved in mlflow under ", mlflow_model_path)

        # # Strip off a prefix like file://, s3://, or similar.
        # model_path = (
        #         str(self.active_run.info.artifact_uri)[7:]
        #         + "/" + model_name + "/model"
        # )
        #
        # print("Epoch[", epoch, "]: Saving model under ", model_path)
        # # save jit model to enable prediction with local model_path
        # save_jit_model(model=model,
        #                transform=transform,
        #                path=model_path,
        #                params_dict=params,
        #                epoch=epoch)

    def log_params(self, params: dict, filename: str = "configuration.yaml", params_section: bool = False) -> None:
        """Logs the parameter dict as artifact

        Note
        ----
        To use integers as keys in the params-dict, we need to use the
        `yaml`-format. The file then be stored correctly in the artifacts
        of the runs, but we are not able to load the dictionary with
        `mlflow.load_artifact()`, because the integers keys cannot be
        decoded with JSON.
        The workaround is to download the params-file as artifact first and
        then specifically load it with `pyyaml`. See issue #110
        """

        # Flatten the nested dictionary structure
        if params_section:
            flattened_params = flatten_dict(params)
            # Log each parameter using mlflow.log_param()
            for key, value in flattened_params.items():
                mlflow.log_param(key, value)

        self.log_dict(params, filename)

    def log_dict(self, dictionary, artifact_file: str) -> None:
        """
        Replaces mlflow.log_dict in order to insure the order of the dictionary keys saved in yaml
        (requires sort_keys=False in safe_dump).

        Parameters
        ----------
        dictionary: dictionary to be logged
        artifact_file: name of the file to be logged to

        """

        with MlflowClient()._log_artifact_helper(self.run_id, artifact_file) as tmp_path:
            with open(tmp_path, "w") as f:
                # Specify `indent` to prettify the output
                yaml.safe_dump(dictionary, f, indent=2, default_flow_style=False, sort_keys=False)

    def log_metrics(self, params: dict):
        """
        Logs metrics using MLflow.

        Flattens the nested dictionary structure of parameters.
        Logs each parameter using `mlflow.log_metric()`.

        Args:
            params (dict): A dictionary containing parameters to be logged.

        Returns:
            None
        """
        # Flatten the nested dictionary structure
        flattened_params = flatten_dict(params)
        # Log each parameter using mlflow.log_param()
        for key, value in flattened_params.items():
            mlflow.log_metric(key=key, value=value, run_id=self.run_id)

    def close(self):
        """Make sure the everthing is closed properly

        For mlflow experiments it is especially important to end all active runs
        of a process. Otherwise this will lead to possible errors down the
        line.
        Typically this can be resolved by using the context manager of mlflow:
        `with mlflow.start_run() as run:`, but in our case this is not
        possible.
        Please make sure to run `logger.close()` after the logger is not needed
        anyore.
        """
        print("closing mlflow logger with close")
        if self._log_dir_tmp is not None:
            self._log_dir_tmp.cleanup()

        mlflow.end_run()

    def __del__(self):
        self.close()

        # if self._log_dir_tmp is not None:
        #     self._log_dir_tmp.cleanup()
        #
        # mlflow.end_run()
