"""
This module implements helper function for dealing with params-files used for
training.

author: MS
date: 01/02/2023
"""
import mlflow
import yaml
import os
import logging
logger = logging.getLogger(__name__)


class MLScoreFeaturesMissingInParamsError(ValueError):
    pass


def get_saved_params(run_id: str, filename="configuration.yaml"):
    """

    Parameters
    ----------
    run_id: the string with either run_id (mlflow) or full path to saved stand-alone *.pth model
    filename:

    Returns
    -------
    params:

    """

    if os.path.splitext(run_id)[1] == ".pth":

        folder_path = os.path.split(run_id)[0]
        local_path = os.path.join(folder_path, filename)
        if not os.path.isfile(local_path):
            logger.error(f"no configuration file {local_path} found")

    else:  # use mlflow run_id

        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=filename)

    # Load the YAML file
    with open(local_path, 'r') as file:
        params = yaml.safe_load(file)

    return params
