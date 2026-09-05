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


# def load_ml_feats_from_params(params_dict: dict) -> List[str]:
#     """Parses the parameter dictionary and outputs the `ml_score_features`
#
#     Parameters
#     ----------
#     params_dict: dict
#         Dictionary which contains the params-file from the deep learning
#         repository (such as `deepclassifier/params/train_params.conf`)
#     Returns
#     -------
#     ml_feats: List[str]
#         List of strings containing the `ml_score`-features
#         Example: ["ml_score_rst", "ml_score_r1f", ...]
#
#     Note:
#         This function assumes that the dictionary contains the following
#         fields:
#             - "create_trainer":
#                 - "ml_score_features":
#                     0: "rst"
#                     1: "r1f"
#                     2: "r1u"
#                     3: "r20"
#                     4: "rna"
#                     5: "l10"
#                     6: "g1n"
#                     7: "g1e"
#                     8: "g1b"
#                     9: "g1m"
#                     10: "t1a"
#     """
#     if "ml_score_features" not in params_dict["create_trainer"]:
#         raise MLScoreFeaturesMissingInParamsError(
#                 "The params-file is missing the `ml_score_features` entry."
#                 "It should be stored in the `create_trainer` "
#                 "configuration part!.")
#     ml_feats_dict = params_dict["create_trainer"]["ml_score_features"]
#
#     ml_score_abbrvs = [ml_feats_dict[key] for key
#                        in sorted(ml_feats_dict.keys(), key=lambda x: int(x))]
#     ml_score_features = [f"ml_score_{abbrv}" for abbrv in ml_score_abbrvs]
#
#     return ml_score_features


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
