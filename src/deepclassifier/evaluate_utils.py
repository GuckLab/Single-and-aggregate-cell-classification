import os
import shutil
import tempfile
from pathlib import Path

from dcml.data.utils import resolve_filepaths

import mlflow
from torch.utils.data import DataLoader
import time
from dcml.preprocessing.io import apply_prediction_to_dir
from dcml.evaluation.helpers import map_abbr_2_ml_score_label, map_weights_abbr_2_labels
from dcml.training.metrics import EvaluationMetrics
from dcml.data import RTDCDataset
import pandas as pd
import torch
from torch import Tensor
from .evaluate_measurements import evaluate_proportions_on_gmm, evaluate_confusion_on_gmm
import yaml


from dcml.evaluation.artifacts import (
    artifact_classification_report,
    artifact_confusion_matrix,
)

from dcml.models import OneModel
from dcml.predict import predict
from dcml.utils.data import create_single_dataset, create_datasets
from .get_device import get_available_device_num_workers
from typing import List
from dcml.utils.params import get_saved_params



def evaluate_models(run_id: str, path_in: str, batch_size: int = 16, model_names: str | List[str] = "best_model_f1"):
    """
    Creates a test dataset and runs on it the evaluation of a model or a list of models from Mlflow repository defined by
    run_id string. Writes evaluation results into mlflow under the given run-id.
    Predictions are not saved.

    :param run_id: string with the mlflow run_id number
    :param model_names: model name (str) or model names (list of strings) from mlflow repository to be evaluated
    :param path_in: input path that was used for training (allows access to the test data defined in yaml file)
    :param batch_size:
    :return:
    """

    MTL_mode = get_saved_params(run_id)["create_trainer"]["architecture"]["type"] == 'multitask'
    dataset_params = get_saved_params(run_id)["create_dataset"]

    required_data = dataset_params["required_data"].copy()
    required_data["target"] = "cell_class_labels"

    path_in = Path(path_in)
    abs_paths = [path_in / Path(yaml_path) for yaml_path in dataset_params["hdf5_paths"]["test"]]
    _, dataset = create_datasets(
        hdf5_paths=abs_paths,
        required_data=required_data,
        target_grouping=dataset_params["target_grouping"],
        crop_size=dataset_params["crop_size"],
        correct_background=dataset_params["correct_background"],
        train_size=0,
        mean=dataset_params.get("mean", None),
        std=dataset_params.get("std", None),
        ml_score_to_int=dataset_params.get('ml_score_to_int', None),
        MTL=MTL_mode,
        unknowns_target_grouping=dataset_params.get('unknowns_target_grouping_test', None)
    )

    if len(dataset) == 0:
        print("created dataset is empty")
        raise
    else:
        print(f"the number of examples in test dataset is {len(dataset)}", flush=True)

    with mlflow.start_run(run_id=run_id, experiment_id=None, nested=False):

        # evaluation on test dataset
        if isinstance(model_names, List):
            for model_name in model_names:
                model_eval(run_id, dataset, model_name, batch_size=batch_size)
        elif isinstance(model_names, str):
            model_eval(run_id, dataset, model_names, batch_size=batch_size)
        else:
            raise Exception(" model names argument should either be string or list of strings")

    # # remove the dataset on the disk that was created with create_datasets(),
    # # which in turn called concatenated_hdf5_data()
    # try:
    #     temp_file_name = dataset.h5data.h5.file.filename
    # except:
    #     print("evaluate: cannot delete temporary file")
    # else:
    #     os.remove(temp_file_name)
    #     dataset.h5data.close()



def evaluate_models_on_gmm(run_id: str, path_in_gmm: str, path_out_gmm_pred: str,
                           model_names: str | List[str] = "best_model_f1", remove_predictions=False,
                           mlflow_folder_name="evaluation_on_gmm_measurements"):
    """

    """

    measurements_params = get_saved_params(run_id)

    if "gmm_based_evaluation" in measurements_params and path_in_gmm and path_out_gmm_pred:
        measurements = measurements_params["gmm_based_evaluation"]["measurements"]

        #cell_type_naming = measurements_params["gmm_based_evaluation"]["measurement_labels"]
        cell_type_naming = measurements_params["gmm_based_evaluation"].get("measurement_labels", None)
        features = measurements_params["create_trainer"]["ml_score_features"].values()
        ml_score_features = ["ml_score_" + feature for feature in features]

        proportions = measurements_params["gmm_based_evaluation"]["proportions"]
        not_wbc_props = proportions['notWBC']
        wbc_props = proportions['WBC']

        single_cell_mode = measurements_params["gmm_based_evaluation"].get("single_cell", True)

        if "full_measurement_label" in measurements_params["gmm_based_evaluation"]:
            full_measurement_label = measurements_params["gmm_based_evaluation"]["full_measurement_label"]
        else:
            full_measurement_label = ""

    else:
        print("no measurements were given to evaluate models on GMM data")
        return

    with mlflow.start_run(run_id=run_id, experiment_id=None, nested=False):

        run_name = mlflow.active_run().info.run_name
        path_out_gmm_pred = os.path.join(path_out_gmm_pred, run_name)

        # evaluation on test dataset
        if isinstance(model_names, List):
            for model_name in model_names:

                model_eval_on_gmm(path_in_gmm, path_out_gmm_pred, run_id, measurements, ml_score_features,
                                  cell_type_naming, not_wbc_props, wbc_props, full_measurement_label, model_name,
                                  cleanup_predictions_after_eval=remove_predictions, single_cell=single_cell_mode,
                                  mlflow_folder_name=mlflow_folder_name)

        elif isinstance(model_names, str):

            model_eval_on_gmm(path_in_gmm, path_out_gmm_pred, run_id, measurements, ml_score_features,
                              cell_type_naming, not_wbc_props, wbc_props, full_measurement_label, model_names,
                              cleanup_predictions_after_eval=remove_predictions, single_cell=single_cell_mode,
                              mlflow_folder_name=mlflow_folder_name)

        else:
            raise Exception(" model names argument should either be string or list of strings")


def model_eval(run_id: str, dataset: RTDCDataset, model_name: str = "", batch_size: int = 16):
    """
    On a given dataset, runs evaluation using the model saved in Mlflow under run_id string. Writes evaluation results
    into mlflow under the given run-id. Predictions are not saved.

    :param run_id: string with the mlflow run_id number or path to stand-alone *.pth model
    :param model_name: name under which model was saved in mlflow repo defined by run_id
    :param path_in: input path that was used for training (allows access to the test data defined in yaml file)
    :param batch_size:
    :return:
    """

    device, num_workers = get_available_device_num_workers()
    #
    # # Get model
    one_model = OneModel(model_path=run_id, device=device, model_name=model_name)
    trainer_params = one_model.params["create_trainer"]
    conds = one_model.params.get("comparative_evaluation", None)

    if conds:
        conds = conds.get("detection_matrix", None)

    bio_relevance_weights = trainer_params.get("bio_relevance_weights", None)
    f1_score_weights = trainer_params.get("f_beta_score_weights", None)

    class_label_dict = trainer_params["class_label_dict"]

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False, # Ok only if the whole test dataset is used
                            num_workers=num_workers)

    targets, prediction_scores, predictions, loss_values = predict(
        model=one_model.model,
        dataloader=dataloader,
        criterion=None,
        device=device,
        class_label_dict=class_label_dict
        #free_dependent=one_model.params["create_dataset"]["free_dependent"]
    )

    mapping = map_abbr_2_ml_score_label(class_label_dict, trainer_params["ml_score_features"])
    bio_relevance_weights_mapped = map_weights_abbr_2_labels(bio_relevance_weights, mapping)
    f1_score_weights_mapped = map_weights_abbr_2_labels(f1_score_weights, mapping)

    metrics = EvaluationMetrics(target_names=class_label_dict, bio_weights=bio_relevance_weights_mapped,
                                beta_weights=f1_score_weights_mapped,
                                used_metrics=trainer_params["performance_metrics"])

    metrics(targets, predictions, prediction_scores)
    # metrics.print_metrics()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print("\n")
    print(pd.DataFrame(metrics.report).transpose())

    mlflow_log(metrics, model_name, eval_name='eval_test')


    # in case of multi-task compare the results with multi-class by making evaluation on the same classes,
    # which are derived for multi-task
    if trainer_params['architecture']['type'] == 'multitask' and conds:

        labels = {n: key for n, key in enumerate(conds.keys())}
        n_observations = targets.shape[0]
        targets_compare = torch.zeros(n_observations, len(labels))
        predictions_compare = torch.zeros(n_observations, len(labels))
        for obs in range(n_observations):
            targets_compare[obs, :] = torch.tensor([eval_cond(conds[label], targets[obs, :]) for label in labels.values()])
            predictions_compare[obs, :] = torch.tensor([eval_cond(conds[label], predictions[obs, :]) for label in labels.values()])

        # targets_compare = targets_compare.float()
        # predictions_compare = predictions_compare.float()

        #targets_compare, predictions_compare, labels = mapping_for_comparison(targets, predictions)

        metrics_to_use = trainer_params["performance_metrics"]
        metrics_to_remove = ["roc_auc_score", "average_precision_score"] # cannot be used as we do not provide scores, but only predicitons
        metrics_to_use = [s for s in metrics_to_use if s not in metrics_to_remove]
        metrics_to_use = [s for s in metrics_to_use if not s.startswith("weighted_") ] # cannot be used as we do not have the correspondng weghts for the composed categories

        metrics_compare = EvaluationMetrics(target_names=labels, used_metrics=metrics_to_use, cls_type="MTL")

        metrics_compare(targets_compare, predictions_compare)
        # metrics.print_metrics()
        print("\n")
        print(pd.DataFrame(metrics_compare.report).transpose())

        mlflow_log(metrics_compare, model_name, eval_name='eval_test_compare')


def model_eval_on_gmm(path_in, path_out, run_id: str, measurements, ml_score_features, cell_type_naming=None, not_wbc_cell_types=None,
                      wbc_cell_types=None, full_measurement_label="", model_name: str = "",
                      with_basin=False, cleanup_predictions=False, cleanup_predictions_after_eval=False, single_cell=True,
                      mlflow_folder_name="evaluation_on_gmm_measurements"):

    path_out = add_suffix_to_path(path_out, model_name)

    device, num_workers = get_available_device_num_workers()

    # if path_out already exists and is not empty add timestamp to not overwrite predictions in the existing folder
    if os.path.isdir(path_out) and os.listdir(path_out):
        print(f"folder {path_out} for predictions already exists")
        unique_id = time.strftime("%d%m%y_%H%M%S")
        path_out = add_suffix_to_path(path_out, unique_id)
        print(f"creating a different folder {path_out}")

    measurements_params = get_saved_params(run_id)
    train_dataset_mean = measurements_params['create_dataset']['mean']
    train_dataset_std = measurements_params['create_dataset']['std']
    train_reference_p99 = measurements_params["create_dataset"].get("p99", None)
    #prefix = measurements_params["gmm_based_evaluation"].get("full_measurement_label", "")

    # run prediction for each measurement separately that begins with "full_measurement_label"
    for measurement in measurements:
        path_in_measurement = os.path.join(path_in, measurement)
        path_out_measurement = os.path.join(path_out, measurement)

        # Calculate statistics for each measurement (using file with a prefix that corresponds to full measurement)
        abs_path_files = [os.path.join(path_in_measurement, el) for el in Path(path_in_measurement).rglob(full_measurement_label + "*.rtdc") if el.is_file()]

        # abs_path_files_temp = resolve_filepaths([path_in_measurement]) could also be used when adding possibility to add prefix

        #print(abs_path_files)
        if abs_path_files:
            print("measurement file: {}".format(abs_path_files[0]))
            if len(abs_path_files) > 1:
                print("a few files with full measurement were found: {}. Using the first one".format(abs_path_files))

            # compute dataset statistics
            dataset = create_single_dataset(
                hdf5_path=abs_path_files[0],
                required_data={"image": "image"},
                crop_size=measurements_params["create_dataset"]["crop_size"],
                correct_background=False,
                mean=None, # forces computation of mean
                std=None, # forces computation of std
                p99_compute=True, # forces computation of p99
                MTL=measurements_params["create_trainer"]["architecture"]["type"] == 'multitask'
            )

            # brightness_factor = train_reference_p99 / dataset.p99 if train_reference_p99 else 1.0
            # print("brightness factor to be applied: {}".format(brightness_factor))
            apply_prediction_to_dir(path_in=path_in_measurement,
                                    path_out=path_out_measurement,
                                    model_path=run_id,
                                    device=device,
                                    num_workers=num_workers,
                                    with_basin=with_basin,
                                    cleanup_predictions=cleanup_predictions,
                                    model_name=model_name,
                                    mean=dataset.mean, #train_dataset_mean, #dataset.mean * brightness_factor, #dataset.mean, train_dataset_mean, ## #, #, #,
                                    std=train_dataset_std, # dataset.std, # #dataset.std * brightness_factor,
                                    brightness_factor=None #brightness_factor #None #,
                                    )

    unique_id = time.strftime("%d%m%y_%H%M%S")
    artifact_path = mlflow_folder_name + '_' + model_name + "_" + unique_id

    # run evaluation of predictions - cell type proportion calculation
    evaluate_proportions_on_gmm(measurements, ml_score_features, path_out, artifact_path=artifact_path,
                    full_measurement=full_measurement_label, not_wbc_cell_types=not_wbc_cell_types, wbc_cell_types=wbc_cell_types, single_cell_mode=single_cell)

    # run evaluation of predictions - confusion matrix, if there is labeled data (measurement_labels in config file)
    if cell_type_naming:
        evaluate_confusion_on_gmm(measurements, ml_score_features, cell_type_naming, path_out, artifact_path=artifact_path, single_cell_mode=single_cell)

    # remove folder with predictions if requested (since it may take a lot of space)
    if cleanup_predictions_after_eval:
        print(f"folder with predictions {path_out} is being deleted")
        shutil.rmtree(path_out)

    return


def add_suffix_to_path(path, suffix):

    if path.endswith("/"):
        path = path.rstrip("/")
        path = path + "_" + suffix + "/"
    else:
        path = path + "_" + suffix

    return path

def mlflow_log(metrics: EvaluationMetrics, model_name: str, eval_name='eval_test'):

    log_dir_tmp = tempfile.TemporaryDirectory()
    log_dir = Path(log_dir_tmp.name)

    artifact_path_dict = {}

    # Add confusion matrix artifact
    artifact_path_dict.update(artifact_confusion_matrix(
        confusion_mat=metrics.confusion_matrix,
        labels=metrics.confusion_matrix_labels,
        artifacts_dir=log_dir))

    # Add classification report artifact
    artifact_path_dict.update(artifact_classification_report(artifacts_dir=log_dir, report=metrics.report))

    """
    Log Artifacts to MLflow Tracking Server.
    Change artifact_path when needed.
    """
    unique_id = time.strftime("%d%m%y_%H%M%S")
    for art_path, local_path in artifact_path_dict.items():
        mlflow.log_artifact(local_path=local_path, artifact_path=f"{eval_name}_{model_name}_" + unique_id)

    log_dir_tmp.cleanup()

    print("{} results were saved under {} suffix".format(eval_name, unique_id))

def eval_cond(cond:dict | int, arr:Tensor) -> Tensor:
    """

    Parameters
    ----------
    cond: dictionary with logical conditions created from yaml comparative evaluation section, e.g. {'or': [4, 5, 6, 7, 8]}
    arr: 1D Tensor - different cell classes for single observation

    Returns
    -------
    Tensor of bool or of nan value
    """
    if isinstance(cond, dict):
        if 'and' in cond:
            # "and" list: all must be true
            output = torch.stack([eval_cond(c, arr) for c in cond['and']]) # tensor of booleans or 1/0 floats and nan
            out_nans = torch.isnan(output)
            if torch.any(out_nans):
            #if torch.all(out_nans):
                return torch.tensor(torch.nan) # return nan if a nan within "and" condition
            else:
                ou = output.all()
                #ou = (output[~out_nans]).all()  # ignore nans
                # ou1 = (output == 1).all() # can also be output.all()
                # assert ou1 == ou, "not correct1"
                return ou

        elif 'or' in cond:
            # "or" list: any must be true
            output = torch.stack([eval_cond(c, arr) for c in cond['or']]) # # tensor of booleans or 1/0 floats and nan
            out_nans = torch.isnan(output)
            if torch.all(out_nans):
                torch.tensor(torch.nan) # return nan if all were nan within "or" condition
            else:
                ou = (output[~out_nans]).any()
                # ou1 = (output == 1).any() # allows to ignore nan
                # assert ou1 == ou, "not correct2"
                return ou
        elif 'not' in cond:
            output = eval_cond(cond['not'], arr) # tensor of boolean or nan
            if torch.isnan(output):
                return torch.tensor(output)
            else:
                return ~output # returns False if output is nan
    elif isinstance(cond, int):
        #return arr[cond].bool()
        if arr[cond] == 1:
            return torch.tensor(True)
        elif arr[cond] == 0:
            return torch.tensor(False)
        else:
            return torch.tensor(torch.nan)
    else:
        raise ValueError('Invalid condition')


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path_in', type=str, help='Input folder path')
#     parser.add_argument('--path_out', type=str, default="", help='Output folder path for predictions on test set')
#     parser.add_argument('--path_in_gmm', type=str, default="", help='Input folder path for GMM test data')
#     parser.add_argument('--path_out_gmm_pred', type=str, default="", help='Output folder path for predictions on GMM set')
#     parser.add_argument('--mlflow_gmm_folder_name', type=str, default="evaluation_on_gmm_measurements",
#                         help='folder name within mlflow where gmm evaluation results will be stored')
#     parser.add_argument('--model', type=str, help='model run_id')
#     parser.add_argument('--batch_size', action='store_true', default=16)
#     parser.add_argument('--rm_pred', action='store_true', default=False, help='removes folder with predictions after evaluation')
#
#     args = parser.parse_args()
#
#
#
#     # evaluate_model(args.model, args.path_in, path_out=args.path_out, model_name = "best_model_f1",
#     #                batch_size=args.batch_size)
#     # evaluate_model(args.model, args.path_in, model_name="best_model_f1", batch_size=args.batch_size)
#
#     # evaluate_models(args.model, args.path_in, model_names=["best_model_f1",
#     #                                                        "best_model_accuracy"], batch_size=args.batch_size)
#
#     #evaluate_models(args.model, args.path_in, model_names=["best_model_f1"], batch_size=args.batch_size)
#
#     evaluate_models(args.model, args.path_in, model_names=["best_model_bal_acc"], batch_size=args.batch_size)
#
#
#     if args.path_in_gmm:
#
#         # model evaluation on GMM data
#         print("starting evaluation on GMM test dataset")
#
#         evaluate_models_on_gmm(args.model, args.path_in_gmm, path_out_gmm_pred=args.path_out_gmm_pred,
#                                model_names=["best_model_bal_acc"], remove_predictions=args.rm_pred,
#                                mlflow_folder_name=args.mlflow_gmm_folder_name)



