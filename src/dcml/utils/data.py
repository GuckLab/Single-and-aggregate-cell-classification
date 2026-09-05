import pathlib
import time
from typing import List, Tuple

import dclab
from dcnum.read import concatenated_hdf5_data, HDF5Data
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler

from ..data import RTDCDataset
from ..data.utils import create_cell_class_array, map_targets, resolve_filepaths, reverse_dict
from ..io import load_ml_score_to_int


# from .models import list_original_module_names


def feature_intersection(features_by_dataset: List[List[str]]) -> List[str]:
    """Returns the intersection of a list of feature-lists"""
    if not features_by_dataset:
        return []

    feat_inter = set(features_by_dataset[0])

    for feature_list in features_by_dataset[1:]:
        feat_inter &= set(feature_list)

    return sorted(list(feat_inter))


def features_for_hdf5_concatenation(hdf5_paths: List[str | pathlib.Path],
                                    unwanted_features: List[str]) -> List[str]:
    """Prepares a list of features that can be used for hdf5-concatenation."""
    features_by_dataset = []
    for ds_path in hdf5_paths:
        with dclab.new_dataset(ds_path) as ds:
            features_by_dataset.append(ds.features_innate)

    features = feature_intersection(features_by_dataset)

    # Remove unwanted features
    features = [el for el in features if el not in unwanted_features]
    return features


def check_for_missing_ml_scores(features: List[str], all_ml_scores: dict = None) -> None:
    """Checks if the feature-list contains all up-to-date ml_score-features"""
    all_ml_scores = list(load_ml_score_to_int(all_ml_scores).keys())
    missing_mls = []
    for ml_score in all_ml_scores:
        if ml_score not in features:
            missing_mls.append(ml_score)
    if missing_mls:
        raise ValueError("Some training data does not contain all "
                         "`ml_score`-features! Make sure to fill all "
                         "missing `ml_scores` with "
                         "'dcml.utils.data.fill_missing_scores()."
                         f"Missing `ml_scores`: {missing_mls}")


def remove_nans(cell_class_labels_mapped, indexes_dataset):
    """
    Removes np.nan from cell_class_labels_mapped if 1D, or rows with all np.nan if 2D and the corresponding entries
    (same indexes) from indexes_dataset

    Parameters
    ----------
    cell_class_labels_mapped: either 1D array of values that can contain np.nan, or 2D array with n-hot rows.
    Row can also be a vector of all nans
    indexes_dataset: array of the same size as cell_class_labels_mapped or as number of rows if 2D array

    Returns
    -------
    cell_class_labels_mapped: cell_class_labels_mapped without np.nans if 1D or without rowas with all np.nan if 2D
    indexes_dataset: indexes_dataset without entries where cell_class_labels_mapped has np.nan in 1D case of nan rows
    in 2D case
    """

    # indices = np.where(np.isnan(cell_class_labels_mapped))
    assert len(indexes_dataset) == len(cell_class_labels_mapped), ("length of cell_class_labels_mapped and "
                                                                   "indexes_dataset should equal")
    if cell_class_labels_mapped.ndim == 1:
        indices = np.flatnonzero(np.isnan(cell_class_labels_mapped))
    elif cell_class_labels_mapped.ndim == 2:
        #indices = np.flatnonzero(np.isnan(cell_class_labels_mapped[:, 0]))
        indices = np.flatnonzero(np.all(np.isnan(cell_class_labels_mapped), axis=1))
    else:
        raise "cell_class_labels_mapped must be one or two dimensional array"

    if len(indices) > 0:
        print("warning: some data in rtdc files is not used because it was not mapped using target grouping "
              "in the configuration file")
    cell_class_labels_mapped = np.delete(cell_class_labels_mapped, indices, axis=0)
    indexes_dataset = np.delete(indexes_dataset, indices)

    return cell_class_labels_mapped, indexes_dataset


def create_data_sampler(sampler_type, weights, num_samples):
    """
        Creates and returns a data sampler
    """
    dataset_train_sampler = None
    if sampler_type.lower() == 'weightedrandomsampler':
        print(f"Prepare weightedRandomSampler: {num_samples} samples.", flush=True)
        dataset_train_sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
    return dataset_train_sampler


# def create_data_sampler(sampler_params, weights, len_dataset):
#     """
#         Creates and returns a data sampler based on
#         the specified sampler parameters.
#     """
#     dataset_train_sampler = None
#     sampler_type = sampler_params.get("type", '')
#     if sampler_type.lower() == 'weightedrandomsampler':
#         num_samples = sampler_params.get("num_samples", len_dataset)
#         print(f"Prepare weightedRandomSampler: "
#               f"{num_samples} samples.", flush=True)
#         dataset_train_sampler = WeightedRandomSampler(
#             weights=weights,
#             num_samples=num_samples,
#             replacement=True)
#     return dataset_train_sampler


def create_single_dataset(hdf5_path: str,
                          required_data: dict,
                          target_grouping: dict = None,
                          augmentation: dict = None,
                          crop_size: int = 80,
                          correct_background: bool = False,
                          mean: float | None = None,
                          std: float | None = None,
                          p99_compute: bool = False,
                          brightness_factor: float | None = None,
                          ml_score_to_int: dict = None,
                          MTL: bool = False,
                          unknowns_target_grouping: dict = None,
                          ) -> RTDCDataset:
    """Create a dataset from a single rtdc file; used for prediction """

    # if target_grouping is None:
    #     target_grouping = {}

    # target_mapping = {}
    # for k, v in target_grouping.items():
    #     target_mapping.update(dict.fromkeys(v, k))

    hdf5_ds = HDF5Data(hdf5_path)
    indexes_dataset = np.arange(0, len(hdf5_ds))

    # Map class_cell_labels if targets (image labels)
    # were requested to be included in the created dataset
    if "target" in required_data:
        target_mapping = reverse_dict(target_grouping)
        unknowns_target_mapping = reverse_dict(unknowns_target_grouping)

        cell_class_labels = create_cell_class_array(hdf5_ds, ml_score_to_int)
        cell_class_labels_mapped = map_targets(target_mapping, cell_class_labels, MTL=MTL,
                                               unknowns_target_mapping=unknowns_target_mapping)

        # remove nans in case of cell labels that are not mapped with target_mapping
        # Remove the corresponding entries in indexes_dataset
        _, indexes_dataset = remove_nans(cell_class_labels_mapped, indexes_dataset)
    else:
        # no target (image labels) was requested to include
        # in the dataset to be created
        cell_class_labels_mapped = None

    # Based on indices, create dataset
    dataset = RTDCDataset(hdf5_data=hdf5_ds,
                          required_data=required_data,
                          indexing_array=indexes_dataset,
                          targets=cell_class_labels_mapped,
                          augm_params=augmentation,
                          crop_size=crop_size,
                          correct_background=correct_background,
                          train=False,
                          mean=mean,
                          std=std,
                          p99_compute=p99_compute,
                          brightness_factor=brightness_factor)

    if len(dataset) == 0:
        print("Warning: the created dataset is empty")

    return dataset


def create_datasets(hdf5_paths: List[str],
                    required_data: dict,
                    target_grouping: dict = None,
                    augmentation: dict = None,
                    crop_size: int = 80,
                    correct_background: bool = False,
                    train_size: float = 0.8,
                    mean: float | None = None,
                    std: float | None = None,
                    p99_compute: bool = False,
                    brightness_factor: float | None = None,
                    ml_score_to_int: dict = None,
                    MTL: bool = False,
                    unknowns_target_grouping: dict = None,
                    ) -> Tuple[RTDCDataset | None, RTDCDataset | None]:
    """Creates training and validation datasets for training"""

    print(f"Resolve filepaths {time.ctime()}", flush=True)
    hdf5_paths = resolve_filepaths(hdf5_paths)



    # Get intersection of features of all datasets
    unwanted_features = ["contour", "trace"]
    print(f"Features for hdf5 concanetantion {time.ctime()}", flush=True)
    features = features_for_hdf5_concatenation(hdf5_paths=hdf5_paths,
                                               unwanted_features=unwanted_features)  # noqa: E501

    # Check if all ml_score-features are present in all datasets;
    # check only if targets (image labels) were requested
    # to be included in the created dataset
    if "target" in required_data:
        print(f"Check for missing ml scores {time.ctime()}", flush=True)
        # Check if all ml_score-features are present in all datasets.
        # If ml_score-features are missing, it raises a ValueError
        check_for_missing_ml_scores(features, ml_score_to_int)

    print(f"Concatenate hdf5 data {time.ctime()}", flush=True)
    # Create Virtual HDF5-dataset of all datasets
    hdf5_ds = concatenated_hdf5_data(hdf5_paths, features=features)
    #hdf5_ds = HDF5Data('/tmp/dcnum_vc_g2h2q2so.hdf5')
    hdf5_ds.h5.tempFile = True  # dataset temporary file on the disk created by concatenated_hdf5_data()
    # to be removed after code ends, with destructor of the RTDCDataset class

    indexes_dataset = np.arange(0, len(hdf5_ds))

    # Map class_cell_labels if targets (image labels)
    # were requested to be included in the created dataset
    if "target" in required_data:
        # class_cell_labels are obtained from the dataset, within which string_labels are saved. String labels are
        # converted to integer cell_class_labels using ml_score_to_int mapping
        cell_class_labels = create_cell_class_array(hdf5_ds, ml_score_to_int)
        target_mapping = reverse_dict(target_grouping)
        unknowns_target_mapping = reverse_dict(unknowns_target_grouping)
        if target_mapping:
            # Map class_cell_labels to target/grouped labels using target_mapping
            cell_class_labels_mapped = map_targets(target_mapping, cell_class_labels, MTL=MTL,
                                                   unknowns_target_mapping=unknowns_target_mapping)
        else:
            assert not bool(target_grouping), "target mapping is empty, but target grouping is not, check the code"
            cell_class_labels_mapped = cell_class_labels
            print("Warning: no mapping to new labels (or groups of labels) was provided. Using the initial "
                  "labels saved in rtdcs")

        # remove nans in case of cell labels that are not mapped with target_mapping
        # Remove the corresponding entries in indexes_dataset
        cell_class_labels_mapped_filt, indexes_dataset = remove_nans(cell_class_labels_mapped, indexes_dataset)
        if MTL:
            cell_class_labels_mapped_filt = None

    else:
        # no target (image labels) was requested to include
        # in the dataset to be created
        cell_class_labels_mapped = None
        cell_class_labels_mapped_filt = None

    dataset_train = None
    dataset_val = None
    indexing_train = None
    indexing_val = None

    if (train_size != 0) and (train_size != 1):
        # Split training data into training and validation dataset
        # stratifying on cell_class_labels (if target (image labels)
        # were requested to be included in the created dataset).
        print(f"Split dataset: train and val {time.ctime()}", flush=True)
        indexing_train, indexing_val = train_test_split(
            indexes_dataset,
            train_size=train_size,
            shuffle=True,
            random_state=42,
            stratify=cell_class_labels_mapped_filt)
    elif train_size == 0:
        indexing_val = indexes_dataset
    elif train_size == 1:
        indexing_train = indexes_dataset
    else:
        raise Exception("train_size parameter should be in [0, 1]")

    if indexing_train is not None:
        print(f"Create dataset: rtdc train dataset {time.ctime()}", flush=True)
        # Based on indices, create training and validation datasets
        dataset_train = RTDCDataset(hdf5_data=hdf5_ds,
                                    required_data=required_data,
                                    indexing_array=indexing_train,
                                    targets=cell_class_labels_mapped,
                                    augm_params=augmentation,
                                    crop_size=crop_size,
                                    correct_background=correct_background,
                                    train=True,
                                    mean=mean,
                                    std=std,
                                    p99_compute=p99_compute,
                                    brightness_factor=brightness_factor)

        assert len(dataset_train) > 0, "created dataset is empty"

    if indexing_val is not None:
        print(f"Create dataset: rtdc val or test dataset {time.ctime()}", flush=True)

        # val dataset applies mean, std of the train dataset.
        if dataset_train is not None:
            mean = dataset_train.mean
            std = dataset_train.std

        dataset_val = RTDCDataset(hdf5_data=hdf5_ds,
                                  required_data=required_data,
                                  indexing_array=indexing_val,
                                  targets=cell_class_labels_mapped,
                                  augm_params=augmentation,
                                  train=False,
                                  mean=mean,
                                  std=std,
                                  p99_compute=p99_compute,
                                  brightness_factor=brightness_factor,
                                  crop_size=crop_size,
                                  correct_background=correct_background)

        assert len(dataset_val) > 0, "created dataset is empty"

    return dataset_train, dataset_val

# def create_prediction_datasets(hdf5_paths,
#                                dataset_params: dict,
#                                transform,
#                                crop_size,
#                                correct_background: bool = False
#                                ) -> RTDCDataset:
#     """
#     Create a prediction dataset with the specified transformations.
#
#     This function prepares a Torch DataLoader for prediction by configuring
#     the dataset parameters and transformations. It reads the augmentation
#     parameters, sets up normalization values and creates an RTDCDataset.
#
#     Args:
#         hdf5_paths (list or str): The path(s) to file(s) contains the dataset.
#         dataset_params (dict): A dictionary containing dataset parameters,
#                                including augmentation settings.
#         transform (callable): The transformation function or pipeline to be
#                               applied to the dataset images.
#         correct_background (bool, optional): A flag indicating if we correct
#                                             the background of the images.
#                                              Defaults to False.
#
#     Returns:
#         RTDCDataset: The prepared dataset ready for prediction.
#     """
#     # Prepare Torch-DataLoader
#     required_data = {"image": "image"}
#     augm_params = {}
#     mean = 0
#     std = 1
#     prediction_aug_params_names = list_original_module_names(transform)
#
#     for params_name in prediction_aug_params_names:
#
#         if params_name == 'ApplyNormalize':
#             mean = dataset_params['mean']
#             std = dataset_params['std']
#
#         elif params_name != 'ApplyToTensor':
#             augm_params[params_name] = dataset_params["augmentation"][params_name]
#
#
#     dataset = RTDCDataset(hdf5_data=hdf5_paths,
#                           required_data=required_data,
#                           augm_params=augm_params,
#                           crop_size=crop_size,
#                           mean=mean,
#                           std=std,
#                           correct_background=correct_background,
#                           train=False)
#
#     if len(dataset) == 0:
#         print("Warning: the created dataset is empty")
#
#     return dataset
