import logging
from typing import Iterable, List
import pathlib
from collections import defaultdict


from dclab.rtdc_dataset import RTDC_HDF5
from dcnum.read import HDF5Data
import numpy as np  # noqa: F401
import torch

from .transforms import ApplyToTensor
from ..io import load_ml_score_to_int

def reverse_dict(d):
    """
    Generates dictionary with keys that are values and values that are keys in the input dictionary.
    Values of the input dictionary are lists with a single or a few numeric values.
    """

    if d is None:

        out = {}
    else:

        reversed_d = defaultdict(list)  # Using defaultdict to create a list for each new key
        for key, values in d.items():
            for value in values:
                reversed_d[value].append(key)  # Append the original key to the list for each value

        # Should I put it in map_targets function?
        # # Convert defaultdict to a regular dictionary and simplify values
        # for key in reversed_d.keys():
        #     # If there's only one key in the list, convert it to an int
        #     if len(reversed_d[key]) == 1:
        #         reversed_d[key] = reversed_d[key][0]

        if reversed_d is None:
            out = {}
        else:
            out = dict(reversed_d)  # Convert back to a normal dictionary

    return out

def flatten_dic_values(dic):
    """
    Extracts all the numbers for dictionary that can have values that are lists of numbers
    """

    values = [item for sublist in dic.values() for item in sublist]

    return values


def map_targets(target_grouping: dict, cell_class_labels: np.ndarray, MTL: bool = False,
                unknowns_target_mapping: dict = None) -> Iterable:
    """

    Maps numeric labels in `cell_class_labels` into the corresponding target labels using target_grouping mapping.
    If class label does not exist in the keys of target_grouping, it will be mapped to np.nan

    :params target grouping: dictionary with keys that are original integer labels and values that are new
    grouped/mapped labels. Each dictionary value is a list of a single or a few integer labels.
    A few labels are only accepted for MTL=True. (Put attention this is a reverse target_grouping relative to the one
    defined in the configuration file)
    :param cell_class_labels: 1D ndarray with the original labels for each event/image (float type to accommodate nan)
    :MTL: set to True if Multi-Task Learning model is used (mapped labels are then hot-vectors instead of scalars)
    :return: 1D ndarray of new mapped (grouped) labels for each event if MTL is False, and 2D array with rows containing
    ones at the indexes corresponding to the mapped labels (n-hot vectors). 1D array can contain nan
    if cell_class_labels does not exist in the keys of target_grouping. 2D array in that case will contain
    the whole row of nans
    unknowns_target_mapping: dictionary with keys that are original integer labels and values that are new/mapped labels.
    Each dictionary value is a list of a single or a few categories that are of unknown label (e.g. Thrombocyte (mapped)
    label might be not known for the image with the original label, e.g. RBC doublet (dict key).

    Max: Could be achieved with `np.vectorize`, but there it is not so easy to
    define the identity to be the default value.
    This could be achived with `np.vectorize` and using `functools.partial`?

    Examples
    --------
    target_grouping = {0: [2], 1: [0], 2: [5, 4], 4: [1, 3], 8: [1]}
    cell_class_labels = np.array([0, 0, 2, 1, 7, 1, 8, 4], dtype=float)
    grouped = map_targets(target_grouping, cell_class_labels, MTL=True)
    print(grouped)
    [ 0.  0.  1.  0.  0.  0.]
    [ 0.  0.  1.  0.  0.  0.]
    [ 0.  0.  0.  0.  1.  1.]
    [ 1.  0.  0.  0.  0.  0.]
    [nan nan nan nan nan nan]
    [ 1.  0.  0.  0.  0.  0.]
    [ 0.  1.  0.  0.  0.  0.]
    [ 0.  1.  0.  1.  0.  0.]]

    target_grouping = {0: [2], 1: [0], 2: [3], 4: [1], 8: [1]}
    cell_class_labels = np.array([0, 0, 2, 1, 7, 1, 8, 4], dtype=float)
    grouped = map_targets(target_grouping, cell_class_labels)
    print(grouped)
    [ 2.  2.  3.  0. nan  0.  1.  1.]
    """

    # flatten the list of lists
    target_grouping_values = flatten_dic_values(target_grouping)
    n_classes = max(target_grouping_values) + 1
    assert len(set(target_grouping_values)) == n_classes, "target classes are not subsequent numbers"

    if not MTL:
        cell_class_labels_grouped = np.empty(len(cell_class_labels), dtype=cell_class_labels.dtype)
    else:
        cell_class_labels_grouped = np.zeros((len(cell_class_labels), n_classes), dtype=cell_class_labels.dtype)

    for n, el in enumerate(cell_class_labels):

        # get nan if original label is not in the keys of the target_grouping mapping
        lab = target_grouping.get(el, np.nan)

        if unknowns_target_mapping is not None:
            unknown_lab = unknowns_target_mapping.get(el, np.nan)
        else:
            unknown_lab = np.nan

        if not MTL: # mapped labels will be 1D array of labels
            if lab is np.nan:
                logging.info(f"data used has label {el} that is not in target_grouping labels")
                cell_class_labels_grouped[n] = lab
            else:
                assert len(lab) == 1, "target grouping is wrong: an original label is mapped to several new labels"
                cell_class_labels_grouped[n] = lab[0]

        else: # mapped labels will be 2D array, n-hot vectors

            if unknown_lab is not np.nan:
                cell_class_labels_grouped[n, unknown_lab] = np.nan

            if lab is np.nan:
                logging.info(f"data used has label {el} that is not in target_grouping labels")
                lab = list(range(n_classes)) # nans for each entry of the vector
                val = np.nan
            else:
                # val = [1] * len(lab)
                # if unknowns_target_grouping is not None:
                #     for n2, la in enumerate(lab):
                #         uknown_influence = unknowns_target_grouping.get(la, None)
                #         if uknown_influence is not None:
                #             if el in uknown_influence:
                #                 val[n2] = 0.5

                val = 1

            cell_class_labels_grouped[n, lab] = val





    return cell_class_labels_grouped


def resolve_filepaths(input_paths: List[str]) -> List[str]:
    """Recursively finds all `rtdc`-files in given paths."""
    input_paths = [pathlib.Path(el) for el in input_paths]
    input_rtdc_paths = []
    for path in input_paths:
        if path.is_dir():
            rtdc_paths = list(path.rglob("*.rtdc"))
            input_rtdc_paths += rtdc_paths
        elif path.suffix == ".rtdc":
            input_rtdc_paths.append(path)
    # Remove duplicates
    input_rtdc_paths = sorted(list(set(input_rtdc_paths)))
    return input_rtdc_paths


def get_group(hdf5_file, group):
    if hdf5_file.filename.endswith('.rtdc'):
        return hdf5_file["events"][group]
    else:
        return hdf5_file[group]


# def get_combined_shape(file_list, group):
#     Commented out during cleanup: unused in current train/evaluate flow.


# def combine_rtdc_to_vds(input_paths, output_path):
#     Commented out during cleanup: deprecated and unused in current train/evaluate flow.


def create_cell_class_array(hdf5_data: HDF5Data | RTDC_HDF5, ml_score_to_int: dict = None) -> np.ndarray:
    """ Creates a single array containing original class labels mapped to integers (ml_score_to_int in yaml)

    Parameter
    ---------
    hdf5_file: h5py.File
        This is the file for which the array of all cell class labels should
        be computed.

    Returns
    -------
    np.ndarray: cell_class_array
        A single np.ndarray containing the combined cell class labels
    """
    ml_score_to_int_dict = load_ml_score_to_int(ml_score_to_int)
    #cell_class_array = np.zeros(len(hdf5_data), dtype=int)
    cell_class_array = np.zeros(len(hdf5_data))
    for score_name, score_int in ml_score_to_int_dict.items():
        score_array = hdf5_data[score_name]
        idxs = np.where(score_array[:] == 1)
        cell_class_array[idxs] = score_int

    return cell_class_array


def compute_dataset_mean_std(torch_ds):
    """
    Computes the mean and standard deviation of the entire dataset of images.

    This function iterates through all the images in the given dataset,
    crops them, applies tensor normalization,and then calculates the overall
    mean and standard deviation of the pixel values.
    """
    # Initialize sum and sum of squares of pixel values as scalars
    pixel_sum = 0.0
    pixel_sum_squared = 0.0

    # Get total pixel count
    crop_size = torch_ds.crop_size
    total_pixel_count = len(torch_ds.data) * crop_size * crop_size
    # Iterate through each image in the dataset
    for data in torch_ds.data:
        # Apply tensor
        image_tensor = ApplyToTensor()(data["image"])

        # Update sum and sum of squares for mean and variance calculation
        pixel_sum += image_tensor.sum().item()
        pixel_sum_squared += (image_tensor ** 2).sum().item()

    # Compute the overall mean and standard deviation
    ds_mean = pixel_sum / total_pixel_count
    ds_variance = (pixel_sum_squared / total_pixel_count) - (ds_mean ** 2)
    ds_std = torch.sqrt(torch.tensor(ds_variance)).item()

    return ds_mean, ds_std


# def retrieve_mean_std(torch_ds):
#     """
#     Retrieve the mean and std of the cropped images.
#     If the normalize mode is mean_std, get mean / std of cropped images.
#     else get mean / std from the augmentation parameters.
#     """
#     normalize_mode = torch_ds.augm_params.get("ApplyNormalize", {}).\
#         get("mode", "default")
#     if normalize_mode == "mean_std":
#         mean, std = compute_dataset_mean_std(torch_ds)
#     else:
#         mean = 0
#         std = 1
#     return mean, std
