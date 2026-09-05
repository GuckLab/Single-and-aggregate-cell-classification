import os
import random
from typing import List, Dict

import numpy as np

from PIL import Image


dirname = os.path.dirname(__file__)


def postprocess_image(dataset: {}, index: int):
    """
    Postprocesses an image from a dataset.

    This function takes an image from a dataset,
    denormalizes it, converts it to a uint8 array,
    and returns it as a PIL Image.

    Args:
        dataset (dict): The dataset containing images.
        index (int): The index of the image to be processed.

    Returns:
        PIL.Image.Image: The postprocessed image.
    """
    # Denormalize the image
    im = (dataset[index]['image'][0].numpy() * dataset.std) + dataset.mean
    im = (im * 255).astype(np.uint8)
    im = Image.fromarray(im)
    return im

# def dict_to_list(dic_in: dict[int, str]) -> List:
#     """
#
#     Parameters
#     ----------
#     dic_in: dictinary with integer keys
#
#     Returns
#     -------
#     List of dictionary values ordered in accordance with integer keys
#
#     """
#
#     list_out = list(dict(sorted(dic_in.items())).values())
#
#     return list_out


def get_key_ordered_values_from_dict(dic: Dict[int, any]) -> (List, List):
    """
    Get values of the dictionary in keys' order
    :param dic:
    :return: list of values
    """

    keys = list(sorted(dic.keys()))
    values = [dic[key] for key in keys]
    return values, keys


def add_ml_score_prefix(in_str: str) -> str:

    return "ml_score_" + in_str


def map_abbr_2_ml_score_label(class_labels: Dict[int, str], features: Dict[int, str]) -> Dict[str, str]:
    """

    Parameters
    ----------
    class_labels: dictionary with values that are class names and keys that are the corresponding integer values
    features: dictionary with values that are feature abbreviations for each class and keys are corresponding
    integer values (same as for class_labels)

    Returns
    -------
    mapping: dictionary with values that are class names and keys that are class abbreviations with "ml_score_" appended

    Example:
    class_labels = {0: "Monocyte", 1: Neutrophil}
    features = {0: 'g1m', 1: 'g1n'}
    mapping = map_abbr_2_ml_score_label(class_labels, features)

    print(mapping) -> {'ml_score_g1m': 'Monocyte', 'ml_score_g1n': 'Neutrophil'}

    """

    mapping = {}
    for k, v in class_labels.items():
        mapping[add_ml_score_prefix(features[k])] = v
        # mapping["ml_score_" + features[k]] = v

    return mapping


def map_weights_abbr_2_labels(weights: Dict[str, float], mapping: Dict[str, str]) -> Dict[str, float]:
    """

    Parameters
    ----------
    weights: dictionary with values that are weights and keys that are the corresponding ml_score abbreviations
    mapping: dictionary with values that are class names and keys that are class abbreviations with "ml_score_" appended

    Returns
    -------
    mapped_weights: dictionary with values that are weights and keys that are class names

    Example:
    weights = {'ml_score_g1m': 0.7, 'ml_score_g1n': 1.0}
    mapping = {'ml_score_g1m': 'Monocyte', 'ml_score_g1n': 'Neutrophil'}

    mapped_weights = map_weights_abbr_2_labels(weights, mapping)
    print(mapped_weights) -> {'Lymphocyte': 1.2, 'Monocyte': 0.7}


    """
    mapped_weights = {}
    for k, v in weights.items():
        mapped_weights[mapping[k]] = v

    return mapped_weights


# def retrieve_bio_relevance_weights(target_names: dict[int, str],
#                                    ml_score_features, all_bio_relevance_weights: dict[str, float] = None,
#                                    filepath='../definitions'
#                                             '/ml_score_to_bio_relevance.json'):
#     """
#     Retrieve bio relevance weights from bio_relevance_weights_filepath
#      and return the weights as a dict.
#
#     Example:
#     bio_relevance_weights = {
#         'Eosinophil': 0.3333333333333333,
#         'Monocyte': 0.3333333333333333,
#         'Neutrophil': 0.3333333333333333,
#     }
#     """
#
#
#     bio_relevance_weights = {}
#
#     target_names = get_key_ordered_values_from_dict(target_names)
#
#     if not all_bio_relevance_weights:
#         with open(os.path.join(dirname, filepath)) as file:
#             all_bio_relevance_weights = json.load(file)
#         print("-->Warning: bio-relevance weights are taken from json file")
#     else:
#         print("bio-relevance weights are taken from the configuration yaml file")
#
#
#     for target_name in target_names:
#         index = target_names.index(target_name)
#         bio_relevance_weights[target_name] = all_bio_relevance_weights[ml_score_features[index]]
#
#     return bio_relevance_weights


# def retrieve_f1_score_weights(target_names: dict[int, str],
#                               ml_score_features, all_f1_score_weights: dict[str, float] = None,
#                               filepath='../definitions'
#                                        '/ml_score_to_f_beta_score.json'):
#     """
#     Retrieve beta weights from weights_f1_score_filepath
#      and return the beta_values as a dict.
#
#     Example:
#     weights_f1_score = {
#         'Eosinophil': beta_value,
#         'Monocyte': beta_value,
#         'Neutrophil': beta_value,
#     }
#     """
#     f1_score_weights = {}
#     target_names = get_key_ordered_values_from_dict(target_names)
#
#     if not all_f1_score_weights:
#         with open(os.path.join(dirname, filepath)) as file:
#             all_f1_score_weights = json.load(file)
#             print("-->Warning: f1 score weights are taken from json file")
#     else:
#         print("f1 score weights are taken from the configuration yaml file")
#
#     for target_name in target_names:
#         index = target_names.index(target_name)
#         f1_score_weights[target_name] = all_f1_score_weights[
#             ml_score_features[index]]
#
#     return f1_score_weights


# def compute_weighted_f1_score(df_report, target_names, f1_score_weights):
#     """
#     Compute the weighted F1 score and add it to the
#     classification report DataFrame.
#
#     This function calculates a weighted F1 score (f_beta_score)
#     for each class in the classification report,
#     using the provided F1 score weights.
#     It then adds this score as a new column
#     in the DataFrame and reorders the columns for clarity.
#
#     Args:
#         df_report (pd.DataFrame): The classification report DataFrame
#                                 containing metrics for each class,
#                                 with columns for 'precision', 'recall',
#                                 'support', and 'f1-score'.
#         target_names (list): A list of target class names.
#         f1_score_weights (dict): A dictionary of F1 score
#         weights for each target class.
#
#     Returns:
#         pd.DataFrame: The updated classification report
#         DataFrame with the 'f_beta_score'
#         column added and columns reordered.
#     """
#     # Set F1 score weights to 0 for metrics not in target_names
#     for index_value in df_report.index.values:
#         if index_value not in target_names:
#             f1_score_weights[index_value] = 0
#
#     # To do: weighted_f1_score = 0 if nan
#     # Calculate the weighted F1 score (f_beta_score)
#     # To fix: Currently produce an "invalid value encountered in scalar divide
#     # (1 + f1_score_weights[row.name] ** 2)" runtime warning.
#     df_report['f_beta_score'] = df_report.apply(
#         lambda row: (
#                 (1 + f1_score_weights[row.name] ** 2)
#                 * (row.precision * row.recall)
#                 / (
#                         (f1_score_weights[row.name] ** 2 * row.precision)
#                         + row.recall
#                 )
#         ) if pd.notnull(row.precision) and pd.notnull(row.recall) else 0,
#         axis=1
#     )
#
#     # Reorder columns
#     df_report = df_report[['precision',
#                            'recall',
#                            'support',
#                            'f1-score',
#                            'f_beta_score',
#                            ]]
#     return df_report
#
#
# def compute_bio_relevance_weighted_avg(report, target_names, weights):
#     """
#     Add 'bio_relevance_weighted_avg' metric to the classification report.
#
#     This function calculates a weighted average of the metrics for each
#     target class, based on their bio relevance weights, and adds the result
#     to the classification report.
#
#     Args:
#         report (dict): The classification report containing
#         metrics for each class.
#         target_names (list): A list of target class names.
#         weights (dict): A dictionary of weights for each target class.
#
#     Returns:
#         dict: The updated classification report
#         with the 'bio_weighted_avg' metric added.
#     """
#     relevant_metrics = {}
#     for label in target_names:
#         for metric in report[label].keys():
#             relevant_metrics[metric] = \
#                 relevant_metrics.get(metric) or 0
#             if metric == 'support':
#                 relevant_metrics[metric] += report[label][metric]
#             else:
#                 relevant_metrics[metric] += report[label][metric] * \
#                                                  weights[label]
#
#     # Calculate sum of weights
#     total_weight = sum(weights.values())
#
#     bio_relevance_weighted_avg = {
#         key: val / total_weight for key, val in relevant_metrics.items()
#     }
#
#     report['bio_weighted_avg'] = bio_relevance_weighted_avg
#     return report

# def shuffle_pairwise(list1=[], list2=[]):
#     """
#     Shuffle two lists simultaneously, maintaining the relative
#     positions of elements.
#
#     This function takes two lists, pairs their elements, shuffles the pairs,
#     and then unzips the shuffled pairs back into two separate lists with their
#     original order preserved between the lists.
#
#     Parameters:
#     list1 (list): The first list to be shuffled.
#     list2 (list): The second list to be shuffled.
#
#     Returns:
#     tuple: A tuple containing two lists:
#         - The first list with elements shuffled.
#         - The second list with elements shuffled
#         in the same order as the first list.
#     """
#     # Pair the elements from the two lists
#     paired_list = list(zip(list1, list2))
#
#     # Shuffle the list of pairs
#     random.shuffle(paired_list)
#
#     # Unzip the shuffled pairs back into two lists
#     shuffled_list1, shuffled_list2 = zip(*paired_list)
#
#     # Convert the shuffled lists from tuples back to lists (if needed)
#     shuffled_list1 = list(shuffled_list1)
#     shuffled_list2 = list(shuffled_list2)
#
#     return shuffled_list1, shuffled_list2
#     # Pair the elements from the two lists
#     paired_list = list(zip(list1, list2))
#
#     # Shuffle the list of pairs
#     random.shuffle(paired_list)
#
#     # Unzip the shuffled pairs back into two lists
#     shuffled_list1, shuffled_list2 = zip(*paired_list)
#
#     # Convert the shuffled lists from tuples back to lists (if needed)
#     shuffled_list1 = list(shuffled_list1)
#     shuffled_list2 = list(shuffled_list2)
#
#     # Print the results
#     return shuffled_list1, shuffled_list2
