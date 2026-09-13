import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
# from dcml.evaluation.helpers import dict_to_list
#from dcml.evaluation.helpers import get_key_ordered_values_from_dict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns

# from sklearn.metrics import (
#     confusion_matrix,
#     classification_report
# )
from .helpers import postprocess_image  # compute_bio_relevance_weighted_avg,  compute_weighted_f1_score


def artifact_confusion_matrix(confusion_mat: np.ndarray,
                              labels: List[str], artifacts_dir: Path,
                              epoch: int = None) -> dict:
    """
    Compute and save confusion matrix plot.

    Parameters:
    targets (list): List of true target labels.
    predictions (list): List of predicted labels.
    labels (list): Class labels for the confusion matrix.
    artifacts_dir (Path): Directory path for saving artifacts.
    epoch (int): Epoch number for the plot filename.

    Returns:
    dict: Dictionary with the file path of the saved plot.
    """

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(confusion_mat,
                annot=True,
                fmt=".2f",
                xticklabels=labels,
                yticklabels=labels)

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion matrix")
    plt.tight_layout()
    # save plot
    if epoch:
        plot_path = artifacts_dir / f"confusion_matrix_plot_epoch_{str(epoch)}.png"
    else:
        plot_path = artifacts_dir / "confusion_matrix_plot.png"

    fig.savefig(plot_path)
    plt.close()

    return {"confusion_matrix_plot_artifact": plot_path, }


def artifact_classification_report(artifacts_dir: Path, report: dict, epoch: int = None) -> dict:
    """
    Compute and save a classification report plot.

    Parameters:
    targets (list): List of true target labels.
    predictions (list): List of predicted labels.
    labels (dict): dictionary from the configuration file with values that are class labels
    (keys are the corresponding integer labels that should also define values for a classifier output).
    bio_relevance_weights (dict): Dictionary mapping labels to bio weights.
    f1_score_weights (dict): Dictionary mapping labels to F1 score weights.
    artifacts_dir (Path): Directory path for saving artifacts.
    epoch (int): Epoch number for the plot filename.

    Returns:
    dict: Dictionary with the file path of the saved plot.
    """

    # takes the values of the dictionary ordered in accordance with the dictionary keys
    # labels = get_key_ordered_values_from_dict(labels)
    #
    # report = classification_report(targets,
    #                                predictions,
    #                                target_names=labels,
    #                                labels=np.arange(0, len(labels)),
    #                                output_dict=True)
    #
    # # rename macro_avg to avg
    # report['avg'] = report.pop('macro avg')
    #
    # # remove weighted_avg from report
    # if 'weighted avg' in report:
    #     del report['weighted avg']
    # # remove accuracy from report
    # if 'accuracy' in report:
    #     del report['accuracy']
    #
    # # add bio relevance
    # report = compute_bio_relevance_weighted_avg(report,
    #                                             labels,
    #                                             bio_relevance_weights)
    df_report = pd.DataFrame(report).transpose()
    # df_report.support = df_report.support.astype(int)
    #
    # df_report = compute_weighted_f1_score(df_report,
    #                                       labels,
    #                                       f1_score_weights)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df_report.iloc[:, :],
                cbar=False,
                square=False,
                annot=True,
                fmt='g',
                cmap=ListedColormap(['white']))
    plt.title("Classification report")
    plt.tight_layout()
    # plt.figtext(0.88,
    #             0,
    #             'Avg: average (averaging the unweighted mean per label)',
    #             horizontalalignment='right',
    #             fontsize=8)
    # save plot
    if epoch:
        plot_name = f"classification_report_plot_epoch_{str(epoch)}"
    else:
        plot_name = "classification_report_plot"

    plot_path = artifacts_dir / (plot_name + ".png")
    dataframe_path = artifacts_dir / (plot_name + ".csv")

    fig.savefig(plot_path)
    df_report.to_csv(dataframe_path, index=True)
    plt.close()

    return {
        "classification_report_plot_artifact": plot_path,
        "classification_report_dataframe": dataframe_path,
    }


def artifact_sample_images_with_targets(dataset: Dict,
                                        targets: List[int],
                                        predictions: List[int],
                                        pred_probs: List[List[float]],
                                        class_label_dict: Dict[int, str],
                                        number_sample: int,
                                        artifacts_dir: Path) -> Dict:
    """
    Save samples of predictions with probabilities for images with targets.

    Parameters:
    dataset (Dict): Dictionary containing images.
    targets (List[int]): List of target labels.
    predictions (List[int]): List of predicted labels.
    pred_probs (List[List[float]]): List of prediction probabilities.
    class_label_dict (Dict[int, str]): Dict mapping class labels to names.
    number_sample (int): Number of sample images to save per prediction.
    artifacts_dir (Path): Directory path where artifacts will be saved.

    Returns:
    Dict: A dictionary with the directory path where sample images with targets
          are saved.
    """
    def save_concatenate_images(image_preds, target, pred, dir_path):
        """
        Save concatenated images with titles in a grid format.

        Parameters:
        image_preds (list): List of dicts with images and their probabilities.
        target (str): Target label.
        pred (str): Prediction label.
        dir_path (Path): Directory path where images will be saved.

        """
        num_images = len(image_preds)
        num_cols = min(5, num_images)
        num_rows = (num_images + num_cols - 1) // num_cols

        subdir_filepath = dir_path / target / pred
        subdir_filepath.mkdir(parents=True, exist_ok=True)

        # sort images in predictions for better format
        images = sorted(image_preds,
                        key=lambda x: x['prob'])

        # set title to format
        title = f"Target: {target} / Prediction: {pred}"

        # Create a subplot with appropriate dimensions
        fig, axes = plt.subplots(num_rows,
                                 num_cols,
                                 figsize=(20, 15))

        # Plot images
        if num_images == 1:
            # When there's only one image, axes is not a list
            axes.imshow(images[0]['image'], cmap="gray")
            axes.set_title(f"score={'%.3f' % (images[0]['prob'])}")
            axes.axis("off")
        else:
            for i, ax in enumerate(axes.flat):
                if i < num_images:
                    ax.imshow(images[i]['image'], cmap="gray")
                    ax.set_title(f"score={'%.3f' % (images[i]['prob'])}")
                    ax.axis("off")
                else:
                    ax.axis('off')

        fig.tight_layout()
        fig.suptitle(title, size=20)
        plt.savefig(Path(subdir_filepath) / "samples.png")
        plt.close(fig)

    dir_filepath = artifacts_dir / 'samples_with_targets'
    int_labels = class_label_dict.keys()

    # Initialize dictionary to count saved samples
    samples = {
        target: {label: [] for label in int_labels}
        for target in int_labels
    }

    # Add samples to save
    for i in range(len(targets)):
        target = targets[i]
        pred = predictions[i]

        if len(samples[target][pred]) < number_sample:
            # save image with probability.png
            image = postprocess_image(dataset, i)
            samples[target][pred].append({'image': image,
                                          'prob': pred_probs[i][pred]})

    # Save samples
    for target in samples:
        for label in samples[target]:
            if len(samples[target][label]) > 0:
                save_concatenate_images(image_preds=samples[target][label],
                                        target=class_label_dict[target],
                                        pred=class_label_dict[label],
                                        dir_path=dir_filepath)

    return {
        "dir_sample_images_with_targets": dir_filepath,
    }


def artifact_sample_images_without_targets(dataset: dict, predictions: list,
                                           pred_probs: list, labels: list,
                                           number_sample: int,
                                           artifacts_dir: Path) -> dict:
    """
    Save samples of predictions along with probabilities of images
    without targets under artifacts_dir.

    Parameters:
    dataset (dict): The dataset containing images.
    predictions (list): List of predicted labels.
    pred_probs (list): List of prediction probabilities for each class.
    labels (list): List of class labels.
    number_sample (int): Number of sample images to save for each prediction.
    artifacts_dir (Path): Directory path where artifacts will be saved.

    Returns:
    dict: A dictionary with the directory path where sample images without
          targets are saved.
    """
    dir_filepath = artifacts_dir / 'samples_without_targets'

    # initialize number_saved_samples
    number_saved_samples = {}
    for prediction in labels:
        number_saved_samples[prediction] = 0

    for i in range(len(predictions)):
        prediction = predictions[i]
        prediction_prob = '%.3f' % (pred_probs[i][prediction])
        if number_saved_samples[prediction] < number_sample:
            subdir_filepath = dir_filepath / labels[prediction]
            subdir_filepath.mkdir(parents=True, exist_ok=True)
            img_filepath = subdir_filepath / (prediction_prob + '.png')
            # save image
            postprocess_image(dataset, i).save(img_filepath)
            number_saved_samples[prediction] = \
                number_saved_samples[prediction] + 1
    return {
        "dir_sample_images_without_targets": dir_filepath,
    }


def artifact_predictions_distribution(predictions: list, labels: list,
                                      artifacts_dir: Path) -> dict:
    """
    Save the distribution of predictions with counts along with their labels.

    Parameters:
    predictions (list): List of predicted labels.
    labels (list): List of labels corresponding to the predictions.
    artifacts_dir (Path): Directory path where artifacts will be saved.

    Returns:
    dict: A dictionary with the file path where the predictions distribution
          CSV file is saved.
    """
    values, counts = np.unique(predictions, return_counts=True)
    results = []
    for i in range(len(values)):
        results.append([labels[values[i]], counts[i]])
    results.append(['total', len(predictions)])
    df = pd.DataFrame(data=results, columns=['cell_type', 'counts'])
    # save plot
    plot_path = artifacts_dir / "predictions_distribution.csv"
    df.to_csv(plot_path, index=False)

    return {
        "predictions_distribution_artifact": plot_path,
    }


def save_concatenate_images(image_preds, target, pred, dir_path):
    """
    Save concatenated images with titles in a grid format.

    Parameters:
    image_preds (list): List of dicts with images and their probabilities.
    target (str): Target label.
    pred (str): Prediction label.
    dir_path (Path): Directory path where images will be saved.

    """
    num_images = len(image_preds)
    num_cols = min(5, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols

    subdir_filepath = dir_path / target / pred
    subdir_filepath.mkdir(parents=True, exist_ok=True)

    # sort images in predictions for better format
    images = sorted(image_preds,
                    key=lambda x: x['prob'])

    # set title to format
    title = f"Target: {target} / Prediction: {pred}"

    # Create a subplot with appropriate dimensions
    fig, axes = plt.subplots(num_rows,
                             num_cols,
                             figsize=(20, 15))

    # Plot images
    if num_images == 1:
        # When there's only one image, axes is not a list
        axes.imshow(images[0]['image'], cmap="gray")
        axes.set_title(f"score={'%.3f' % (images[0]['prob'])}")
        axes.axis("off")
    else:
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                ax.imshow(images[i]['image'], cmap="gray")
                ax.set_title(f"score={'%.3f' % (images[i]['prob'])}")
                ax.axis("off")
            else:
                ax.axis('off')

    fig.tight_layout()
    fig.suptitle(title, size=20)
    plt.savefig(Path(subdir_filepath) / "samples.png")
    plt.close(fig)
