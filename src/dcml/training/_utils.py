import torch

from PIL import Image
import os
import numpy as np


class SaveImage:
    def __init__(self, path_to_save, file_prefix='', ext='png', stat: dict = None, str_labels=None, n_images=50):
        """
        Callable that saves images to the disk. It assumes that either the images are in the range [0, 1]
        or they were in this range before normalization.
        ----------
        stat: dictionary with "mean" and "std" keys for mean and standard deviation respectively
        path_to_save: path to save images
        ext: extension of the images, e.g. 'png'
        file_prefix: - prefix in the file name followed by digits assigned to images sequentially
        str_labels: mapping from integer labels to strings
        n_images: number of images to save if not images need to be saved

        """

        if stat:
            self.mean = stat["mean"]
            self.std = stat["std"]
        else:
            self.mean = 0.524
            self.std = 0.093

        self.counter = 0
        self.path = path_to_save
        self.ext = '.' + ext
        self.prefix = file_prefix
        self.str_labels = str_labels
        self.batch_message = False
        self.n_images = n_images

    def __call__(self, im, label=None):

        # check if it is a batch
        if not self.batch_message:
            if len(im.shape) == 4:
                self.batch_message = True
                print(
                    "warning: only first image from the batch will be taken. If all images need to be saved use batch size equal one")

        # take first image from the batch and first channel from the image
        im = im[0].squeeze(0)

        # if label is given, use it for defining the output folder
        if label is not None:
            if torch.is_tensor(label):
                label = int(label[0].item())
            else:
                raise "expected label is of torch tensor type"

            if self.str_labels is not None:
                label = self.str_labels[label]
            else:
                label = str(label)

            path = os.path.join(self.path, label)
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = os.path.join(self.path, 'no_label')
            if not os.path.exists(path):
                os.makedirs(path)

        file_name = os.path.join(path, self.prefix + '_' + str(self.counter) + self.ext)

        # denormalize image and put to an appropriate range for saving
        im = im * self.std + self.mean
        im = np.uint8(im * 255)

        # save an image to the disk
        im = Image.fromarray(im)
        im.save(file_name)

        self.counter += 1

    pass


def print_batch(epoch: int, batch_idx: int, loss_value: float) -> None:
    """Prints out basic information about current epoch, batch and loss"""
    print(f"Epoch: {epoch} - Batch: {batch_idx} - Loss: {loss_value}")


# def print_metrics(scores, class_label_dict):
#     """Pretty-Prints out Precision, Recall and F1-Score for all classes"""
#     class_names = [v for k, v in sorted(class_label_dict.items())]
#     metric_types = ["Precision", "Recall", "F1"]
#     print(tabulate(
#         [[metr, *(scores[idx])] for idx, metr in enumerate(metric_types)],
#         headers=["Metrics"]+class_names), flush=True)


# def create_matrix_penalization(penalization_filepath) -> dict:
#     Commented out during cleanup: focal-loss support helper is unused.


# def retrieve_penalization_weights(predictions, targets, penalties):
#     Commented out during cleanup: focal-loss support helper is unused.


# def create_support_penalization(dataset_train_targets):
#     Commented out during cleanup: focal-loss support helper is unused.


# def retrieve_support_weights(targets, penalties):
#     Commented out during cleanup: focal-loss support helper is unused.


def param_dict_to_list(dic_params: dict, class_label_dict: dict) -> list:

    """

    Parameters
    ----------
    dic_params: dictionary with keys as names of the MTL classes and values as possitive weights for training binary classifiers
    class_label_dict: dictionary with values as names of the classes (can be both MTL and MC) and keys as integers for the correspondent classes

    Returns
    -------
    list_params: list of positive weights ordered in accordance with integers (from class_label_dict) for the MTL classes
    """
    n = len(dic_params) # max(class_label_dict.keys()) + 1
    list_params = [float('nan')] * n
    MTL_classes = dic_params.keys()

    for cls_int, cls_name in class_label_dict.items():
        if cls_name in MTL_classes:
            list_params[cls_int] = dic_params[cls_name]

    return list_params
