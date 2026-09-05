import logging
import typing  # noqa: F401
import warnings

from dcnum.read import HDF5Data, concatenated_hdf5_data
import numpy as np  # noqa: F401
import torch

from .transforms import TransformFactory
from .transforms import ApplyToTensor

from ..preprocessing import crop_image, event_coordinates, correct_bg_on_image
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


class RTDCDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_data, required_data, indexing_array=None,
                 targets=None, augm_params=None, train=True, mean:float | None = None,
                 std:float | None = None, p99_compute:bool = False, brightness_factor:float | None = 1.0, crop_size=80, correct_background=True):
        """Create a torch-Dataset for training with RTDC data

        Parameters
        ----------
        hdf5_data: h5py.File or pathlib.Path or list of pathlib.Paths
            (Virtual) Dataset in form of h5py.File of (multiple)
            *.rtdc Dataset(s)
        required_data: dict
            Key-value pairs that define which feature value in the *.rtdc
            datasets, defined in `hdf5_data` are assigned to which key-value
            in the data-dictionary which is returned by __getitem__()-function
            Example: {"image": "image", "target": "ml_score_ctt"}
            makes the __getitem__(idx=5)-function return:
                data = {"image": rtdc["image"][5],
                        "target": rtdc["ml_score_ctt"][5]}
        indexing_array: np.ndarray
            An indexing array that contains the indices of elements in dataset
            that should be used. This is helpful if you dynamically want to
            create a training dataset and validation dataset from the same
            RTDC-Dataset without actually splitting the dataset into two.
        targets: list | np.ndarray
            This iterable contains the target-class index. Has to have the
            same length as `hdf5_data`. Can also be 2D array with n-hot rows
        augm_params: dict
            Dictionary used to instantiate the image-transformations.
            Will be passed through to `TransformFactory()`
        train: boolean
            defines if the returned data should be transformed according to
            training or validation data
        transform: torchvision.transforms.Compose
            A composition of transforms applied to the imagery data
        crop_size: int for square patch or tuple for rectangular patch
        with the first entry being width and second the
        hight. Defines the size of the crop of the imagery data
        correct_background: bool
            Apply background correction on "image" in __getitem__() if True.
            Requires "image" to be in required_data and "image_bg" to be
            present in dataset.
        mean: mean values to be used for normalization of the images. If None, mean will be computed from the dataset.
        std: standard deviation values to be used for normalization of the images. If None, std will be computed from the dataset.
        p99_compute: if True, computes 99th percentile gray level across all images in the dataset (can be used for brightness correction).
        brightness_factor: If not None, this factor will be used to correct the brightness of the images by multiplying them with this factor.
        If mean or std were None, then they will be computed from the dataset and corrected by multiplying with the brightness factor.
        If they were given as arguments, they will be used as they are (without correction).
                """
        super(RTDCDataset, self).__init__()

        try:
            len(hdf5_data)
        except TypeError:
            # hdf5_data might be a Path
            pass
        else:
            if targets is not None:
                # hdf5_data is dcnum.read.hdf5_data.HDF5Data object and targets argument is given
                assert len(hdf5_data) == len(targets), ("the length of the dataset should be equal to "
                                                        "the number of targets")

        if isinstance(hdf5_data, list):
            print("RTDCDataset: Concatenate hdf5 data", flush=True)
            self.h5data = concatenated_hdf5_data(paths=hdf5_data)
            self.h5data.h5.tempFile = True  # dataset temporary file on the disk created by concatenated_hdf5_data()
            # to be removed after code ends, with destructor of the class

        elif isinstance(hdf5_data, HDF5Data):
            self.h5data = hdf5_data
        else:
            self.h5data = HDF5Data(hdf5_data)

        if augm_params is None:
            augm_params = {}
        if indexing_array is None:
            # self.indexing_array = np.array(range(len(self.h5data)))
            self.indexing_array = np.arange(0, len(self.h5data))
        else:
            self.indexing_array = indexing_array
        self.crop_size = crop_size
        self.pxs, self.pys = event_coordinates(
            self.h5data["pos_x"][:],
            self.h5data["pos_y"][:],
            crop_size=self.crop_size,
            pixel_size=self.h5data.pixel_size)

        self.required_data = required_data.copy()

        # Drop 'target' if it is given in dictionary
        self.required_data.pop("target", None)
        self.targets = targets
        self.augm_params = augm_params
        self.train = train
        self.correct_background = correct_background

        self.mean = mean
        self.std = std
        self.p99 = None

        if brightness_factor is None:
            brightness_factor = 1.0

        if mean is None or std is None or p99_compute:
            mean_computed, std_computed, p99_computed = self.get_mean_std_p99()
            if mean is None:
                self.mean = mean_computed * brightness_factor
            if std is None:
                self.std = std_computed * brightness_factor
            if p99_compute:
                self.p99 = p99_computed * brightness_factor

        self.transform = TransformFactory(augm_params, mean=self.mean, std=self.std, brightness_factor=brightness_factor).create(train=train)

    def  get_mean_std_p99(self):
        """
        Calculates mean and standard deviation of the dataset of cropped images
        (using requested indexing_array). Additionally, returns the 99th
        percentile gray level across all cropped pixels.
        """
        ds_mean = 0.0
        ds_mean_squared = 0.0
        ds_p99 = 0.0
        count = 0
        print("Getting mean and standard deviation from the dataset",
              flush=True)
        print("The size of the dataset is {}".format(len(self.indexing_array)),
              flush=True)
        for _idx in tqdm(self.indexing_array):
            for key, value in self.required_data.items():
                if key == "image":
                    image = self.h5data[value][_idx]
                    if self.correct_background and value == "image":
                        image_bg = self.h5data["image_bg"][_idx]
                        image = correct_bg_on_image(image, image_bg)
                    image = crop_image(image,
                                       self.pxs[_idx],
                                       self.pys[_idx],
                                       self.crop_size)

                    image_tensor = ApplyToTensor()(image)

                    # Update sum and sum of squares for mean and variance
                    ds_mean += image_tensor.mean().item()
                    ds_mean_squared += (image_tensor ** 2).mean().item()
                    # Accumulate per-image 99th percentile (avoids building a
                    # huge pixel vector; equivalent to global p99 when all
                    # images have the same size)
                    ds_p99 += float(np.percentile(
                        image_tensor.cpu().numpy(), 99))
                    count += 1

        ds_mean = ds_mean / count
        ds_mean_squared = ds_mean_squared / count
        ds_p99 = ds_p99 / count

        ds_variance = ds_mean_squared - ds_mean ** 2
        ds_std = np.sqrt(ds_variance)
        print("mean: {}, std: {}, gray p99: {}".format(ds_mean, ds_std, ds_p99), flush=True)

        return float(ds_mean), float(ds_std), float(ds_p99)

    def __enter__(self):
        print("Entering Context Manager of HDF5Dataset:")
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __len__(self):
        return len(self.indexing_array)

    def __getitem__(self, idx):
        # TODO: refactor this for speed. A few suggestions:
        #  - Try to avoid for-loops
        #  - Use self.h5data.image instead of self.h5data["image"] for speed
        #  - Use the self.h5data.image_corr instead of correcting the
        #    background on-the-fly here.
        #  - If self.correct_background is set only during init, then maybe
        #    it makes sense to set `self.image` to `h5data.image` or
        #    `h5data.image_corr` during init (saves an if-clause)
        #  - Note that if your indexing array is totally random
        #    (e.g. for training), then the speed-up might not be so good.
        #    But if the training data are small (a few 1000's), then we
        #    could increase the cache_size of dcnum's ImageCache.
        _idx = self.indexing_array[idx]
        data = {}
        for key, value in self.required_data.items():
            if key == "image":
                image = self.h5data[value][_idx]
                if (self.correct_background and value == "image"):
                    # assumes key:value-> {"image":"image"} for background
                    # correction
                    image_bg = self.h5data["image_bg"][_idx]
                    image = correct_bg_on_image(image, image_bg)
                image = crop_image(image,
                                   self.pxs[_idx],
                                   self.pys[_idx],
                                   self.crop_size)
                if self.transform is not None:
                    image = self.transform(image)
                data[key] = image
            else:
                data[key] = self.h5data[value][_idx]
        if self.targets is not None:
            data["target"] = self.targets[_idx]
        return data


    def calculate_sample_weights(self):

        """
        Calculate the sample weights for each class in dataset to generate uniform distribution for classes

        Returns:
            torch.Tensor: A tensor of weights for each sample in the dataset,
            the weight is inversely proportional to the frequency
            of the sample's class.
        """

        # Extract the training labels using the indices in self.indexing_array
        dataset_targets = [self.targets[i] for i in self.indexing_array]

        if isinstance(dataset_targets[0], np.ndarray):
            # multitask classification: labels are row n-hot vectors

            # Count the number of samples for each class
            count_samples_per_class = np.sum(np.array(dataset_targets) == 1, axis=0) # target can be also 0.5 for MTL

            # Compute the inverse of the class counts to get the weights
            weights_class = 1 / count_samples_per_class

            weights_samples_class = []
            for target in dataset_targets:
                idxs = np.nonzero(target == 1) # target can be also 0.5 for MTL

                # when there are several 1-labels, the more rare label is considered by using "max".
                # This gives all single cells being equally sampled
                max_weight = np.max(weights_class[idxs])

                weights_samples_class.append(max_weight)

        else:

            # multiclass classification: labels are indexes of correct class

            # Count the number of samples for each class in labels
            class_labels = np.unique(dataset_targets)
            count_samples_per_class = np.array([len(np.where(dataset_targets == t)[0])
                                                for t in class_labels])

            # Compute the inverse of the class counts to get the weights
            weights_class = 1 / count_samples_per_class

            # Map the class weights to all samples
            weights_samples_class = [weights_class[int(t)] for t in dataset_targets]


        weights_samples_class = torch.tensor(weights_samples_class).double()

        return weights_samples_class

    # def calculate_sample_weights(self):
    #     """
    #     Calculate the sample weights for each class in dataset.
    #
    #     Returns:
    #         torch.Tensor: A tensor of weights for each sample in the dataset,
    #         the weight is inversely proportional to the frequency
    #         of the sample's class.
    #     """
    #     # Extract the training labels using the indices in self.indexing_array
    #     dataset_targets = [self.targets[i] for i in self.indexing_array]
    #
    #     # Count the number of samples for each class in labels
    #     weights_samples_class = self._sample_weights(dataset_targets)
    #     # count_samples_class = np.array([len(np.where(dataset_targets == t)[0])
    #     #                                 for t in np.unique(dataset_targets)])
    #
    #     # # Compute the inverse of the class counts to get the weights
    #     # weights_class = 1 / count_samples_class
    #     #
    #     # # Map the class weights to all samples
    #     # weights_samples_class = torch.tensor([weights_class[int(t)] for t in dataset_targets])
    #
    #     # Convert the sample weights to a double-precision tensor
    #     #weights_samples_class = weights_samples_class.double()
    #     weights_samples_class = torch.tensor(weights_samples_class).double()
    #     return weights_samples_class

    @property
    def hdf5_ds(self):
        warnings.warn("`hdf5_ds` is deprecated, please use `h5data` instead!",
                      DeprecationWarning)
        return self.h5data

    def close(self):
        # Close hdf5-file
        #print("I am closing")
        self.h5data.close()

    def __del__(self):

        #print("I am am deleting")
        #print(type(self))

        #if getattr(self, 'h5data', False):
            #print(type(self.h5data))

            #if self.h5data:
                #print(type(self.h5data.h5))
                #print(self.h5data.h5.tempFile)
                #print(bool(self.h5data.h5))
                #print(self.h5data.h5.file.filename)


        if self.h5data.h5:
            if getattr(self.h5data.h5, 'tempFile', False):

                temp_file_name = self.h5data.h5.file.filename
                # TODO consider using only one condition for deletion (tempFile field or '/dcnum_vc_ prefix')
                if ('/dcnum_vc_' in temp_file_name) and os.path.isfile(temp_file_name):
                    logger.info(f"{temp_file_name} is being deleted")
                    #print(f"{temp_file_name} is being deleted")
                    os.remove(temp_file_name)

            self.close()
