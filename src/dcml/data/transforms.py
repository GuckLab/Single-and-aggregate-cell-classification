import random
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import v2


class ApplyToTensor(nn.Module):
    apply_test = True  # will be applied in any case

    def __init__(self):
        super().__init__()

    def forward(self, image):
        """
        Returns a tensor of an image.
        """
        min_val = 0
        max_val = 255
        tensor = torch.tensor(
            (image - min_val) / (max_val - min_val),
            dtype=torch.float32)
        return tensor.unsqueeze(0)

    def __repr__(self):
        repr_str = f"{self.__class__.__name__} "
        return repr_str


class ApplyNormalize(nn.Module):
    apply_test = True  # will be applied in any case

    def __init__(self, mean, std):
        """
        Apply normalization on image with mean and std.
        Parameters
        ----------
        "mean": float
        "std": float
        """
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        # TODO: Implement channel-size agnostic version of this method
        normalized_tensor = transforms.functional.normalize(tensor,
                                                            [self.mean, ],
                                                            [self.std, ])
        return normalized_tensor

    def __repr__(self):
        return f"{self.__class__.__name__} (mean={self.mean}, std={self.std})"



class BrightnessCorrection(nn.Module):

    apply_test = True # will be applied in any case

    def __init__(self, brightness_factor: float=1.0):
        """
        Multiplies brightness of an image by a given factor.

        Parameters
        ----------

        """
        super().__init__()
        self.brightness_factor = brightness_factor

    def forward(self, tensor):

        return transforms.functional.adjust_brightness(tensor, self.brightness_factor)


    def __repr__(self):
        repr_str = (f"{self.__class__.__name__} "
                    + f"(brightness_factor={self.brightness_factor})")
        return repr_str



class ApplyRandomVerticalFlip(nn.Module):

    # will be applied only when creating the dataset for training: TransformFactory.create(train=True)
    apply_test = False

    def __init__(self, p=0.5):
        """
        Vertically flip an image randomly with a given probability.
        Parameters
        ----------
        "p": float
        Probability of applying horizontal flip on an image
        """
        super().__init__()
        self.p = p

    def forward(self, tensor):
        if random.random() < self.p:
            return transforms.functional.vflip(tensor)
        else:
            return tensor

    def __repr__(self):
        return f"{self.__class__.__name__} (p={self.p})"


class AdjustGamma(nn.Module):

    # will be applied only when creating the dataset for training: TransformFactory.create(train=True)
    apply_test = False

    def __init__(self, gamma_range=(0.85, 1.15), p=0.5):
        """
        Makes gamma correction for an  image with.
        Parameters
        ----------
        gamma_range: List[float] or Tuple
            Min and Max value of uniform distribution for sampling the
            gamma factor. Example: `[0.85, 1.15]` will
            sample for `gamma_factor`-values between `0.85` and `1.15`
        p: float
            Probability of applying brightness adjustment on an image
        """
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def forward(self, tensor):
        if random.random() < self.p:
            gamma = random.uniform(self.gamma_range[0],
                                               self.gamma_range[1])
            return transforms.functional.adjust_gamma(tensor, gamma=gamma)
        else:
            return tensor

    def __repr__(self):
        repr_str = (f"{self.__class__.__name__} "
                    + f"(gamma_range={self.gamma_range})")
        return repr_str


# class AdjustBrightnessMult(nn.Module):
#     Commented out during cleanup: not used by current YAML augmentation configs.


class AdjustBrightnessShift(nn.Module):

    # will be applied only when creating the dataset for training: TransformFactory.create(train=True)
    apply_test = False

    def __init__(self, brightness_range=(-1, 1), p=0.5):
        """
        Adjust brightness of an image with a Summand given a brightness range.
        Parameters
        ----------
        brightness_range: List[float] or Tuple
            Min and Max value of uniform distribution for sampling the
            brightness factor to increase/reduce with. Example: `[-1, 1]` will
            sample for `brightness_factor` -values between `-1` and `1`
        p: float
            Probability of applying brightness adjustment on an image
        """
        super().__init__()
        self.brightness_range = brightness_range
        self.p = p

    def forward(self, tensor):
        if random.random() < self.p:
            brightness_factor = random.uniform(self.brightness_range[0],
                                               self.brightness_range[1])
            return tensor + brightness_factor
        else:
            return tensor

    def __repr__(self):
        repr_str = (f"{self.__class__.__name__} "
                    + f"(brightness_range={self.brightness_range})")
        return repr_str


# class AdjustSharpness(nn.Module):
#     Commented out during cleanup: not used by current YAML augmentation configs.


class AdjustContrast(nn.Module):

    # will be applied only when creating the dataset for training: TransformFactory.create(train=True)
    apply_test = False

    def __init__(self, contrast_range=(1.5, 2.0), p=0.5):
        """
        Adjust contrast of an image. Only work for RGB-images.
        Parameters
        ----------
        contrast_range: List[float] or Tuple
            Min and Max value of uniform distribution for sampling the
            contrast factor. Example: `[0.85, 1.15]` will sample
            for `contrast_factor`-values between `0.85` and `1.15`
        p: float
            Probability of rotation an image
        """
        super().__init__()
        self.contrast_range = contrast_range
        self.p = p

    def forward(self, tensor):
        if random.random() < self.p:
            contrast_factor = random.uniform(self.contrast_range[0],
                                             self.contrast_range[1])
            return transforms.functional.adjust_contrast(tensor,
                                                         contrast_factor)
        else:
            return tensor

    def __repr__(self):
        repr_str = (f"{self.__class__.__name__} "
                    + f"(contrast_range={self.contrast_range})")
        return repr_str


# class ApplyRandomRotation(nn.Module):
#     Commented out during cleanup: not used by current YAML augmentation configs.


class ApplyInversion(nn.Module):

    # will be applied only when creating the dataset for training: TransformFactory.create(train=True)
    apply_test = False

    def __init__(self, p=0.5):
        """
        Invert the colors of an image.
        Parameters
        ----------
        "p": float
        Probability of applying the color inversion.
        """
        super().__init__()
        self.p = p

    def forward(self, tensor):
        if random.random() < self.p:
            return transforms.functional.invert(img=tensor)
        else:
            return tensor

    def __repr__(self):
        return f"{self.__class__.__name__} (p={self.p})"


class ApplyGaussianBlur(nn.Module):

    # will be applied only when creating the dataset for training: TransformFactory.create(train=True)
    apply_test = False

    def __init__(self, kernel_size=(), sigma=(0.5, 1.2), p=0.5):
        """
        Apply Gaussian blurring by given kernel on image.
        Parameters
        ----------
        "p": float
        Probability of gaussian blurring.
        "kernel_size": List[int]
        Gaussian kernel size.
        """
        super().__init__()
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, tensor):
        if random.random() < self.p:
            return v2.GaussianBlur(kernel_size=self.kernel_size,
                                   sigma=self.sigma)(tensor)
        else:
            return tensor

    def __repr__(self):
        return f"{self.__class__.__name__} (p={self.p})"


class ApplyGaussianNoise(nn.Module):

    # will be applied only when creating the dataset for training: TransformFactory.create(train=True)
    apply_test = False

    def __init__(self, mean=0., sigma=0.02, p=0.5):
        """
        Add Gaussian noise to an image.
        Parameters
        ----------
        mean: float
        std: float
        p: float
            Probability of adding Gaussian noise on an image
        """
        super().__init__()
        self.sigma = sigma
        self.mean = mean
        self.p = p

    def forward(self, tensor):
        if random.random() < self.p:
            return v2.GaussianNoise(mean=self.mean,
                                    sigma=self.sigma)(tensor)
        else:
            return tensor

    def __repr__(self):
        return f"{self.__class__.__name__} (mean={self.mean}," \
               f"sigma={self.sigma})"


class TransformFactory:
    def __init__(self, augm_params, mean=None, std=None, convert_to_tensor=True, brightness_factor=1.0):
        """
        Transform Factory to apply augmentations on image.
        It follows the order of augmentations in the augm_params.
        Augmentations are applied in the "train" mode (TransformFactory.create(train=True)) or when transformation
        class method attribute "apply_test" is True

        If mean or std is given adds normalization as a last transformation
        If convert_to_tensor is True (default) adds conversion to tensor as a first transformation

        Parameters
        ----------
        augm_params: dictionary of augmentations to apply on image.
        mean: mean value to normalize
        std: std value to normalize
        convert_to_image: if True adds conversion to Tensor as first transformation

        """
        self.augm_params = augm_params

        self.normalize = True
        if (mean is None) and (std is None):
            self.normalize = False
        elif mean is None:
            self.mean = 0.0
        elif std is None:
            self.std = 1.0
        else:
            self.mean = mean
            self.std = std

        if (brightness_factor is None) or (brightness_factor == 1):
            self.brightness_correction = False
            self.brightness_factor = 1
        else:
            self.brightness_correction = True
            self.brightness_factor = brightness_factor


        self.convert_to_tensor = convert_to_tensor

    def create(self, train: bool) -> nn.Sequential:

        transform_list = []
        if self.convert_to_tensor:
            transform_list.append(ApplyToTensor())

        # if self.normalize:
        #     transform_list.append(ApplyNormalize(mean=self.mean, std=self.std))

        for augm_name, augm_kwargs in self.augm_params.items():

            augmentation_class = globals()[augm_name]
            if train or augmentation_class.apply_test:
                # # ApplyNormalize need the computed mean and std
                # if augm_name == 'ApplyNormalize':
                #     augmentation = augmentation_class(mean=self.mean, std=self.std)
                # else:
                augmentation = augmentation_class(**augm_kwargs)
                transform_list.append(augmentation)


        if self.brightness_correction:
            transform_list.append(BrightnessCorrection(brightness_factor=self.brightness_factor))

        if self.normalize:
            transform_list.append(ApplyNormalize(mean=self.mean, std=self.std))

        return nn.Sequential(*transform_list)
