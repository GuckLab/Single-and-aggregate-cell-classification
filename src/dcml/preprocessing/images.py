import numpy as np
import numpy.typing as npt


def correct_bg_on_image(
        image: npt.NDArray[np.uint8],
        image_bg: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Subtract image background and readjust value range.

    Parameters
    ---------
    image: npt.NDArray
        Image of cell of dimension 2
    image_bg: npt.NDArray
        Background image of dimension 2
    Returns
    -------
    image_corr: npt.NDArray[np.int8]
        Background corrected image of dimension 2
    """
    if image.ndim != 2 or image_bg.ndim != 2:
        raise ValueError("Input images need to be of dimension 2!")

    if image.shape != image_bg.shape:
        raise ValueError("'image' and 'image_bg' need to have the same shape!")

    bg_mean = image_bg.mean(axis=(-2, -1))
    bg_mean = np.rint(bg_mean).astype(np.int16)

    image_16 = image.astype(np.int16)
    image_bg_16 = image_bg.astype(np.int16)

    image_corr_16 = image_16 - image_bg_16
    image_corr_16 += bg_mean
    image_corr = np.clip(image_corr_16, 0, 255).astype(np.uint8)

    return image_corr


# def _subtract_rolling_median(images_slice, kernel_size=100):
#     Commented out during cleanup: unused in current train/evaluate flow.


# def _subtract_rolling_median_on_mp_array(row_idx, images_slice, kernel_size=100):
#     Commented out during cleanup: unused in current train/evaluate flow.


# def init_mp_global(arr):
#     Commented out during cleanup: unused in current train/evaluate flow.


# def _correct_background_on_subset(images, kernel_size=100, processes=None, use_mp_array=True):
#     Commented out during cleanup: unused in current train/evaluate flow.


# def correct_background(images, kernel_size=100, chunk_size=10000, processes=None, use_mp_array=True):
#     Commented out during cleanup: unused in current train/evaluate flow.

#    images_corr = np.array(fpr)
#    # Close internal mmap and delete numpy.memmap before unlinking.
#    # Otherwise Windows OS will throw PermissionError WinError32
#    # delete reference to np.memmap from variable `fpr`
#    fpr._mmap.close()
#    del fpr
#
#    # Delete np.memmap-file
#    temp_filename.unlink()
#
#    return images_corr


def event_coordinates(pos_x,
                      pos_y,
                      crop_size,
                      pixel_size,
                      max_x=250,
                      max_y=80):
    """Returns center coordinates of all cells

    If the center of the cell is too close to boundary, causing the crop
    to go out-of-bounds, it will shift the coordinates such that the crop
    starts at the boundary.

    Parameters
    ----------
    pos_x: np.ndarray
        array containing position-x and of ROI-center
    pos_y: np.ndarray
        array containing position-y and of ROI-center
    crop_size: int for square patch or tuple for rectangular patch
    with first entry corresponding to the width and
        second the height of the resulting cropped image
        Sets parameter of crop size, so for boundary cases it will shift
        computed coordinates accordingly
    pixel_size: float
        length of pixel in micrometer
    max_x: int
        maximum number of pixels along x axis
    max_y: int
        maximum number of pixels along y axis
    Returns
    -------
    pxs : array_like
        Array containing the x-coordinates of the cropped images
    pys : array_like
        Array containing the y-coordinates
    """

    if isinstance(crop_size, int):
        sh_x = crop_size//2
        sh_y = crop_size//2
    else:
        sh_x = crop_size[0] // 2
        sh_y = crop_size[1] // 2

    pxs = pos_x/pixel_size
    pys = pos_y/pixel_size

    # If necessary shift image, so that complete crop is contained in image
    pxs = np.minimum(np.maximum(pxs, sh_x),
                     max_x-sh_x).astype(np.int16)
    pys = np.minimum(np.maximum(pys, sh_y),
                     max_y-sh_y).astype(np.int16)
    return pxs, pys


def crop_image(im, px, py, crop_size):
    """Crops a single image

    Parameter
    ---------
    im: np.ndarray
    px: int
        x-position of center of the event in given image
    py: int
        y-position of center of the event in given image
    crop_size: int for square patch or tuple for rectangular patch
    with first entry corresponding to the width and
    second the height of the resulting cropped image

    Returns
    -------
    np.ndarray
        2D-Array of cropped image
    """
    if isinstance(crop_size, int):
        sh_x = crop_size
        sh_y = crop_size
    else:
        sh_x = crop_size[0]
        sh_y = crop_size[1]

    sh_x = sh_x//2
    sh_y = sh_y//2

    im_cropped = im[py-sh_y:py+sh_y,
                    px-sh_x:px+sh_x]
    return im_cropped


# def crop_images(images, pxs, pys, crop_size, processes=None):
#     Commented out during cleanup: unused in current train/evaluate flow.


# def crop(ds, crop_size=64, bg_corr=False, processes=None, use_mp_array=True):
#     Commented out during cleanup: unused in current train/evaluate flow.
