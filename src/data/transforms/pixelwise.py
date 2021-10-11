import cv2
import numpy as np

import albumentations.augmentations.functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform, to_tuple

__all__ = ["InstanceNormalize", "ToGrayCustom", "MotionBlur", "OtsuBinarize", "AdaptiveBinarize"]


class InstanceNormalize(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(InstanceNormalize, self).__init__(always_apply, p)

    def apply(self, image, **params):
        mean = image.mean(axis=(0, 1), keepdims=True)
        std = image.std(axis=(0, 1), keepdims=True)
        std[std == 0] = 1
        return F.normalize(image, mean, std, 1)

    def get_transform_init_args_names(self):
        return ()


class ToGrayCustom(ImageOnlyTransform):
    """Convert the input RGB image to grayscale.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class MotionBlur(ImageOnlyTransform):
    """Apply motion blur with a random direction to the image.
    Args:
        kernel_size ((int, int)): range of a size of a kernel that will be applied to imitate motion blur.
                                  Default: (7, 9)
        direction (str): direction of moving the image, 5 values are available:
                         'vertical', 'horizontal', 'diagonal', 'antidiagonal', and None.
                         If None then direction assigned randomly.
                         Default: None.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, kernel_size=(7, 9), direction=None, always_apply=False, p=0.5):
        super(MotionBlur, self).__init__(always_apply, p)
        assert direction in (None, 'vertical', 'horizontal', 'diagonal', 'antidiagonal'), \
            "The value of parameter 'direction' should be one of the following: " \
            "'vertical', 'horizontal', 'diagonal', 'antidiagonal', or None"

        self.kernel_size = to_tuple(kernel_size)
        self.direction = direction

    def apply(self, img, kernel_size=3, **params):
        # get kernel
        kernel = self.get_motion_blur_kernel(kernel_size=kernel_size)
        return cv2.filter2D(img.astype('float32'), -1, kernel=kernel).astype('uint8')

    def get_motion_blur_kernel(self, kernel_size=3):
        """
        Get kernel for motion blur in form:
        [[value, 0, 0],
         [value, 0, 0],
         [value, 0, 0]]
        or
        [[0, 0, 0],
         [value, value, value],
         [0, 0, 0]]
        or
        [[value, 0, 0],
         [0, value, 0],
         [0, 0, value]]
        or
        [[0, 0, value],
         [0, value, 0],
         [value, 0, 0]]
        NB: randomly it could be that diagonal, a certain row, or a certain column to be non-zeroed,
        i.e. direction of moving assigned randomly

        :param kernel_size: int, height and width of kernel for motion blur, default=3
        :return: np.array of shape (kernel_size, kernel_size) of zeros except for one row, column, diagonal
                 or antidiagonal
        """
        value = 1. / kernel_size

        # define zeros kernel
        kernel = np.zeros((kernel_size, kernel_size))

        # randomize if necessary
        if self.direction is None:
            direction = np.random.choice(['vertical', 'horizontal', 'diagonal', 'antidiagonal'],
                                         size=1, p=[1. / 4, 1. / 4, 1. / 4, 1. / 4])[0]
        else:
            direction = self.direction

        # in a quarter of cases the kernel will have non-zero main diagonal
        if direction == 'diagonal':
            kernel = self._get_diagonal_movement(kernel, value)
        # in a quarter of cases the kernel will have non-zero reverse diagonal
        elif direction == 'antidiagonal':
            kernel = self._get_antidiagonal_movement(kernel, value)
        # in a quarter of cases the kernel will have non-zero column
        elif direction == 'vertical':
            kernel = self._get_vertical_movement(kernel, value)
        # in other quarter of cases the kernel will have non-zero row
        elif direction == 'horizontal':
            kernel = self._get_horizontal_movement(kernel, value)

        return kernel

    def get_params(self):
        return {"kernel_size": np.random.randint(self.kernel_size[0], self.kernel_size[1])}

    @staticmethod
    def _get_horizontal_movement(kernel, value):
        """
        Get kernel for motion blur in form:
        [[0, 0, 0],
         [0, 0, 0],
         [value, value, value]]
        i.e. all-zeroed kernel with one randomly assigned non-zeroed row
        :param kernel: np.array of shape (H, H) of zeros
        :param value: float, specific value for a non-zero values of a kernel, equal 1 / kernel_size.
        :return: np.array of shape (H, H) of zeros except for one row
        """
        row = np.random.randint(low=0, high=kernel.shape[0])
        kernel[row] = value
        return kernel

    @staticmethod
    def _get_vertical_movement(kernel, value):
        """
        Get kernel for motion blur in form:
        [[value, 0, 0],
         [value, 0, 0],
         [value, 0, 0]]
        i.e. all-zeroed kernel with one randomly assigned non-zeroed column
        :param kernel: np.array of shape (H, H) of zeros
        :param value: float, specific value for a non-zero values of a kernel, equal 1 / kernel_size.
        :return: np.array of shape (H, H) of zeros except for one column
        """
        col = np.random.randint(low=0, high=kernel.shape[1])
        kernel[..., col] = value
        return kernel

    @staticmethod
    def _get_diagonal_movement(kernel, value):
        """
        Get kernel for motion blur in form:
        [[value, 0, 0],
         [0, value, 0],
         [0, 0, value]]
        i.e. diagonal matrix
        :param kernel: np.array of shape (H, H) of zeros
        :param value: float, specific value for a non-zero values of a kernel, equal 1 / kernel_size.
        :return: np.array of shape (kernel_size, kernel_size) of zeros except for diagonal
        """
        np.fill_diagonal(kernel, value)
        return kernel

    def _get_antidiagonal_movement(self, kernel, value):
        """
        Get kernel for motion blur in form:
        [[0, 0, value],
         [0, value, 0],
         [value, 0, 0]]
        i.e. antidiagonal matrix
        :param kernel: np.array of shape (H, H) of zeros
        :param value: float, specific value for a non-zero values of a kernel, equal 1 / kernel_size.
        :return: np.array of shape (kernel_size, kernel_size) of zeros except for antidiagonal
        """
        return self._get_diagonal_movement(kernel, value)[..., ::-1]


class OtsuBinarize(ImageOnlyTransform):
    """Convert the input RGB image to binary form with OTSU threshold.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, otsu_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_img = cv2.cvtColor(otsu_img, cv2.COLOR_GRAY2RGB)

        return otsu_img


class AdaptiveBinarize(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        bin_image = cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 2) * 255
        bin_image = np.dstack([bin_image] * image.shape[-1])
        return bin_image

    def get_transform_init_args_names(self):
        return ()
