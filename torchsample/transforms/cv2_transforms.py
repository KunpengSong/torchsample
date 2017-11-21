"""
CV2 Augmentations
Source: https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
"""

import random

import cv2
import numpy as np
import torch


class TensorToCV2Image(object):
    """Convert tensor to cv2 image."""

    def __init__(self):
        """Init transform."""
        pass

    def __call__(self, tensor):
        """Call transform."""
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))


class CV2ImageToTensor(object):
    """Convert cv2 image to tensor."""

    def __init__(self):
        """Init transform."""
        pass

    def __call__(self, cvimage):
        """Call transform."""
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)


class ConvertFromInts(object):
    """Convert the type of cv2 image from uint8 to float32."""

    def __init__(self):
        """Init transform."""
        pass

    def __call__(self, image):
        """Call transform."""
        return image.astype(np.float32)


class ConvertColor(object):
    """convert coloe space of the input cv2 image."""

    def __init__(self, current="BGR", transform="HSV"):
        """Init transform."""
        self.transform = transform
        self.current = current

    def __call__(self, img):
        """Call transform."""
        if self.current == "BGR" and self.transform == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == "HSV" and self.transform == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        elif self.current == "RGB" and self.transform == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise NotImplementedError
        return img


class Resize(object):
    """Resize cv2 image."""

    def __init__(self, size=(224, 224)):
        """Init transform."""
        self.size = size

    def __call__(self, img):
        """Call transform."""
        img = cv2.resize(img, self.size)
        return img


class RandomContrast(object):
    """Alter random contrast on the input cv2 image."""

    def __init__(self, lower=0.5, upper=1.5):
        """Init transform."""
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, img):
        """Call transform."""
        if random.randint(0, 1):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img


class RandomBrightness(object):
    """Alter random brightness on the input cv2 image."""

    def __init__(self, delta=32):
        """Init transform."""
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img):
        """Call transform."""
        if random.randint(0, 1):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img


class RandomSaturation(object):
    """Alter random saturation on the input cv2 image."""

    def __init__(self, lower=0.5, upper=1.5):
        """Init transform."""
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img):
        """Call transform."""
        if random.randint(0, 1):
            img[:, :, 1] *= random.uniform(self.lower, self.upper)

        return img


class RandomHue(object):
    """Alter random hue on the input cv2 image."""

    def __init__(self, delta=18.0):
        """Init transform."""
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, img):
        """Call transform."""
        if random.randint(0, 1):
            img[:, :, 0] += random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
        return img


# ===================== #
# Source: https://github.com/EKami/carvana-challenge/blob/master/src/img/augmentation.py
def random_hue_saturation_value_cv2(image, hue_shift_limit=(-180, 180),
                                sat_shift_limit=(-255, 255),
                                val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def random_shift_scale_rotate_cv2(image, mask,
                              shift_limit=(-0.0625, 0.0625),
                              scale_limit=(-0.1, 0.1),
                              rotate_limit=(-45, 45), aspect_limit=(0, 0),
                              borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def random_horizontal_flip_cv2(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def random_saturation_cv2(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
    return img


def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img


def random_gray(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return img


def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img