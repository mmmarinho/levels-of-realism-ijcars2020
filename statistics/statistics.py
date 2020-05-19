"""
The Effects of Different Levels of Photorealism on the Training of CNNs with only Synthetic Images for the Semantic Segmentation of Robotic Instruments in a Head Phantom
Copyright (C) 2019 Murilo Marques Marinho

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
"""

import numpy as np


def get_binary_iou(a, b, threshold=0.5):
    """
    Gets the intersection over union between two binary images (composed of zeros or ones).
    :param a: a numpy image with size=(height, width, 1), pixel depth float32 between 0 and 1.
    :param b: same type as b.
    :param threshold: values above the threshold will be considered as ones and below as zeros.
    :return: (zero_iou, one_iou) which is the IOU of the class 0 and the IOU for class 1.
    """
    if a.shape != b.shape:
        raise ValueError('The shape of a and b have to be the same {}!={}'.format(a.shape, b.shape))

    zero_intersection = np.sum(a < threshold and b < threshold)
    zero_union = np.sum(a < threshold or b < threshold)

    ones_intersection = np.sum(a > threshold and b > threshold)
    ones_union = np.sum(a > threshold or a > threshold)

    return (zero_intersection/zero_union), (ones_intersection/ones_union)


