"""
The Effects of Different Levels of Photorealism on the Training of CNNs with only Synthetic Images for the Semantic Segmentation of Robotic Instruments in a Head Phantom
Copyright (C) 2020 Murilo Marques Marinho

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
"""
import PIL
from conversions import pil_image_to_numpy_array


def open_greyscale_8bit_image_resize_and_convert_to_numpy_float_array(image_path, image_size):
    """
    Loads the image at image_path, checks if it is 8bit depth, resizes the image to image_size, converts
    it to a numpy float32 array, and divides it by 255 to give each bit a value between 0 and 1.
    :param image_path: the path to the image as a string (e.g. 'data/image.png')
    :param image_size: a tuple with the desired image size (e.g. (256, 256))
    :return: a float32 numpy array version of the image
    :except: raises a ValueError if the image is not 8bit
    """
    pil_image = PIL.Image.open(image_path).resize(image_size)
    if pil_image.mode != 'L':
        raise ValueError(
            'The input image at {} has incorrect mode={}. Check if it is a 8bit greyscale image'.format(image_path,
                                                                                                        pil_image.mode))
    return pil_image_to_numpy_array(pil_image)


def open_color_24bit_image_resize_and_convert_to_numpy_float_array(image_path, image_size):
    """
    Loads the image at image_path, checks if it is 24bit depth, resizes the image to image_size, converts
    it to a numpy float32 array, and divides it by 255 to give each bit a value between 0 and 1.
    :param image_path: the path to the image as a string (e.g. 'data/image.png')
    :param image_size: a tuple with the desired image size (e.g. (256, 256, 3))
    :return: a float32 numpy array version of the image
    :except: raises a ValueError if the image is not 24bit
    """
    pil_image = PIL.Image.open(image_path).resize(image_size)
    if pil_image.mode != 'RGB':
        raise ValueError(
            'The input image at {} has incorrect mode={}. Check if it is a 8bit greyscale image'.format(image_path,
                                                                                                        pil_image.mode))
    return pil_image_to_numpy_array(pil_image)
