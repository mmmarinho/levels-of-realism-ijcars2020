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
import numpy


def numpy_array_to_pil_image(image, color_mode):
    if color_mode == 'grey':
        return one_channel_numpy_array_to_pil_image(image)
    elif color_mode == 'color':
        return three_channel_numpy_array_to_pil_image(image)
    else:
        raise ValueError('Unknown color_mode={}'.format(color_mode))


def pil_image_to_numpy_array(pil_image):
    return numpy.array(pil_image, dtype='float32') / 255


def one_channel_numpy_array_to_pil_image(image):
    return PIL.Image.fromarray((image * 255).astype('uint8'), 'L')


def three_channel_numpy_array_to_pil_image(image):
    return PIL.Image.fromarray((image * 255).astype('uint8'), 'RGB')
