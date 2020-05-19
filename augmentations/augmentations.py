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

import PIL
import PIL.ImageEnhance
import random
import numpy


def _numpy_array_to_pil_image(image, color_mode):
    if color_mode == 'grey':
        return _one_channel_numpy_array_to_pil_image(image)
    elif color_mode == 'color':
        return _three_channel_numpy_array_to_pil_image(image)
    else:
        raise ValueError('Unknown color_mode={}'.format(color_mode))


def _one_channel_numpy_array_to_pil_image(image):
    return PIL.Image.fromarray((image * 255).astype('uint8'), 'L')


def _three_channel_numpy_array_to_pil_image(image):
    return PIL.Image.fromarray((image * 255).astype('uint8'), 'RGB')


#class GSAugmentationImageNormalizer:
#    def augment(self, image, label):
#        normalized_image = numpy.divide(numpy.subtract(image, numpy.average(image)), numpy.std(image))
#        return normalized_image, label

class AugmentationSharpness:
    def __init__(self, min, max, color_mode):
        if min < 0.0 or max > 2.:
            raise ValueError('Min should be => 0.0 and max <= 2')
        self.min = min
        self.max = max
        self.color_mode = color_mode

    def augment(self, image, label):
        pil_image = _numpy_array_to_pil_image(image, self.color_mode)
        enhancer = PIL.ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(random.uniform(self.min, self.max))
        ret_image = numpy.array(pil_image, dtype='float32') / 255
        return ret_image, label


class AugmentationContrast:
    def __init__(self, min, max, color_mode):
        if min < 0.0 or max > 1.:
            raise ValueError('Min should be => 0.0 and max <= 1.0')
        self.min = min
        self.max = max
        self.color_mode = color_mode

    def augment(self, image, label):
        pil_image = _numpy_array_to_pil_image(image, self.color_mode)
        enhancer = PIL.ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(random.uniform(self.min, self.max))
        ret_image = numpy.array(pil_image, dtype='float32') / 255
        return ret_image, label


class AugmentationColor:
    def __init__(self, min, max, color_mode):
        if min < 0.0 or max > 1.:
            raise ValueError('Min should be => 0.0 and max <= 1.0')
        self.min = min
        self.max = max
        self.color_mode = color_mode

    def augment(self, image, label):
        pil_image = _numpy_array_to_pil_image(image, self.color_mode)
        enhancer = PIL.ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(random.uniform(self.min, self.max))
        ret_image = numpy.array(pil_image, dtype='float32') / 255
        return ret_image, label


class AugmentationBrightness:
    def __init__(self, min, max, color_mode):
        if min < 0.0 or max > 2.0:
            raise ValueError('Min should be > 0.0 and max < 2.0')
        self.min = min
        self.max = max
        self.color_mode = color_mode

    def augment(self, image, label):
        pil_image = _numpy_array_to_pil_image(image, self.color_mode)
        enhancer = PIL.ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(random.uniform(self.min, self.max))
        ret_image = numpy.array(pil_image, dtype='float32') / 255
        return ret_image, label


class AugmentationPixelWiseRandom:
    def __init__(self, min, max, distribution='uniform'):

        if min < -1.0 or max > 1.0:
            raise ValueError(
                'Error initializing {}. min should be > -1.0 (min={}) and max should be < 1.0 (max={})'.format(
                    self.__name__,
                    min,
                    max))

        self.min = float(min)
        self.max = float(max)
        self.distribution = distribution

    def augment(self, image, label):
        # Add random noise
        if self.distribution == 'uniform':
            augmented_image = image + numpy.random.uniform(low=self.min, high=self.max, size=image.shape)
        else:
            raise ValueError('Unknown distribution={}'.format(self.distribution))
        # Clamp pixel values to be within the acceptable range
        augmented_image[augmented_image > 1.0] = 1.0
        augmented_image[augmented_image < 0.0] = 0.0
        return augmented_image, label


class AugmentationAffine:
    def __init__(self, angle_min, angle_max, translation_min, translation_max, color_mode):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.translation_min = translation_min
        self.translation_max = translation_max
        self.color_mode = color_mode

    def augment(self, image, label):
        angle = random.randint(self.angle_min,
                               self.angle_max)
        translation = random.randint(self.translation_min[0], self.translation_max[0]), random.randint(
            self.translation_min[1], self.translation_max[1])

        pil_ret_image = _numpy_array_to_pil_image(image, self.color_mode)
        ret_image = numpy.array(pil_ret_image.rotate(angle=angle, translate=translation,
                                                                           fillcolor='black'), dtype='float32') / 255
        pil_ret_label = _numpy_array_to_pil_image(label, 'grey')
        ret_label = numpy.array(pil_ret_label.rotate(angle=angle, translate=translation,
                                                                           fillcolor='black'), dtype='float32') / 255
        return ret_image, ret_label
