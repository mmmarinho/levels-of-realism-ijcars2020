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
import numpy


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
