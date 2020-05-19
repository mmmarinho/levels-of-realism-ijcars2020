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
import random
import numpy
import PIL
from conversions import numpy_array_to_pil_image


class AugmentationBrightness:
    def __init__(self, min, max, color_mode):
        if min < 0.0 or max > 2.0:
            raise ValueError('Min should be > 0.0 and max < 2.0')
        self.min = min
        self.max = max
        self.color_mode = color_mode

    def augment(self, image, label):
        pil_image = numpy_array_to_pil_image(image, self.color_mode)
        enhancer = PIL.ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(random.uniform(self.min, self.max))
        ret_image = numpy.array(pil_image, dtype='float32') / 255
        return ret_image, label
