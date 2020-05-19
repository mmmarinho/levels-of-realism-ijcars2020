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
from conversions import numpy_array_to_pil_image


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

        pil_ret_image = numpy_array_to_pil_image(image, self.color_mode)
        ret_image = numpy.array(pil_ret_image.rotate(angle=angle, translate=translation,
                                                     fillcolor='black'), dtype='float32') / 255
        pil_ret_label = numpy_array_to_pil_image(label, 'grey')
        ret_label = numpy.array(pil_ret_label.rotate(angle=angle, translate=translation,
                                                     fillcolor='black'), dtype='float32') / 255
        return ret_image, ret_label
