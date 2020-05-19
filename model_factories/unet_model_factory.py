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
import models.unet


class UnetModelFactory:
    def __init__(self, batch_size, optimizer):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.name = 'unet'

    def get_model(self, input_size=None, pretrained_weights=None):
        return models.unet.get_model(pretrained_weights=pretrained_weights, input_size=input_size,
                                     optimizer=self.optimizer, show_summary=False)

    def get_name(self):
        return self.name + '_' + str(self.batch_size) + '_' + str(self.optimizer.get_config()['learning_rate'])

    def get_batch_size(self):
        return self.batch_size

    def __str__(self):
        return self.get_name()
