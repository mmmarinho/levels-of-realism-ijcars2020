"""
The Effects of Different Levels of Photorealism on the Training of CNNs with only Synthetic Images for the Semantic
Segmentation of Robotic Instruments in a Head Phantom
Copyright (C) 2020 Murilo Marques Marinho

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
"""
import time
import tensorflow as tf
from multiprocessing import Process
from training import train_greyscale_model
from model_factories import UnetModelFactory
from augmentations import *
import yaml

# All the configurable parameters for the overall training
configuration_yaml = open("configuration.yml")
configuration = yaml.load(configuration_yaml, Loader=yaml.FullLoader)

case_names = [
    '1_flat_renderer',
    '2_basic_renderer',
    '3_realistic_renderer',
]

# Models to be trained
model_factories = [
    UnetModelFactory(batch_size=configuration['BATCH_SIZE'],
                     optimizer=tf.keras.optimizers.SGD(configuration['LEARNING_RATE'])),
]

# Augmentations will be applied to the input data in the same order as they are in the list
augmentations = [
    AugmentationAffine(angle_min=-45,
                       angle_max=45,
                       translation_min=(-20, -20),
                       translation_max=(20, 20),
                       color_mode=configuration['GREY_OR_COLOR']),
    AugmentationBrightness(min=0.5,
                           max=1.5,
                           color_mode=configuration['GREY_OR_COLOR']),
    AugmentationPixelWiseRandom(min=-0.1,
                                max=0.1),
]


class TrainingManager:
    def __init__(self, device_name):
        self.device_name = device_name
        self.running = False
        self.subprocess = None


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    subprocess_training_manager_list = []
    for gpu in gpus:
        print('Registering GPU {}...'.format(gpu.name.replace('physical_', '')))
        subprocess_training_manager_list.append(TrainingManager(gpu.name.replace('physical_', '')))

    for case_name in case_names:
        print('Running case {}...'.format(case_name))

        for model_factory in model_factories:
            print('Running model {}...'.format(model_factory))

            for model_iteration in range(configuration['ITERATION_START'],
                                         configuration['ITERATION_START'] + configuration['ITERATION_COUNT']):

                model_assigned_to_gpu = False
                while not model_assigned_to_gpu:
                    for training_manager in subprocess_training_manager_list:
                        if training_manager.subprocess is None or not training_manager.subprocess.is_alive():
                            if training_manager.subprocess is None:
                                print('Running process on {} for the first time.'.format(
                                    training_manager.device_name))
                            else:
                                print('Running process on {}, because subprocess alive={}'.format(
                                    training_manager.device_name,
                                    training_manager.subprocess.is_alive()))
                            training_manager.subprocess = Process(target=train_greyscale_model,
                                                                  args=(
                                                                      training_manager.device_name,
                                                                      configuration,
                                                                      case_name,
                                                                      model_factory,
                                                                      model_iteration,
                                                                      augmentations))
                            training_manager.subprocess.start()
                            model_assigned_to_gpu = True
                            break
                        time.sleep(1)
