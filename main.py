"""
The Effects of Different Levels of Photorealism on the Training of CNNs with only Synthetic Images for the Semantic
Segmentation of Robotic Instruments in a Head Phantom
Copyright (C) 2019 Murilo Marques Marinho

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
from unet.training import train_greyscale_model
from unet.model_unet import UNETModelFactory
from unet.augmentations import *

# All the configurable parameters for the overall training
configuration = {
    'ITERATION_START': 0,  # Initial iteration number
    'ITERATION_COUNT': 20,  # Number of iterations of each model
    'EPOCHS': 40,  # Number of epochs, in practice the number of checkpoints
    'STEPS_PER_EPOCH': 1000,  # Number of training steps before each checkpoint
    'EPOCHS_TO_REDUCE_LR': 15,  # Number of epochs after which the LR will be divided by 'LR DIVISOR'
    'LR_DIVISOR': 2.,  # The number the learning rate will be divided by after 'EPOCHS_TO_REDUCE_LR'
    'VALIDATION_STEPS': 1,  # Validate through the data only once
    'TRAIN_MODEL': True,  # Train the model or load an existing file
    'DEBUG_OUTPUT_IMAGES': False,  # Show output images while the network is being trained
    'PRINT_OUTPUT_IMAGES': True,  # Print the output images while the network is being trained
    'FIT_VERBOSITY': 2,  # Verbosity of the Keras fit_generator function
    'BATCH_GENERATION_MODE': 'sequential',  # The batch generation mode
    'TRAINING_PATH': 'data/',  # The root path where your 'case_names' are. E.g. 'data/'
    'RESULTS_PATH': 'data/',  # Where to save the training results. E.g. 'data/'
    'VALIDATION_PATH': 'database/merge/',  # Where your validation images are stored. E.g. 'database/05/cropped/'
    'IMAGE_SIZE': (256, 256),  # Only height and width, (e.g. (256, 256) ) Channels are defied by 'GREY_OR_COLOR'
    'GREY_OR_COLOR': 'grey'  # Either 'grey' (8bit 1 channel) or 'color' (8bit 3 channel). Raises Exception if wrong
}

# Simulation dataset names
case_names = [
    #'1_membrane_flat_circle',
    #'2_membrane_basic_circle',
    #'3_membrane_realistic_circle',
    #'4_membrane_realistic_circle_withCameraVariation',
    #"5_membrane_photo_background_circle",
    "6_membrane_photo_background_tool_circle",
    "7_membrane_photo_tool_circle"
]


# Models to be trained
model_factories = [
    UNETModelFactory(name='unet', batch_size=8, optimizer=tf.keras.optimizers.SGD(lr=0.02)),
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


class SubProcessTrainingManager:
    def __init__(self, device_name):
        self.device_name = device_name
        self.running = False
        self.subprocess = None


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    subprocess_training_manager_list = []
    for gpu in gpus:
        print('Registering GPU {}...'.format(gpu.name.replace('physical_', '')))
        subprocess_training_manager_list.append(SubProcessTrainingManager(gpu.name.replace('physical_', '')))

    for case_name in case_names:
        print('Running case {}...'.format(case_name))

        for model_factory in model_factories:
            print('Running model {}...'.format(model_factory))

            for model_iteration in range(configuration['ITERATION_START'],
                                         configuration['ITERATION_START'] + configuration['ITERATION_COUNT']):

                model_assigned_to_gpu = False
                while not model_assigned_to_gpu:
                    for subprocess_training_manager in subprocess_training_manager_list:
                        if subprocess_training_manager.subprocess is None or not subprocess_training_manager.subprocess.is_alive():
                            if subprocess_training_manager.subprocess is None:
                                print('Running process on {} for the first time.'.format(
                                    subprocess_training_manager.device_name))
                            else:
                                print('Running process on {}, because subprocess alive={}'.format(
                                    subprocess_training_manager.device_name,
                                    subprocess_training_manager.subprocess.is_alive()))
                            subprocess_training_manager.subprocess = Process(target=train_greyscale_model,
                                                                             args=(
                                                                                 subprocess_training_manager.device_name,
                                                                                 configuration,
                                                                                 case_name,
                                                                                 model_factory,
                                                                                 model_iteration,
                                                                                 augmentations))
                            subprocess_training_manager.subprocess.start()
                            model_assigned_to_gpu = True
                            break
                        time.sleep(1)
