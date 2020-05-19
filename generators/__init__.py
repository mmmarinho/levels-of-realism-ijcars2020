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

import pathlib
import random
import numpy
from io import *


def _get_sequential_batch(incremental_index,
                          batch_size,
                          shuffle,
                          sample_size,
                          training_images_paths,
                          training_labels_paths,
                          image_size,
                          augmentations,
                          color_mode):
    """
    If shuffle is False, sequentially move through the dataset (in alphabetical order of the files), if shuffle is True
    get a batch of randomly sampled images (with replacement).

    :param incremental_index: an index that persists between executions of this function. Useful when the data has to
    be evaluated sequentially.
    :param batch_size: an integer representing the batch size, e.g. 16.
    :param shuffle: True for randomly moving through the dataset and False to going sequentially.
    :param sample_size: the length of training_images_paths (so that we don't have to calculate it all the time.
    :param training_images_paths: a list of paths to the images in the dataset.
    :param training_labels_paths: a list of paths to the labels in the dataset.
    :param image_size: a tuple in the form (height, width), e.g. (256, 256).
    :param augmentations: the list of augmentations to be applied to the data. See the augmentation module.
    :param color_mode should be 'grey' or 'color'

    :return: A numpy array with dimension (batch_size, height, width, 1) ready to be used for fitting in a Keras model.
    """

    # Run for a single batch
    for batch_counter in range(0, batch_size):

        # Get a random index
        if shuffle:
            random_sample_index = random.randint(0, sample_size - 1)
            image_path = training_images_paths[random_sample_index]
            label_path = training_labels_paths[random_sample_index]
        # Or not
        else:
            image_path = training_images_paths[incremental_index]
            label_path = training_labels_paths[incremental_index]
            incremental_index += 1
            if incremental_index >= sample_size:
                incremental_index = 0

        # Load, resize, and cast images to float
        if color_mode == 'grey':
            image = open_greyscale_8bit_image_resize_and_convert_to_numpy_float_array(image_path, image_size)
        else:
            image = open_color_24bit_image_resize_and_convert_to_numpy_float_array(image_path, image_size)
        label = open_greyscale_8bit_image_resize_and_convert_to_numpy_float_array(label_path, image_size)

        # Check image and label
        if image.shape[0:2] != label.shape[0:2]:
            # The first two dimensions have to be compatible
            raise Exception('Image and label shapes are not compatible {} != {}'.format(
                image.shape[0:2],
                label.shape[0:2]))
        if image_path.stem != label_path.stem:
            raise Exception('Image and label do not have the same filename {} != {}'.format(image_path.stem,
                                                                                            label_path.stem))

        # Augment
        for augmentation in augmentations:
            image, label = augmentation.augment(image, label)

        # Threshold label
        label[label > 0.5] = 1.0
        label[label <= 0.5] = 0.0

        # Stack into a batch
        if batch_counter == 0:
            image_batch = image
            label_batch = label
        else:
            image_batch = numpy.concatenate((image_batch, image), axis=0)
            label_batch = numpy.concatenate((label_batch, label), axis=0)

    if color_mode == 'grey':
        (height, width) = image.shape
        channel = 1
    else:
        (height, width, channel) = image.shape
    return incremental_index, \
           image_batch.reshape(batch_size, height, width, channel), \
           label_batch.reshape(batch_size, height, width, 1)


def get_size_of_data(path):
    training_images_dir = pathlib.Path(path / 'image')
    training_images_paths = list(training_images_dir.glob('*.png'))

    training_labels_dir = pathlib.Path(path / 'label')
    training_labels_paths = list(training_labels_dir.glob('*.png'))

    if len(training_images_paths) != len(training_labels_paths):
        raise Exception('The same number of images and labels are required {} != {}'.format(
            len(training_images_paths),
            len(training_labels_paths)
        ))
    return len(training_images_paths)


def image_generator(path,
                    image_size,
                    shuffle=True,
                    augmentations=[],
                    batch_size=1,
                    batch_generation_mode='',
                    color_mode='grey'):
    """ Receives the path, loads, resizes, and converts the images on the subfolders
       "image" and "label" into a meaningful format for tensorflow training.

       :param path a pathlib.Path() instance with the path to the parent folder of the "image" and "label" folders
       :param image_size a tuple with the size of the image. e.g (256, 256)
       :param shuffle defines whether the inputs will be returned in shuffled order or not
       :param augmentations a list of augmentation instances. Anything with an .augment() method will work.
       :param batch_size the size of the batch that will be returned by the generator
       :param batch_generation_mode The batch generation mode (e.g. 'sequential')
       :param color_mode should be 'grey' or 'color'

       ::return a tuple (image, label) in which each element of the tuple has size (batch_size, image_size[0],
       image_size[1], 1)

       ::except Number of images in "image" is different from the number of images in "label"
       ::except Size of an "image" is different from its respective "label"
       ::except The filename of an "image" is different from its respective "label"
       ::except ValueError if color_mode is not 'grey' or 'color'
       ::except RuntimeError if no images were found
       """

    if color_mode != 'grey' and color_mode != 'color':
        raise ValueError("color_mode needs to be either 'grey' or 'color'")

    training_images_dir = pathlib.Path(path / 'image')
    training_images_paths = list(training_images_dir.glob('*.png'))

    training_labels_dir = pathlib.Path(path / 'label')
    training_labels_paths = list(training_labels_dir.glob('*.png'))

    if len(training_images_paths) != len(training_labels_paths):
        raise Exception('The same number of images and labels are required {} != {}'.format(
            len(training_images_paths),
            len(training_labels_paths)
        ))
    sample_size = len(training_images_paths)
    if sample_size < 1:
        raise RuntimeError(
            'No images were found in {} and/or {}'.format(str(training_images_dir), str(training_labels_dir)))

    print('Found {} images and {} labels.'.format(len(training_images_paths), len(training_labels_paths)))
    print('The batch generation mode is {}.'.format(str(batch_generation_mode)))

    incremental_index = 0
    while True:

        if batch_generation_mode == 'sequential':
            incremental_index, image_batch, label_batch = _get_sequential_batch(
                incremental_index=incremental_index,
                batch_size=batch_size,
                shuffle=shuffle,
                sample_size=sample_size,
                training_images_paths=training_images_paths,
                training_labels_paths=training_labels_paths,
                image_size=image_size,
                augmentations=augmentations,
                color_mode=color_mode)
        else:
            raise Exception('Unknown batch_generation_mode={}'.format(batch_generation_mode))

        yield image_batch, label_batch
