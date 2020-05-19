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
import os
import pathlib
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf

from generators import image_generator, get_size_of_data


def train_greyscale_model(device, configuration, case_name, model_factory, model_iteration, augmentations):
    with tf.device(device):

        # Some helpful variables
        image_size = configuration['IMAGE_SIZE']  # e.g. (256, 256)
        if configuration['GREY_OR_COLOR'] == 'grey':
            image_size_with_channel_and_batch = (1,) + image_size + (1,)  # e.g. (1, 256, 256, 1)
        else:
            image_size_with_channel_and_batch = (1,) + image_size + (3,)  # e.g. (1, 256, 256, 3)

        highest_metrics = {
            'binary_accuracy': 0.,
            'val_binary_accuracy': 0.,
            'precision': 0.,
            'val_precision': 0.,
            'recall': 0.,
            'val_recall': 0.,
        }

        print('Iteration {}...'.format(model_iteration))

        print('Getting model...')
        # Get Keras model as specified in the factory
        # Get the dimensions without the batch
        model = model_factory.get_model(input_size=image_size_with_channel_and_batch[1:])
        model_name = str(model_iteration) + '_' + model_factory.get_name()

        # If the model should be trained, we train it:
        if configuration['TRAIN_MODEL']:

            print("For training case {}, training model {}...".format(case_name, model_name))

            # Get Keras model as specified in the factory
            train_path = pathlib.Path(configuration['TRAINING_PATH'] + case_name + '/train/')
            validation_path = pathlib.Path(configuration['VALIDATION_PATH'])

            tg = image_generator(train_path,
                                 image_size=image_size,
                                 shuffle=True,
                                 augmentations=augmentations,
                                 batch_size=model_factory.get_batch_size(),
                                 batch_generation_mode=configuration['BATCH_GENERATION_MODE'],
                                 color_mode=configuration['GREY_OR_COLOR'])
            vg = image_generator(validation_path,
                                 image_size=image_size,
                                 shuffle=False,  # No shuffling
                                 augmentations=[],
                                 batch_size=1,
                                 batch_generation_mode='sequential',
                                 color_mode=configuration['GREY_OR_COLOR'])
            vg_size = get_size_of_data(validation_path)

            # Manually spin through the EPOCHS
            for i in range(0, configuration['EPOCHS']):
                print('For training case {}, training model {}, iteration {}, EPOCH {} of {}'.format(case_name,
                                                                                                     model_name,
                                                                                                     model_iteration,
                                                                                                     i, configuration[
                                                                                                         'EPOCHS']))
                # Verify checkpoint to reduce learning rate
                if configuration['EPOCHS_TO_REDUCE_LR'] is not None:
                    if i > 1 and i % configuration['EPOCHS_TO_REDUCE_LR'] == 0:
                        old_learning_rate = tf.keras.backend.get_value(model.optimizer.lr)
                        new_learning_rate = old_learning_rate / configuration['LR_DIVISOR']
                        print('Epoch {}/{} : Reducing learning rate from {} to {}'.format(i,
                                                                                          configuration['EPOCHS'],
                                                                                          old_learning_rate,
                                                                                          new_learning_rate))
                        tf.keras.backend.set_value(model.optimizer.lr, new_learning_rate)

                # Let tensorflow take care of the STEPS_PER_EPOCH
                history = model.fit_generator(tg,
                                              steps_per_epoch=configuration['STEPS_PER_EPOCH'],
                                              epochs=1,
                                              verbose=configuration['FIT_VERBOSITY'],
                                              validation_data=vg,
                                              validation_steps=vg_size*configuration['VALIDATION_STEPS'],
                                              workers=1,
                                              use_multiprocessing=False,
                                              )

                # Append the histories of the different EPOCHS
                if i == 0:
                    full_history = history.history
                else:
                    for key in full_history.keys():
                        full_history[key] += history.history[key]

                # Show DEBUG output images or PRINT them
                if configuration['PRINT_OUTPUT_IMAGES']:

                    # Depending on the number of STEPS_PER_EPOCH the generator can be accessed here before
                    # the fit_generator stops using it. So we try a few times in case an exception is raised
                    for _ in range(0, 5):
                        try:
                            (image, label) = next(tg)
                            (val_image, val_label) = next(vg)
                            break
                        except ValueError as ve:
                            print("Caught exception {}, will try again".format(ve))
                            time.sleep(1)

                    # Get the prediction of a simulation image
                    output = model.predict(image[0].reshape(image_size_with_channel_and_batch))
                    # Get the prediction of a validation image (real)
                    val_output = model.predict(val_image[0].reshape(image_size_with_channel_and_batch))

                    # Save output images to disk
                    if configuration['PRINT_OUTPUT_IMAGES']:
                        output_path = train_path / ('output/' + model_name + '/')

                        if not os.path.exists(output_path):
                            os.makedirs(output_path)

                        for key in highest_metrics:
                            if highest_metrics[key] < history.history[key][0]:
                                print("{} improved from {} to {}, saving model...".format(
                                    key, highest_metrics[key], history.history[key][0]))
                                highest_metrics[key] = history.history[key][0]
                                try:
                                    model.save(str(output_path / (key + '_model.h5')))
                                except Exception as e:
                                    print('Failed to save model with message {}'.format(e))

                        def save_grey(path, image):
                            plt.imsave(str(path), image, cmap=plt.cm.gray)

                        def save_color(path, image):
                            plt.imsave(str(path), image)

                        # Print some output images that can be useful in the future
                        save_grey(str(output_path / (str(i) + '_output.png')), output.reshape(image_size))
                        save_grey(str(output_path / (str(i) + '_label.png')), label[0].reshape(image_size))
                        save_grey(str(output_path / (str(i) + '_output_val.png')), val_output.reshape(image_size))
                        save_grey(str(output_path / (str(i) + '_label_val.png')), val_label[0].reshape(image_size))
                        if configuration['GREY_OR_COLOR'] == 'grey':
                            save_grey(str(output_path / (str(i) + '_input.png')), image[0].reshape(image_size))
                            save_grey(str(output_path / (str(i) + '_input_val.png')), val_image[0].reshape(image_size))
                        else:
                            save_color(str(output_path / (str(i) + '_input.png')), image[0].reshape(image_size + (3,)))
                            save_color(str(output_path / (str(i) + '_input_val.png')), val_image[0].reshape(image_size + (3,)))

            # Save history
            sio.savemat(configuration['RESULTS_PATH'] + '_' + case_name + '_' + model_name + '.mat', full_history)
            # Clean the graph
            tf.compat.v1.reset_default_graph()
            tf.keras.backend.clear_session()
