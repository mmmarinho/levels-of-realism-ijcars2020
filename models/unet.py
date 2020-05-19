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

import tensorflow as tf


def get_model(pretrained_weights=None, input_size=(256, 256, 1), show_summary=False,
              optimizer=tf.keras.optimizers.SGD()):
    """
    U-NET as described in https://arxiv.org/pdf/1505.04597.pdf
    The main difference here is that they crop the images and here I adjust the convolutions
    in a way that the size of the output is the same as the input.

    :param pretrained_weights: To load a saved network
    :param input_size: The size of the input as a tuple, e.g. (256, 256, 1)
    :param show_summary: Whether to print the summary of the network or not. (True or False)
    :param optimizer: The tf.keras.optimizer for this model

    :return: The compiled model
    """

    Conv2D = tf.keras.layers.Conv2D
    Conv2DTranspose = tf.keras.layers.Conv2DTranspose
    MaxPooling2D = tf.keras.layers.MaxPooling2D
    concatenate = tf.keras.layers.concatenate

    inputs = tf.keras.Input(input_size)
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    skip1 = Conv2D(64, 3, activation='relu', padding='same')(x)

    # Encoder path
    x = MaxPooling2D(pool_size=(2, 2))(skip1)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    skip2 = Conv2D(128, 3, activation='relu', padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(skip2)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    skip3 = Conv2D(256, 3, activation='relu', padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(skip3)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    skip4 = Conv2D(512, 3, activation='relu', padding='same')(x)

    # Central
    x = MaxPooling2D(pool_size=(2, 2))(skip4)
    x = Conv2D(1024, 3, activation='relu', padding='same')(x)
    x = Conv2D(1024, 3, activation='relu', padding='same')(x)

    # Decoder
    x = Conv2DTranspose(512, 2, strides=(2, 2), activation='relu', padding='same')(x)
    x = concatenate([x, skip4], axis=3)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)

    x = Conv2DTranspose(256, 2, strides=(2, 2), activation='relu', padding='same')(x)
    x = concatenate([x, skip3], axis=3)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)

    x = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same')(x)
    x = concatenate([x, skip2], axis=3)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)

    x = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same')(x)
    x = concatenate([x, skip1], axis=3)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    # Output
    output = Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(),
                      tf.keras.metrics.Precision(thresholds=0.5),
                      tf.keras.metrics.Recall(thresholds=0.5),
                      tf.keras.metrics.TruePositives(thresholds=0.5),
                      tf.keras.metrics.TrueNegatives(thresholds=0.5),
                      tf.keras.metrics.FalsePositives(thresholds=0.5),
                      tf.keras.metrics.FalseNegatives(thresholds=0.5)
                  ]
                  )

    if show_summary:
        model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
