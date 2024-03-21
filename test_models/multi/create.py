# coding: utf-8

"""
Test script to create a multi input/output model for AOT compilation tests.

Signature: f32(4), f64(4) -> f32(2), b(2)
"""

import os

import cmsml


def create_model(model_dir):
    # get tensorflow (suppressing the usual device warnings and logs)
    tf, _, tf_version = cmsml.tensorflow.import_tf()
    print("creating multi model")
    print(f"location  : {model_dir}")
    print(f"TF version: {'.'.join(map(str, tf_version))}")

    # set random seeds to get deterministic results for testing
    tf.keras.utils.set_random_seed(1)

    # define architecture
    n_in, n_out, n_layers, n_units = 4, 2, 5, 128

    # define input layers
    x1 = tf.keras.Input(shape=(n_in,), dtype=tf.float32, name="input1")
    x2 = tf.keras.Input(shape=(n_in,), dtype=tf.float64, name="input2")
    x = tf.keras.layers.Concatenate(axis=1)([x1, x2])

    # model layers
    a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(x)
    for _ in range(n_layers):
        a = tf.keras.layers.Dense(n_units, activation="tanh")(a)
        a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(a)

    # output layers
    y1 = tf.keras.layers.Dense(n_out, activation="softmax", name="output1", dtype=tf.float32)(a)
    y2 = tf.keras.layers.Reshape((n_out,), name="output2")(y1 > 0.5)

    # define the model
    model = tf.keras.Model(inputs=[x1, x2], outputs=[y1, y2])

    # test evaluation
    inputs = [
        tf.constant([list(range(n_in))], dtype=tf.float32),
        tf.constant([list(range(n_in))], dtype=tf.float64),
    ]
    print(model(inputs))

    # save it
    tf.saved_model.save(model, model_dir)


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    create_model(os.path.join(this_dir, "saved_model"))


if __name__ == "__main__":
    main()
