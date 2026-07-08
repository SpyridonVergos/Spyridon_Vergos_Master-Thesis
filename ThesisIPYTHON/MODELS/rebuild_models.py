"""Version-independent loaders for the thesis models.

The original MLP2.h5 / LSTM.h5 files were saved with Keras 2.0.9 / 2.1.5
and can only be opened by Keras 2 (TensorFlow <= 2.15). This module
rebuilds the same architectures in code and loads the exported weights
(MLP2_weights.npz / LSTM_weights.npz), so the models keep working on
Keras 3 and later.

Note: Keras 2's `hard_sigmoid` is clip(0.2 * x + 0.5, 0, 1), while Keras 3
redefined it as relu6(x + 3) / 6. The LSTM was trained with the former, so
we pass the original definition explicitly to reproduce the thesis results.

Usage:
    from MODELS.rebuild_models import load_mlp, load_lstm
    mlp = load_mlp()    # input shape (batch, 5)
    lstm = load_lstm()  # stateful, batch_input_shape (1, 3, 1)
"""

import os

import numpy as np
import keras
from keras import layers


_HERE = os.path.dirname(os.path.abspath(__file__))


if hasattr(keras, "ops"):  # Keras 3

    def hard_sigmoid_k2(x):
        """hard_sigmoid as defined in Keras 2 (used at training time)."""
        return keras.ops.clip(0.2 * x + 0.5, 0.0, 1.0)

else:  # Keras 2
    from keras import backend as K

    def hard_sigmoid_k2(x):
        """hard_sigmoid as defined in Keras 2 (used at training time)."""
        return K.clip(0.2 * x + 0.5, 0.0, 1.0)


def _load_weights(model, npz_name, layer_names):
    data = np.load(os.path.join(_HERE, npz_name))
    for layer_name in layer_names:
        weights = []
        j = 0
        while f"{layer_name}__{j}" in data:
            weights.append(data[f"{layer_name}__{j}"])
            j += 1
        model.get_layer(layer_name).set_weights(weights)
    return model


def load_mlp():
    """Dense(12, relu) -> Dense(8, relu) -> Dense(1). Input: (batch, 5)."""
    inputs = keras.Input(shape=(5,))
    x = layers.Dense(12, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(8, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(1, name="dense_3")(x)
    model = keras.Model(inputs, outputs, name="MLP2")
    return _load_weights(model, "MLP2_weights.npz",
                         ["dense_1", "dense_2", "dense_3"])


def load_lstm():
    """Stateful LSTM(4) -> LSTM(4) -> Dense(1). Input: batch_shape (1, 3, 1)."""
    inputs = keras.Input(batch_shape=(1, 3, 1))
    x = layers.LSTM(4, return_sequences=True, stateful=True,
                    recurrent_activation=hard_sigmoid_k2,
                    name="lstm_1")(inputs)
    x = layers.LSTM(4, stateful=True,
                    recurrent_activation=hard_sigmoid_k2,
                    name="lstm_2")(x)
    outputs = layers.Dense(1, name="dense_1")(x)
    model = keras.Model(inputs, outputs, name="LSTM")
    return _load_weights(model, "LSTM_weights.npz",
                         ["lstm_1", "lstm_2", "dense_1"])


if __name__ == "__main__":
    mlp = load_mlp()
    lstm = load_lstm()
    print(f"Keras {keras.__version__}: both models rebuilt successfully.")
    demo = mlp.predict(np.zeros((1, 5), dtype="float32"), verbose=0)
    demo2 = lstm.predict(np.zeros((1, 3, 1), dtype="float32"),
                         batch_size=1, verbose=0)
    print("MLP(0) =", demo[0], " LSTM(0) =", demo2[0])
