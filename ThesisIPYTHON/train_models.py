"""Training code for the thesis models (MLP and stateful LSTM).

Reproduces the training that produced MODELS/MLP2.h5 and MODELS/LSTM.h5,
using the same data pipeline as ATHEXNEURALNETS.ipynb and the training
configuration recorded inside the original h5 files:

  - optimizer: Adam, learning rate 0.001
  - loss: mean squared error
  - LSTM: batch size 1, 200 epochs (thesis hyperparameter list),
    stateful, look_back = 3
  - MLP:  look_back = 5, architecture Dense(12, relu) -> Dense(8, relu)
    -> Dense(1)

Requires the pinned environment (Python 3.9-3.11, tensorflow-cpu==2.15.1,
see requirements.txt). Since the original random seeds and exact epoch
counts for the MLP were not recorded, retrained weights will not be
bit-identical to the originals, but reach comparable error scores.

Retrained models are written next to the originals with a `_retrained`
suffix so the thesis artifacts are never overwritten:

    python train_models.py                       # full training
    python train_models.py --lstm-epochs 5       # quick smoke run
"""

import argparse
import math
import os

import numpy
from pandas import read_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

_HERE = os.path.dirname(os.path.abspath(__file__))


def create_dataset(dataset, look_back=1):
    """Same windowing function as in the notebook."""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def load_data():
    """Load and scale the returns series exactly as the notebook does.

    Note: as in the original notebook, the scaler is fit on the full
    series before the train/test split (kept for faithfulness to the
    thesis; fit on the training slice only to avoid the leakage).
    """
    dataframe = read_csv(os.path.join(_HERE, 'DATASETS/2001-2017-ATHEX-RETURNS.csv'),
                         usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.67)
    return dataset[0:train_size, :], dataset[train_size:, :], scaler


def report_scores(label, model, trainX, trainY, testX, testY, scaler, batch_size):
    trainPredict = model.predict(trainX, batch_size=batch_size, verbose=0)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size, verbose=0)
    model.reset_states()
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    trainY = scaler.inverse_transform([trainY])
    testY = scaler.inverse_transform([testY])
    for name, y, p in [('Train', trainY[0], trainPredict[:, 0]),
                       ('Test', testY[0], testPredict[:, 0])]:
        mae = mean_absolute_error(y, p)
        rmse = math.sqrt(mean_squared_error(y, p))
        print(f'{label} {name}: MAE={mae:.4f}  RMSE={rmse:.4f}')


def train_mlp(train, test, scaler, epochs, batch_size):
    look_back = 5
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    model = Sequential([
        Dense(12, input_dim=look_back, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1),
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

    report_scores('MLP', model, trainX, trainY, testX, testY, scaler, batch_size=1)
    return model


def train_lstm(train, test, scaler, epochs):
    look_back = 3
    batch_size = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    model = Sequential([
        LSTM(4, batch_input_shape=(batch_size, look_back, 1),
             stateful=True, return_sequences=True),
        LSTM(4, stateful=True),
        Dense(1),
    ])
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

    # Stateful training: no shuffling, and the cell state is reset by hand
    # after each pass over the series.
    for epoch in range(epochs):
        hist = model.fit(trainX, trainY, epochs=1, batch_size=batch_size,
                         verbose=0, shuffle=False)
        model.reset_states()
        print(f'LSTM epoch {epoch + 1}/{epochs}  loss={hist.history["loss"][0]:.6f}')

    report_scores('LSTM', model, trainX, trainY, testX, testY, scaler, batch_size)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mlp-epochs', type=int, default=200)
    parser.add_argument('--mlp-batch-size', type=int, default=2)
    parser.add_argument('--lstm-epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-dir', default=os.path.join(_HERE, 'MODELS'))
    args = parser.parse_args()

    numpy.random.seed(args.seed)
    train, test, scaler = load_data()

    mlp = train_mlp(train, test, scaler, args.mlp_epochs, args.mlp_batch_size)
    mlp.save(os.path.join(args.output_dir, 'MLP2_retrained.h5'))

    lstm = train_lstm(train, test, scaler, args.lstm_epochs)
    lstm.save(os.path.join(args.output_dir, 'LSTM_retrained.h5'))

    print('Saved MLP2_retrained.h5 and LSTM_retrained.h5 to', args.output_dir)


if __name__ == '__main__':
    main()
