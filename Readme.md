#                            Abstract
The fields of big data analysis and deep learning are developing fast
recently mainly due to the vast availability of data sets online and the
unpreceded cheap computational power of cloud services. LSTM
Neural Networks is a specialized function approximator for time-series
problems. It involves memory neurons that can overcome the training
problems faced by classical Recurrent Neural Networks with very deep
architectures. In this Thesis, we use an LSTM Neural Network to test the
Efficient Market Hypothesis introduced by Eugene Fama by attempting
to forecast the movement of the Greek Index Future FTSE/ATHEX Large
Cap. Additionally, we compare the performance of our LSTM model
performance with a Multilayered Perceptron with a time window using as variables autoregressive terms of our
dataset. We conclude that the performed accuracy is impressive
nevertheless it might be attributed to the good fitting of the model in the
out of sample data and lack of exposure to LSTM of the particular
market. Finally, we suggest ideas for further research.

Keywords: Algorithmic Trading, Efficient Market Hypothesis, Artificial
Neural Networks.



## How to run

Requires Python 3.9–3.11 (TensorFlow 2.15 does not support newer versions).

```bash
pip install -r ThesisIPYTHON/requirements.txt
jupyter notebook ThesisIPYTHON/ATHEXNEURALNETS.ipynb
```

Then run all cells. The notebook loads the pre-trained models from
`ThesisIPYTHON/MODELS/` — no training is needed.

**Why TensorFlow is pinned to 2.15:** the pre-trained models
(`MLP2.h5`, `LSTM.h5`) were saved with Keras 2.0.9/2.1.5 and cannot be
loaded by Keras 3 (bundled with TensorFlow 2.16+). TensorFlow 2.15 is the
last release that ships Keras 2.

**Using the models on modern Keras 3:** the weights are also exported in
plain NumPy format (`MODELS/*_weights.npz`) together with
`MODELS/rebuild_models.py`, which rebuilds the architectures in code and
reproduces the original models' predictions on both Keras 2 and Keras 3
(it preserves the original Keras 2 `hard_sigmoid` definition, which
Keras 3 changed):

```python
from MODELS.rebuild_models import load_mlp, load_lstm
mlp = load_mlp()
lstm = load_lstm()
```

## Retraining the models

`ThesisIPYTHON/train_models.py` reproduces the training that created the
pre-trained models, using the notebook's exact data pipeline and the
training configuration recorded inside the original h5 files (Adam,
learning rate 0.001, MSE loss; batch size 1 and 200 epochs for the
stateful LSTM). Run it inside the pinned environment:

```bash
cd ThesisIPYTHON
python train_models.py                 # full training (~1-2 h on CPU)
python train_models.py --lstm-epochs 5 --mlp-epochs 20   # quick check
```

Retrained models are saved as `MODELS/*_retrained.h5` so the original
thesis artifacts are never overwritten. Because the original random
seeds were not recorded, retrained weights differ from the originals
but reach comparable MAE/RMSE scores.
