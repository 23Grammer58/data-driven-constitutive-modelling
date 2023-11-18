import os
from typing import Any
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class MLPRegressorWrapper:

    def __init__(self, hidden_layer_sizes = (100,), activation='relu', solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                 max_iter=200, shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=False,
                 momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8, model_name="binn"):
        # self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                  alpha=alpha, batch_size=batch_size, learning_rate=learning_rate,
                                  learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
                                  shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose,
                                  warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum,
                                  early_stopping=early_stopping, validation_fraction=validation_fraction,
                                  beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        self.model_name = model_name
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_scale_params(self):
        return self.data_min_, self.scale_, self.data_range_

    def set_scale_params(self, range, min, scale):
        self.data_min_ = min
        self.scale_ = scale
        self.data_range_ = range

    def save_model(self):
        file = os.path.join('pretrained_models', self.model_name + '.joblib')
        joblib.dump(self, file)
    