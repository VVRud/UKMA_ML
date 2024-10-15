from pathlib import Path

import numpy as np

from logger import get_logger


class MyLinearRegression:
    def __init__(self, log_path: Path, weights_init='random', add_bias=True, learning_rate=1e-4,
                 num_iterations=150_000, evaluate_every=1000, max_error=1e-6, verbose=True):
        ''' Linear regression model using gradient descent

        # Arguments
            weights_init: str
                weights initialization option ['random', 'zeros']
            add_bias: bool
                whether to add bias term
            learning_rate: float
                learning rate value for gradient descent
            num_iterations: int
                maximum number of iterations in gradient descent
            max_error: float
                error tolerance term, after reaching which we stop gradient descent iterations
            verbose: bool
                enabling verbose output
            num_messages: int
                number of messages to print during verbose output
        '''
        if weights_init not in ['random', 'zeros']:
            raise ValueError('weights_init should be either `random` or `zeros`')

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights_init = weights_init
        self.add_bias = add_bias
        self.verbose = verbose
        self.max_error = max_error
        self.evaluate_every = evaluate_every
        self.logger = get_logger("linear_regression", log_path)

    def initialize_weights(self, n_features):
        ''' weights initialization function '''
        match self.weights_init:
            case 'random':
                weights = np.random.normal(loc=0, scale=np.sqrt(n_features), size=(n_features, 1))
            case 'zeros':
                weights = np.zeros(shape=(n_features, 1))
            case _:
                raise NotImplementedError
        return weights.astype(np.float32)

    def initialize_bias(self):
        ''' bias initialization function '''
        if not self.add_bias:
            return np.float32(0)
        return np.float32(1)

    def cost(self, target, pred):
        ''' calculate cost function

            # Arguments:
                target: np.array
                    array of target floating point numbers
                pred: np.array
                    array of predicted floating points numbers
        '''
        return np.mean((target - pred) ** 2)

    def fit(self, x, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.bias = self.initialize_bias()
        self.weights = self.initialize_weights(x.shape[1])

        verbose_per = int(self.num_iterations / 100)
        current_loss = None
        for it in range(self.num_iterations):
            y_hat = self._predict(x)
            err = y_hat - y
            w_grad = 2 * x.T @ err / y.size
            self.weights -= self.learning_rate * w_grad

            if self.add_bias:
                b_grad = 2 * np.sum(err) / y.size
                self.bias -= self.learning_rate * b_grad

            if self.verbose and it % verbose_per == 0:
                self.logger.debug(f'Iteration {it}')

            if it % self.evaluate_every == 0:
                new_loss = self.cost(y, y_hat)
                if self.verbose:
                    self.logger.debug(f'Iteration {it}, loss: {new_loss}')
                if current_loss is not None and self.max_error is not None and abs(current_loss - new_loss) < self.max_error:
                    if self.verbose:
                        self.logger.debug(f'Converged after {it} iterations.')
                    break
                current_loss = new_loss

    def _predict(self, x):
        ''' prediction function '''
        return x @ self.weights + self.bias

    def predict(self, x):
        return self._predict(x)
