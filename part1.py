import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

os.remove('part1.log') if os.path.exists('part1.log') else None

formatter = logging.Formatter('%(name)s - %(message)s')

logger = logging.getLogger("part1")
logger.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('part1.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class MyLinearRegression:
    def __init__(self, weights_init='random', add_bias=True, learning_rate=1e-4,
        num_iterations=150_000, max_error=1e-6, verbose=True, num_messages=50):
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
        self.verbose_per = num_iterations / num_messages
        self.max_error = max_error

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
        loss = np.mean((target - pred) ** 2)
        return loss

    def fit(self, x, y):
        self.bias = self.initialize_bias()
        self.weights = self.initialize_weights(x.shape[1])

        current_loss = None
        for it in range(self.num_iterations):
            y_hat = self.predict(x)
            new_loss = self.cost(y, y_hat)
            if self.verbose and it % self.verbose_per == 0:
                logger.debug(f'Iteration {it}, loss: {new_loss}')
            if current_loss is not None and abs(new_loss - current_loss) < self.max_error:
                if self.verbose:
                    logger.debug(f'Converged after {it} iterations.')
                break

            err = y_hat - y
            w_grad = 2 * x.T @ err / y.size
            self.weights -= self.learning_rate * w_grad

            if self.add_bias:
                b_grad = 2 * np.sum(err) / y.size
                self.bias -= self.learning_rate * b_grad

            current_loss = new_loss
    
    def predict(self, x):
        ''' prediction function '''
        y_hat = x @ self.weights + self.bias
        return y_hat



def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y



if __name__ == "__main__":
    os.makedirs('imgs', exist_ok=True)

    # generating data samples
    x = np.linspace(-5.0, 5.0, 100)[:, np.newaxis]
    y = 29 * x + 40 * np.random.rand(100,1)

    # normalization of input data
    x /= np.max(x)

    plt.title('Data samples')
    plt.scatter(x, y)
    plt.savefig('imgs/data_samples.png')
    plt.close()

    # Sklearn linear regression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    y_hat_sklearn = sklearn_model.predict(x)

    plt.title('Data samples with sklearn model')
    plt.scatter(x, y)
    plt.plot(x, y_hat_sklearn, color='r')
    plt.savefig('imgs/sklearn_model.png')
    plt.close()
    logger.info(f'Sklearn MSE: {mean_squared_error(y, y_hat_sklearn)}')

    # Your linear regression model
    my_model = MyLinearRegression()
    my_model.fit(x, y)
    y_hat = my_model.predict(x)

    plt.title('Data samples with my model')
    plt.scatter(x, y)
    plt.plot(x, y_hat, color='r')
    plt.savefig('imgs/my_model.png')
    plt.close()
    logger.info(f'My MSE: {mean_squared_error(y, y_hat)}')

    # Normal equation without bias
    weights = normal_equation(x, y)
    y_hat_normal = x @ weights

    plt.title('Data samples with normal equation (without bias)')
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal, color='r')
    plt.savefig('imgs/normal_equation_without_bias.png')
    plt.close()
    logger.info(f'Normal equation without bias MSE: {mean_squared_error(y, y_hat_normal)}')

    # Normal equation with bias
    x_biased = np.hstack((np.ones((x.shape[0], 1)), x))
    weights = normal_equation(x_biased, y)
    y_hat_normal_with_bias = x_biased @ weights

    plt.title('Data samples with normal equation (with bias)')
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal_with_bias, color='r')
    plt.savefig('imgs/normal_equation_with_bias.png')
    plt.close()
    logger.info(f'Normal equation with bias MSE: {mean_squared_error(y, y_hat_normal_with_bias)}')