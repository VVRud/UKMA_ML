from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from linear_regression import MyLinearRegression
from logger import get_logger

base_path = Path('logs') / 'part1'
imgs_dir = base_path / 'imgs'
imgs_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger("part1", base_path)

def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y



if __name__ == "__main__":
    # generating data samples
    x = np.linspace(-5.0, 5.0, 100)[:, np.newaxis]
    y = 29 * x + 40 * np.random.rand(100,1)

    # normalization of input data
    x /= np.max(x)

    plt.title('Data samples')
    plt.scatter(x, y)
    plt.savefig(imgs_dir / 'data_samples.png')
    plt.close()

    # Sklearn linear regression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    y_hat_sklearn = sklearn_model.predict(x)

    plt.title('Data samples with sklearn model')
    plt.scatter(x, y)
    plt.plot(x, y_hat_sklearn, color='r')
    plt.savefig(imgs_dir / 'sklearn_model.png')
    plt.close()
    logger.info(f'Sklearn MSE: {mean_squared_error(y, y_hat_sklearn)}')

    # Your linear regression model
    my_model = MyLinearRegression(log_path=base_path)
    my_model.fit(x, y)
    y_hat = my_model.predict(x)

    plt.title('Data samples with my model')
    plt.scatter(x, y)
    plt.plot(x, y_hat, color='r')
    plt.savefig(imgs_dir / 'my_model.png')
    plt.close()
    logger.info(f'My MSE: {mean_squared_error(y, y_hat)}')

    # Normal equation without bias
    weights = normal_equation(x, y)
    y_hat_normal = x @ weights

    plt.title('Data samples with normal equation (without bias)')
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal, color='r')
    plt.savefig(imgs_dir / 'normal_equation_without_bias.png')
    plt.close()
    logger.info(f'Normal equation without bias MSE: {mean_squared_error(y, y_hat_normal)}')

    # Normal equation with bias
    x_biased = np.hstack((np.ones((x.shape[0], 1)), x))
    weights = normal_equation(x_biased, y)
    y_hat_normal_with_bias = x_biased @ weights

    plt.title('Data samples with normal equation (with bias)')
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal_with_bias, color='r')
    plt.savefig(imgs_dir / 'normal_equation_with_bias.png')
    plt.close()
    logger.info(f'Normal equation with bias MSE: {mean_squared_error(y, y_hat_normal_with_bias)}')