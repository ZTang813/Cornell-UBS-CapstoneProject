import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm, inv, eig


def get_covariance(sigma: float, delta: float, theta: np.ndarray) -> np.ndarray:
    theta_p = theta + theta.T
    return (sigma ** 2.0) * inv(theta_p) * (np.eye(theta.shape[0]) - expm(-theta_p * delta))


def sample_gaussian(n: int, covariance: np.ndarray) -> np.ndarray:
    d, v = eig(covariance)
    a = np.dot(v, np.diag(np.sqrt(np.real(d))))
    g = np.random.normal(0.0, 1.0, (a.shape[0], n))
    return np.dot(a, g)


def sample_mean_reversion(n: int, x0: np.ndarray, mu: np.ndarray, sigma: float, delta: float,
                          theta: np.ndarray) -> np.ndarray:
    if not positive_eigenvalues(theta):
        raise AssertionError("Input theta does not have all positive eigenvalues")
    covariance = get_covariance(sigma, delta, theta)
    if not positive_eigenvalues(covariance):
        raise AssertionError("Covariance does not have all positive eigenvalues")
    gaussian_matrix = sample_gaussian(n, covariance)
    sample_paths = np.ndarray(gaussian_matrix.shape)
    sample_paths[:, [0]] = x0
    exp_theta = expm(-theta * delta)
    for i in range(1, sample_paths.shape[1]):
        prev = sample_paths[:, [i - 1]]
        sample_paths[:, [i]] = mu + np.dot(exp_theta, (prev - mu)) + gaussian_matrix[:, [i - 1]]
    return sample_paths


def positive_eigenvalues(theta: np.ndarray) -> bool:
    d, v = eig(theta)
    return np.all(np.real(d) > 0.0)


specs = pd.DataFrame(data={'p0': [100.0, 110.0, 150.0],
                           'mu': [100.0, 110.0, 80.0],
                           'theta': [0.05, 0.02, 0.0003],
                           'sigma': [0.05, 0.02, 0.1],
                           'lam': [0.05, 2.0, 0.05],
                           'date': ['20200901', '20200909', '20200916']},
                     index=[1, 2, 3])

# demo script for classification (binary or multiclass) using classic, axis-normal splits
if __name__ == '__main__':
    episode = 3
    train = False
    episode_specs = specs.loc[episode]
    date = episode_specs.date
    file_name = 'sample_data_' + date + '_' + str(episode) + ('_train.csv' if train else '_test.csv')
    seed = 0
    p0 = episode_specs['p0']  # 100.0, 110.0
    mu = episode_specs['mu']  # 100.0, 110.0
    theta = episode_specs['theta']  # 0.05, 0.02
    sigma = episode_specs['sigma']  # 0.05, 0.02
    n = 36_000
    display = 30_000
    lam = episode_specs['lam']  # 0.05, 2.0
    resolution = 1.0 / 10.0  # per dollar
    q_min = 50.0
    q_max = 100.0

    np.random.seed(seed)
    default_font_size = 16
    model_type = 'tree'  # it can be 'tree' or 'nn'
    plt.rc('axes', titlesize=default_font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=default_font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=default_font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=default_font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=default_font_size)  # legend fontsize
    plt.rc('figure', titlesize=default_font_size)  # fontsize of the figure title

    gaussian_vector = np.random.normal(0.0, 1.0, 2 * n)
    poisson_vector = 1 + np.random.poisson(lam, 2 * n)
    quantities = np.round(np.random.uniform(size=(2 * n, 2)) * (q_max - q_min) + q_min)
    mid = np.ndarray(2 * n)
    p = p0
    for i in range(2 * n):
        pm1 = p0 if i == 0 else mid[i - 1]
        mid[i] = pm1 + np.floor((theta * (mu - pm1) + sigma * gaussian_vector[i]) / resolution + 0.5) * resolution
    spread = resolution * poisson_vector
    order_books = pd.DataFrame(data={'bidPrice': mid - 0.5 * spread, 'offerPrice': mid + 0.5 * spread,
                                     'bidQty': quantities[:, 0], 'offerQty': quantities[:, 1]})
    order_books.index.name = 'time'
    order_books = order_books.iloc[:n] if train else order_books.iloc[n:]
    plt.step(order_books.index[:display], order_books.bidPrice.iloc[:display], where='post', label='bid')
    plt.step(order_books.index[:display], order_books.offerPrice.iloc[:display], where='post', label='offer')
    # plt.hlines(mu, 0, n, linestyles=['-'], zorder=100)
    plt.title('Stock price')
    ax = plt.gca()
    # ax.set_xlim([0, n])
    # ax.set_ylim([90, 120])
    # plt.savefig('trading_example_prices.png')
    plt.show()

    formats = {'bidPrice': '{:.2f}', 'offerPrice': '{:.2f}', 'bidQty': '{:.0f}', 'offerQty': '{:.0f}'}
    for col, f in formats.items():
        order_books[col] = order_books[col].map(lambda x: f.format(x))
    order_books.to_csv(file_name, float_format='%.3f')
