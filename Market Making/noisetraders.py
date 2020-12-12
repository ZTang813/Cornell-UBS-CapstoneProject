import numpy as np
import pandas as pd
from enum import Enum


class Side(Enum):
    BID = 1.0
    OFFER = -1.0

    def __str__(self):
        return 'BUY' if self.value > 0.0 else 'SELL'


if __name__ == '__main__':
    train = False
    file_name = 'noise_trades_' + ('_train.csv' if train else '_test.csv')
    seed = 0
    n = 36_000
    q_min = 5.0
    q_max = 60.0
    p_order = 0.1

    np.random.seed(seed)
    uniform_vector = np.random.uniform(size=2 * n)
    submit = uniform_vector <= p_order
    quantities = np.round(np.random.uniform(size=2 * n) * (q_max - q_min) + q_min)
    side = np.array([Side.BID if value < 0.5 else Side.OFFER for value in np.random.uniform(size=2 * n)])
    orders = pd.DataFrame(index=range(2 * n))
    orders.loc[submit, 'side'] = side[submit]
    orders.loc[submit, 'qty'] = quantities[submit]
    orders.index.name = 'time'
    orders = orders.iloc[:n] if train else orders.iloc[n:]
    formats = {'qty': '{:.0f}'}
    for col, f in formats.items():
        orders[col] = orders[col].map(lambda x: f.format(x))
    orders.to_csv(file_name, float_format='%.3f')
