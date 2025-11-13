from sklearn.datasets import load_breast_cancer, load_wine, load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from .voting_utils import get_voting_weights
from .airport_utils import get_airport_costs


def load_dataset(name, test_size=0.2, random_state=42, data_home=None):
    """
    通用数据加载函数（统一在末尾做标准化与返回）
    支持: iris, wine, cancer, mnist, diabetes

    参数
    ----
    name : str
        数据集名称: 'iris' | 'wine' | 'cancer' | 'mnist' | 'diabetes'
    scale : bool
        是否进行StandardScaler标准化（基于训练集fit，测试集transform）
    test_size : float
        测试集比例（除MNIST外，MNIST用每类100/20规则）
    random_state : int
        随机种子
    data_home : str | None
        fetch_openml 的本地缓存目录（可选）
    """
    name = name.lower()
    rng = np.random.RandomState(random_state)

    # 先占位，分支里赋值
    x_train = x_test = y_train = y_test = None

    scale_data = True

    if name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
 
    elif name == "cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    elif name == "mnist":
        # 手动每类取 100 训练 / 20 测试
        print("Loading MNIST dataset...")
        X, y = fetch_openml(
            'mnist_784', version=1, return_X_y=True, parser="auto", data_home=data_home
        )
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        print(X.shape, y.shape)

        X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
        for digit in range(10):
            idx = np.where(y == digit)[0]
            chosen = rng.choice(idx, 120, replace=False)
            train_idx, test_idx = chosen[:100], chosen[100:]
            X_train_list.append(X[train_idx])
            y_train_list.append(y[train_idx])
            X_test_list.append(X[test_idx])
            y_test_list.append(y[test_idx])

        x_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        x_test = np.vstack(X_test_list)
        y_test = np.hstack(y_test_list)

    elif name == "diabetes":
        data = fetch_openml(name='diabetes', version=1, as_frame=True, data_home=data_home)
        X: pd.DataFrame = data.data
        y_raw = data.target

        # 常见是 'tested_negative' / 'tested_positive'
        if set(pd.Series(y_raw).unique()) == {'tested_negative', 'tested_positive'}:
            y = (y_raw == 'tested_positive').astype(int)
        else:
            # 若已是0/1或其他二分类编码，尽量转为数值
            y = pd.Series(y_raw).astype('category').cat.codes if not np.issubdtype(pd.Series(y_raw).dtype, np.number) else y_raw

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 转为 numpy
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.to_numpy()
            x_test = x_test.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

    elif name == "voting":
        if voting_weights is None:
            weights = get_voting_weights()
        else:
            weights = np.asarray(voting_weights, dtype=float).reshape(-1)
        x_train = weights.reshape(-1, 1).astype(float)
        y_train = np.zeros_like(weights, dtype=int)
        x_test = x_train.copy()
        y_test = y_train.copy()
        scale_data = False

    elif name == "airport":
        costs = get_airport_costs()
        x_train = costs.reshape(-1, 1).astype(float)
        y_train = np.zeros_like(costs, dtype=int)
        x_test = x_train.copy()
        y_test = y_train.copy()
        scale_data = False

    else:
        raise ValueError(f"Unsupported dataset: {name}")


    if scale_data:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test
