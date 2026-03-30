import numpy as np
from genbart.bart import bart

def test_loading_data():
    X = np.array([0, 1, 0, 2, 0, 3]).reshape((3, 2,))
    y = np.array([1, 2, 3])
    model = bart()

    model.fit(X, y)

    assert model.X.ndim == 2
    assert model.X.shape == (3, 2)
    assert model.y.ndim == 1
    assert model.y.shape == (3, )
    assert model.y_scale == 2.0
    assert model.y_shift == 2.0
    assert model.y[0] == -0.5
    assert model.y[2] == 0.5
    assert model.n == 3
    assert model.p == 2
    assert model._inverse_transform_y(model.y[0]) == 1.0

    assert len(model.trees) == 200
    assert model.sigma < 1e-15