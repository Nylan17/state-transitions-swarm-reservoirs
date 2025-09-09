from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def train_readout(states, targets, lam):
    X = states[:-1]          # t
    y = targets[1:]          # t+1
    model = Ridge(alpha=lam, fit_intercept=True)
    model.fit(X, y)
    return model

def train_kernel_readout(states, targets, lam, gamma=None):
    X = states[:-1]
    y = targets[1:]
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    model = KernelRidge(alpha=lam, kernel="rbf", gamma=gamma)
    model.fit(X, y)
    return model

def train_poly_readout(states, targets, lam, degree=2):
    X = states[:-1]
    y = targets[1:]
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_poly)

    model = Ridge(alpha=lam, fit_intercept=True)
    model.fit(X_scaled, y)

    # package transform objects for use during prediction
    class _PolyModel:
        def __init__(self, model, poly, scaler):
            self._m = model
            self._p = poly
            self._s = scaler

        def predict(self, X_raw):
            Xp = self._p.transform(X_raw)
            Xp = self._s.transform(Xp)
            return self._m.predict(Xp)

    return _PolyModel(model, poly, scaler_x) 