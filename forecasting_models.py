"""
K8sight - forecasting_models.py

Shared forecasting model classes, used by both forecast_workload.py (which
fits and backtests them) and right_sizing_engine.py (which only calls
.predict() on the already-fitted models saved in forecasters.joblib).

This module exists so joblib/pickle can locate these classes by a stable
module path ("forecasting_models.DriftModel", etc.) regardless of which
script loads forecasters.joblib. Defining these classes inside a script's
own __main__ instead would make the saved model unloadable from any other
script - pickle records the class's module at save time, and a different
script's __main__ is a different module even with identical code.

All classes share one interface: fit(y) -> self, predict(h) -> array of h
forecasted values. SEASONAL_PERIOD (8 samples = 120s) matches the Periodic
pattern's own load-generation cycle.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASONAL_PERIOD = 8   # 8 * 15s = 120s, matches PeriodicShape's cycle


class NaiveModel:
    name = "naive"

    def fit(self, y):
        self.last = float(y[-1])
        return self

    def predict(self, h):
        return np.full(h, self.last)


class DriftModel:
    name = "drift"

    def fit(self, y):
        n = len(y)
        x = np.arange(n, dtype=float)
        self.slope, self.intercept = np.polyfit(x, y, 1)
        self.n = n
        return self

    def predict(self, h):
        x = np.arange(self.n, self.n + h, dtype=float)
        return self.slope * x + self.intercept


class SeasonalNaiveModel:
    name = "seasonal_naive"

    def fit(self, y):
        self.y = np.asarray(y, dtype=float)
        return self

    def predict(self, h):
        n = len(self.y)
        return np.array([self.y[n - SEASONAL_PERIOD + (i % SEASONAL_PERIOD)]
                         if n >= SEASONAL_PERIOD else self.y[-1] for i in range(h)])


class HoltModel:
    name = "holt"

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        try:
            self.fitted = ExponentialSmoothing(y, trend="add", seasonal=None).fit()
            self.ok = True
        except Exception:
            self.fallback = NaiveModel().fit(y)
            self.ok = False
        return self

    def predict(self, h):
        if self.ok:
            return np.asarray(self.fitted.forecast(h))
        return self.fallback.predict(h)


class HoltWintersModel:
    name = "holt_winters"

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        try:
            if len(y) < 2 * SEASONAL_PERIOD:
                raise ValueError("not enough points for a seasonal model")
            self.fitted = ExponentialSmoothing(y, trend="add", seasonal="add",
                                               seasonal_periods=SEASONAL_PERIOD).fit()
            self.ok = True
        except Exception:
            self.fallback = SeasonalNaiveModel().fit(y)
            self.ok = False
        return self

    def predict(self, h):
        if self.ok:
            return np.asarray(self.fitted.forecast(h))
        return self.fallback.predict(h)


class ArimaModel:
    name = "arima"
    ORDER_CANDIDATES = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0)]

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        for order in self.ORDER_CANDIDATES:
            try:
                self.fitted = ARIMA(y, order=order).fit()
                self.order = order
                self.ok = True
                return self
            except Exception:
                continue
        self.fallback = NaiveModel().fit(y)
        self.ok = False
        self.order = None
        return self

    def predict(self, h):
        if self.ok:
            return np.asarray(self.fitted.forecast(h))
        return self.fallback.predict(h)


def model_registry_for(pattern):
    reg = {"naive": NaiveModel, "drift": DriftModel, "holt": HoltModel, "arima": ArimaModel}
    if pattern == "periodic":
        reg["seasonal_naive"] = SeasonalNaiveModel
        reg["holt_winters"] = HoltWintersModel
    return reg

def model_registry_for(pattern):
    reg = {"naive": NaiveModel, "drift": DriftModel, "holt": HoltModel, "arima": ArimaModel}
    if pattern == "periodic":
        reg["seasonal_naive"] = SeasonalNaiveModel
        reg["holt_winters"] = HoltWintersModel
    return reg
