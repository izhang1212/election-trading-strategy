"""Thin wrapper: the simulation calls the shared top-level estimator (../estimate_naive.py), so the paper's
simulated validation runs exactly the code applied to the real night. Same API as before."""
import importlib.util, os
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "estimate_naive_shared", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "estimate_naive.py"))
_shared = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_shared)
W_GRID, LAGS, KAPPAS = _shared.W_GRID, _shared.LAGS, _shared.KAPPAS


def _call(R, called):
    return np.where(called, (R["P_S"] > 0.5).astype(float), np.nan)   # the media call is observable on the night


def naive_signal(R, called, lag):
    return _shared.naive_signal(R["RAW"], R["FRAC"], _call(R, called), R["PRIOR"], lag)


def estimate(R, mid, called, tmax=180):
    return _shared.estimate(R["P_I"], R["RAW"], R["FRAC"], _call(R, called), mid, R["PRIOR"], tmax)["w"]
