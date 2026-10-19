"""Shared deterministic Gaussian-process helpers for the ISABE 2026 experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
GPR_MAX_TRAIN_SAMPLES = 650


class DeterministicTrainingCapRegressor(RegressorMixin, BaseEstimator):
    """Fit a cloned regressor on a deterministic capped subset of rows."""

    def __init__(
        self,
        estimator: BaseEstimator,
        max_train_samples: int = GPR_MAX_TRAIN_SAMPLES,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.estimator = estimator
        self.max_train_samples = max_train_samples
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape[0] > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            indices = np.sort(
                rng.choice(x.shape[0], size=self.max_train_samples, replace=False)
            )
            x_fit = x[indices]
            y_fit = y[indices]
        else:
            x_fit = x
            y_fit = y
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x_fit, y_fit)
        self.n_training_samples_seen_ = int(x.shape[0])
        self.n_training_samples_used_ = int(x_fit.shape[0])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(x)


def gpr_kernels() -> Dict[str, BaseEstimator]:
    """Return the fixed kernels used in every GPR comparison."""

    return {
        "RBF": ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=1.0, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed"),
        "Matern1.5": ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(length_scale=1.0, length_scale_bounds="fixed", nu=1.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed"),
    }


def gpr_estimator(kernel: BaseEstimator, alpha: float) -> BaseEstimator:
    """Build one standardized, deterministically capped GPR estimator."""

    return make_pipeline(
        StandardScaler(),
        DeterministicTrainingCapRegressor(
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=True,
                optimizer=None,
                random_state=RANDOM_STATE,
            )
        ),
    )


def gpr_candidates() -> List[Tuple[str, BaseEstimator]]:
    """Return the common casewise GPR hyperparameter candidates."""

    candidates: List[Tuple[str, BaseEstimator]] = []
    for kernel_name, kernel in gpr_kernels().items():
        for alpha in (1e-6, 1e-4):
            estimator = gpr_estimator(kernel, alpha)
            candidates.append(
                (
                    f"kernel={kernel_name},alpha={alpha:g},cap={GPR_MAX_TRAIN_SAMPLES}",
                    estimator,
                )
            )
    return candidates
