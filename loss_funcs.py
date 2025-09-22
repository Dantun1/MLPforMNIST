import numpy as np
from typing import TypeAlias
from numpy.typing import NDArray

ProbArray: TypeAlias = NDArray[np.float64]
OneHot: TypeAlias = NDArray[np.int64]


def cross_entropy(y_pred: ProbArray, y_true: OneHot) -> float:
    """Cross-entropy loss function.

    Compute and return cross-entropy loss for a single sample.

    Args:
        y_pred: shape (n,), Predicted probability distribution.
        y_true: shape (n,), One-hot encoded true labels.

    Returns:
        float: Scalar cross entropy loss

    Raises:
        ValueError: If y_pred and y_true have different shapes or are not 1D.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    if y_pred.ndim != 1 or y_true.ndim != 1 or y_pred.shape != y_true.shape:
        raise ValueError("Expected y_pred and y_true with shape (C,)")

    # For floating point stability
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    return float(-np.sum(y_true * np.log(y_pred)))




