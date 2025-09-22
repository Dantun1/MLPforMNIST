import numpy as np
from numpy.typing import NDArray

class ActivationStrategy:
    """Base class for activation functions.

    Methods:
        forward: computes the activation function
        backward: computes dL/dz from dL/da (error signal from the next layer)
    """
    def forward(self, z):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class ReLU(ActivationStrategy):
    def forward(self, z: NDArray) -> NDArray:
        """
        Activate with ReLU, store pre-activation values.

        Args:
            z: array of previous layer's activations.

        Returns:
            float: The activated value.
        """
        self.z = z
        return np.maximum(0, z)

    def backward(self,  grad_output: NDArray) -> NDArray:
        """
        Pass back the gradient of loss wrt to pre-activation input z

        dL/dz = dL/da * da/dz

        Where da/dz = 1 if z > 0 else 0 (ReLU derivative)

        Args:
            grad_output: array of gradient signal from the next layer. dL/da

        Returns:
            NDArray: gradient of loss wrt to pre-activation z.
        """
        return grad_output * (self.z > 0)

class Softmax(ActivationStrategy):
    def forward(self, z: NDArray) -> NDArray:
        """
        Activate with Softmax, store pre-activation values.

        Args:
            z: array of previous layer's activations.

        Returns:
            NDArray: Softmax activation values.
        """
        self.z = z

        # To prevent float overflow, subtract constant from all zs
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)

        return exp_z / np.sum(exp_z)

    def backward(self, grad_output):
        """
        Pass back the gradient of loss wrt to pre-activation input z.

        In my setup, dL/dz is simply y_pred - y_true when using cross-entropy loss and softmax activation.

        Hence, backward is a pass-through because the correct gradient (y_pred - y_true)
        is computed externally in the MLP training loop.
        """
        return grad_output
