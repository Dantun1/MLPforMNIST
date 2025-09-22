from activation_funcs import *
from loss_funcs import *


class DenseLayer:
    """
    Represents a dense layer in a neural network.

    Methods:
        forward: computes the current layer's activation values.
        backward: computes the gradients of the layer's weights and biases, returns error signal for the previous layer.
    """
    def __init__(self, input_size, output_size, activation_func: ActivationStrategy =ReLU()):
        self.input_size = input_size
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.b = np.zeros(output_size)
        self.activation = activation_func

        # Cache for backward passes.
        self.inputs = None

        # Gradients computed during backward pass
        self.dW = None
        self.db = None

    def forward(self, inputs):
        """ Forward pass of the layer.

        Compute and return activation values, a, for use in backward pass. Cache inputs for use in backward pass.

        Args:
            inputs: shape (input_size, ), array of previous layer's activations.

        Returns:
            self.a: shape (output_size, ), array of current layer's activations.
        """
        self.inputs = inputs
        z = np.matmul(self.W, inputs) + self.b
        a = self.activation.forward(z)
        return a

    def backward(self, grad_output):
        """ Backward pass of the layer.

        Compute dL/dW, dL/db, and return the error signal for the previous layer.

        Args:
            grad_output: shape (output_size, ), array of gradient signal from the next layer.

        Returns:
            grad_input: shape (input_size, ), array of gradient signal to the previous layer.
        """
        grad_z = self.activation.backward(grad_output)

        self.dW = np.outer(grad_z, self.inputs)
        self.db = grad_z

        grad_input = np.matmul(self.W.T, grad_z)

        return grad_input

if __name__ == "__main__":
    input_vec = np.random.randn(784)
    layers = (
        DenseLayer(784, 20),
        DenseLayer(20, 10, Softmax())
    )
    current_vec = input_vec
    for layer in layers:
        current_vec = layer.forward(current_vec)
    print(current_vec)
    print(cross_entropy(current_vec, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])))



