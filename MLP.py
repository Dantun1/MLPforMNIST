from activation_funcs import Softmax
from layers import DenseLayer
from loss_funcs import cross_entropy


class MLP:
    """
    Class for a simple multi-layer perceptron with 1 hidden layer.

    """
    def __init__(self, input_size: int = 784, hidden_size: int = 120, output_size: int = 10, lr: float = 0.01) -> None :
        self.layers = [
            DenseLayer(input_size, hidden_size),
            DenseLayer(hidden_size, output_size, Softmax())
        ]
        self.lr = lr

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred, y_true):
        grad_loss = y_pred - y_true

        for layer in reversed(self.layers):
            grad_loss = layer.backward(grad_loss)

    def update(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

    def train(self, x, y_true):
        y_pred = self.forward(x)
        loss = cross_entropy(y_pred, y_true)
        self.backward(y_pred, y_true)
        self.update()
        return loss


