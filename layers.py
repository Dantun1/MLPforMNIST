import numpy as np
from activation_funcs import *


class DenseLayer:
    def __init__(self, input_size, output_size, activation_func: ActivationStrategy =ReLU()):
        self.input_size = input_size
        self.W = np.random.randn(output_size, input_size)
        self.b = np.zeros(output_size)
        self.activation = activation_func

    def forward(self, inputs):
        lin_comb = np.matmul(self.W, inputs) + self.b
        return self.activation.forward(lin_comb)

    def backward(self):
        ...





if __name__ == "__main__":
    input_vec = np.random.randn(784)
    layers = (
        DenseLayer(784, 20),
        DenseLayer(20, 10, Softmax())
    )
    current_vec = input_vec
    for layer in layers:
        current_vec = layer.forward(current_vec)

