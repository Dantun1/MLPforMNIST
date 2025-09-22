import pandas as pd
import numpy as np
from MLP import MLP

mnist_train = pd.read_csv('mnist_train.csv')
mnist_test = pd.read_csv('mnist_test.csv')

my_mlp = MLP(lr = 0.001)

# Train the MLP

X = mnist_train.drop(columns=['label']).to_numpy()
y = mnist_train['label'].to_numpy()
y_onehot = np.eye(10)[y]

#FIXME: The loss converges on 2.3 as network predicts uniform probabilities

epochs = 5
for epoch in range(epochs):
    losses = []
    for i in range(len(X)):
        x = X[i]
        y = y_onehot[i]
        loss = my_mlp.train(x, y)

        if i % 1000 == 0:
            print(f"Training sample {i}, Loss: {loss}")
        losses.append(loss)
    print(f"Epoch {epoch+1}/{epochs}, Average loss: {np.mean(losses)}")







