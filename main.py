import pandas as pd
import numpy as np
from MLP import MLP

mnist_train = pd.read_csv('mnist_train.csv')
mnist_test = pd.read_csv('mnist_test.csv')

my_mlp = MLP(lr = 0.01)

# Train the MLP

X = mnist_train.drop(columns=['label']).to_numpy()
y = mnist_train['label'].to_numpy()
y_onehot = np.eye(10)[y]


X = X / 255.0

epochs = 5
for epoch in range(epochs):
    losses = []
    for i in range(len(X)):
        x = X[i]
        y = y_onehot[i]
        loss = my_mlp.train(x, y)

        if i % 10000 == 0:
            print(f"Training sample {i}, Loss: {loss}")
        losses.append(loss)
    print(f"Epoch {epoch+1}/{epochs}, Average loss: {np.mean(losses)}")


# Test it

X_test = mnist_train.drop(columns=['label']).to_numpy()
y_test = mnist_train['label'].to_numpy()
y_onehot_test = np.eye(10)[y_test]

X_test = X_test / 255.0

correct = 0

for i in range(len(X_test)):
    y_pred = my_mlp.forward(X_test[i])
    if np.argmax(y_pred) == np.argmax(y_onehot_test[i]):
        correct += 1

print(f"Correct vs wrong: {correct} out of {len(X_test)}, Accuracy: {correct/len(X_test)}")












