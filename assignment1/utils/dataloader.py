from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

mnist = fetch_openml('mnist_784', cache=False)
mnist.data.shape



X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

X = np.concatenate([X[np.where(y==0)[0]], X[np.where(y==1)[0]]], axis=0)
y = np.concatenate([np.zeros(sum(y==0)), np.ones(sum(y==1))], axis=0)


X = X / 255.0


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print('X train shape', X_train.shape)
print('y train shape', y_train.shape)


# define the batch loader of the dataset
def batch_loader(X, y, batch, shuffle=True):
    num_samples = len(X)
    pointer = 0
    while True:
        if shuffle:
            idx = np.random.choice(num_samples, batch)
        else:
            if pointer + batch <= num_samples:
                idx = np.arange(pointer, pointer+batch)
                pointer = pointer + batch
            else:
                pointer = 0
                idx = np.arange(pointer, pointer+batch)
                pointer = pointer + batch
        yield X[idx], y[idx]