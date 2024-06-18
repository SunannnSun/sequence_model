# src/plot.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt



def plot_result(X_train, X_test):
    if torch.is_tensor(X_train):
        x_train = X_train.cpu().numpy()
    else:
        x_train = X_train
    if torch.is_tensor(X_test):
        x_test = X_test.cpu().numpy()
    else:
        x_test = X_test

    plt.figure(figsize=(10, 6))
    plt.plot(x_train[:, 0], x_train[:, 1], 'bo', label='Actual') 
    plt.plot(x_test[:, 0], x_test[:, 1], 'ro', label='Predictions')  
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.title('Actual vs Predictions')
