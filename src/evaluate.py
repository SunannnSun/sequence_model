# src/evaluate.py

import torch
import torch.nn as nn
import math


def evaluate_model(model, criterion, X_test, Y_test):
    model.eval()  
    with torch.no_grad():  
        predictions = model(X_test) 
        test_loss = criterion(predictions, Y_test)  

    print(f'Test Loss: {test_loss}')

    return test_loss.item(), predictions.cpu().numpy()  

