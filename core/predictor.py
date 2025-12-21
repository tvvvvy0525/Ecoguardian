import numpy as np
import math


class EfficiencyPredictor:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.weights = np.zeros(6)  # [清单8] 5个特征 + 1个 Bias
        self.training_count = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def predict_prob(self, features):
        x = np.append(features, 1.0)
        return self.sigmoid(np.dot(self.weights, x))

    def train(self, features, label):
        x = np.append(features, 1.0)
        pred = self.sigmoid(np.dot(self.weights, x))
        self.weights -= self.lr * (pred - label) * x
        self.training_count += 1
        print(f"[Learn] Label:{label} Pred:{pred:.2f} -> NewWeights:{self.weights}")
