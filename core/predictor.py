import numpy as np
import math


class EfficiencyPredictor:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.base_lr = learning_rate
        self.weights = np.array([
            0.5,   # W_Prox (+): 离得越近越好 (特征是 1-dist，所以要正权重)
            0.5,   # W_Sev  (+): 火大比较值得去 (收益高)
            0.5,   # W_Bat  (+): 电多好
            0.5,   # W_Wat  (+): 水多好
            -0.5,  # W_Obs  (-): 障碍物多不好 (负权重)
            -0.3,  # W_Wind
            1.0    # Bias   (+): 总体保持乐观
        ])

        self.training_count = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def predict_prob(self, features):
        x = np.append(features, 1.0)
        return self.sigmoid(np.dot(self.weights, x))

    def train(self, features, label):
        x = np.append(features, 1.0)
        pred = self.sigmoid(np.dot(self.weights, x))

        # 学习率衰减
        current_lr = max(0.01, self.base_lr * (0.9995 ** self.training_count))

        error = pred - label
        self.weights -= current_lr * error * x

        # 权重限制
        self.weights = np.clip(self.weights, -10.0, 10.0)
        self.weights[0] = max(0.4, self.weights[0]) # W_Prox 至少为正
        self.weights[1] = max(0.1, self.weights[1]) # W_Sev 至少为正
        self.weights[2] = max(0.3, self.weights[2]) # W_Bat
        self.weights[3] = max(0.3, self.weights[3]) # W_Wat
        self.weights[5] = min(-0.2, self.weights[5]) # 强制风向权重至少是 -0.2
        self.training_count += 1
        print(f"[Learn] Label:{label} Pred:{pred:.2f} -> NewWeights:{np.round(self.weights, 3)}")