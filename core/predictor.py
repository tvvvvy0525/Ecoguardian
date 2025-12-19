# core/predictor.py
import numpy as np
import math

class EfficiencyPredictor:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        # 初始化权重: [w_diff_dist, w_fire_size, bias]
        # 初始假设: 
        # w1 (diff_dist) 应该是负数 (我比队友越近越好 -> diff越小越好 -> 概率越高)
        # 但为了让它自己学，我们初始化为 0 或小随机数
        self.weights = np.array([0.0, 0.0, 0.0]) 
        self.training_count = 0

    def sigmoid(self, x):
        # 防止溢出
        if x > 20: return 0.9999
        if x < -20: return 0.0001
        return 1 / (1 + math.exp(-x))

    def predict_prob(self, diff_dist, fire_size):
        """
        预测任务成功的概率 P(Success)
        diff_dist = (MyDist - TeammateDist)
        fire_size = 火团大小
        """
        # 特征向量 [x1, x2, 1] (1是bias input)
        features = np.array([diff_dist, fire_size, 1.0])
        
        # 线性组合 z = w . x
        z = np.dot(self.weights, features)
        
        # Sigmoid 激活
        prob = self.sigmoid(z)
        return prob

    def train(self, features, label):
        """
        在线训练 (SGD 更新)
        features: (diff_dist, fire_size)
        label: 1 (成功) / 0 (失败)
        """
        diff_dist, fire_size = features
        x = np.array([diff_dist, fire_size, 1.0])
        
        # 前向传播
        pred = self.sigmoid(np.dot(self.weights, x))
        
        # 计算梯度 (Cross Entropy Loss 的梯度)
        # Gradient = (Pred - Label) * x
        error = pred - label
        gradient = error * x
        
        # 更新权重
        self.weights -= self.lr * gradient
        self.training_count += 1
        
        # print(f"[Learn] Label:{label} Pred:{pred:.2f} -> NewWeights:{self.weights}")