import numpy as np
import pandas as pd
from random_forest import DecisionTree


class RandomForest():
    def __init__(self, n_estimators, sample_size, min_leaf=1):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.min_leaf = min_leaf
        self.trees = None

    def create_tree(self, x, y):
        idxs = np.random.randint(len(y), size=self.sample_size)
        return DecisionTree(x.iloc[idxs], y.iloc[idxs], self.min_leaf)

    def fit(self, x, y):
        self.trees = None
        self.trees = [self.create_tree(x, y) for i in range(self.n_estimators)]

    def predict(self, x):
        tree_predictions = [tree.predict(x) for tree in self.trees]
        prediction = np.mean(tree_predictions, axis=0)
        return prediction
