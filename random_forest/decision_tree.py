import pandas as pd
import numpy as np
import math


class DecisionTree():
    def __init__(self, x, y, min_leaf):
        self.x = x
        self.y = y
        self.min_leaf = min_leaf
        self.row_count = x.shape[0]
        self.idxs = np.array(range(self.row_count))
        self.category_count = x.shape[1]
        self.val = np.mean(y.values[self.idxs])
        self.score = float('inf')

        self.left_decision_tree = None
        self.right_decision_tree = None
        self.splitting_category_id = None
        self.split_val = None

        self.find_split_category()

    def find_split_category(self):
        for i in range(self.category_count):
            self.check_category_for_split(i)

        if self.is_leaf:
            return

        x = self.split_col

        lhs = np.nonzero(x <= self.split_val)[0]
        rhs = np.nonzero(x > self.split_val)[0]

        self.left_decision_tree = DecisionTree(
            self.x.iloc[lhs], self.y.iloc[lhs], self.min_leaf)
        self.right_decision_tree = DecisionTree(
            self.x.iloc[rhs], self.y.iloc[rhs], self.min_leaf)

    def check_category_for_split(self, category_id):
        x = self.x.values[self.idxs, category_id]
        y = self.y.values[self.idxs]

        sorted_idx = np.argsort(x)
        sorted_x = x[sorted_idx]
        sorted_y = y[sorted_idx]

        # Standard deviation criterion specific
        # TODO Refactor code and create other criterion

        rhs_count = self.row_count
        rhs_sum = sorted_y.sum()
        rhs_square_sum = (sorted_y ** 2).sum()

        lhs_count = 0
        lhs_sum = 0.0
        lhs_square_sum = 0.0

        for i in range(0, self.row_count - self.min_leaf):
            x_i = sorted_x[i]
            y_i = sorted_y[i]

            lhs_count += 1
            rhs_count -= 1

            lhs_sum += y_i
            rhs_sum -= y_i

            lhs_square_sum += y_i ** 2
            rhs_square_sum -= y_i ** 2

            if i < self.min_leaf - 1 or x_i == sorted_x[i + 1]:
                continue

            lhs_std = self.std_deviation_score(
                lhs_count, lhs_sum, lhs_square_sum)
            rhs_std = self.std_deviation_score(
                rhs_count, rhs_sum, rhs_square_sum)
            curr_score = lhs_std * lhs_count + rhs_std * rhs_count

            if curr_score < self.score:
                self.splitting_category_id = category_id
                self.score = curr_score
                self.split_val = x_i

    def std_deviation_score(self, count, sum_val, square_sum):
        return math.sqrt((square_sum / count) - (sum_val / count) ** 2)

    @property
    def split_name(self):
        return self.x.columns[self.splitting_category_id]

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.splitting_category_id]

    @property
    def is_leaf(self):
        return self.score == float('inf')

    def predict(self, x):
        rows = [self.predict_row(x_i[1]) for x_i in x.iterrows()]
        return np.array(rows)

    def predict_row(self, x_i):
        if self.is_leaf:
            return self.val

        if x_i[self.splitting_category_id] <= self.split_val:
            return self.left_decision_tree.predict_row(x_i)
        else:
            return self.right_decision_tree.predict_row(x_i)
