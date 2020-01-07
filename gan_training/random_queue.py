import numpy as np


class Random_queue(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.length = 0
        self.data = []
        self.label = []

    def set_data(self, samples, y=None):
        if self.length < self.capacity:
            for i in range(samples.shape[0]):
                self.data.append(samples[i:i + 1])
                if y is not None:
                    self.label.append(y[i:i + 1])
                self.length += 1
        else:
            permutation = np.random.permutation(self.length)
            for i in range(samples.shape[0]):
                self.data[permutation[i]] = samples[i:i + 1]
                if y is not None:
                    self.label[permutation[i]] = y[i:i + 1]

    def get_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # print(batch_size, self.length)
        if batch_size > self.length:
            if len(self.label) == 0:
                return np.concatenate(self.data, 0)
            else:
                return np.concatenate(self.data,
                                      0), np.concatenate(self.label, 0)

        results, results_l = [], []
        permutation = np.random.permutation(self.length)
        for i in range(batch_size):
            results.append(self.data[permutation[i]])
            if len(self.label) > 0:
                results_l.append(self.label[permutation[i]])

        if len(self.label) == 0:
            return np.concatenate(results, 0)
        else:
            return np.concatenate(results, 0), np.concatenate(results_l, 0)
