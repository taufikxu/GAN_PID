import numpy as np


class Random_queue(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.length = 0
        self.data = []

    def set_data(self, samples):
        if self.length < self.capacity:
            for i in range(samples.shape[0]):
                self.data.append(samples[i:i + 1])
                self.length += 1
        else:
            permutation = np.random.permutation(self.length)
            for i in range(samples.shape[0]):
                self.data[permutation[i]] = samples[i:i + 1]

    def get_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size > self.length:
            return np.concatenate(self.data, 0)

        results = []
        permutation = np.random.permutation(self.length)
        for i in range(batch_size):
            results.append(self.data[permutation[i]])
        return np.concatenate(results, 0)
