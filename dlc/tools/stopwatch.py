import math
import time


class StopWatch:
    def __init__(self):
        self.last_ref = None
        # For statistics computation
        self.n = 0.0
        self.sum = 0.0
        self.sum_sq = 0.0

    def start(self):
        self.last_ref = time.perf_counter()

    def stop(self):
        if self.last_ref is not None:
            now = time.perf_counter()
            elapsed_s = now - self.last_ref
            self.n += 1.0
            self.sum += elapsed_s
            self.sum_sq += elapsed_s * elapsed_s
            self.last_ref = None

    def mean(self):
        return self.sum / self.n

    def std(self):
        mean = self.sum / self.n
        mean_sq = self.sum_sq / self.n
        var = mean_sq - mean ** 2
        std = math.sqrt(var)
        return std
