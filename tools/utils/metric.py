import numpy as np


def create_metric(metric_type):
    if metric_type == 'avg':
        return AvgMetric()

    if metric_type == 'max':
        return MaxMetric()

    if metric_type == 'min':
        return MinMetric()


class Metric(object):

    def __init__(self):
        self.curr_val = None

    def update(self):
        raise NotImplementedError

    def val(self):
        return self.curr_val

    def reset(self):
        self.curr_val = None


class MinMetric(Metric):

    def __init__(self):
        super().__init__()
        self.curr_val = np.inf

    def update(self, update_val):
        self.curr_val = min(self.curr_val, update_val)
        return self.curr_val


class MaxMetric(Metric):

    def __init__(self):
        super().__init__()
        self.curr_val = -np.inf

    def update(self, update_val):
        self.curr_val = max(self.curr_val, update_val)
        return self.curr_val


class AvgMetric(Metric):

    def __init__(self):
        super().__init__()
        self.curr_sum = 0
        self.tot_count = 0

    def update(self, update_val, count=1):
        self.tot_count += count
        self.curr_sum += update_val
        self.curr_val = self.curr_sum / self.tot_count
        return self.curr_val

    def reset(self):
        self.tot_count = 0
        self.curr_sum = 0
        super().reset()
