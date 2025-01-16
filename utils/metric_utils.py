import numpy as np

def create_metric(key_list):
    return {key: DataProcessor() for key in key_list}

def add_metric(metric, key_list):
    for key in key_list:
        metric[key] = DataProcessor()
    return metric

class DataProcessor:
    def __init__(self):
        self.data = [] # for normal data analysis
        self.sub_processors = []  # for server, process the data collected

    def clear(self):
        self.data = []
        self.sub_processors = []

    def is_empty(self):
        return len(self.data) == 0

    def append(self, item):
        self.data.append(item)

    def avg(self):
        if self.is_empty(): return -1
        return np.mean(self.data)

    def std(self):
        if self.is_empty(): return -1
        return np.std(self.data)

    def min(self):
        if self.is_empty(): return -1
        return np.min(self.data)

    def max(self):
        if self.is_empty(): return -1
        return np.max(self.data)

    def last(self):
        if self.is_empty(): return -1
        return self.data[-1]

    # ===== server specific =====
    def add_sub_processor(self, sub_processor):
        self.sub_processors.append(sub_processor)

    def clear_sub_processor(self):
        self.sub_processors = []

    # return avg and std
    def avg_sub_processor(self):
        return np.mean([sub_processor.last() for sub_processor in self.sub_processors]), np.std([sub_processor.last() for sub_processor in self.sub_processors])

    # return a list
    def avg2_sub_processor(self):
        return np.mean([sub_processor.last() for sub_processor in self.sub_processors], axis=0), np.std([sub_processor.last() for sub_processor in self.sub_processors], axis=0)