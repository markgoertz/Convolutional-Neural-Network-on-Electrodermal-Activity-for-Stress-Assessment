import os

class DatasetConfig:
    def __init__(self, name, target_rate, window_duration_seconds, base_path):
        self.name = name
        self.target_rate = target_rate
        self.window_duration_seconds = window_duration_seconds
        self.base_path = base_path

    def __repr__(self):
        return (f"DatasetConfig(name={self.name}, target_rate={self.target_rate}, "
                f"window_duration_seconds={self.window_duration_seconds}, base_path={self.base_path})")