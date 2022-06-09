import numpy as np


class Rotations:
    def __init__(self):
        self.rotations = [
            lambda data: data,
            lambda data: np.rot90(data, axes=(2, 1)),
            lambda data: np.rot90(np.rot90(data, axes=(2, 1)), axes=(2, 3)),
            lambda data: np.flip(np.rot90(data, axes=(2, 1)), axis=(2, 3)),
            lambda data: np.rot90(np.rot90(data, axes=(2, 1)), axes=(3, 2)),
            lambda data: np.flip(data, axis=(2, 1)),
            lambda data: np.rot90(np.flip(data, axis=(2, 1)), axes=(2, 3)),
            lambda data: np.flip(data, axis=(3, 1)),
            lambda data: np.rot90(np.flip(data, axis=(2, 1)), axes=(3, 2)),
            lambda data: np.rot90(data, axes=(1, 2)),
            lambda data: np.rot90(np.rot90(data, axes=(1, 2)), axes=(2, 3)),
            lambda data: np.flip(np.rot90(data, axes=(1, 2)), axis=(2, 3)),
            lambda data: np.rot90(np.rot90(data, axes=(1, 2)), axes=(3, 2)),
            lambda data: np.rot90(data, axes=(2, 3)),
            lambda data: np.flip(data, axis=(2, 3)),
            lambda data: np.rot90(data, axes=(3, 2)),
            lambda data: np.rot90(np.rot90(data, axes=(2, 1)), axes=(3, 1)),
            lambda data: np.rot90(np.rot90(data, axes=(3, 2)), axes=(1, 2)),
            lambda data: np.rot90(np.flip(data, axis=(2, 1)), axes=(3, 1)),
            lambda data: np.rot90(data, axes=(1, 3)),
            lambda data: np.rot90(np.rot90(data, axes=(1, 2)), axes=(3, 1)),
            lambda data: np.rot90(np.rot90(data, axes=(2, 1)), axes=(1, 3)),
            lambda data: np.rot90(data, axes=(3, 1)),
            lambda data: np.rot90(np.flip(data, axis=(2, 3)), axes=(3, 1))
        ]

    def rotation(self, data, k):
        return self.rotations[k](data)
