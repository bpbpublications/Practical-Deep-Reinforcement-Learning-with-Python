from collections import deque
import numpy as np


class ScreenMotion:
    frame_number = 5

    def __init__(self) -> None:
        self.frames = deque(maxlen = ScreenMotion.frame_number + 1)

    def add(self, state):
        self.frames.append(state)

    def get_frames(self):
        F = ScreenMotion.frame_number
        stacks = []
        for i in range(1, F + 1):
            stacks.append(self.frames[i])
        return np.stack(stacks)

    def get_prev_frames(self):
        F = ScreenMotion.frame_number
        return np.stack([
            self.frames[i]
            for i in range(0, F)
        ])

    def is_full(self):
        return len(self.frames) == ScreenMotion.frame_number + 1
