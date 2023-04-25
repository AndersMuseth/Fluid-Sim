import numpy as np


class Particle2d:
    def __init__(self, x_p, y_p, v_x, v_y):
        self.pos = np.array([x_p, y_p])
        self.vel = np.array([v_x, v_y])
        self.m = 1
        self.grid = None
