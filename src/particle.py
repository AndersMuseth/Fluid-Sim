import numpy as np
class Particle:
    def __init__(self, x_p, y_p, v_x, v_y, g):
        self.pos = np.array([x_p, y_p])
        self.vel = np.array([x_p, y_p])
        self.F = np.array([0.0, g])
        self.m = 1