import numpy as np

class Particle:

    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.m = 1
        self.grid = None
    def __init__(self, x_p, y_p, z_p,  v_x, v_y, v_z):
        self.pos = np.array([x_p, y_p, z_p])
        self.vel = np.array([v_x, v_y, v_z])
        self.m = 1
        self.grid = None