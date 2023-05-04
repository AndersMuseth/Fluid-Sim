import numpy as np


class Particle:

    def __init__(self, x_p=0, y_p=0, z_p=0, v_x=0, v_y=0, v_z=0):
        self.pos = np.array([x_p, y_p, z_p])
        self.vel = np.array([v_x, v_y, v_z])
        self.m = 1
        self.grid = None

    #def __sub__(self, other):
    #    self.vel += other.vel
    #    return self