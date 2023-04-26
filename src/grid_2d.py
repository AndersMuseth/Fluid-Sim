
import numpy as np
import particle_2d as particle

class GridCell:

    def __init__(self, x, y):
        self.pos = np.array((x, y))
        self.n_particles = 0
        self.u = x
        self.v = y
        self.particles = []



