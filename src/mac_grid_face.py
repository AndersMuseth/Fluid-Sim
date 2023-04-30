import numpy as np


class Grid_Face:

    def __init__(self, down_grid=None, up_grid=None, v=0, s=0):
        """This object is the velocity of a face in the 3d staggered grid
        up and down are pointers to the two grid cells that are adjacent to the face
        s is the term used to force incompressibility and r is the running sum of weights
        and v is the velocity perpendicular to the face (scalar/float)"""
        self.v = v
        self.s = s
        self.r = 0
        self.up = up_grid
        self.down = down_grid
