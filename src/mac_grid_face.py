import numpy as np


class Grid_Face:

    def __init__(self):
        self.v = 0
        self.s = 0
        self.r = 0
        self.up = None
        self.down = None

    def __init__(self, down_grid, up_grid):
        self.v = 0
        self.s = 0
        self.r = 0
        self.up = up_grid
        self.down = down_grid

    #def __init__(self, down_grid, up_grid, v, s):
    #    self.v = v
    #    self.s = s
    #    self.r = 0
    #    self.up = up_grid
    #    self.down = down_grid
