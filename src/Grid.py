import numpy as np
import grid_cell
import mac_grid_face


class Grid:

    def __init__(self, num_cells, cell_size):
        self.num_cells = num_cells
        self.cell_size = cell_size
        self.grid_cells = \
            np.array([[[grid_cell.Grid_Cell() for i in range(num_cells)] for j in range(num_cells)] for k in
                      range(num_cells)])
        num_faces = num_cells + 1
        self.x_grid_faces = [[[0] * num_cells] * num_cells] * num_faces
        for i in range(num_faces):
            for j in range(num_cells):
                for k in range(num_cells):
                    if i == 0:
                        self.x_grid_faces[i][j][k] = mac_grid_face.Grid_Face(None, self.grid_cells[i][j][k])
                    elif i == self.num_cells:
                        self.x_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i - 1][j][k], None)
                    else:
                        self.x_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i - 1][j][k],
                                                                             self.grid_cells[i][j][k])
        self.y_grid_faces = [[[0] * num_cells] * num_faces] * num_cells
        for i in range(num_cells):
            for j in range(num_faces):
                for k in range(num_cells):
                    if j == 0:
                        self.y_grid_faces[i][j][k] = mac_grid_face.Grid_Face(None, self.grid_cells[i][j][k])
                    elif j == self.num_cells:
                        self.y_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i][j - 1][k], None)
                    else:
                        self.y_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i][j - 1][k],
                                                                             self.grid_cells[i][j][k])
        self.z_grid_faces = [[[0] * num_faces] * num_cells] * num_cells
        for i in range(num_cells):
            for j in range(num_cells):
                for k in range(num_faces):
                    if k == 0:
                        self.z_grid_faces[i][j][k] = mac_grid_face.Grid_Face(None, self.grid_cells[i][j][k])
                    elif k == self.num_cells:
                        self.z_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i][j][k - 1], None)
                    else:
                        self.z_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i][j][k - 1],
                                                                             self.grid_cells[i][j][k])


