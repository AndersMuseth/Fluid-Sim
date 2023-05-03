import numpy as np
import grid_cell
import mac_grid_face


class Grid:

    def __init__(self, num_cells, cell_size):
        """x_grid_faces is the faces in the staggered grid that effect the x velocities
        y_grid_faces is the faces in the staggered grid that effect the y velocities
        z_grid_faces is the faces in the staggered grid that effect the z velocities"""
        self.num_cells = num_cells
        self.cell_size = cell_size
        self.grid_cells = \
            np.array([[[grid_cell.Grid_Cell(int(0 != i and 0 != j and 0 != k and i != num_cells - 1
                                                and j != num_cells - 1 and k != num_cells - 1)) for i in range(num_cells)]
                       for j in range(num_cells)] for k in range(num_cells)])
        num_faces = num_cells + 1
        self.x_grid_faces = [[[0 for _ in range(num_cells)] for _ in range(num_cells)] for _ in range(num_faces)]
        self.y_grid_faces = [[[0 for _ in range(num_cells)] for _ in range(num_faces)] for _ in range(num_cells)]
        self.z_grid_faces = [[[0 for _ in range(num_faces)] for _ in range(num_cells)] for _ in range(num_cells)]
        for i in range(num_faces):
            for j in range(num_cells):
                for k in range(num_cells):
                    if i == 0:
                        self.x_grid_faces[i][j][k] = mac_grid_face.Grid_Face(None, self.grid_cells[i][j][k], s=0)
                    elif i == self.num_cells:
                        self.x_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i - 1][j][k], None, s=0)
                    else:
                        self.x_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i - 1][j][k],
                                                                             self.grid_cells[i][j][k], s=1)
        for i in range(num_cells):
            for j in range(num_faces):
                for k in range(num_cells):
                    if j == 0:
                        self.y_grid_faces[i][j][k] = mac_grid_face.Grid_Face(None, self.grid_cells[i][j][k], s=0)
                    elif j == self.num_cells:
                        self.y_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i][j - 1][k], None, s=0)
                    else:
                        self.y_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i][j - 1][k],
                                                                             self.grid_cells[i][j][k], s=1)
        for i in range(num_cells):
            for j in range(num_cells):
                for k in range(num_faces):
                    if k == 0:
                        self.z_grid_faces[i][j][k] = mac_grid_face.Grid_Face(None, self.grid_cells[i][j][k], s=0)
                    elif k == self.num_cells:
                        self.z_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i][j][k - 1], None, s=0)
                    else:
                        self.z_grid_faces[i][j][k] = mac_grid_face.Grid_Face(self.grid_cells[i][j][k - 1],
                                                                             self.grid_cells[i][j][k], s=1)
        self.x_grid_faces = np.array(self.x_grid_faces)
        self.y_grid_faces = np.array(self.y_grid_faces)
        self.z_grid_faces = np.array(self.z_grid_faces)

        def set_zero(face):
            face.v = 0
            face.r = 0

        self.vf = np.vectorize(set_zero)

        def divide_r(face):
            if face.r == 0:
                face.v = 0
            else:
                face.v = face.v / face.r

        self.rf = np.vectorize(divide_r)

    def clear_faces(self):
        self.vf(self.x_grid_faces)
        self.vf(self.y_grid_faces)
        self.vf(self.z_grid_faces)

    def r_div(self):
        self.rf(self.x_grid_faces)
        self.rf(self.y_grid_faces)
        self.rf(self.z_grid_faces)
