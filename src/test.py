import numpy as np
import Grid
import particle

num_cells = 3
cell_size = 1
overRelaxation = 1.9
grid = Grid.Grid(num_cells, cell_size)
grid_cells = grid.grid_cells
# print(grid_cells[0, 1, 2].s, grid_cells[1, 1, 2].s, grid_cells[num_cells - 1, 1, 2].s)
particles = []
particles.append(particle.Particle(0, 1, 1, 10.0, 10.0, 10.0))
particles.append(particle.Particle(0, 1.2, 1.2, 5.0, 5.0, 5.0))
for i in particles:
    px = i.pos[0] // cell_size
    py = i.pos[1] // cell_size
    pz = i.pos[2] // cell_size
    print(px, py, pz)
    i.grid = grid_cells[int(px)][int(py)][int(pz)]
    i.grid.particles.append(i)
n_iters = 1
for i in range(num_cells):
    for j in range(num_cells):
        for k in range(num_cells):
            grid_cells[i, j, k].s = 1
print(np.array(grid.x_grid_faces).shape)
print(np.array(grid.y_grid_faces).shape)
print(np.array(grid.z_grid_faces).shape)


def force_incompression():
    indx = set()
    for p in particles:
        p_x = int(p.pos[0] / cell_size)
        p_y = int(p.pos[1] / cell_size)
        p_z = int(p.pos[2] / cell_size)
        indx.add((p_x, p_y, p_z))
    for _ in range(n_iters):
        for i in indx:
            ind_1x = tuple(np.array(i) + np.array((1, 0, 0)))
            ind_1y = tuple(np.array(i) + np.array((0, 1, 0)))
            ind_1z = tuple(np.array(i) + np.array((0, 0, 1)))
            ind_n1x = tuple(np.array(i) + np.array((-1, 0, 0)))
            ind_n1y = tuple(np.array(i) + np.array((0, -1, 0)))
            ind_n1z = tuple(np.array(i) + np.array((0, 0, -1)))
            d = grid.x_grid_faces[ind_1x].v - grid.x_grid_faces[i].v + \
                grid.y_grid_faces[ind_1y].v - grid.y_grid_faces[i].v + \
                grid.z_grid_faces[ind_1z].v - grid.z_grid_faces[i].v
            print("vel x: ", grid.x_grid_faces[ind_1x].v, grid.x_grid_faces[i].v, "vel y", grid.y_grid_faces[ind_1y].v, grid.y_grid_faces[i].v, "vel z", grid.z_grid_faces[ind_1z].v, grid.z_grid_faces[i].v)
            print("before", d)
            # d *= overRelaxation
            # d -= grid_cells[i].density - rest_density[0]
            s = grid_cells[ind_1x].s + grid_cells[ind_n1x].s + \
                grid_cells[ind_1y].s + grid_cells[ind_n1y].s + \
                grid_cells[ind_1z].s + grid_cells[ind_n1z].s
            #if s < 6:
            #    print(s, grid_cells[ind_n1y].s, grid_cells[ind_1y].s, grid_cells[ind_n1x].s, grid_cells[ind_1x].s,
            #          grid_cells[ind_n1z].s, grid_cells[ind_1z].s)
            grid.x_grid_faces[i].v += d * grid_cells[ind_n1x].s / s
            grid.x_grid_faces[ind_1x].v -= d * grid_cells[ind_1x].s / s
            grid.y_grid_faces[i].v += d * grid_cells[ind_n1y].s / s
            grid.y_grid_faces[ind_1y].v -= d * grid_cells[ind_1y].s / s
            grid.z_grid_faces[i].v += d * grid_cells[ind_n1z].s / s
            grid.z_grid_faces[ind_1z].v -= d * grid_cells[ind_1z].s / s
            d = grid.x_grid_faces[ind_1x].v - grid.x_grid_faces[i].v + \
                grid.y_grid_faces[ind_1y].v - grid.y_grid_faces[i].v + \
                grid.z_grid_faces[ind_1z].v - grid.z_grid_faces[i].v
            print("after", d)
            print("vel x: ", grid.x_grid_faces[ind_1x].v, grid.x_grid_faces[i].v, "vel y", grid.y_grid_faces[ind_1y].v,
                  grid.y_grid_faces[i].v, "vel z", grid.z_grid_faces[ind_1z].v, grid.z_grid_faces[i].v)


def particle_to_grid(p):
    def calc_weights_and_update_grid(pos, offset, grid_faces, vel):
        q_indices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
        q1_i = (pos - offset) // cell_size
        dxyz = pos - (q1_i * cell_size + offset)
        q1_i = q1_i.astype(int)
        q1_i = tuple(q1_i)
        #print(q1_i)
        #print(dxyz)
        weights = np.zeros(8)
        for i, q_idx in enumerate(q_indices):
            t = tuple(np.array(q1_i) + np.array(q_idx))
            #if not grid_faces[t].is_undefined():
            weights[i] = (1 - abs((dxyz[0] / cell_size) - q_idx[0])) * \
                         (1 - abs((dxyz[1] / cell_size) - q_idx[1])) * \
                         (1 - abs((dxyz[2] / cell_size) - q_idx[2]))
            grid_faces[t].v += weights[i] * vel
            #print(grid_faces[t].v)
            grid_faces[t].r += weights[i]
        print("weights ptg ", weights)
    calc_weights_and_update_grid(p.pos, np.array([0, cell_size / 2, cell_size / 2]), grid.x_grid_faces, p.vel[0])
    calc_weights_and_update_grid(p.pos, np.array([cell_size / 2, 0, cell_size / 2]), grid.y_grid_faces, p.vel[1])
    calc_weights_and_update_grid(p.pos, np.array([cell_size / 2, cell_size / 2, 0]), grid.z_grid_faces, p.vel[2])

def grid_to_particle(p):
    #epsilon = 1e-8

    def calc_weights_and_interp(pos, offset, grid_faces):
        q_indices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
        q1_i = (pos - offset) // cell_size
        dxyz = pos - (q1_i * cell_size + offset)
        q1_i = q1_i.astype(int)
        q1_i = tuple(q1_i)

        weights = np.zeros(8)
        for i, q_idx in enumerate(q_indices):
            idx = tuple(np.array(q1_i) + np.array(q_idx))
            if not grid_faces[idx].is_undefined():
                weights[i] = (1 - abs((dxyz[0] / cell_size) - q_idx[0])) * \
                             (1 - abs((dxyz[1] / cell_size) - q_idx[1])) * \
                             (1 - abs((dxyz[2] / cell_size) - q_idx[2]))

        print("weights gtp ", weights)
        qp = sum(
            weights[i] * grid_faces[tuple(np.array(q1_i) + np.array(q_idx))].v for i, q_idx in enumerate(q_indices))
        total_weights = np.sum(weights)
        if total_weights != 0:
            qp /= total_weights
        else:
            qp = 0

        return qp

    qpx = calc_weights_and_interp(p.pos, np.array([0, cell_size / 2, cell_size / 2]), grid.x_grid_faces)
    qpy = calc_weights_and_interp(p.pos, np.array([cell_size / 2, 0, cell_size / 2]), grid.y_grid_faces)
    qpz = calc_weights_and_interp(p.pos, np.array([cell_size / 2, cell_size / 2, 0]), grid.z_grid_faces)

    return np.array([qpx, qpy, qpz])

def r_divide():
    epsilon = 1e-8

    def r_divide_face(pos, offset, grid_faces):
        q_indices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
        q1_i = ((pos - offset) // cell_size).astype(int)
        dxyz = pos - (q1_i * cell_size + offset)
        q1_i = tuple(q1_i)

        for i, q_idx in enumerate(q_indices):
            t = tuple(np.array(q1_i) + np.array(q_idx))
            if grid_faces[t].r == 0:
                pass
            else:
                grid_faces[t].v /= grid_faces[t].r
                grid_faces[t].r = 0

    for p in particles:
        r_divide_face(p.pos, np.array([0, cell_size / 2, cell_size / 2]), grid.x_grid_faces)
        r_divide_face(p.pos, np.array([cell_size / 2, 0, cell_size / 2]), grid.y_grid_faces)
        r_divide_face(p.pos, np.array([cell_size / 2, cell_size / 2, 0]), grid.z_grid_faces)


def main():
    particle_to_grid(particles[0])
    particle_to_grid(particles[1])
    r_divide()
    print("gtp", grid_to_particle(particles[0]))
    print("gtp", grid_to_particle(particles[1]))
    force_incompression()
    print(grid_to_particle(particles[0]))
    grid_cells[1, 1, 0].s = 0
    particle_to_grid(particles[0])
    force_incompression()
    print(grid_to_particle(particles[0]))


if __name__ == "__main__":
    main()

