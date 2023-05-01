import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm
import particle
import grid_cell
import mac_grid_face
import Grid

n_steps = 10
max_particles = 100
num_cells = 50
cell_size = 2
overRelaxation = 1.9
rest_density = 0
plot_every = 10
scatter_dot_size = 50
domain_x_lim = np.array([1, num_cells * cell_size - 1])
domain_y_lim = np.array([1, num_cells * cell_size - 1])
domain_z_lim = np.array([1, num_cells * cell_size - 1])
time_step = 0.01
time_steps = 3000
damping_coef = -0.9

grid = Grid.Grid(num_cells, cell_size)
grid_cells = grid.grid_cells
particles = np.array([particle.Particle(20.0 + 10 * i, 50.0, 50.0, 0.0, 0.0, 0.0) for i in range(6)])
for i in particles:
    px = i.pos[0] // cell_size
    py = i.pos[1] // cell_size
    pz = i.pos[2] // cell_size
    i.grid = grid_cells[int(px)][int(py)][int(pz)]
    i.grid.particles.append(i)
print(np.array(grid.x_grid_faces).shape)
print(np.array(grid.y_grid_faces).shape)
print(np.array(grid.z_grid_faces).shape)


def integrate_particles(dt, gravity):
    for i in particles:
        i.vel[1] += dt * gravity
        i.pos += i.vel * dt
        # print(i.pos[0])
        p_x = int(i.pos[0] // cell_size)
        p_y = int(i.pos[1] // cell_size)
        p_z = int(i.pos[2] // cell_size)
        # print(p_x, p_y, p_z)
        if i.grid != grid_cells[p_x][p_y][p_z]:
            i.grid.particles.remove(i)
            i.grid = grid_cells[p_x][p_y][p_z]
            i.grid.particles.append(i)


def handle_out_of_bounds():
    tempx = [particles[i].pos[0] for i in range(len(particles))]
    tempy = [particles[i].pos[1] for i in range(len(particles))]
    tempz = [particles[i].pos[2] for i in range(len(particles))]
    out_of_left = tempx < domain_x_lim[0]
    out_of_right = tempx > domain_x_lim[1]
    out_of_bottom = tempy < domain_y_lim[0]
    out_of_top = tempy > domain_y_lim[1]
    out_of_nz = tempz < domain_z_lim[0]
    out_of_pz = tempz > domain_z_lim[1]

    for p in particles[out_of_left]:
        p.vel[0] *= damping_coef
        p.pos[0] = domain_x_lim[0]

    for p in particles[out_of_right]:
        p.vel[0] *= damping_coef
        p.pos[0] = domain_x_lim[1]

    for p in particles[out_of_bottom]:
        p.vel[1] *= damping_coef
        p.pos[1] = domain_y_lim[0]

    for p in particles[out_of_top]:
        p.vel[1] *= damping_coef
        p.pos[1] = domain_y_lim[1]

    for p in particles[out_of_nz]:
        p.vel[2] *= damping_coef
        p.pos[2] = domain_z_lim[0]

    for p in particles[out_of_pz]:
        p.vel[2] *= damping_coef
        p.pos[2] = domain_z_lim[1]


def l_grid_to_particle(p):
    x_q1_i = ((p.pos - np.array([0, cell_size / 2, cell_size / 2])) // cell_size).astype(int)
    x_q2_i = tuple(x_q1_i + np.array([0, 1, 0]))
    x_q3_i = tuple(x_q1_i + np.array([0, 1, 1]))
    x_q4_i = tuple(x_q1_i + np.array([0, 0, 1]))
    x_q5_i = tuple(x_q1_i + np.array([1, 0, 0]))
    x_q6_i = tuple(x_q1_i + np.array([1, 1, 0]))
    x_q7_i = tuple(x_q1_i + np.array([1, 1, 1]))
    x_q8_i = tuple(x_q1_i + np.array([1, 0, 1]))

    h = cell_size
    dxyz = p.pos - (x_q1_i * cell_size + np.array([0, cell_size / 2, cell_size / 2]))
    x_q1_i = tuple(x_q1_i)
    w1 = 0

    if not grid.x_grid_faces[x_q1_i].is_undefined():
        w1 = (1 - dxyz[0] / h) * (1 - dxyz[1] / h) * (1 - dxyz[2] / h)
    w2 = 0
    if not grid.x_grid_faces[x_q2_i].is_undefined():
        w2 = (1 - dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)
    w3 = 0
    if not grid.x_grid_faces[x_q3_i].is_undefined():
        w3 = (1 - dxyz[0] / h) * (dxyz[1] / h) * (dxyz[2] / h)
    w4 = 0
    if not grid.x_grid_faces[x_q4_i].is_undefined():
        w4 = (1 - dxyz[0] / h) * (1 - dxyz[1] / h) * (dxyz[2] / h)
    w5 = 0
    if not grid.x_grid_faces[x_q5_i].is_undefined():
        w5 = (dxyz[0] / h) * (1 - dxyz[1] / h) * (1 - dxyz[2] / h)
    w6 = 0
    if not grid.x_grid_faces[x_q6_i].is_undefined():
        w6 = (dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)
    w7 = 0
    if not grid.x_grid_faces[x_q7_i].is_undefined():
        w7 = (dxyz[0] / h) * (dxyz[1] / h) * (dxyz[2] / h)
    w8 = 0
    if not grid.x_grid_faces[x_q8_i].is_undefined():
        w8 = (dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)

    qpx = w1 * grid.x_grid_faces[x_q1_i].v + w2 * grid.x_grid_faces[x_q2_i].v + w3 * grid.x_grid_faces[x_q3_i].v + \
          w4 * grid.x_grid_faces[x_q4_i].v + w5 * grid.x_grid_faces[x_q5_i].v + w6 * grid.x_grid_faces[x_q6_i].v + \
          w7 * grid.x_grid_faces[x_q7_i].v + w8 * grid.x_grid_faces[x_q8_i].v
    total_weights_x = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8
    if total_weights_x != 0:
        qpx /= total_weights_x
    else:
        qpx = 0

    y_q1_i = ((p.pos - np.array([cell_size / 2, 0, cell_size / 2])) // cell_size).astype(int)
    y_q2_i = tuple(y_q1_i + np.array([0, 1, 0]))
    y_q3_i = tuple(y_q1_i + np.array([0, 1, 1]))
    y_q4_i = tuple(y_q1_i + np.array([0, 0, 1]))
    y_q5_i = tuple(y_q1_i + np.array([1, 0, 0]))
    y_q6_i = tuple(y_q1_i + np.array([1, 1, 0]))
    y_q7_i = tuple(y_q1_i + np.array([1, 1, 1]))
    y_q8_i = tuple(y_q1_i + np.array([1, 0, 1]))

    dxyz = p.pos - (y_q1_i * cell_size + np.array([cell_size / 2, 0, cell_size / 2]))
    y_q1_i = tuple(y_q1_i)
    w1 = 0
    if not grid.y_grid_faces[y_q1_i].is_undefined():
        w1 = (1 - dxyz[0] / h) * (1 - dxyz[1] / h) * (1 - dxyz[2] / h)
    w2 = 0
    if not grid.y_grid_faces[y_q2_i].is_undefined():
        w2 = (1 - dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)
    w3 = 0
    if not grid.y_grid_faces[y_q3_i].is_undefined():
        w3 = (1 - dxyz[0] / h) * (dxyz[1] / h) * (dxyz[2] / h)
    w4 = 0
    if not grid.y_grid_faces[y_q4_i].is_undefined():
        w4 = (1 - dxyz[0] / h) * (1 - dxyz[1] / h) * (dxyz[2] / h)
    w5 = 0
    if not grid.y_grid_faces[y_q5_i].is_undefined():
        w5 = (dxyz[0] / h) * (1 - dxyz[1] / h) * (1 - dxyz[2] / h)
    w6 = 0
    if not grid.y_grid_faces[y_q6_i].is_undefined():
        w6 = (dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)
    w7 = 0
    if not grid.y_grid_faces[y_q7_i].is_undefined():
        w7 = (dxyz[0] / h) * (dxyz[1] / h) * (dxyz[2] / h)
    w8 = 0
    if not grid.y_grid_faces[y_q8_i].is_undefined():
        w8 = (dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)

    qpy = w1 * grid.y_grid_faces[y_q1_i].v + w2 * grid.y_grid_faces[y_q2_i].v + w3 * grid.y_grid_faces[y_q3_i].v + \
          w4 * grid.y_grid_faces[y_q4_i].v + w5 * grid.y_grid_faces[y_q5_i].v + w6 * grid.y_grid_faces[y_q6_i].v + \
          w7 * grid.y_grid_faces[y_q7_i].v + w8 * grid.y_grid_faces[y_q8_i].v
    total_weights_y = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8
    if total_weights_y != 0:
        qpy /= total_weights_y
    else:
        qpy = 0

    z_q1_i = ((p.pos - np.array([cell_size / 2, cell_size / 2, 0])) // cell_size).astype(int)
    z_q2_i = tuple(z_q1_i + np.array([0, 1, 0]))
    z_q3_i = tuple(z_q1_i + np.array([0, 1, 1]))
    z_q4_i = tuple(z_q1_i + np.array([0, 0, 1]))
    z_q5_i = tuple(z_q1_i + np.array([1, 0, 0]))
    z_q6_i = tuple(z_q1_i + np.array([1, 1, 0]))
    z_q7_i = tuple(z_q1_i + np.array([1, 1, 1]))
    z_q8_i = tuple(z_q1_i + np.array([1, 0, 1]))

    dxyz = p.pos - (z_q1_i * cell_size + np.array([cell_size / 2, cell_size / 2, 0]))
    z_q1_i = tuple(z_q1_i)
    w1 = 0
    if not grid.z_grid_faces[z_q1_i].is_undefined():
        w1 = (1 - dxyz[0] / h) * (1 - dxyz[1] / h) * (1 - dxyz[2] / h)
    w2 = 0
    if not grid.z_grid_faces[z_q2_i].is_undefined():
        w2 = (1 - dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)
    w3 = 0
    if not grid.z_grid_faces[z_q3_i].is_undefined():
        w3 = (1 - dxyz[0] / h) * (dxyz[1] / h) * (dxyz[2] / h)
    w4 = 0
    if not grid.z_grid_faces[z_q4_i].is_undefined():
        w4 = (1 - dxyz[0] / h) * (1 - dxyz[1] / h) * (dxyz[2] / h)
    w5 = 0
    if not grid.z_grid_faces[z_q5_i].is_undefined():
        w5 = (dxyz[0] / h) * (1 - dxyz[1] / h) * (1 - dxyz[2] / h)
    w6 = 0
    if not grid.z_grid_faces[z_q6_i].is_undefined():
        w6 = (dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)
    w7 = 0
    if not grid.z_grid_faces[z_q7_i].is_undefined():
        w7 = (dxyz[0] / h) * (dxyz[1] / h) * (dxyz[2] / h)
    w8 = 0
    if not grid.z_grid_faces[z_q8_i].is_undefined():
        w8 = (dxyz[0] / h) * (dxyz[1] / h) * (1 - dxyz[2] / h)

    qpz = w1 * grid.z_grid_faces[z_q1_i].v + w2 * grid.z_grid_faces[z_q2_i].v + w3 * grid.z_grid_faces[z_q3_i].v + \
          w4 * grid.z_grid_faces[z_q4_i].v + w5 * grid.z_grid_faces[z_q5_i].v + w6 * grid.z_grid_faces[z_q6_i].v + \
          w7 * grid.z_grid_faces[z_q7_i].v + w8 * grid.z_grid_faces[z_q8_i].v
    total_weights_z = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8
    if total_weights_z != 0:
        qpz /= total_weights_z
    else:
        qpz = 0

    return np.array([qpx, qpy, qpz])


def grid_to_particle(p):
    def calc_weights_and_interp(pos, offset, grid_faces):
        q_indices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
        q1_i = ((pos - offset) // cell_size).astype(int)
        dxyz = pos - (q1_i * cell_size + offset)
        q1_i = tuple(q1_i)

        weights = np.zeros(8)
        for i, q_idx in enumerate(q_indices):
            idx = tuple(np.array(q1_i) + np.array(q_idx))
            if not grid_faces[idx].is_undefined():
                weights[i] = (1 - dxyz[0] / cell_size * q_idx[0]) * (1 - dxyz[1] / cell_size * q_idx[1]) * (
                            1 - dxyz[2] / cell_size * q_idx[2])

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


def main():
    grid_to_particle(particles[0])
    for t in tqdm(range(time_steps)):
        plt.style.use("dark_background")
        integrate_particles(time_step, -9.8)

        handle_out_of_bounds()
        

        if t % plot_every == 0:
            ax = plt.axes(projection="3d")
            ax.scatter(
                [particles[i].pos[0] for i in range(len(particles))],
                [particles[i].pos[2] for i in range(len(particles))],
                [particles[i].pos[1] for i in range(len(particles))],
                s=scatter_dot_size,
                c=[particles[i].pos[1] for i in range(len(particles))],
                cmap="Blues_r"
            )
            plt.xlim(domain_x_lim)
            plt.ylim(domain_y_lim)
            ax.set_zlim(domain_z_lim)
            plt.draw()
            plt.pause(0.0001)
            plt.clf()


if __name__ == "__main__":
    main()
