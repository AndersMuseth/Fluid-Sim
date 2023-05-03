import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import particle
import grid_cell
import mac_grid_face
import Grid

n_steps = 10
max_particles = 100
num_cells = 30
cell_size = 0.75
overRelaxation = 1.9
rest_density = [0]
plot_every = 5
scatter_dot_size = 50
domain_x_lim = np.array([cell_size, (num_cells - 1) * cell_size - 1e-8])
domain_y_lim = np.array([cell_size, (num_cells - 1) * cell_size - 1e-8])
domain_z_lim = np.array([cell_size, (num_cells - 1) * cell_size - 1e-8])
time_step = 0.01
time_steps = 3000
damping_coef = -0.9
n_iters = 20
new_p_every = 5
particle_radius = cell_size / 2

grid = Grid.Grid(num_cells, cell_size)
grid_cells = grid.grid_cells
print(grid_cells[0, 1, 2].s, grid_cells[1, 1, 2].s, grid_cells[num_cells - 1, 1, 2].s)
particles = []  # [particle.Particle(20.0 + 10 * i, 50.0, 50.0, 0.0, 0.0, 0.0) for i in range(6)]
for i in range(10):
    for j in range(10):
        for k in range(10):
            particles.append(particle.Particle(7.5 + 1 * i, 10.5 + 1 * j, 7.5 + 1 * k, 0.0, -20.0, 0.0))
for i in range(10):
    for j in range(10):
        for k in range(10):
            particles.append(particle.Particle(7.5 + 1 * i, 10.0 + 1 * j, 7.0 + 1 * k, 0.0, -20.0, 0.0))
for i in range(10):
    for j in range(10):
        for k in range(10):
            particles.append(particle.Particle(7.0 + 1 * i, 10.0 + 1 * j, 7.5 + 1 * k, 0.0, -20.0, 0.0))
for i in range(10):
    for j in range(10):
        for k in range(10):
            particles.append(particle.Particle(7.5 + 1 * i, 10.0 + 1 * j, 7.5 + 1 * k, 0.0, -20.0, 0.0))
#for i in range(4):
#    for j in range(4):
#        for k in range(4):
#            particles.append(particle.Particle(3.0 + 0.5 * i, 6.0 + 1 * j, 3.0 + 0.5 * k, 0.0, 0.0, 0.0))
#particles.append(particle.Particle(10.5, 10.5, 10.5, 0.0, 0.0, 0.0))
#particles.append(particle.Particle(10.5, 12.0, 10.5, 0.0, 0.0, 0.0))
for i in particles:
    px = i.pos[0] // cell_size
    py = i.pos[1] // cell_size
    pz = i.pos[2] // cell_size
    i.grid = grid_cells[int(px)][int(py)][int(pz)]
    i.grid.particles.append(i)
print(np.array(grid.x_grid_faces).shape)
print(np.array(grid.y_grid_faces).shape)
print(np.array(grid.z_grid_faces).shape)


def handle_out_of_bounds():
    tempx = np.array([particles[i].pos[0] for i in range(len(particles))])
    tempy = np.array([particles[i].pos[1] for i in range(len(particles))])
    tempz = np.array([particles[i].pos[2] for i in range(len(particles))])
    out_of_left = tempx < domain_x_lim[0]
    out_of_right = tempx > domain_x_lim[1]
    out_of_bottom = tempy < domain_y_lim[0]
    out_of_top = tempy > domain_y_lim[1]
    out_of_nz = tempz < domain_z_lim[0]
    out_of_pz = tempz > domain_z_lim[1]
    particlesnp = np.array(particles)
    #print("in handle", particlesnp[0].vel[1])
    for p in particlesnp[out_of_left]:
        p.vel[0] *= damping_coef
        p.pos[0] = domain_x_lim[0]

    for p in particlesnp[out_of_right]:
        p.vel[0] *= damping_coef
        p.pos[0] = domain_x_lim[1]

    for p in particlesnp[out_of_bottom]:
        p.vel[1] *= damping_coef
        p.pos[1] = domain_y_lim[0]

    for p in particlesnp[out_of_top]:
        p.vel[1] *= damping_coef
        p.pos[1] = domain_y_lim[1]

    for p in particlesnp[out_of_nz]:
        p.vel[2] *= damping_coef
        p.pos[2] = domain_z_lim[0]

    for p in particlesnp[out_of_pz]:
        p.vel[2] *= damping_coef
        p.pos[2] = domain_z_lim[1]


def integrate_particles(dt, gravity):
    for i in particles:
        i.vel[1] += dt * gravity
        i.pos += i.vel * dt


def update_grids():
    for i in particles:
        p_x = int(i.pos[0] // cell_size)
        p_y = int(i.pos[1] // cell_size)
        p_z = int(i.pos[2] // cell_size)
        if i.grid != grid_cells[p_x][p_y][p_z]:
            i.grid.particles.remove(i)
            i.grid.density = 0
            i.grid = grid_cells[p_x][p_y][p_z]
            i.grid.particles.append(i)
        i.grid.density = 0


def clear_faces():
    def clear_grid_face(pos, offset, grid_faces):
        q_indices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
        q1_i = ((pos - offset) // cell_size).astype(int)
        dxyz = pos - (q1_i * cell_size + offset)
        q1_i = tuple(q1_i)

        for i, q_idx in enumerate(q_indices):
            t = tuple(np.array(q1_i) + np.array(q_idx))
            grid_faces[t].v = 0
            grid_faces[t].r = 0

    for p in particles:
        clear_grid_face(p.pos, np.array([0, cell_size / 2, cell_size / 2]), grid.x_grid_faces)
        clear_grid_face(p.pos, np.array([cell_size / 2, 0, cell_size / 2]), grid.y_grid_faces)
        clear_grid_face(p.pos, np.array([cell_size / 2, cell_size / 2, 0]), grid.z_grid_faces)


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


def particle_to_grid(p):
    def calc_weights_and_update_grid(pos, offset, grid_faces, vel):
        q_indices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
        q1_i = (pos - offset) // cell_size
        dxyz = pos - (q1_i * cell_size + offset)
        q1_i = q1_i.astype(int)
        q1_i = tuple(q1_i)

        weights = np.zeros(8)
        for i, q_idx in enumerate(q_indices):
            weights[i] = (1 - (q_idx[0] * 1) - (1 - 2 * q_idx[0]) * (dxyz[0] / cell_size)) * \
                         (1 - (q_idx[1] * 1) - (1 - 2 * q_idx[1]) * (dxyz[1] / cell_size)) * \
                         (1 - (q_idx[2] * 1) - (1 - 2 * q_idx[2]) * (dxyz[2] / cell_size))
            t = tuple(np.array(q1_i) + np.array(q_idx))
            grid_faces[t].v += weights[i] * vel
            grid_faces[t].r += weights[i]

    calc_weights_and_update_grid(p.pos, np.array([0, cell_size / 2, cell_size / 2]), grid.x_grid_faces, p.vel[0])
    calc_weights_and_update_grid(p.pos, np.array([cell_size / 2, 0, cell_size / 2]), grid.y_grid_faces, p.vel[1])
    calc_weights_and_update_grid(p.pos, np.array([cell_size / 2, cell_size / 2, 0]), grid.z_grid_faces, p.vel[2])

def grid_to_particle(p):
    epsilon = 1e-8

    def calc_weights_and_interp(pos, offset, grid_faces):
        q_indices = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]
        q1_i = ((pos - offset) // cell_size).astype(int)
        dxyz = pos - (q1_i * cell_size + offset)
        q1_i = tuple(q1_i)

        weights = np.zeros(8)
        for i, q_idx in enumerate(q_indices):
            idx = tuple(np.array(q1_i) + np.array(q_idx))
            if not grid_faces[idx].is_undefined():
                weights[i] = (1 - (q_idx[0] * 1) - (1 - 2 * q_idx[0]) * (dxyz[0] / cell_size)) * \
                             (1 - (q_idx[1] * 1) - (1 - 2 * q_idx[1]) * (dxyz[1] / cell_size)) * \
                             (1 - (q_idx[2] * 1) - (1 - 2 * q_idx[2]) * (dxyz[2] / cell_size))

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
            d *= overRelaxation
            d -= grid_cells[i].density - rest_density[0]
            s = grid_cells[ind_1x].s + grid_cells[ind_n1x].s + \
                grid_cells[ind_1y].s + grid_cells[ind_n1y].s + \
                grid_cells[ind_1z].s + grid_cells[ind_n1z].s
            if s == 0:
                continue
            grid.x_grid_faces[i].v += d * grid_cells[ind_n1x].s / s
            grid.x_grid_faces[ind_1x].v -= d * grid_cells[ind_1x].s / s
            grid.y_grid_faces[i].v += d * grid_cells[ind_n1y].s / s
            grid.y_grid_faces[ind_1y].v -= d * grid_cells[ind_1y].s / s
            grid.z_grid_faces[i].v += d * grid_cells[ind_n1z].s / s
            grid.z_grid_faces[ind_1z].v -= d * grid_cells[ind_1z].s / s


def calculate_density():
    def add_density_to_grid(pos, offset):
        q1_i = ((pos - offset) // cell_size).astype(int)
        dxyz = pos - (q1_i * cell_size + offset)
        q1_i = tuple(q1_i)
        # q_indices = []
        q_indices = {(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)}
        if q1_i[0] < 0:
            q_indices.discard((0, 0, 0)), q_indices.discard((0, 1, 0))
            q_indices.discard((0, 1, 1)), q_indices.discard((0, 0, 1))
        if q1_i[0] >= num_cells:
            q_indices.discard((1, 0, 0)), q_indices.discard((1, 1, 0))
            q_indices.discard((1, 1, 1)), q_indices.discard((1, 0, 1))
        if q1_i[1] < 0:
            q_indices.discard((0, 0, 0)), q_indices.discard((1, 0, 1))
            q_indices.discard((1, 0, 0)), q_indices.discard((0, 0, 1))
        if q1_i[1] >= num_cells:
            q_indices.discard((0, 1, 0)), q_indices.discard((1, 1, 1))
            q_indices.discard((1, 1, 0)), q_indices.discard((0, 1, 1))
        if q1_i[2] < 0:
            q_indices.discard((0, 0, 0)), q_indices.discard((1, 1, 0))
            q_indices.discard((1, 0, 0)), q_indices.discard((0, 1, 0))
        if q1_i[2] >= num_cells:
            q_indices.discard((0, 0, 1)), q_indices.discard((1, 1, 1))
            q_indices.discard((1, 0, 1)), q_indices.discard((0, 1, 1))

        for q_idx in q_indices:
            weight = (1 - (q_idx[0] * 1) - (1 - 2 * q_idx[0]) * (dxyz[0] / cell_size)) * \
                     (1 - (q_idx[1] * 1) - (1 - 2 * q_idx[1]) * (dxyz[1] / cell_size)) * \
                     (1 - (q_idx[2] * 1) - (1 - 2 * q_idx[2]) * (dxyz[2] / cell_size))
            t = tuple(np.array(q1_i) + np.array(q_idx))
            grid.grid_cells[t].density += weight

    for p in particles:
        add_density_to_grid(p.pos, np.array([cell_size / 2, cell_size / 2, cell_size / 2]))


def separate_particles():
    epsilon = 1e-8
    for p in particles:
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    p_x = int(p.pos[0] / cell_size)
                    p_y = int(p.pos[1] / cell_size)
                    p_z = int(p.pos[2] / cell_size)
                    if 0 <= i + p_x < num_cells and 0 <= j + p_y < num_cells and 0 <= k + p_z < num_cells:
                        for p_c in grid_cells[p_x + i, p_y + j, p_z + k].particles:
                            l = LA.norm(p.pos - p_c.pos)
                            if p != p_c and l < 2 * particle_radius:
                                dir = (p.pos - p_c.pos) / (l + epsilon)
                                p.pos += dir * (2 * particle_radius - l) / 2
                                p_c.pos -= dir * (2 * particle_radius - l) / 2


def main():
    # grid_to_particle(particles[0])
    metadata = dict(title="sph_2d", artist="matlib", comment='')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    filename = "flip_test6.mp4"
    plt.style.use("dark_background")
    fig = plt.figure()
    calculate_density()
    sum_d = 0
    for p in particles:
        sum_d += p.grid.density
    rest_density[0] = sum_d / len(particles)
    #rest_density[0] = 0
    with writer.saving(fig, filename, dpi=160):
    #if True:
        for t in tqdm(range(time_steps)):

            #if t % new_p_every == 0:
            #    new_p = particle.Particle(4 + 3 * np.random.rand(), 8.0, 4 * np.random.rand() + 3.0,
            #                          -2.0, 0.0, -2.0)
            #    new_p.grid = \
            #        grid_cells[int(new_p.pos[0] / cell_size)][int(new_p.pos[1] / cell_size)][
            #        int(new_p.pos[0] / cell_size)]
            #    new_p.grid.particles.append(new_p)
            #    particles.append(new_p)
                #if len(particles) == 1:
                #    for p in particles:
                #        sum_d = p.grid.density
                #        rest_density[0] = sum_d / len(particles)

            integrate_particles(time_step, -9.8)
            separate_particles()
            handle_out_of_bounds()
            update_grids()
            clear_faces()
            #grid.clear_faces()
            for p in particles:
                particle_to_grid(p)
                if np.isnan(p.pos[0]) or np.isnan(p.pos[1]) and np.isnan(p.pos[2].isnan):
                    print(p.pos)
            r_divide()
            #grid.r_div()
            calculate_density()
            force_incompression()
            sum_p = 0
            for p in particles:
                sum_p += 1
                #print(sum_p, "before", p.vel[1])
                p_v = grid_to_particle(p)
                p.vel = p_v
                #print("after", p.vel[1])
                if np.isnan(p.pos[0]) or np.isnan(p.pos[1]) and np.isnan(p.pos[2].isnan):
                    print(p.pos, "gtp")
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
                # plt.draw()
                #plt.draw()
                #plt.pause(0.0001)
                writer.grab_frame()
                plt.clf()


if __name__ == "__main__":
    main()
