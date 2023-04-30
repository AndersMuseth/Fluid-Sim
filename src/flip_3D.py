import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm
import particle
import grid_cell


n_steps = 10
max_particles = 100
num_cells = 50
cell_size = 2
overRelaxation = 1.9
rest_density = 0
plot_every = 10
scatter_dot_size = 50
domain_x_lim = np.array([5, num_cells * cell_size - 5])
domain_y_lim = np.array([5, num_cells * cell_size - 5])
domain_z_lim = np.array([5, num_cells * cell_size - 5])
time_step = 0.01
time_steps = 3000
damping_coef = -0.9

grid_cells = \
    np.array([[[grid_cell.Grid_Cell() for i in range(num_cells)]for j in range(num_cells)] for k in range(num_cells)])
particles = np.array([particle.Particle(20.0 + 10 * i, 50.0, 50.0, 0.0, 0.0, 0.0) for i in range(6)])
for i in particles:
    px = i.pos[0] // cell_size
    py = i.pos[1] // cell_size
    pz = i.pos[2] // cell_size
    i.grid = grid_cells[int(px)][int(py)][int(pz)]
    i.grid.particles.append(i)
    print(grid_cells[int(px)][int(py)][int(pz)])


def integrate_particles(dt, gravity):
    for i in particles:
        i.vel[1] += dt * gravity
        i.pos += i.vel * dt
        # print(i.pos[0])
        p_x = int(i.pos[0] // cell_size)
        p_y = int(i.pos[1] // cell_size)
        p_z = int(i.pos[2] // cell_size)
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


def main():
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