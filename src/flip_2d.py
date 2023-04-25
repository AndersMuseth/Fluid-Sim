import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm
import particle_2d
import grid_2d


max_particles = 100
num_cells = 50
cell_size = 2
c_u = np.zeros(num_cells + 1)
c_v = np.zeros(num_cells + 1)
s_u = np.ones(num_cells + 1)
s_u[0] = 0
s_u[num_cells] = 0
s_v = np.ones(num_cells + 1)
s_v[0] = 0
s_v[num_cells] = 0
overRelaxation = 1.9
rest_density = 0
grid_cells = np.array([[grid_2d.GridCell(i, j) for i in range(num_cells)] for j in range(num_cells)])
particles = np.array([particle_2d.Particle2d(20.0 + 10 * i, 50.0, 0.0, 0.0) for i in range(6)])
for i in particles:
    px = i.pos[0] // cell_size
    py = i.pos[1] // cell_size
    i.grid = grid_cells[int(py)][int(px)]
    i.grid.n_particles += 1
plot_every = 5
scatter_dot_size = 1000
domain_x_lim = np.array([5, num_cells * cell_size - 5])
domain_y_lim = np.array([5, num_cells * cell_size - 5])
time_step = 0.01
time_steps = 3000
damping_coef = -0.9


def integrateParticles(dt, gravity):
    for i in particles:
        i.vel[1] += dt * gravity
        i.pos += i.vel * dt
        p_x = int(i.pos[0] // cell_size)
        p_y = int(i.pos[1] // cell_size)
        if i.grid != grid_cells[p_y][p_x]:
            i.grid.n_particles -= 1
            i.grid = grid_cells[p_y][p_x]


def gtp(particle):
    pass

def handle_out_of_bounds():
    tempx = [particles[i].pos[0] for i in range(len(particles))]
    tempy = [particles[i].pos[1] for i in range(len(particles))]
    out_of_left = tempx < domain_x_lim[0]
    out_of_right = tempx > domain_x_lim[1]
    out_of_bottom = tempy < domain_y_lim[0]
    out_of_top = tempy > domain_y_lim[1]

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

def main():
    for t in tqdm(range(time_steps)):
        plt.style.use("dark_background")
        integrateParticles(time_step, -9.8)
        handle_out_of_bounds()


        if t % plot_every == 0:
            plt.scatter(
                [particles[i].pos[0] for i in range(len(particles))],
                [particles[i].pos[1] for i in range(len(particles))],
                s=scatter_dot_size,
                c=[particles[i].pos[1] for i in range(len(particles))],
                cmap="Blues_r"
            )
            plt.xlim(domain_x_lim)
            plt.ylim(domain_y_lim)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()



if __name__ == "__main__":
    main()