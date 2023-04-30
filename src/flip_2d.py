import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm
import particle_2d
import grid_2d

n_steps = 10
max_particles = 100
num_cells = 50
cell_size = 2
c_u = np.zeros((num_cells + 1, num_cells + 1))
r_u = np.zeros((num_cells + 1, num_cells + 1))
c_v = np.zeros((num_cells + 1, num_cells + 1))
r_v = np.zeros((num_cells + 1, num_cells + 1))
s_u = np.ones((num_cells + 1, num_cells + 1))
s_u[0, :] = 0
s_u[num_cells, :] = 0
s_u[:, 0] = 0
s_u[:, num_cells] = 0
s_v = np.ones((num_cells + 1, num_cells + 1))
s_v[0, :] = 0
s_v[num_cells, :] = 0
s_v[:, 0] = 0
s_v[:, num_cells] = 0
overRelaxation = 1.9
rest_density = 0
grid_cells = np.array([[grid_2d.GridCell(i, j) for i in range(num_cells)] for j in range(num_cells)])
particles = np.array([particle_2d.Particle2d(20.0 + 10 * i, 50.0, 0.0, 0.0) for i in range(6)])
for i in particles:
    px = i.pos[0] // cell_size
    py = i.pos[1] // cell_size
    i.grid = grid_cells[int(px)][int(py)]
    i.grid.n_particles += 1
    i.grid.particles.append(i)
plot_every = 5
scatter_dot_size = 1000
domain_x_lim = np.array([5, num_cells * cell_size - 5])
domain_y_lim = np.array([5, num_cells * cell_size - 5])
time_step = 0.01
time_steps = 3000
damping_coef = -0.9


def integrate_particles(dt, gravity):
    for i in particles:
        i.vel[1] += dt * gravity
        i.pos += i.vel * dt
        # print(i.pos[0])
        p_x = int(i.pos[0] // cell_size)
        p_y = int(i.pos[1] // cell_size)
        if i.grid != grid_cells[p_x][p_y]:
            i.grid.n_particles -= 1
            i.grid.particles.remove(i)
            i.grid = grid_cells[p_x][p_y]
            i.grid.particles.append(i)


def cell_solve(p):
    u1 = np.array([p.pos[0], p.pos[1] - cell_size / 2])
    u1 = u1 // cell_size
    u2 = u1 + np.array((1, 0))
    u3 = u1 + np.array((0, 1))
    u4 = u1 + np.array((1, 1))

    v1 = np.array([p.pos[0] - cell_size / 2, p.pos[1]])
    v1 = v1 // cell_size
    v2 = v1 + np.array((1, 0))
    v3 = v1 + np.array((0, 1))
    v4 = v1 + np.array((1, 1))

    return [u1, u2, u3, u4], [v1, v2, v3, v4]


def ptg(p):
    u1 = np.array([p.pos[0], p.pos[1] - cell_size / 2])
    u1 = u1 // cell_size
    u2 = u1 + np.array((1, 0))
    u3 = u1 + np.array((0, 1))
    u4 = u1 + np.array((1, 1))

    dp = p.pos - u1 * cell_size
    w1 = (1 - dp[0] / cell_size) * (1 - dp[1] / cell_size)
    w2 = (dp[0] / cell_size) * (1 - dp[1] / cell_size)
    w3 = (1 - dp[0] / cell_size) * (dp[1] / cell_size)
    w4 = (dp[0] / cell_size) * (dp[1] / cell_size)

    r_u[int(u1[0]), int(u1[1])] += w1
    r_u[int(u2[0]), int(u2[1])] += w2
    r_u[int(u3[0]), int(u3[1])] += w3
    r_u[int(u4[0]), int(u4[1])] += w4

    c_u[int(u1[0]), int(u1[1])] += w1 * p.vel[0]
    c_u[int(u2[0]), int(u2[1])] += w2 * p.vel[0]
    c_u[int(u3[0]), int(u3[1])] += w3 * p.vel[0]
    c_u[int(u4[0]), int(u4[1])] += w4 * p.vel[0]

    v1 = np.array([p.pos[0] - cell_size / 2, p.pos[1]])
    v1 = v1 // cell_size
    v2 = v1 + np.array((1, 0))
    v3 = v1 + np.array((0, 1))
    v4 = v1 + np.array((1, 1))

    dp = p.pos - v1 * cell_size
    w1 = (1 - dp[0] / cell_size) * (1 - dp[1] / cell_size)
    w2 = (dp[0] / cell_size) * (1 - dp[1] / cell_size)
    w3 = (1 - dp[0] / cell_size) * (dp[1] / cell_size)
    w4 = (dp[0] / cell_size) * (dp[1] / cell_size)

    r_v[int(v1[0]), int(v1[1])] += w1
    r_v[int(v2[0]), int(v2[1])] += w2
    r_v[int(v3[0]), int(v3[1])] += w3
    r_v[int(v4[0]), int(v4[1])] += w4

    c_v[int(v1[0]), int(v1[1])] += w1 * p.vel[1]
    c_v[int(v2[0]), int(v2[1])] += w2 * p.vel[1]
    c_v[int(v3[0]), int(v3[1])] += w3 * p.vel[1]
    c_v[int(v4[0]), int(v4[1])] += w4 * p.vel[1]


def gtp(p):
    u1 = np.array([p.pos[0], p.pos[1] - cell_size / 2])
    u1 = u1 // cell_size
    u2 = u1 + np.array((1, 0))
    u3 = u1 + np.array((0, 1))
    u4 = u1 + np.array((1, 1))

    dp = p.pos - u1 * cell_size
    w1 = (1 - dp[0] / cell_size) * (1 - dp[1] / cell_size)
    w2 = (dp[0] / cell_size) * (1 - dp[1] / cell_size)
    w3 = (1 - dp[0] / cell_size) * (dp[1] / cell_size)
    w4 = (dp[0] / cell_size) * (dp[1] / cell_size)

    vx = w1 * c_u[int(u1[0]), int(u1[1])] + w2 * c_u[int(u2[0]), int(u2[1])] \
         + w3 * c_u[int(u3[0]), int(u3[1])] + w4 * c_u[int(u4[0]), int(u4[1])]
    vx /= w1 + w2 + w3 + w4

    v1 = np.array([p.pos[0] - cell_size / 2, p.pos[1]])
    v1 = v1 // cell_size
    v2 = v1 + np.array((1, 0))
    v3 = v1 + np.array((0, 1))
    v4 = v1 + np.array((1, 1))

    dp = p.pos - v1 * cell_size
    w1 = (1 - dp[0] / cell_size) * (1 - dp[1] / cell_size)
    w2 = (dp[0] / cell_size) * (1 - dp[1] / cell_size)
    w3 = (1 - dp[0] / cell_size) * (dp[1] / cell_size)
    w4 = (dp[0] / cell_size) * (dp[1] / cell_size)

    vy = w1 * c_v[int(v1[0]), int(v1[1])] + w2 * c_v[int(v2[0]), int(v2[1])] \
         + w3 * c_v[int(v3[0]), int(v3[1])] + w4 * c_v[int(v4[0]), int(v4[1])]
    vy /= w1 + w2 + w3 + w4

    return np.array([vx, vy])


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


def force_incompress():
    for _ in range(n_steps):
        for i in range(num_cells):
            for j in range(num_cells):
                d = (c_v[i, j + 1] - c_v[i, j] + c_u[i + 1, j] - c_u[i, j])
                s = (s_v[i, j + 1] + s_v[i, j] + s_u[i + 1, j] + s_u[i, j])
                # if(s == 0):
                #    print("hey I'm zero")
                c_u[i + 1, j] -= d / 4
                c_u[i, j] += d / 4
                c_v[i + 1, j] -= d / 4
                c_v[i, j] += d / 4


def set_zeros():
    c_u[:] = np.zeros((num_cells + 1, num_cells + 1))
    r_u[:] = np.zeros((num_cells + 1, num_cells + 1))
    c_v[:] = np.zeros((num_cells + 1, num_cells + 1))
    r_v[:] = np.zeros((num_cells + 1, num_cells + 1))

def norm_r():
    c_v[:] = np.divide(c_v, r_v, out=np.zeros_like(c_v), where=r_v != 0)
    c_u[:] = np.divide(c_u, r_u, out=np.zeros_like(c_u), where=r_u != 0)

def main():
    for t in tqdm(range(time_steps)):
        plt.style.use("dark_background")
        integrate_particles(time_step, -9.8)
        handle_out_of_bounds()
        set_zeros()
        for p in particles:
            ptg(p)

        norm_r()

        force_incompress()

        for p in particles:
            vel_change = gtp(p)
            p.vel = vel_change

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
