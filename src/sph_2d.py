import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from sklearn import neighbors
from tqdm import tqdm

domain_width = 50
domain_height = 100

max_time = 2500
max_particle = 100
particle_spacing = 2
smoothness_len = 5 #2 * particle_spacing
time_step = 0.01
particles_every = 50
isotropic_exp = 20
base_density = 1
p_mass = 1
dynamic_viscosity = 0.5
damping_coef = - 0.9
g = np.array([[0, -0.1]])

figure_size = (4,6)
plot_every = 5
scatter_dot_size = 2000

domain_x_lim = np.array([smoothness_len, domain_width - smoothness_len])
domain_y_lim = np.array([smoothness_len, domain_height - smoothness_len])

normalization_density = (315 * p_mass) / (64 * np.pi * smoothness_len**9)
normalization_pressure = (-45 * p_mass) / (np.pi * smoothness_len**6)
normalization_viscous = (45 * dynamic_viscosity * p_mass) / (np.pi * smoothness_len**6)

# 2D cubic spline kernel function
def kernel_f(r, h):
    q = r / h
    if q > 2:
        return 0
    kernel_const = 10 / (7 * np.pi * h ** 2)
    if 1 < q <= 2:
        return (kernel_const / 4) * (2 - q) ** 3
    else:
        return kernel_const * (1 - (3 / 2) * q ** 2 * (1 - q / 2))


def main():
    n_particles = 1

    positions = np.zeros((n_particles, 2))
    velocities = np.zeros((n_particles, 2))
    forces = np.zeros((n_particles, 2))
    metadata = dict(title="sph_2d", artist="matlib", comment='')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    filename = "sph_2d.mp4"
    plt.style.use("dark_background")
    fig = plt.figure(figsize=figure_size, dpi=160)
    with writer.saving(fig, filename, dpi=160):
        for t in tqdm(range(max_time)):
            if t % particles_every == 0 and n_particles < max_particle:
                new_pos = np.array([
                    [20 + np.random.rand(), domain_y_lim[1]],
                    [25 + np.random.rand(), domain_y_lim[1]],
                    [30 + np.random.rand(), domain_y_lim[1]]
                ])

                new_vel = np.array([
                    [-3, -15],
                    [-3, -15],
                    [-3, -15]
                ])
                n_particles += 3
                positions = np.concatenate((positions, new_pos), axis=0)
                velocities = np.concatenate((velocities, new_vel), axis=0)

            neighbor_id, distances = neighbors.KDTree(positions).query_radius(positions, smoothness_len, return_distance=True, sort_results=True)
            densities = np.zeros(n_particles)

            for i in range(n_particles):
                for j, k in enumerate(neighbor_id[i]):
                    densities[i] += normalization_density * (smoothness_len**2 - distances[i][j]**2)**3

            pressures = isotropic_exp * (densities - base_density)

            forces = np.zeros((n_particles, 2))
            neighbor_id = [np.delete(x, 0) for x in neighbor_id]
            distances = [np.delete(x, 0) for x in distances]
            for i in range(n_particles):
                for j, k in enumerate(neighbor_id[i]):
                    forces[i] += normalization_pressure * - (positions[k] - positions[i]) / (distances[i][j]) * (pressures[k] + pressures[i]) / (2 * densities[k]) * (smoothness_len - distances[i][j])**2
                    forces[i] += normalization_viscous * (velocities[k] - velocities[i]) / densities[k] * (smoothness_len - distances[i][j])

            forces += g * densities[:, np.newaxis]

            velocities = velocities + time_step * forces / densities[:, np.newaxis]
            positions = positions + time_step * velocities

            out_of_left = positions[:, 0] < domain_x_lim[0]
            out_of_right = positions[:, 0] > domain_x_lim[1]
            out_of_bottom = positions[:, 1] < domain_y_lim[0]
            out_of_top = positions[:, 1] > domain_y_lim[1]

            velocities[out_of_left, 0] *= damping_coef
            positions[out_of_left, 0] = domain_x_lim[0]
            velocities[out_of_right, 0] *= damping_coef
            positions[out_of_right, 0] = domain_x_lim[1]

            velocities[out_of_bottom, 1] *= damping_coef
            positions[out_of_bottom, 1] = domain_y_lim[0]
            velocities[out_of_top, 1] *= damping_coef
            positions[out_of_top, 1] = domain_y_lim[1]

            if t % plot_every == 0:
                plt.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    s=scatter_dot_size,
                    c=positions[:, 1],
                    cmap="Blues_r"
                )
                plt.xlim(domain_x_lim)
                plt.ylim(domain_y_lim)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.tight_layout()
                plt.draw()
                plt.pause(0.0001)
                writer.grab_frame()
                plt.clf()

if __name__ == "__main__":
    main()
