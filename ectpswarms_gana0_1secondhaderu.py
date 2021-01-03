import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random
from IPython.display import HTML
from matplotlib.animation import PillowWriter
from PIL import Image, ImageDraw

# 目的関数
# def objective_function(position):
#       return np.sum(position * position)
fig = plt.figure()
ims = []

def objective_function(position):


    # RASTRIGIN FUNCTION
    t1 = 10 * len(position)
    t2 = np.sum(position ** 2)
    t3 = -10 * np.sum(np.cos(2 * np.pi * position))
    return t1 + t2 + t3
    """
    # ACKLEY FUNCTION
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / len(position) * np.sum(position ** 2, axis=0)))
    t3 = np.e
    t4 = -np.exp(1.0 / len(position) * np.sum(np.cos(2 * np.pi * position), axis=0))
    return t1 + t2 + t3 + t4
    """

# 描画のための初期化
def init_plot(xy_min, xy_max):
    matplot_x = np.arange(xy_min, xy_max, 0.1)
    matplot_y = np.arange(xy_min, xy_max, 0.1)

    matplot_mesh_X, matplot_mesh_Y = np.meshgrid(matplot_x, matplot_y)

    Z = []
    for i in range(len(matplot_mesh_X)):
        z = []
        for j in range(len(matplot_mesh_X[0])):
            result = objective_function(np.array([matplot_mesh_X[i][j], matplot_mesh_Y[i][j]]))
            z.append(result)
        Z.append(z)
    Z = np.array(Z)
    plt.ion()
    #fig = plt.figure()
    fig.suptitle("PSO")
    axes = fig.add_subplot(111, projection='3d')
    #axes = Axes3D(fig)

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('f(x, y)')
    """
    ax = fig.add_subplot(112, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    """

    mesh_XYZ = {
        "X": matplot_mesh_X,
        "Y": matplot_mesh_Y,
        "Z": Z
    }

    return axes, mesh_XYZ

# 描画
def play_plot(axes, mesh_XYZ, positions, personal_best_positions, personal_best_scores, global_best_particle_position, velocities):
    #cm = plt.get_cmap("cool")
    im1 = axes.plot_surface(mesh_XYZ['X'], mesh_XYZ['Y'], mesh_XYZ['Z'], alpha=0.3, cmap='viridis')
    im2 = axes.scatter(positions[:,0], positions[:,1], np.apply_along_axis(objective_function, 1, positions), c="crimson", linewidth=1, label="position")
    im3 = axes.scatter(personal_best_positions[:,0], personal_best_positions[:,1], personal_best_scores, linewidth=1, c='darkslateblue', label="personal_best_positions")
    im4 = axes.scatter(global_best_particle_position[0], global_best_particle_position[1], objective_function(global_best_particle_position), linewidth=3, marker='o', c='black', label="global_best_particle_position")

    im5 = axes.quiver(positions[:,0], positions[:,1], np.apply_along_axis(objective_function, 1, positions), velocities[:,0], velocities[:,1],np.zeros(len(velocities)), color='gray', label="velocity")

    #ims.append([im1,im2,im3,im4,im5])

    plt.draw()
    #ims.append([im1])
    #plt.pause(0.1)
    #plt.cla()
    return im1,im2,im3,im4,im5


# 各粒子の位置更新
def update_positions(positions, velocities):
    positions += velocities
    return positions


# 各粒子の速度更新
def update_velocities(positions, velocities, personal_best_positions, global_best_particle_position, w=0.5, ro_max=0.14):
    rc1 = random.uniform(0, ro_max)
    rc2 = random.uniform(0, ro_max)

    velocities = velocities * w + rc1 * (personal_best_positions - positions) + rc2 * (global_best_particle_position - positions)

    return velocities

def main():
    print("Particles: ")
    number_of_particles = int(input())

    print("Dimensions: ")
    dimensions = int(input())

    print("LimitTimes: ")
    limit_times = int(input())

    xy_min, xy_max = -5, 5

    # グラフの初期化
    axes, mesh_XYZ = init_plot(xy_min, xy_max)

    # 各粒子の位置
    positions = np.array([[random.uniform(xy_min, xy_max) for _ in range(dimensions)] for _ in range(number_of_particles)])

    # 各粒子の速度
    velocities = np.zeros(positions.shape)

    # 各粒子ごとのパーソナルベスト位置
    personal_best_positions = np.copy(positions)

    # 各粒子ごとのパーソナルベストの値
    personal_best_scores = np.apply_along_axis(objective_function, 1, personal_best_positions)

    # グローバルベストの粒子ID
    global_best_particle_id = np.argmin(personal_best_scores)

    # グローバルベスト位置
    global_best_particle_position = personal_best_positions[global_best_particle_id]

    # 規定回数
    for T in range(limit_times):

        # 速度更新
        velocities = update_velocities(positions, velocities, personal_best_positions, global_best_particle_position)

        # 位置更新
        positions = update_positions(positions, velocities)

        # パーソナルベストの更新
        for i in range(number_of_particles):
            score = objective_function(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]

        # グローバルベストの更新
        global_best_particle_id = np.argmin(personal_best_scores)
        global_best_particle_position = personal_best_positions[global_best_particle_id]

        im1 = axes.plot_surface(mesh_XYZ['X'], mesh_XYZ['Y'], mesh_XYZ['Z'], alpha=0.3, cmap='viridis')
        im2 = axes.scatter(positions[:,0], positions[:,1], np.apply_along_axis(objective_function, 1, positions), c="crimson", linewidth=1, label="position")
        im3 = axes.scatter(personal_best_positions[:,0], personal_best_positions[:,1], personal_best_scores, linewidth=1, c='darkslateblue', label="personal_best_positions")
        im4 = axes.scatter(global_best_particle_position[0], global_best_particle_position[1], objective_function(global_best_particle_position), linewidth=3, marker='o', c='black', label="global_best_particle_position")

        im5 = axes.quiver(positions[:,0], positions[:,1], np.apply_along_axis(objective_function, 1, positions), velocities[:,0], velocities[:,1],np.zeros(len(velocities)), color='gray', label="velocity")

        #ims.append([im1,im2,im3,im4,im5])

        plt.draw()
        ims.append([im1])
        #ims.append([im1])
        #plt.pause(0.1)
        #plt.cla()
        # グラフ描画
        #play_plot(axes, mesh_XYZ, positions, personal_best_positions, personal_best_scores, global_best_particle_position, velocities)

    Writer = animation.writers['pillow']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    ani = animation.ArtistAnimation(fig, ims)
    ani.save('im.gif', writer=writer)
    plt.show()
    #ani = animation.ArtistAnimation(fig,ims,  blit=True)
    #ani.save('popopo.gif', writer="pillow", dpi=100)

if __name__ == '__main__':
    main()
