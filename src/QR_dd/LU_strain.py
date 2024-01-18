import os
import numpy as np
import matplotlib.pyplot as plt

# Параметры
start_iteration = 1
iterations = 60
points = 49

# Цикл по всем итерациям
def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix_flat = [float(value) for line in lines for value in line.strip().split()]
        return np.array(matrix_flat)


# Функция для создания матрицы 2x2 из производных
def create_2x2_matrix(dudx, dudy, dvdx, dvdy):
    return np.array([[dudx, dudy], [dvdx, dvdy]])


# Главная функция для обработки файлов
def process_files(directory):
    result_array = np.zeros((31, 50, 2, 2))

    for iteration in range(1, 30):  # 30 итераций
        for point in range(49):  # 49 точек
            dudx_file = f"dudx{iteration:02d}.dat"
            dudy_file = f"dudy{iteration:02d}.dat"
            dvdx_file = f"dvdx{iteration:02d}.dat"
            dvdy_file = f"dvdy{iteration:02d}.dat"

            dudx_matrix = read_matrix_from_file(os.path.join(directory, dudx_file))
            dudy_matrix = read_matrix_from_file(os.path.join(directory, dudy_file))
            dvdx_matrix = read_matrix_from_file(os.path.join(directory, dvdx_file))
            dvdy_matrix = read_matrix_from_file(os.path.join(directory, dvdy_file))

            result_matrix = create_2x2_matrix(
                dudx_matrix[point],
                dudy_matrix[point],
                dvdx_matrix[point],
                dvdy_matrix[point]
            )

            result_array[iteration, point] = result_matrix

    return result_array + np.eye(2)


def cauchy_green(F):
    C = F.transpose() @ F
    return np.array(C)


def upper_triangle(C):
    F = np.zeros((2, 2))
    F[0, 0] = np.sqrt(C[0, 0])
    F[0, 1] = C[0, 1] / F[0, 0]
    F[1, 1] = np.sqrt(C[1, 1] - F[0, 1]**2)
    return F


def xie(F):
    x = np.zeros(3)
    x[0] = np.log(F[0, 0])
    x[1] = np.log(F[1, 1])
    x[2] = F[0, 1] / F[0, 0]
    return x


def plot2d(x, y, z):
    # plt.style.use('_mpl-gallery')

    # size and color:
    colors = x
    size1 = 0.1
    sizes2 = 0.1
    colors2 = z
    # colors =

    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('мера деформации, индуцированная QR разложением')

    scatter1 = ax1.scatter(x, y, c=colors, s=size1)
    ax1.set_ylabel('ksi3')
    ax1.set_xlabel('ksi1')

    scatter2 = ax2.scatter(z, y, c=colors2, s=sizes2)
    ax2.set_xlabel('(ksi1 + ksi2) / sqrt(2)')
    ax2.set_ylabel('ksi3')

    fig = plt.figure(figsize=(50, 60))
    fig.tight_layout()
    # .subplot_tool()

    # scatter = ax.scatter(x, y, s=sizes)
    # ax.scatter(x, y)

    # produce a legend with a cross-section of sizes from the scatter
    # handles, labels = scatter1.legend_elements(prop="sizes", alpha=0.6)
    # legend1 = ax1.legend(handles, labels, loc="upper right", title="Sizes")
    #
    # handles1, labels1 = scatter2.legend_elements(prop="sizes", alpha=0.6)
    # legend2 = ax2.legend(handles1, labels1, loc="upper right", title="Sizes")
    ax1.grid()
    ax2.grid()

    plt.show()


def plot3d(x, y, z, title = ""):

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(5, 15)

    fig.suptitle(title)

    ax.set_xlabel('ksi1')
    ax.set_ylabel('ksi3')
    ax.set_zlabel('ksi2')

    fig = plt.figure(figsize=(50, 50))
    ax.scatter(x, y, z, s=0.1)

    # produce a legend with a cross-section of sizes from the scatter
    # handles, labels = scatter1.legend_elements(prop="sizes", alpha=0.6)
    # legend1 = ax1.legend(handles, labels, loc="upper right", title="Sizes")
    #
    # handles1, labels1 = scatter2.legend_elements(prop="sizes", alpha=0.6)
    # legend2 = ax2.legend(handles1, labels1, loc="upper right", title="Sizes")
    ax.grid()

    plt.show()


def plot3d_double(x1, y1, z1, x2, y2, z2):

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    fig.suptitle("VHB")

    ax.set_xlabel('ksi1')
    ax.set_ylabel('ksi3')
    ax.set_zlabel('ksi2')

    # plot two sets of data with different colors
    ax.scatter(x1, y1, z1, s=0.1, c='red', label='ShearFree')
    ax.scatter(x2, y2, z2, s=0.1, c='blue', label='EquiBiaxial')

    ax.view_init(2, 25)

    ax.grid()
    ax.legend()

    plt.show()

def gen_tensors(exp_path):

    folder_path = os.path.join('experiments', exp_path)

    Fs = process_files(folder_path)
    # print(Fs[2])
    F2s = np.zeros(Fs.shape)
    Cs = np.zeros(Fs.shape)
    xies = np.zeros((30, 49, 3))
    for t in range(1, 30):
        for point in range(points):
            Cs[t][point] = cauchy_green(Fs[t][point])
            F2s[t][point] = upper_triangle(Cs[t][point])
            xies[t][point] = xie(F2s[t][point])
    return xies

if __name__ == "__main__":
    experiment_name1 = "VHBEquiBiaxial"
    experiment_name = "shear_free"

    xies = gen_tensors(experiment_name)
    xies1 = gen_tensors(experiment_name1)
    # print(Fs[-1])
    # print("Cs:\n", Cs[-1])
    # print("F2s:\n",F2s[-1])
    # print("ksi:\n",xies[-1])

    xie1 = xies[:, :, 0]
    xie2 = xies[:, :, 1]
    xie3 = xies[:, :, 2]
    xie_norm = (xie1 + xie2) / np.sqrt(2)

    xie1_1 = xies1[:, :, 0]
    xie2_1 = xies1[:, :, 1]
    xie3_1 = xies1[:, :, 2]

    # plot2d(xie2, xie3, xie_norm)
    # plot3d(xie1, xie3, xie2, experiment_name)
    plot3d_double(xie1, xie3, xie2, xie1_1, xie3_1, xie2_1)
    # print(xie3[2])
    # plt.plot(range(1, 31), xie3)
    # plt.show()
    # print(xie1)
    # print(xie3)
    # xie1_flat = xie1.flatten()
    # xie2_flat = xie2.flatten()
    # xie_norm_flat = xie_norm.flatten()
    # # print(xie1_flat.shape)
    # print("Дисперсия первого графика =", np.var(np.column_stack([xie2_flat, xie1_flat])))
    # print("Дисперсия второго графика =", np.var(np.column_stack([xie2_flat, xie_norm_flat])))
    # print(np.var(np.concatenate(xie1.flat, xie3.flat)))
    # plot((xie1 + xie2)/np.sqrt(2), xie3)
    # print(tensor_gradient.shape)
    # print(tensor_gradient[29, 0])
    # print(Cs[5])
