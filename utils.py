import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


def db2power(dB_value):
    return 10 ** (dB_value / 10)

def sum_rate(H, V, sigma2, R, I, alpha):
    rate = np.zeros(shape=(I, 1))
    for i in range(I):
        denominator = np.zeros(shape=(R, R)) + 1j*np.zeros(shape=(R, R))
        for j in range(I):
            denominator = denominator + H[:,:,i]@V[:,:,j]@np.conjugate(V[:,:,j]).T@np.conjugate(H[:,:,i]).T
        numerator = H[:,:,i]@V[:,:,i]@np.conjugate(V[:,:,i]).T@np.conjugate(H[:,:,i]).T
        denominator = denominator - numerator + sigma2*np.eye(R)

        rate[i] = np.log2(np.linalg.det(np.eye(R)+numerator@np.linalg.inv(denominator)))
    system_rate = np.real(np.sum(rate*alpha))

    return system_rate



def compute_circular_gaussian_channel(R, T, I):
    H = np.zeros(shape=(R, T, I)) + 1j*np.zeros(shape=(R, T, I))
    for i in range(I):
        H[:,:,i] = np.sqrt(0.5)*(np.random.normal(loc=0.0, scale=1.0, size=(R,T))+1j*np.random.normal(loc=0.0, scale=1.0, size=(R,T)))
    return H

def compute_channel(K, R, T, I, total_power, path_loss_option=True, path_loss_min=-5, path_loss_max=5):
    H = np.zeros(shape=(R, T, I)) + 1j*np.zeros(shape=(R, T, I))

    for i in range(I):
        regularization_parameter_for_RZF_solution = 0
        path_loss = 0
        if path_loss_option == True:
            path_loss = np.random.uniform(path_loss_min, path_loss_max)
            regularization_parameter_for_RZF_solution = regularization_parameter_for_RZF_solution+1/((10**(path_loss/10))*total_power)

        result_real = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size=(R,T))
        result_imag = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size=(R,T))
        H[:,:,i] = result_real+1j*result_imag

    return H

def compute_circular_symmetric_standard_normal_channel(K, R, T, I, snr, distance_range=[0.1, 0.3]):
    H = np.zeros(shape=(R, T, I)) + 1j*np.zeros(shape=(R, T, I))
    distances = np.random.uniform(low=distance_range[0],
                                 high=distance_range[1],
                                 size=I)
    logH = 0.0

    for i in range(I):
        path_loss = 128.1 + 37.6 * np.log10(distances[i])

        result_real = np.sqrt(10**(-path_loss/10))*np.sqrt(0.5)*np.random.normal(loc=0.0, scale=1.0, size=(R,T))
        result_imag = np.sqrt(10**(-path_loss/10))*np.sqrt(0.5)*np.random.normal(loc=0.0, scale=1.0, size=(R,T))
        H[:,:,i] = result_real+1j*result_imag

        logH += np.log10(np.linalg.norm(H[:,:,i], 'fro')**2/R)


    sigma2 = 10**(logH/I)*(10**(-(snr/10)))

    return H, sigma2


def plot_ites(rates, model_name, K, T, R, d, snr, epsilon):
    x = np.arange(len(rates))
    fig, ax = plt.subplots(figsize=(8, 5))

    line, = ax.plot(
        x, rates,
        color="#8B0000",  # deep red
        marker="^",  # filled triangles
        markersize=8,  # markers size
        markerfacecolor="#8B0000",
        markeredgecolor="k",
        linestyle="-",
        linewidth=2,
        label=model_name
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        color="gray",
        alpha=0.7
    )


    ax.legend(
        loc="best",
        frameon=True,
        shadow=True
    )

    # 设置坐标轴标签
    ax.set_xlabel("Iterations", fontsize=12)
    ax.set_ylabel("Sum rate (bits per channel use)", fontsize=12)
    ax.set_title(r"{}, K={}, T={}, R={}, d={}, {}dB, $\epsilon$={}".format(model_name, K, T, R, d, snr, epsilon), fontsize=14)

    plt.tight_layout()
    plt.show()




