import sys
import numpy as np

from utils import sum_rate


class Nonhomo_QT():
    def __init__(self, T, R, I, d, sigma, total_power, alpha, max_ite):
        """
        R: Number of receiving antennas.
        d: Number of data streams.
        I: Number of users.
        """
        self.T = T
        self.R = R
        self.d = d
        self.I = I
        self.sigma = sigma
        self.P = total_power
        self.alpha = alpha
        self.max_ite = max_ite

    def name(self):
        return "nonhomo_qt"

    def forward(self, H, V):

        # compute Z
        Z = V

        # compute Y
        J = np.zeros(shape=(self.R, self.R, self.I)) + 1j*np.zeros(shape=(self.R, self.R, self.I))
        Y = np.zeros(shape=(self.R, self.d, self.I)) + 1j*np.zeros(shape=(self.R, self.d, self.I))
        for i in range(self.I):
            for j in range(self.I):
                J[:,:,i] = J[:,:,i]+H[:,:,i]@V[:,:,j]@np.conjugate(V[:,:,j]).T@np.conjugate(H[:,:,i]).T
            Y[:,:,i] = np.linalg.inv(J[:,:,i]+self.sigma*np.eye(self.R))@H[:,:,i]@V[:,:,i]

        # compute gamma
        Gamma = np.zeros(shape=(self.d, self.d, self.I)) + 1j*np.zeros(shape=(self.d, self.d, self.I))
        for i in range(self.I):
            denominator = np.zeros(shape=(self.d, self.d)) + 1j * np.zeros(shape=(self.d, self.d))
            for j in range(self.I):
                denominator = denominator + H[:, :, i] @ V[:, :, j] @ np.conjugate(V[:, :, j]).T @ np.conjugate(H[:, :, i]).T
            denominator = denominator - H[:, :, i] @ V[:, :, i] @ np.conjugate(V[:, :, i]).T @ np.conjugate(H[:, :, i]).T + self.sigma * np.eye(self.R)

            Gamma[:,:,i] = np.conjugate(V[:,:,i]).T@np.conjugate(H[:,:,i]).T@denominator@H[:,:,i]@V[:,:,i]

        # compute V
        V0 = np.zeros(shape=(self.T, self.d, self.I)) + 1j*np.zeros(shape=(self.T, self.d, self.I))
        D = np.zeros(shape=(self.T, self.T)) + 1j * np.zeros(shape=(self.T, self.T))
        for j in range(self.I):
            D = D + self.alpha[j, 0] * (1 + Gamma[:, :, j]) * np.conjugate(H[:, :, j]).T @ Y[:, :, j] @ np.conjugate(Y[:, :, j]).T @ H[:, :, j]
        lambda_max = max(np.linalg.eigvals(D), key=lambda x: np.abs(x))
        for i in range(self.I):
            V0[:,:,i] = Z[:,:,i]+(1/lambda_max)*(self.alpha[i, 0]*(1+Gamma[:,:,i])*np.conjugate(H[:,:,i]).T@Y[:,:,i]-D@Z[:,:,i])

        P_tem = 0
        for i in range(self.I):
            P_tem = P_tem + np.real(np.trace(V0[:,:,i]@np.conjugate(V0[:,:,i]).T))
        if P_tem > self.P:
            V0 = V0*(np.sqrt(self.P/P_tem))

        return Z, Y, Gamma, V0