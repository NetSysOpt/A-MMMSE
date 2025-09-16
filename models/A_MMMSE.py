import sys
import numpy as np

from utils import sum_rate


class A_MMMSE():
    def __init__(self, T, R, I, d, sigma, total_power, alpha, max_ite, lr, lr_type, tol):
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
        self.lr = lr
        self.lr_type = lr_type
        self.tol = tol
        self.grads = None

    def name(self):
        return "a_mmmse"

    def forward(self, H, V, lr, rel_rate):
        # compute U
        J = np.zeros(shape=(self.R, self.R, self.I)) + 1j*np.zeros(shape=(self.R, self.R, self.I))
        U = np.zeros(shape=(self.R, self.d, self.I)) + 1j*np.zeros(shape=(self.R, self.d, self.I))
        for i in range(self.I):
            for j in range(self.I):
                J[:,:,i] = J[:,:,i]+H[:,:,i]@V[:,:,j]@np.conjugate(V[:,:,j]).T@np.conjugate(H[:,:,i]).T
            U[:,:,i] = np.linalg.inv(J[:,:,i]+self.sigma*np.eye(self.R))@H[:,:,i]@V[:,:,i]

        if rel_rate <= self.tol:
            # compute W
            W = np.zeros(shape=(self.d, self.d, self.I)) + 1j * np.zeros(shape=(self.d, self.d, self.I))
            for i in range(self.I):
                if rel_rate <= self.tol:
                    W[:, :, i] = np.linalg.inv(np.eye(self.d) - np.conjugate(U[:, :, i]).T @ H[:, :, i] @ V[:, :, i])
                else:
                    W[:, :, i] = np.eye(self.d)

            # compute V
            J = np.zeros(shape=(self.T, self.T)) + 1j*np.zeros(shape=(self.T, self.T))
            V0 = np.zeros(shape=(self.T, self.d, self.I)) + 1j*np.zeros(shape=(self.T, self.d, self.I))
            WUhH = []
            HhU = []
            for j in range(self.I):
                wuhh = W[:,:,j]@(np.conjugate(U[:,:,j]).T)@H[:,:,j]
                hhu = (np.conjugate(H[:, :, j]).T) @ U[:, :, j]
                J = J + self.alpha[j, 0] * hhu @ wuhh
                WUhH.append(wuhh)
                HhU.append(hhu)

            if self.lr_type == "lipschitz":
                L = max(np.linalg.eigvals(J), key=lambda x: np.abs(x))

            # calculate gradient
            grads = []
            for i in range(self.I):
                grad = 2 * (J @ V[:, :, i] - (self.alpha[i, 0] * (np.conjugate(H[:, :, i]).T @ U[:, :, i] @ W[:, :, i])))
                grads.append(grad)

            for i in range(self.I):
                # calculate step size
                if self.lr_type == "fixed":
                    lr = lr

                elif self.lr_type == "lipschitz":
                    lr = 1/L

                # update precoding matrix
                V0[:,:,i] = V[:,:,i] - lr*grads[i]

            # save old gradients
            self.grads = grads

            P_tem = 0
            for i in range(self.I):
                P_tem = P_tem + np.real(np.trace(V0[:,:,i]@np.conjugate(V0[:,:,i]).T))
            if P_tem > self.P:
                V0 = V0*(np.sqrt(self.P/P_tem))

            V = V0

        else:
            # compute V
            J = np.zeros(shape=(self.T, self.T)) + 1j * np.zeros(shape=(self.T, self.T))
            V0 = np.zeros(shape=(self.T, self.d, self.I)) + 1j * np.zeros(shape=(self.T, self.d, self.I))
            UhH = []
            HhU = []
            for j in range(self.I):
                uhh = (np.conjugate(U[:, :, j]).T) @ H[:, :, j]
                hhu = (np.conjugate(H[:, :, j]).T) @ U[:, :, j]
                J = J + self.alpha[j, 0] * hhu @ uhh
                UhH.append(uhh)
                HhU.append(hhu)

            if self.lr_type == "lipschitz":
                L = max(np.linalg.eigvals(J), key=lambda x: np.abs(x))

            # calculate gradient
            grads = []
            for i in range(self.I):
                grad = 2 * (J @ V[:, :, i] - (self.alpha[i, 0] * (np.conjugate(H[:, :, i]).T @ U[:, :, i])))
                grads.append(grad)

            for i in range(self.I):
                # calculate step size
                if self.lr_type == "fixed":
                    lr = lr

                elif self.lr_type == "lipschitz":
                    lr = 1 / L

                # update precoding matrix
                V0[:, :, i] = V[:, :, i] - lr * grads[i]

            # save old gradients
            self.grads = grads

            P_tem = 0
            for i in range(self.I):
                P_tem = P_tem + np.real(np.trace(V0[:, :, i] @ np.conjugate(V0[:, :, i]).T))
            if P_tem > self.P:
                V0 = V0 * (np.sqrt(self.P / P_tem))

            V = V0



        return U, V0