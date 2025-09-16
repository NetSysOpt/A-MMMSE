import sys
import numpy as np


class WMMSE():
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
        return "wmmse"

    def forward(self, H, V):
        # compute U
        J = np.zeros(shape=(self.R, self.R, self.I)) + 1j*np.zeros(shape=(self.R, self.R, self.I))
        U = np.zeros(shape=(self.R, self.d, self.I)) + 1j*np.zeros(shape=(self.R, self.d, self.I))
        for i in range(self.I):
            for j in range(self.I):
                J[:,:,i] = J[:,:,i]+H[:,:,i]@V[:,:,j]@np.conjugate(V[:,:,j]).T@np.conjugate(H[:,:,i]).T
            U[:,:,i] = np.linalg.inv(J[:,:,i]+self.sigma*np.eye(self.R))@H[:,:,i]@V[:,:,i]

        # compute W
        W = np.zeros(shape=(self.d, self.d, self.I)) + 1j * np.zeros(shape=(self.d, self.d, self.I))
        for i in range(self.I):
            W[:,:,i] = np.linalg.inv(np.eye(self.d)-np.conjugate(U[:,:,i]).T@H[:,:,i]@V[:,:,i])

        # compute V
        J = np.zeros(shape=(self.T, self.T)) + 1j*np.zeros(shape=(self.T, self.T))
        V = np.zeros(shape=(self.T, self.d, self.I)) + 1j*np.zeros(shape=(self.T, self.d, self.I))
        for j in range(self.I):
            J = J + self.alpha[j,0]*(np.conjugate(H[:,:,j]).T)@U[:,:,j]@W[:,:,j]@(np.conjugate(U[:,:,j]).T)@H[:,:,j]

        mu_min = 0
        mu_max = 10
        ite = 0
        while True:
            mu1 = (mu_min+mu_max)/2
            P_tem = 0

            for i in range(self.I):
                V_tem = np.linalg.inv(J+mu1*np.eye(self.T))@(self.alpha[i,0]*(np.conjugate(H[:,:,i]).T@U[:,:,i]@W[:,:,i]))
                P_tem = P_tem+np.real(np.trace(V_tem@np.conjugate(V_tem).T))

            if P_tem > self.P:
                mu_min = mu1
            else:
                mu_max = mu1

            ite = ite + 1

            if (np.abs(mu_max-mu_min)<0.00001) or (ite>self.max_ite):
                break

        mu = mu1

        for i in range(self.I):
            V[:,:,i] = np.linalg.inv(J+mu*np.eye(self.T))@(self.alpha[i,0]*(np.conjugate(H[:,:,i]).T@U[:,:,i]@W[:,:,i]))


        return U, W, V
