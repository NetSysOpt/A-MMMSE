import sys
import numpy as np
from scipy.linalg import block_diag


class R_WMMSE():
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
        return "r_wmmse"

    def forward(self, H_hat, X):
        # compute U
        J = np.zeros(shape=(self.R, self.R, self.I)) + 1j*np.zeros(shape=(self.R, self.R, self.I))
        IN = np.zeros(shape=(self.R, self.R, self.I)) + 1j*np.zeros(shape=(self.R, self.R, self.I))
        U = np.zeros(shape=(self.R, self.d, self.I)) + 1j*np.zeros(shape=(self.R, self.d, self.I))
        for i in range(self.I):
            for j in range(self.I):
                J[:,:,i] = J[:,:,i]+H_hat[self.R*i:(self.R)*(i+1),:]@X[:,:,j]@np.conjugate(X[:,:,j]).T@np.conjugate(H_hat[self.R*i:(self.R)*(i+1),:]).T
                IN[:,:,i] = IN[:,:,i]+((self.sigma/self.P)*np.trace(H_hat@X[:,:,j]@np.conjugate(X[:,:,j]).T))*np.eye(self.R)
            U[:,:,i] = np.linalg.inv(J[:,:,i]+IN[:,:,i])@H_hat[self.R*i:self.R*(i+1),:]@X[:,:,i]

        # compute W
        W = np.zeros(shape=(self.d, self.d, self.I)) + 1j * np.zeros(shape=(self.d, self.d, self.I))
        for i in range(self.I):
            W[:,:,i] = np.linalg.inv(np.eye(self.d)-np.conjugate(U[:,:,i]).T@H_hat[self.R*i:self.R*(i+1),:]@X[:,:,i])

        # compute X
        eta = self.alpha[0,0]*np.trace(U[:,:,0]@W[:,:,0]@np.conjugate(U[:,:,0]).T)
        W_hat = self.alpha[0,0]*W[:,:,0]
        U_hat = U[:,:,0]

        for i in range(1, self.I):
            W_hat = block_diag(W_hat, self.alpha[i,0]*W[:,:,i])
            U_hat = block_diag(U_hat, U[:,:,i])
            eta = eta+self.alpha[i,0]*np.trace(U[:,:,i]@W[:,:,i]@np.conjugate(U[:,:,i]).T)

        eta = eta*(self.sigma/self.P)
        X_hat = U_hat@np.linalg.inv(eta*np.linalg.inv(W_hat)+np.conjugate(U_hat).T@H_hat@U_hat)

        X = np.zeros(shape=(self.R*self.I, self.d, self.I)) + 1j * np.zeros(shape=(self.R*self.I, self.d, self.I))
        for i in range(self.I):
            X[:,:,i] = X_hat[:, self.d*i:self.d*(i+1)]

        # # compute X
        # J = np.zeros(shape=(self.I*self.R, self.I*self.R, self.I)) + 1j * np.zeros(shape=(self.I*self.R, self.I*self.R, self.I))
        # IN = np.zeros(shape=(self.I, 1)) + 1j * np.zeros(shape=(self.I, 1))
        # X = np.zeros(shape=(self.I*self.R, self.d, self.I)) + 1j * np.zeros(shape=(self.I*self.R, self.d, self.I))
        #
        # for i in range(self.I):
        #     for l in range(self.I):
        #         J[:,:,i] = J[:,:,i]+self.alpha[l,0]*np.conjugate(H_hat[self.R*l:self.R*(l+1),:]).T@U[:,:,l]@W[:,:,l]@np.conjugate(U[:,:,l]).T@H_hat[self.R*l:self.R*(l+1),:]
        #         IN[i,:] = IN[i,:]+self.alpha[l,0]*np.trace(U[:,:,l]@W[:,:,l]@np.conjugate(U[:,:,l]).T)
        #
        #     X[:,:,i] = np.linalg.inv(J[:,:,i]+IN[i,:]*H_hat*self.sigma/self.P)@(self.alpha[i,0]*np.conjugate(H_hat[self.R*i:self.R*(i+1),:]).T@U[:,:,i]@W[:,:,i])


        return U, W, X
