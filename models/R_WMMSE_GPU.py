import sys
import torch
import numpy as np
from scipy.linalg import block_diag


class R_WMMSE_GPU():
    def __init__(self, T, R, I, d, sigma, total_power, alpha, max_ite, device):
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
        self.device = device

    def name(self):
        return "r_wmmse_gpu"

    def forward(self, H_hat, X):
        """
        H_hat: [R*I, R*I]
        X: [I, R*I, d]
        """

        # compute U
        J = torch.complex(torch.zeros(size=(self.I, self.R, self.R), device=self.device),
                          torch.zeros(size=(self.I, self.R, self.R), device=self.device))
        IN = torch.complex(torch.zeros(size=(self.I, self.R, self.R), device=self.device),
                          torch.zeros(size=(self.I, self.R, self.R), device=self.device))
        for i in range(self.I):
            for j in range(self.I):
                J[i,:,:] = J[i,:,:]+H_hat[self.R*i:(self.R)*(i+1),:]@X[j,:,:]@torch.conj(X[j,:,:]).T@torch.conj(H_hat[self.R*i:(self.R)*(i+1),:]).T
                IN[i,:,:] = IN[i,:,:]+((self.sigma/self.P)*torch.trace(H_hat@X[j,:,:]@torch.conj(X[j,:,:]).T))*torch.eye(self.R, device=self.device)
        U = torch.bmm(torch.linalg.inv(J+IN), torch.bmm(H_hat.reshape(shape=(self.I,self.R,self.I*self.R)), X))


        # compute W
        W = torch.linalg.inv(torch.diag_embed(torch.ones(size=(self.I, self.d), device=self.device)) -
                                 torch.bmm(torch.bmm(torch.conj(U).permute(0, 2, 1), H_hat.reshape(shape=(self.I,self.R,self.I*self.R))), X))

        # compute X

        W_hat = torch.block_diag(*[W[i] for i in range(self.I)])
        U_hat = torch.block_diag(*[U[i] for i in range(self.I)])
        eta = (self.alpha*torch.diagonal(torch.bmm(torch.bmm(U, W), torch.conj(U).permute(0,2,1)), dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)).sum()

        eta = eta*(self.sigma/self.P)
        X_hat = U_hat@torch.linalg.inv(eta*torch.linalg.inv(W_hat)+torch.conj(U_hat).T@H_hat@U_hat)

        X = torch.complex(torch.zeros(size=(self.I, self.R*self.I, self.d), device=self.device),
                          torch.zeros(size=(self.I, self.R*self.I, self.d), device=self.device))
        for i in range(self.I):
            X[i,:,:] = X_hat[:, self.d*i:self.d*(i+1)]


        return U, W, X