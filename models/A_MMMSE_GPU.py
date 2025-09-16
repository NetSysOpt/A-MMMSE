import sys
import torch
import torch.nn as nn
import numpy as np


class A_MMMSE_GPU(nn.Module):
    def __init__(self, T, R, I, d, sigma, total_power, alpha, max_ite, lr, lr_type, tol, device):
        super(A_MMMSE_GPU, self).__init__()
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
        self.device = device

    def name(self):
        return "a_mmmse_gpu"

    def forward(self, H, V, lr, rel_rate):
        # compute U
        U_denom = (torch.bmm(H@(torch.bmm(V, torch.conj(V).permute(0,2,1)).sum(0)), torch.conj(H).permute(0,2,1))+
                   self.sigma*torch.diag_embed(torch.ones(size=(self.I, self.R), device=self.device)))
        U = torch.bmm(torch.linalg.inv(U_denom), torch.bmm(H, V))


        if rel_rate <= self.tol:
            # compute W
            W = torch.linalg.inv(torch.diag_embed(torch.ones(size=(self.I, self.d), device=self.device))-
                                 torch.bmm(torch.bmm(torch.conj(U).permute(0,2,1), H), V))

            # compute V
            HhU = torch.bmm(torch.conj(H).permute(0,2,1), U)
            WUhH = torch.bmm(W, torch.conj(HhU).permute(0,2,1))
            J = (self.alpha*torch.bmm(HhU, WUhH)).sum(0)

            if self.lr_type == "lipschitz":
                L = torch.linalg.eigvals(J).real.max()

            grad = J@V - self.alpha * (torch.bmm(torch.conj(H).permute(0,2,1), torch.bmm(U, W)))

            if self.lr_type == "fixed":
                lr = self.lr
            elif self.lr_type == "lipschitz":
                lr = 1/L

            V = V - lr * grad

            # projection
            P_tem = torch.real(torch.diagonal(torch.bmm(V, torch.conj(V).permute(0, 2, 1)), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)).sum()
            if P_tem > self.P:
                V = V * (torch.sqrt(self.P / P_tem))

        else:
            # compute V
            HhU = torch.bmm(torch.conj(H).permute(0, 2, 1), U)
            UhH = torch.conj(HhU).permute(0, 2, 1)
            J = (self.alpha * torch.bmm(HhU, UhH)).sum(0)

            if self.lr_type == "lipschitz":
                L = torch.linalg.eigvals(J).real.max()

            grad = J @ V - self.alpha * (torch.bmm(torch.conj(H).permute(0, 2, 1), U))

            if self.lr_type == "fixed":
                lr = self.lr
            elif self.lr_type == "lipschitz":
                lr = 1/L

            V = V - lr * grad

            # projection
            P_tem = torch.real(torch.diagonal(torch.bmm(V, torch.conj(V).permute(0, 2, 1)), dim1=-2, dim2=-1).sum(dim=-1,keepdim=True).unsqueeze(-1)).sum()
            if P_tem > self.P:
                V = V * (torch.sqrt(self.P / P_tem))


        return U, V