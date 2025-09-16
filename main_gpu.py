import os
import sys
import pickle
import torch
import random
import warnings
import numpy as np
import configargparse

from models.A_MMMSE_GPU import A_MMMSE_GPU
from models.R_WMMSE_GPU import R_WMMSE_GPU
from utils import db2power, compute_circular_gaussian_channel, compute_channel, sum_rate, plot_ites

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = configargparse.ArgumentParser(description='train')
parser.add_argument('-c', '--config', is_config_file=True, type=str)

# system settings
parser.add_argument('--K', type=int, help='Number of base stations.')
parser.add_argument('--T', type=int, help='Number of transmitting antennas.')
parser.add_argument('--R', type=int, help='Number of receiving antennas.')
parser.add_argument('--sigma', type=float, help='Power of noise.')
parser.add_argument('--snr', type=float, help='Signal to noise ratio.')
parser.add_argument('--I', type=int, help='Number of users.')
parser.add_argument('--d', type=int, help='Number of data streams.')
parser.add_argument('--path_loss_option', type=bool, help='Whether include path loss.')
parser.add_argument('--channel_type', type=str, choices=["circular", "fading", "symmetric_circular"], help='The type of channel.')

# model settings
parser.add_argument('--device', type=str, help="The running device.")
parser.add_argument('--lr', type=float, help='Step size.')
parser.add_argument('--lr_type', type=str, choices=['fixed', 'optimal', 'lipschitz'], help="Method of calculating step size.")
parser.add_argument('--model_name', type=str, help='Model names.')
parser.add_argument('--max_ite', type=int, help='Maximum iteration.')
parser.add_argument('--tol', type=float, help='Tolerance of convergence.')
parser.add_argument('--mmmse_tol', type=float, help='Tolerance of MMMSE.')
parser.add_argument('--epsilon', type=float, help='The parameter of AdaGrad.')

# test settings
parser.add_argument('--save_dir', type=str, default='./results/', help='Save path for the best model.')
parser.add_argument('--save_result', action='store_true', help='Save the results.')
parser.add_argument('--test_numb', type=int, help='The number of simulation times.')
parser.add_argument('--model_ites', type=int, help='The number of model iterations.')



args, _ = parser.parse_known_args()

# user weight
alpha = torch.ones(size=(args.I, args.K, 1), device=args.device)


all_rates = []
final_rates = []
times = []
ites = []

for num in range(args.test_numb):

    if args.channel_type == "circular":
        H = compute_circular_gaussian_channel(args.R, args.T, args.I)
        logH = 0.0
        for i in range(args.I):
            logH += np.log10(np.linalg.norm(H[:, :, i], 'fro') ** 2 / args.R)
        sigma = 10 ** (logH / args.I) * (10 ** (-(args.snr / 10)))
        total_power = 10
        # [I, R, T]
        H = torch.tensor(H, device=args.device).permute(2, 0, 1).to(torch.complex64)
    elif args.channel_type == "fading":
        sigma = args.sigma
        total_power = db2power(args.snr) * args.sigma
        H = compute_channel(args.K, args.R, args.T, args.I, total_power)

    H_full = H.reshape(shape=(args.I * args.R, args.T))
    H_hat = H_full @ torch.conj(H_full).T

    if args.model_name == "A_MMMSE_GPU":
        model = A_MMMSE_GPU(args.T, args.R, args.I, args.d, sigma, total_power, alpha,
                                args.max_ite, args.lr, args.lr_type, args.mmmse_tol, args.device)
    elif args.model_name == "R_WMMSE_GPU":
        model = R_WMMSE_GPU(args.T, args.R, args.I, args.d, sigma, total_power, alpha, args.max_ite, args.device)


    # record solution time
    total_time = 0.0

    start_event_1 = torch.cuda.Event(enable_timing=True)
    end_event_1 = torch.cuda.Event(enable_timing=True)

    # rates
    rates = []

    torch.cuda.init()

    start_event_1.record()
    # initialize U, W
    U = torch.complex(torch.randn(size=(args.I, args.R, args.d), device=args.device),
                          torch.randn(size=(args.I, args.R, args.d), device=args.device))
    W = torch.complex(torch.diag_embed(torch.ones(size=(args.I, args.d), device=args.device)),
                          torch.zeros(size=(args.I, args.d, args.d), device=args.device))

    # initialize V
    if args.model_name in ["R_WMMSE_GPU", "R_MMMSE"]:
        X0 = torch.complex(torch.normal(0, 1, (args.I, args.R * args.I, args.d), device=args.device),
                               torch.normal(0, 1, (args.I, args.R * args.I, args.d), device=args.device))
        X = torch.sqrt(total_power / (
                        args.I * torch.diagonal(H_hat @ torch.bmm(X0, torch.conj(X0).permute(0, 2, 1)), dim1=-2,
                                                dim2=-1).sum(-1, keepdim=True).unsqueeze(-1))) * X0
        V = torch.conj(H_full).T @ X
        X1 = X
    else:
        V = torch.sqrt(torch.tensor(0.5, device=args.device)) * (
                torch.complex(torch.randn(size=(args.I, args.T, args.d), device=args.device),
                              torch.randn(size=(args.I, args.T, args.d), device=args.device)))
        V = (torch.sqrt(total_power / (
                        args.I * torch.diagonal(torch.bmm(V, torch.conj(V).permute(0, 2, 1)), dim1=-2, dim2=-1).sum(
                    dim=-1, keepdim=True))).unsqueeze(-1)) * V

    V_bar = V

    end_event_1.record()

    torch.cuda.synchronize()

    init_time = start_event_1.elapsed_time(end_event_1) / 1000.0
    total_time += init_time


    start_event_2 = torch.cuda.Event(enable_timing=True)
    end_event_2 = torch.cuda.Event(enable_timing=True)

    start_event_2.record()

    # compute rate
    rate_old = sum_rate(H.permute(1, 2, 0).cpu().numpy(), V.permute(1, 2, 0).cpu().numpy(),
                            sigma, args.R, args.I, alpha.squeeze(-1).cpu().numpy())
    rates.append(rate_old)

    num_ite = 1
    rel_rate = 1.0
    V1 = V
    lr = 1
    grad_squares = 0
    while True:
        if args.model_name == "A_MMMSE_GPU":
            # extrapolation
            if num_ite >= 2:
                V0 = V1
                V1 = V
                V = V1 + 0.9 * (V1 - V0)

            U, V = model.forward(H, V, args.lr, rel_rate)

        elif args.model_name == "R_WMMSE_GPU":
            U, W, X = model.forward(H_hat, X)

            # compute beta
            beta = (torch.diagonal(H_hat @ torch.bmm(X, torch.conj(X).permute(0, 2, 1)), dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)).sum()
            beta = total_power / beta

            # compute V
            V = torch.sqrt(beta) * torch.conj(H_full).T @ X

        rate_new = sum_rate(H.permute(1, 2, 0).cpu().numpy(), V.permute(1, 2, 0).cpu().numpy(),
                                sigma, args.R, args.I, alpha.squeeze(-1).cpu().numpy())

        rates.append(rate_new)
        num_ite += 1

        rel_rate = np.abs((rate_new - rate_old) / rate_old)
        if (rel_rate < args.tol) or (num_ite > args.max_ite):
                break
        rate_old = rate_new

    end_event_2.record()
    torch.cuda.synchronize()
    total_time += start_event_2.elapsed_time(end_event_2) / 1000.0


    all_rates.append(rates)
    final_rates.append(rates[-1])
    times.append(total_time)
    ites.append(num_ite)



if args.save_result:
    data_dict = {"final_rate": np.array(final_rates),
                 "time": np.array(times),
                 "ite":np.array(ites)}

    # save results
    dir_path = os.path.join(args.save_dir, args.model_name)
    os.makedirs(dir_path, exist_ok=True)
    pkl_path = os.path.join(dir_path, "GPUTime_{}_K{}_T{}_R{}_I{}_d{}_SNR{}_{}_{}.pkl".format(args.channel_type, args.K, args.T,
                                                                                          args.R, args.I, args.d, args.snr, args.test_numb, args.model_ites))

    with open(pkl_path, 'wb') as file:
        pickle.dump(data_dict, file)

print("Elapsed Time: {}.".format(np.array(times).mean()))
print("Iteration Number: {}.".format(np.array(ites).mean()))
print("Final Rate: {}".format(np.array(final_rates).mean()))
plot_ites(rates, args.model_name, args.K, args.T, args.R, args.d, args.snr, args.epsilon)