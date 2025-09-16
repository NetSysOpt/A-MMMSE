
import os
import sys
import pickle
import random
import warnings
import numpy as np
import configargparse
import time

from models.ZF import ZF
from models.WMMSE import WMMSE
from models.MMMSE import MMMSE
from models.A_MMMSE import A_MMMSE
from models.R_MMMSE import R_MMMSE
from models.R_WMMSE import R_WMMSE
from models.Nonhomo_QT import Nonhomo_QT
from utils import db2power, compute_circular_symmetric_standard_normal_channel, compute_circular_gaussian_channel, compute_channel, sum_rate, plot_ites

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
parser.add_argument('--lr', type=float, help='Step size.')
parser.add_argument('--lr_type', type=str, choices=['fixed', 'optimal', 'line_search', 'lipschitz'],
                    help="Method of calculating step size.")
parser.add_argument('--omega', type=float, help='The extrapolation coefficient.')
parser.add_argument('--model_name', type=str, help='Model names.')
parser.add_argument('--max_ite', type=int, help='Maximum iteration.')
parser.add_argument('--tol', type=float, help='Tolerance of convergence.')
parser.add_argument('--mmmse_tol', type=float, help='Tolerance of MMMSE.')
parser.add_argument('--epsilon', type=float, help='The parameter of AdaGrad.')

parser.add_argument('--save_dir', type=str, default='./results/', help='Save path for the best model.')
parser.add_argument('--save_result', action='store_true', help='Save the results.')



args, _ = parser.parse_known_args()

# user weight
alpha = np.ones(shape=(args.I, args.K))



if args.channel_type == "circular":
    H = compute_circular_gaussian_channel(args.R, args.T, args.I)
    logH = 0.0
    for i in range(args.I):
        logH += np.log10(np.linalg.norm(H[:,:,i], 'fro')**2/args.R)
    sigma = 10**(logH/args.I)*(10**(-(args.snr/10)))
    total_power = 10

elif args.channel_type == "fading":
    sigma = args.sigma
    total_power = db2power(args.snr) * args.sigma
    H = compute_channel(args.K, args.R, args.T, args.I, total_power)
elif args.channel_type == "symmetric_circular":
    H, sigma = compute_circular_symmetric_standard_normal_channel(args.K, args.R, args.T, args.I, args.snr)
    total_power = 10


H_full = np.zeros(shape=(args.I*args.R, args.T)) + 1j*np.zeros(shape=(args.I*args.R, args.T))
for i in range(args.I):
    H_full[i*args.R:(i+1)*args.R,:] = H[:,:,i]
H_hat = H_full@np.conjugate(H_full).T



if args.model_name == "WMMSE":
    model = WMMSE(args.T, args.R, args.I, args.d, sigma, total_power, alpha, args.max_ite)
elif args.model_name == "MMMSE":
    model = MMMSE(args.T, args.R, args.I, args.d, sigma, total_power, alpha, args.max_ite, args.mmmse_tol)
elif args.model_name == "R_WMMSE":
    model = R_WMMSE(args.T, args.R, args.I, args.d, sigma, total_power, alpha, args.max_ite)
elif args.model_name == "R_MMMSE":
    model = R_MMMSE(args.T, args.R, args.I, args.d, sigma, total_power, alpha, args.max_ite, args.mmmse_tol)
elif args.model_name == "A_MMMSE":
    model = A_MMMSE(args.T, args.R, args.I, args.d, sigma, total_power, alpha, args.max_ite, args.lr, args.lr_type, args.mmmse_tol)
elif args.model_name == "Nonhomo_QT":
    model = Nonhomo_QT(args.T, args.R, args.I, args.d, sigma, total_power, alpha, args.max_ite)
elif args.model_name == "ZF":
    model = ZF(args.T, args.R, args.I, args.d, sigma, total_power, alpha)

# record solution time
total_time = 0.0

# rates
rates = []

start_time = time.time()
# initialize U, W
U = np.random.normal(loc=0.0, scale=1.0, size=(args.R,args.d,args.I)) + 1j*np.random.normal(loc=0.0, scale=1.0, size=(args.R,args.d,args.I))
W = np.zeros(shape=(args.d,args.d,args.I)) + 1j*np.zeros(shape=(args.d,args.d,args.I))
for i in range(args.I):
    W[:,:,i] = np.eye(args.d)

# initialize V
if args.model_name in ["R_WMMSE", "R_MMMSE"]:
    X = np.zeros(shape=(args.R*args.I, args.d, args.I)) + 1j * np.zeros(shape=(args.R*args.I, args.d, args.I))
    V = np.zeros(shape=(args.T, args.d, args.I)) + 1j * np.zeros(shape=(args.T, args.d, args.I))
    for i in range(args.I):
        x = np.sqrt(1/2)*(np.random.normal(loc=0.0, scale=1.0, size=(args.R*args.I, args.d))+1j*np.random.normal(loc=0.0, scale=1.0, size=(args.R*args.I, args.d)))
        v = np.conjugate(H_full).T@x
        X[:,:,i] = np.sqrt(total_power/(args.I*np.trace(H_hat@x@np.conjugate(x).T)))*x
        V[:,:,i] = np.conjugate(H_full).T@X[:,:,i]
    X1 = X
else:
    V = np.zeros(shape=(args.T, args.d, args.I)) + 1j*np.zeros(shape=(args.T, args.d, args.I))
    for i in range(args.I):
        v = np.sqrt(1/2)*(np.random.normal(loc=0.0, scale=1.0, size=(args.T, args.d))+1j*np.random.normal(loc=0.0, scale=1.0, size=(args.T, args.d)))
        V[:, :, i] = np.sqrt(total_power/(args.I*np.trace(v@np.conjugate(v).T)))*v


V_bar = V

end_time = time.time()

init_time = end_time - start_time
print("Initialization Time: {}".format(init_time))

total_time += (end_time-start_time)

# compute rate
rate_old = sum_rate(H, V, sigma, args.R, args.I, alpha)
rates.append(rate_old)


num_ite = 1
rel_rate = 1.0
V1 = V
lr = 1
G = np.zeros(shape=(args.I, args.T, args.R))
grad_squares = 0
while True:
    start_time = time.time()

    if args.model_name == "A_MMMSE":
        # extrapolation
        # Extrapolation does not guarantee monotonicity, especially when the calculation method of
        # learning rate is one tenth of the Lipschitz constant
        if num_ite >= 2:
            V0 = V1
            V1 = V
            V = V1+args.omega*(V1-V0)

        U, V = model.forward(H, V, args.lr, rel_rate)


    elif args.model_name == "R_MMMSE":
        U, X = model.forward(H_hat, X, rel_rate)

        # compute beta
        beta = 0

        for i in range(args.I):
            beta = beta + np.trace(H_hat@X[:,:,i]@np.conjugate(X[:,:,i]).T)

        beta = total_power / beta

        # compute V
        for i in range(args.I):
            V[:,:,i] = np.sqrt(beta)*np.conjugate(H_full).T@X[:,:,i]

    elif args.model_name == "R_WMMSE":
        U, W, X = model.forward(H_hat, X)

        # compute beta
        beta = 0

        for i in range(args.I):
            beta = beta + np.trace(H_hat@X[:,:,i]@np.conjugate(X[:,:,i]).T)

        beta = total_power / beta

        # compute V
        for i in range(args.I):
            V[:,:,i] = np.sqrt(beta)*np.conjugate(H_full).T@X[:,:,i]

    elif args.model_name == "WMMSE":
        U, W, V = model.forward(H, V)

    elif args.model_name == "MMMSE":
        U, V = model.forward(H, V, rel_rate)

    elif args.model_name == "Nonhomo_QT":
        # extrapolation
        if num_ite >= 2:
            V0 = V1
            V1 = V
            V = V1 + 0.9 * (V1 - V0)

        Z, Y, Gamma, V = model.forward(H, V)

    rate_new = sum_rate(H, V, sigma, args.R, args.I, alpha)
    rates.append(rate_new)
    num_ite += 1

    rel_rate = np.abs((rate_new-rate_old)/rate_old)

    end_time = time.time()
    total_time += (end_time - start_time)

    if (rel_rate<args.tol) or (num_ite > args.max_ite):
        break

    rate_old = rate_new

# save results
if args.save_result:
    dir_path = os.path.join(args.save_dir, args.model_name)
    os.makedirs(dir_path, exist_ok=True)
    pkl_path = os.path.join(dir_path, "{}_K{}_T{}_R{}_d{}_SNR{}.pkl".format(args.channel_type, args.K, args.T, args.R, args.d, args.snr))

    with open(pkl_path, 'wb') as file:
        pickle.dump(rates, file)

print("Elapsed Time: {}.".format(total_time))
print("Iteration Number: {}.".format(num_ite))
print("Final Rate: {}".format(rate_new))
plot_ites(rates, args.model_name, args.K, args.T, args.R, args.d, args.snr, args.epsilon)



