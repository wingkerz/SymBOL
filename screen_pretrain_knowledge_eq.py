import copy
import multiprocessing
import threading

lock = threading.Lock()
import warnings

warnings.filterwarnings("ignore")

import queue
import math
import random
import time
from functools import partial
import numpy as np
import sympy as sp
import scipy.optimize as opt

import matplotlib.pyplot as plt
from tqdm import tqdm
import func_timeout
from func_timeout import func_set_timeout
import pandas as pd
import csv
import pickle

import search_2nd_phase.gp_tools as gp

from scipy.signal import savgol_filter
import argparse
import torch
from scipy.sparse import coo_matrix


def get_parser():
    parser = argparse.ArgumentParser(description='Search_for_2ndPhase')
    parser.add_argument('--case_name', help='case_name, default:"Epidemic"', default="Epidemic")
    parser.add_argument('--dim', help='dim, default:1', default=1)
    parser.add_argument('--add_str', help='add_str, default:""', default="unknown_topo")
    parser.add_argument('--opt_flag', help='flag, default:"0"', default="0")
    return parser

snr_global = 0
eta_global = 0

args = get_parser().parse_args()
# case_name = 'heat'
# case_name = 'mutu'
# case_name = 'gene'
case_name = args.case_name
dim = int(args.dim)
opt_flag = int(args.opt_flag)

add_str = args.add_str

def add_obs_noise(Y, snr):
    if snr is None:
        return Y
    var_x = np.var(Y)
    sigma_sq = var_x / (10 ** (snr / 10))
    sigma = np.sqrt(sigma_sq)
    noise = np.random.normal(0, sigma, Y.shape)
    return Y + noise
def add_poisson_noise(Y, snr):
    
    if snr is None:
        return Y   
    var_x = np.var(Y)
    lam = var_x / (10 ** (snr / 10))
    lam = max(lam, 0) 
    noise = np.random.poisson(lam, Y.shape) - lam
    
    return Y + noise
def add_phase_noise(Y, snr):
   
    if snr is None:
        return Y
        
    
    var_x = np.var(Y)
    
    
    sigma_sq = var_x / (10 ** (snr / 10))
    sigma = np.sqrt(sigma_sq)
    
    
    epsilon_phase = np.random.normal(0, sigma, Y.shape)
    
   
    return np.mod(Y + epsilon_phase, 2 * np.pi)
def add_topo_noise(edge_index, eta, num_nodes):

    if eta == 0:
        return edge_index

   
    keep_mask = np.random.uniform(0, 1, edge_index.shape[1]) >= eta
    edge_index_remained = edge_index[:, keep_mask]

    
    num_possible_empty = num_nodes * (num_nodes - 1) - edge_index.shape[1]
    num_spurious = int(max(0, num_possible_empty * eta))

    if num_spurious > 0:
        spurious_edges = np.random.randint(0, num_nodes, (2, num_spurious))
        edge_index_noisy = np.concatenate([edge_index_remained, spurious_edges], axis=1)
    else:
        edge_index_noisy = edge_index_remained

    no_self_loop_mask = edge_index_noisy[0] != edge_index_noisy[1]
    edge_index_final = edge_index_noisy[:, no_self_loop_mask]

    return edge_index_final.astype(np.int64)
def apply_savgol_smoothing(Y, dt=0.01, window_length=11, polyorder=3):
    Y_smoothed = savgol_filter(Y, window_length=window_length, polyorder=polyorder, axis=0)
    diff_Y_smoothed = savgol_filter(Y, window_length=window_length, polyorder=polyorder,
                                    deriv=1, delta=dt, axis=0)

    return Y_smoothed, diff_Y_smoothed

def load_csv_as_sparse_edge_index_nomarl(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    adj_array = df.to_numpy(dtype=float)
    np.fill_diagonal(adj_array, 0)
    adj_min = adj_array.min()
    adj_max = adj_array.max()
    if adj_max - adj_min > 1e-9:  
        adj_array = (adj_array - adj_min) / (adj_max - adj_min)
    else:
        adj_array = adj_array - adj_min
    mask = adj_array > 0.1
    adj_array[mask] = 1.0
    adj_array[~mask] = 0.0
    sparse_A = coo_matrix(adj_array)

    row = torch.from_numpy(sparse_A.row).long()
    col = torch.from_numpy(sparse_A.col).long()

    edge_index = torch.stack([row, col], dim=0)
    return np.array(edge_index.numpy(), dtype=int), adj_array
def load_data(case_name, dim, add_str="",snr = snr_global ,eta = eta_global): # Kuramoto 40

    test_data_set_path = 'data/test_data_on_%s.pickle' % (case_name)
    with open(test_data_set_path, 'rb') as f:
        batch_data = pickle.load(f)
    batch_data = batch_data[2]
    case_name_list = ["%s_dim=%s"%(case_name, i) for i in range(dim)]

    sparse_A = np.array(batch_data['adj'].numpy(), dtype=float)
    
    # sparse_A, adj_learn = load_csv_as_sparse_edge_index_nomarl('/home/liqifei/SymBOL_Code/data/ai_learned_adj.csv')


    X0 = batch_data['X0'].numpy()[2:-2,:]
    t_range = batch_data['t_target'].numpy()
    Mask = batch_data['obs_mask'].numpy()[2:-2,:,:]
    Y = batch_data['obs_state'].numpy()

    time_steps_to_keep = 100
    t_range = t_range[:time_steps_to_keep]
    Mask = Mask[:time_steps_to_keep][2:-2,:,:]
    Y = Y[:time_steps_to_keep]
    Dim = X0.shape[1]
    N = X0.shape[0]

    if eta > 0:
        N = batch_data['X0'].shape[0]
        sparse_A = add_topo_noise(sparse_A, eta, N)
    if snr is not None and snr > 0:
        Y = add_phase_noise(Y, snr)
    if snr is not None and snr > 0:
        Y, diff_Y = apply_savgol_smoothing(Y, dt=0.01, window_length=time_steps_to_keep-1, polyorder=2)
    else:
        Y, diff_Y = gp.finite_difference(Y, 0.01)

    return case_name_list, sparse_A, X0, t_range, Mask, Y, diff_Y, Dim, N
def load_data_discrete(case_name, dim, add_str=""):

    test_data_set_path = 'data/test_data_on_%s%s.pickle' % (case_name, add_str)

    with open(test_data_set_path, 'rb') as f:
        batch_data = pickle.load(f)


    case_name_list = ["%s_dim=%s"%(case_name, i) for i in range(dim)]
    sparse_A = np.array(batch_data['adj'].numpy(), dtype=float)

    X0 = batch_data['X0'].numpy()
    
    t_range = batch_data['t_target'].numpy()

    Mask = batch_data['obs_mask'].numpy()
   
    Y = batch_data['obs_state'].numpy()
    Y1, diff_Y = gp.finite_difference(Y, 1)
   
    Dim = X0.shape[1]
    
    N = X0.shape[0]
    period = 45
    limit = 100
    country_select_index = []
    time_select_index = []
    for country_index in range(Y.shape[1]):
        first_no0_index = (np.nonzero(Y[:, country_index])[0])[0]
        try:
            if len(Y[first_no0_index:, country_index]) >= period and (Y[first_no0_index:, country_index])[
                period - 1] >= limit:
                country_select_index.append(country_index)
                time_select_index.append([first_no0_index, first_no0_index + period - 1])
        except Exception as e:
            continue
   
    X0 = np.zeros(Y.shape[1], dtype=np.float32)

   
    for idx, country_index in enumerate(country_select_index):
        start_time = time_select_index[idx][0]
        X0[country_index] = Y[start_time, country_index]
    X0 = X0.reshape(-1, 1)  # shape: (N, 1)
    data_list = []

    for idx, (country, (start, end)) in enumerate(zip(country_select_index, time_select_index)):
        data_segment = Y[start:end + 1, country]  # shape: (T,)
        data_list.append(data_segment.reshape(-1, 1))  # reshape to (T, 1)

    true = np.stack(data_list, axis=1)  
    # print(N)
    # print(t_range.shape[0])
    return case_name_list, sparse_A, X0, t_range, Mask, Y, diff_Y, Dim, N,true[:, :9], country_select_index[:9], time_select_index


def load_data_SR(case_name, dim, add_str=""):
    test_data_set_path = 'data/lsr_transform/test_data_on_%s%s.pickle' % (case_name, add_str)
    # test_data_set_path = 'data/test_data_on_%s%s.pickle' % (case_name, add_str)

    with open(test_data_set_path, 'rb') as f:
        batch_data = pickle.load(f)
    batch_data = batch_data[0]


    case_name_list = ["%s_dim=%s" % (case_name, i) for i in range(dim)]
    # print(case_name_list)
    # adj sparse_adj
    # sparse_A = np.array(batch_data['adj'].numpy(), dtype=int)
    # print("sparse_A shape:", sparse_A.shape)
    X0 = batch_data['X0'].numpy()

    Y = batch_data['obs_state'].numpy()
    Dim = X0.shape[1]
    N = X0.shape[0]
    return case_name_list, X0, Y, Dim, N
print("case_name:", case_name)
if case_name in ["H1N1", "COVID19", "Sars"]:
    case_name_list, sparse_A, X0, t_range, Mask, Y, diff_Y, Dim, N ,true_data,country_select_index,time_select_index = load_data_discrete(case_name, dim)
elif case_name in ["Epidemic", "Neural", "Gene", "Population", "Mutualistic_6.12_before", "Heat", "Lotka_Volterra","Mutualistic","high_order","high_order1","high_order2","high_order_noliner","high_order_noliner1","Kuramoto","Kuramoto_01"]:
    case_name_list, sparse_A, X0, t_range, Mask, Y, diff_Y, Dim, N = load_data(case_name, dim)
else:
    case_name_list, X0, Y, Dim, N = load_data_SR(case_name, dim)

min_ = 1
max_ = 5

tol_err = 5e-3   # for mutu # for heat, gene

termination_condition_err = 1e-4

# timeout_second = 10.0*60
timeout_second = 10.0*60

# This value is related to the maximum complexity of the equation to be optimized,
# and the larger the value, the higher the complexity of the equation to be optimized.
# num_choose = multiprocessing.cpu_count()
num_choose = 20

# distribution of constants
min_const = -3.0
max_const = 3.0
sampling_const = partial(random.uniform, min_const, max_const)


# ------------------------------------
# build psets of f and g based on dim
# ------------------------------------
def build_psets(dim=1):
    # ---------- f -----------------
    pset_f = gp.PrimitiveSet("F", dim, prefix='x')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.add(x, y), nan=0),
                        2, name='Add')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.subtract(x, y), nan=0),
                        2, name='Sub')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.multiply(x, y), nan=0),
                        2, name='Mul')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.divide(x, y), nan=0),
                        2, name='Div')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.exp(x), nan=0),
                        1, name='exp')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.sin(x), nan=0),
                        1, name='sin')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.cos(x), nan=0),
                        1, name='cos')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.tan(x), nan=0),
                        1, name='tan')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.abs(x), nan=0),
                        1, name='Abs')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.log(x), nan=0),
                        1, name='log')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.power(x, y), nan=0),
                       2, name='Pow')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.tanh(x), nan=0), 1, name='tanh')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.sinh(x), nan=0), 1, name='sinh')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.cosh(x), nan=0), 1, name='cosh')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.arcsin(np.clip(x, -1, 1)), nan=0), 1, name='arcsin')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.arccos(np.clip(x, -1, 1)), nan=0), 1, name='arccos')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.arctan(x), nan=0), 1, name='arctan')
    pset_f.addPrimitive(lambda x: np.nan_to_num(1 / np.cos(x), nan=0, posinf=0, neginf=0), 1, name='sec')
    pset_f.addPrimitive(lambda x: np.nan_to_num(1/np.sin(x), nan=0, posinf=0, neginf=0), 1, name='csc')
    pset_f.addPrimitive(lambda x: np.nan_to_num(1/np.tan(x), nan=0, posinf=0, neginf=0), 1, name='cot')
    pset_f.addTerminal(np.e, name='e')  # 添加指数 e
    pset_f.addTerminal(np.pi, name='pi') #添加常数pi
    pset_f.addEphemeralConstant('C',
                                sampling_const)

    # ---------- g -----------------
    #高阶
    # pset_g = gp.PrimitiveSet("G", int(dim * 3), prefix='x')
    pset_g = gp.PrimitiveSet("G", int(dim * 2), prefix='x')

    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.add(x, y), nan=0),
                        2, name='Add')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.subtract(x, y), nan=0),
                        2, name='Sub')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.multiply(x, y), nan=0),
                        2, name='Mul')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.divide(x, y), nan=0),
                        2, name='Div')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.exp(x), nan=0),
                        1, name='exp')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.sin(x), nan=0),
                        1, name='sin')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.cos(x), nan=0),
                        1, name='cos')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.tan(x), nan=0),
                        1, name='tan')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.abs(x), nan=0),
                        1, name='Abs')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.log(x), nan=0),
                        1, name='log')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.power(x, y)),
                       2, name='Pow')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.tanh(x), nan=0), 1, name='tanh')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.sinh(x), nan=0), 1, name='sinh')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.cosh(x), nan=0), 1, name='cosh')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.arcsin(np.clip(x, -1, 1)), nan=0), 1, name='arcsin')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.arccos(np.clip(x, -1, 1)), nan=0), 1, name='arccos')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.arctan(x), nan=0), 1, name='arctan')
    pset_g.addPrimitive(lambda x: np.nan_to_num(1 / np.cos(x), nan=0, posinf=0, neginf=0), 1, name='sec')
    pset_g.addPrimitive(lambda x: np.nan_to_num(1/np.sin(x), nan=0, posinf=0, neginf=0), 1, name='csc')
    pset_g.addPrimitive(lambda x: np.nan_to_num(1/np.tan(x), nan=0, posinf=0, neginf=0), 1, name='cot')
    pset_g.addTerminal(np.e, name='e')  # 添加指数 e
    pset_g.addTerminal(np.pi, name='pi')
    pset_g.addEphemeralConstant('C',
                                sampling_const)

    pset = (pset_f, pset_g)

    return pset_f, pset_g, pset
def build_psets_noliner(dim=1):
    # ---------- f -----------------
    pset_f = gp.PrimitiveSet("F", dim, prefix='x')

    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.add(x, y), nan=0),
                        2, name='Add')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.subtract(x, y), nan=0),
                        2, name='Sub')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.multiply(x, y), nan=0),
                        2, name='Mul')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.divide(x, y), nan=0),
                        2, name='Div')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.exp(x), nan=0),
                        1, name='exp')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.sin(x), nan=0),
                        1, name='sin')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.cos(x), nan=0),
                        1, name='cos')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.tan(x), nan=0),
                        1, name='tan')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.abs(x), nan=0),
                        1, name='Abs')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.log(x), nan=0),
                        1, name='log')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.power(x, y), nan=0),
                       2, name='Pow')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.tanh(x), nan=0), 1, name='tanh')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.sinh(x), nan=0), 1, name='sinh')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.cosh(x), nan=0), 1, name='cosh')
    # === 互补三角函数 ===
    pset_f.addPrimitive(lambda x: np.nan_to_num(1 / np.tan(x), nan=0, posinf=0, neginf=0), 1, name='cot')
    pset_f.addPrimitive(lambda x: np.nan_to_num(1 / np.cos(x), nan=0, posinf=0, neginf=0), 1, name='sec')
    pset_f.addPrimitive(lambda x: np.nan_to_num(1 / np.sin(x), nan=0, posinf=0, neginf=0), 1, name='csc')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.arctan(x), nan=0),
                        1, name='arctan')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.arcsin(x), nan=0),
                        1, name='arcsin')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.arccos(x), nan=0),
                        1, name='arccos')
    pset_f.addTerminal(np.e, name='e')  # 添加指数 e
    pset_f.addTerminal(np.pi, name='pi') #添加常数pi
    pset_f.addEphemeralConstant('C',
                                sampling_const)

    # ---------- g -----------------
    pset_g = gp.PrimitiveSet("G", int(dim * 2), prefix='x')
    
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.add(x, y), nan=0),
                        2, name='Add')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.subtract(x, y), nan=0),
                        2, name='Sub')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.multiply(x, y), nan=0),
                        2, name='Mul')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.divide(x, y), nan=0),
                        2, name='Div')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.exp(x), nan=0),
                        1, name='exp')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.sin(x), nan=0),
                        1, name='sin')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.cos(x), nan=0),
                        1, name='cos')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.tan(x), nan=0),
                        1, name='tan')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.abs(x), nan=0),
                        1, name='Abs')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.log(x), nan=0),
                        1, name='log')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.power(x, y)),
                       2, name='Pow')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.tanh(x), nan=0), 1, name='tanh')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.sinh(x), nan=0), 1, name='sinh')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.cosh(x), nan=0), 1, name='cosh')
    pset_g.addTerminal(np.e, name='e')  # 添加指数 e
    pset_g.addTerminal(np.pi, name='pi')
    pset_g.addEphemeralConstant('C',
                                sampling_const)

    pset_s = gp.PrimitiveSet("S", int(dim), prefix='x')
    # for i in range(dim * 2):
    #     xi = f"x{i}"
    #     pset_g.addPrimitive(lambda c, x: c * x, 2, name=f"{xi}_weighted")
    #     pset_g.addTerminal(i, name=xi)
    pset_s.addPrimitive(lambda x, y: np.nan_to_num(np.add(x, y), nan=0),
                        2, name='Add')
    pset_s.addPrimitive(lambda x, y: np.nan_to_num(np.subtract(x, y), nan=0),
                        2, name='Sub')
    pset_s.addPrimitive(lambda x, y: np.nan_to_num(np.multiply(x, y), nan=0),
                        2, name='Mul')
    pset_s.addPrimitive(lambda x, y: np.nan_to_num(np.divide(x, y), nan=0),
                        2, name='Div')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.exp(x), nan=0),
                        1, name='exp')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.sin(x), nan=0),
                        1, name='sin')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.cos(x), nan=0),
                        1, name='cos')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.tan(x), nan=0),
                        1, name='tan')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.abs(x), nan=0),
                        1, name='Abs')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.log(x), nan=0),
                        1, name='log')
    pset_s.addPrimitive(lambda x, y: np.nan_to_num(np.power(x, y)),
                        2, name='Pow')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.tanh(x), nan=0), 1, name='tanh')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.sinh(x), nan=0), 1, name='sinh')
    pset_s.addPrimitive(lambda x: np.nan_to_num(np.cosh(x), nan=0), 1, name='cosh')
    pset_s.addTerminal(np.e, name='e')  # 添加指数 e
    pset_s.addTerminal(np.pi, name='pi')
    pset_s.addEphemeralConstant('C',
                                sampling_const)

    pset = (pset_f, pset_g, pset_s)

    return pset_f, pset_g, pset_s, pset

pset_f, pset_g, pset = build_psets(Dim)
pset_f1, pset_g1, pset_s1, pset1 = build_psets_noliner(Dim)

converter = {
    'Sub': lambda x, y: x - y,
    'Div': lambda x, y: x / y,
    'Mul': lambda x, y: x * y,
    'Add': lambda x, y: x + y,
    'Neg': lambda x: -x,
    'Pow': lambda x, y: x ** y,
    'Inv': lambda x: x ** (-1),
    'Sqrt': lambda x: sp.sqrt(x),
    'exp': lambda x: sp.exp(x),
    'sin': lambda x: sp.sin(x),
    'cos': lambda x: sp.cos(x),
    'tan': lambda x: sp.tan(x),
    'log': lambda x: sp.log(x),
    'Abs': lambda x: sp.Abs(x)
}


# used to save the complexity of searched equations

def stat_complex(pop):
    Stat_Complex = {}
    for p_i in pop:
        complex_of_eq = len(p_i[0][0]) + len(p_i[0][1])
        if complex_of_eq not in Stat_Complex.keys():
            Stat_Complex[complex_of_eq] = 1
        else:
            Stat_Complex[complex_of_eq] += 1
    return Stat_Complex


def eval_func_noliner(ind_i_1_2_3, pset, X0, sparse_A, Y, Mask, t_start=0, t_end=10, t_inc=0.01, Stat_Complex=None):
    
    pset_f_, pset_g_, pset_s_ = pset

    ind_f, ind_g, ind_s = ind_i_1_2_3

    def compile_expr(exprs, pset):
        compiled = gp.compile(exprs, pset)
        if opt_flag is not None and Dim is not None:
            zero_func = gp.compile(gp.PrimitiveTree.from_string_sympy("x0", pset), pset)
            result = [zero_func for _ in range(Dim)]
            result[opt_flag] = compiled
            return result
        else:
            if isinstance(exprs, list):
                return [gp.compile(e, pset) for e in exprs]
            else:
                return [compiled for _ in range(Dim)]

    if Dim == 1:
        eval_func_f_list = gp.compile(ind_f, pset_f_)
        eval_func_g_list = gp.compile(ind_g, pset_g_)
        eval_func_s_list = gp.compile(ind_s, pset_s_)  
    else:
        eval_func_f_list = compile_expr(ind_f, pset_f_)
        eval_func_g_list = compile_expr(ind_g, pset_g_)
        eval_func_s_list = compile_expr(ind_s, pset_s_)

    
    soluation_Y = gp.solve_ivp_diff_noliner(
        eval_func_f_list,
        eval_func_g_list,
        eval_func_s_list,  
        X0, sparse_A, Y, diff_Y,
        t_start=0, t_end=1, t_inc=0.01
    )

    # 4. 计算 Loss
    dim = soluation_Y.shape[-1]
    loss_list = []
    for d in range(dim):
        if d == opt_flag:
            pred_d = soluation_Y[..., d]
            true_d = diff_Y[..., d]

            mse_d = np.mean((pred_d - true_d) ** 2)
            denom_d = np.var(true_d)
            
            loss_d = mse_d / denom_d if denom_d != 0 else 1e30
            loss_list.append(loss_d)

    if Dim == 1:
        total_loss = loss_list[0]
    else:
        total_loss = np.mean(loss_list)

    if Stat_Complex is not None:
        def get_complex(ind):
            return sum(len(e) for e in (ind if isinstance(ind, list) else [ind]))

        complex_of_eq = get_complex(ind_f) + get_complex(ind_g) + get_complex(ind_s)

        total_loss *= np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))

    return np.nan_to_num(total_loss, nan=1e30)
def eval_func(ind_i_1_2, pset, X0, sparse_A, Y, Mask, Stat_Complex=None):
    pset_f_, pset_g_ = pset

    
    def compile_expr(exprs, pset):
        # 编译表达式
        compiled = gp.compile(exprs, pset)

        if opt_flag is not None and Dim is not None:
            zero_func = gp.compile(gp.PrimitiveTree.from_string_sympy("x0", pset), pset)

            result = [zero_func for _ in range(Dim)]

            result[opt_flag] = compiled

            return result

        else:
            if isinstance(exprs, list):
                return [gp.compile(e, pset) for e in exprs]
            else:
                return [compiled for _ in range(Dim)]
 

    def compile_expr1(exprs, pset):
        if isinstance(exprs, list):
            return [gp.compile(e, pset) for e in exprs]
        else:
            return [gp.compile(exprs, pset)]

    
    if Dim == 1:
        eval_func_f_list = gp.compile(ind_i_1_2[0], pset_f_)
        eval_func_g_list = gp.compile(ind_i_1_2[1], pset_g_)
    else:
        eval_func_f_list = compile_expr(ind_i_1_2[0], pset_f_)
        eval_func_g_list = compile_expr(ind_i_1_2[1], pset_g_)

   
    if case_name == "high_order2":
        soluation_Y = gp.solve_ivp_diff_high_order(
            eval_func_f_list,
            eval_func_g_list,
            X0, sparse_A, Y, diff_Y,
            t_start=0, t_end=1, t_inc=0.01
        )
    else:
        soluation_Y = gp.solve_ivp_diff(
            eval_func_f_list,
            eval_func_g_list,
            X0, sparse_A, Y, diff_Y,
            t_start=0, t_end=1, t_inc=0.01
        )
    
    dim = soluation_Y.shape[-1] 
    loss_list = []
    for d in range(dim):
        if d == opt_flag:
            pred_d = soluation_Y[..., d]
            # print(pred_d.shape)
            true_d = diff_Y[..., d]
  
            mask_d = Mask[..., d]
            
            mse_d = np.mean((pred_d - true_d) ** 2)
            # denom_d = np.mean(true_d[mask_d == 1] ** 2)
            denom_d = np.var(true_d)
            loss_d = mse_d / denom_d if denom_d != 0 else 1e30
            loss_list.append(loss_d)

   
    if Dim == 1:
        total_loss = loss_list[0]
    else:
       total_loss = np.mean(loss_list)
       
    if Stat_Complex is not None:
        complex_of_eq = sum(len(e) for e in (ind_i_1_2[0] if isinstance(ind_i_1_2[0], list) else [ind_i_1_2[0]])) \
                        + sum(len(e) for e in (ind_i_1_2[1] if isinstance(ind_i_1_2[1], list) else [ind_i_1_2[1]]))
        total_loss *= np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))
    return np.nan_to_num(total_loss, nan=1e30)

def eval_func_NMSE_discrete(ind_i_1_2, pset, X0, sparse_A, Y, Mask,t_start=0, t_end=10, t_inc=0.01,Stat_Complex=None):
    # print('.')
    pset_f_, pset_g_ = pset
    def compile_expr(exprs, pset):
    
        compiled = gp.compile(exprs, pset)
    
        if opt_flag is not None and Dim is not None:
            zero_func = gp.compile(gp.PrimitiveTree.from_string_sympy("x0", pset), pset)
            result = [zero_func for _ in range(Dim)]

            result[opt_flag] = compiled
            return result
        else:
            if isinstance(exprs, list):
                return [gp.compile(e, pset) for e in exprs]
            else:
                return [compiled for _ in range(Dim)]
    def compile_expr1(exprs, pset):
        if isinstance(exprs, list):
            return [gp.compile(e, pset) for e in exprs]
        else:
            return [gp.compile(exprs, pset)]

    if Dim == 1:
        eval_func_f_list = gp.compile(ind_i_1_2[0], pset_f_)
        eval_func_g_list = gp.compile(ind_i_1_2[1], pset_g_)
    else:
        eval_func_f_list = compile_expr(ind_i_1_2[0], pset_f_)
        eval_func_g_list = compile_expr(ind_i_1_2[1], pset_g_)
    
    soluation_Y = gp.solve_ivp_diff_modified_local(eval_func_f_list,
        eval_func_g_list,
        Y, country_select_index,time_select_index,sparse_A)
    
    dim = soluation_Y.shape[-1]  # 最后一维就是变量数量
    loss_list = []
    for d in range(dim):
        if d == opt_flag:
            pred_d = soluation_Y[..., d]
            true = true_data[..., d]
            X0 = X0.reshape(-1)  
            cumulative_pred = np.vstack([X0, X0 + np.cumsum(pred_d, axis=0)])  # shape: (45, 21)
            sub_matrix = cumulative_pred[:, country_select_index]

            

            mse_per_country = np.mean((true - sub_matrix) ** 2, axis=0)
            var_per_country = np.var(true, axis=0)
            loss_d = np.max(mse_per_country/var_per_country)


            
            if loss_d > 10 or loss_d == 1e30:
                loss_d = random.uniform(3.8, 3.99)

            loss_list.append(loss_d)
            
    if Dim == 1:
        total_loss = loss_list[0]
    else:
       total_loss = np.mean(loss_list)
       
    if Stat_Complex is not None:
        complex_of_eq = sum(len(e) for e in (ind_i_1_2[0] if isinstance(ind_i_1_2[0], list) else [ind_i_1_2[0]])) \
                        + sum(len(e) for e in (ind_i_1_2[1] if isinstance(ind_i_1_2[1], list) else [ind_i_1_2[1]]))
        total_loss *= np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))
    return np.nan_to_num(total_loss, nan=1e30)

def evaluate_gp_function(eval_func, x):
    
    diff_f = np.array(
        eval_func(*[x[:, m].reshape(-1, 1) for m in range(x.shape[1])]),
        dtype=np.float32
    )

    if diff_f.ndim < 2:
        diff_f = diff_f.reshape(-1, 1)

    return diff_f
def eval_func_SR(ind_i_1_2, pset, X0, Y, Stat_Complex=None):
        pset_f_, pset_g_ = pset

        def compile_expr(exprs, pset):
            compiled = gp.compile(exprs, pset)

           
            if opt_flag is not None and Dim is not None:
                
                zero_func = gp.compile(gp.PrimitiveTree.from_string_sympy("x0", pset), pset)

                result = [zero_func for _ in range(Dim)]

                result[opt_flag] = compiled

                return result

            else:
                if isinstance(exprs, list):
                    return [gp.compile(e, pset) for e in exprs]
                else:
                    return [compiled for _ in range(Dim)]


        def compile_expr1(exprs, pset):
            if isinstance(exprs, list):
                return [gp.compile(e, pset) for e in exprs]
            else:
                return [gp.compile(exprs, pset)]

            
        if Dim == 1:
            eval_func_f_list = gp.compile(ind_i_1_2[0], pset_f_)
        else:
            eval_func_f_list = compile_expr(ind_i_1_2[0], pset_f_)

       
        soluation_Y = evaluate_gp_function(eval_func_f_list[0],X0)
       
        dim = soluation_Y.shape[-1]  # 最后一维就是变量数量
        loss_list = []
        for d in range(dim):
            if d == opt_flag:
               
                pred_d = soluation_Y[..., d]
                true_d = Y

                valid_mask = ~np.isnan(pred_d)
                pred_d = pred_d[valid_mask]
                true_d = true_d[valid_mask]

                mse_d = np.mean((pred_d - true_d) ** 2)
                var_d = np.var(true_d)  

                nmse_d = mse_d / var_d if var_d != 0 else 1e30
                loss_list.append(nmse_d)
               
        if Dim == 1:
            total_loss = loss_list[0]
        else:
            total_loss = np.mean(loss_list)
          
        if Stat_Complex is not None:
            complex_of_eq = sum(len(e) for e in (ind_i_1_2[0] if isinstance(ind_i_1_2[0], list) else [ind_i_1_2[0]])) \
                            + sum(len(e) for e in (ind_i_1_2[1] if isinstance(ind_i_1_2[1], list) else [ind_i_1_2[1]]))
            total_loss *= np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))
        return np.nan_to_num(total_loss, nan=1e30)

@func_set_timeout(timeout_second)
def optimize_in_scipy(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options):
    minimizer_kwargs = {"method": method}
    tol = 1e-12
    res = opt.minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    res = opt.minimize(fun, res.x, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    res = opt.minimize(fun, res.x, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    return res


def optimize_in_scipy_with_timeout_check(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol,
                                         callback, options):
    return optimize_in_scipy(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback,
                                options)



def multistart(on_core_i, fun, x0, nrestart=1, full_output=False, args=(), method=None, jac=None, hess=None, hessp=None,
               bounds=None, constraints=(), tol=None, callback=None, options = None):
    options={'xatol': 1e-12, 'fatol': 1e-12}
    
    res_list = np.empty(nrestart, dtype=object)
    res = optimize_in_scipy_with_timeout_check(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints,
                                               tol, callback, options)
    res_list[0] = res
    for i in range(nrestart - 1):
        new_x0 = x0 + np.array([random.gauss(0, 0.1) * 1. for _ in range(x0.shape[0])])
        res = optimize_in_scipy_with_timeout_check(on_core_i, fun, new_x0, args, method, jac, hess, hessp, bounds,
                                                   constraints, tol, callback, options)
        res_list[i + 1] = res
        print(res)

    res_fun_list = []
    for res in res_list:
        if res is None:
            res_fun_list.append(1e30)
        else:
            res_fun_list.append(res.fun)
    sort_res_list = res_list[np.argsort(res_fun_list)]
    if full_output:
        return sort_res_list[0], sort_res_list
    else:
        return sort_res_list[0]

def optimizeConstants_SR(individual, X0, Y, on_core_i, opt_const_dim=-1):
    # Optimize the constants
    constants_f = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[0]))))
    if opt_const_dim > 0:
        constants_f = constants_f[:opt_const_dim]
    num_constants_f = len(constants_f)
    constants_g = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[1]))))
    if opt_const_dim > 0:
        constants_g = constants_g[:opt_const_dim]
    num_constants_g = len(constants_g)
    constants = constants_f
   
    if constants.size > 0:
        def setConstants(individual, constants):
            optIndividual = individual
            c = 0
            for i in range(0, len(optIndividual)):
                if isinstance(optIndividual[i], gp.Terminal) and not isinstance(optIndividual[i].value, str) and c < len(constants):
                    optIndividual[i].value = np.nan_to_num(constants[c])
                    optIndividual[i].name = str(optIndividual[i].value)
                    c += 1
            return optIndividual

        def evaluate(constants, individual):
            
            individual = setConstants(individual, constants)
            individual_f_ = setConstants(individual[0], constants[:num_constants_f])
            individual_g_ = setConstants(individual[1], constants[num_constants_f:])
           
            return eval_func_SR((individual_f_, individual_g_), pset, X0,Y)
           


        def evaluateLM(constants):

            return evaluate(constants, individual)


        res = multistart(on_core_i, evaluateLM, constants, nrestart=1,method='Nelder-Mead')
  
        if res is not None:
            individual_f = setConstants(individual[0], res.x[:num_constants_f])
            individual_g = setConstants(individual[1], res.x[num_constants_f:])
        else:
            individual_f = setConstants(individual[0], np.array([sampling_const() for _ in range(num_constants_f)]))
            individual_g = setConstants(individual[1], np.array([sampling_const() for _ in range(num_constants_g)]))
       
        return (individual_f, individual_g)
    else:
        return individual
def optimizeConstants_noliner(individual, X0, sparse_A, Y, Mask, on_core_i, opt_const_dim=-1):
    
    constants_f = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[0]))))
    if opt_const_dim > 0:
        constants_f = constants_f[:opt_const_dim]
    num_constants_f = len(constants_f)

    constants_g = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[1]))))
    if opt_const_dim > 0:
        constants_g = constants_g[:opt_const_dim]
    num_constants_g = len(constants_g)

    constants_s = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[2]))))
    if opt_const_dim > 0:
        constants_s = constants_s[:opt_const_dim]
    num_constants_s = len(constants_s)

    constants = np.concatenate([constants_f, constants_g, constants_s], axis=-1)
    
    if constants.size > 0:
        def setConstants(individual_comp, constants_slice):
            optIndividual = individual_comp
            c = 0
            for i in range(0, len(optIndividual)):
                if isinstance(optIndividual[i], gp.Terminal) and not isinstance(optIndividual[i].value,
                                                                                str) and c < len(constants_slice):
                    optIndividual[i].value = np.nan_to_num(constants_slice[c])
                    optIndividual[i].name = str(optIndividual[i].value)
                    c += 1
            return optIndividual

        def evaluateLM(constants):
           
            idx_g = num_constants_f + num_constants_g

            individual_f_ = setConstants(individual[0], constants[:num_constants_f])
            individual_g_ = setConstants(individual[1], constants[num_constants_f:idx_g])
            individual_s_ = setConstants(individual[2], constants[idx_g:])

            return eval_func_noliner((individual_f_, individual_g_, individual_s_), pset1, X0, sparse_A, Y, Mask)

      
        res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='Nelder-Mead')

        if res is not None:
            idx_g = num_constants_f + num_constants_g
            individual_f = setConstants(individual[0], res.x[:num_constants_f])
            individual_g = setConstants(individual[1], res.x[num_constants_f:idx_g])
            individual_s = setConstants(individual[2], res.x[idx_g:])
        else:
            individual_f = setConstants(individual[0], np.array([sampling_const() for _ in range(num_constants_f)]))
            individual_g = setConstants(individual[1], np.array([sampling_const() for _ in range(num_constants_g)]))
            individual_s = setConstants(individual[2], np.array([sampling_const() for _ in range(num_constants_s)]))

        return (individual_f, individual_g, individual_s)
    else:
        return individual
# Define our evaluation function
def optimizeConstants(individual, X0, sparse_A, Y, Mask, on_core_i, opt_const_dim=-1):
    # s_time = time.time()
    # Optimize the constants
    constants_f = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[0]))))
    if opt_const_dim > 0:
        constants_f = constants_f[:opt_const_dim]
    num_constants_f = len(constants_f)
    constants_g = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[1]))))
    if opt_const_dim > 0:
        constants_g = constants_g[:opt_const_dim]
    num_constants_g = len(constants_g)
    constants = np.concatenate([constants_f, constants_g], axis=-1)
    
    if constants.size > 0:
        def setConstants(individual, constants):
            optIndividual = individual
            c = 0
            for i in range(0, len(optIndividual)):
                if isinstance(optIndividual[i], gp.Terminal) and not isinstance(optIndividual[i].value, str) and c < len(constants):
                    optIndividual[i].value = np.nan_to_num(constants[c])
                    optIndividual[i].name = str(optIndividual[i].value)
                    c += 1
            return optIndividual

        def evaluate(constants, individual):
            individual_f_ = setConstants(individual[0], constants[:num_constants_f])
            individual_g_ = setConstants(individual[1], constants[num_constants_f:])
           
            return eval_func((individual_f_, individual_g_), pset, X0, sparse_A, Y, Mask)

        def evaluateLM(constants):
           
            return evaluate(constants, individual)


        res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='Nelder-Mead')
        
        if res is not None:
            individual_f = setConstants(individual[0], res.x[:num_constants_f])
            individual_g = setConstants(individual[1], res.x[num_constants_f:])
        else:
            individual_f = setConstants(individual[0], np.array([sampling_const() for _ in range(num_constants_f)]))
            individual_g = setConstants(individual[1], np.array([sampling_const() for _ in range(num_constants_g)]))

        return (individual_f, individual_g)
    else:
        return individual
def optimizeConstants_discrete(individual, X0, sparse_A, Y, Mask, on_core_i, opt_const_dim=-1):
    
    constants_f = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[0]))))
    if opt_const_dim > 0:
        constants_f = constants_f[:opt_const_dim]
    num_constants_f = len(constants_f)
    constants_g = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[1]))))
    if opt_const_dim > 0:
        constants_g = constants_g[:opt_const_dim]
    num_constants_g = len(constants_g)
    constants = np.concatenate([constants_f, constants_g], axis=-1)
    
    if constants.size > 0:
        def setConstants(individual, constants):
            optIndividual = individual
            c = 0
            for i in range(0, len(optIndividual)):
                if isinstance(optIndividual[i], gp.Terminal) and not isinstance(optIndividual[i].value, str) and c < len(constants):
                    optIndividual[i].value = np.nan_to_num(constants[c])
                    optIndividual[i].name = str(optIndividual[i].value)
                    c += 1
            return optIndividual

        def evaluate(constants, individual):
            individual_f_ = setConstants(individual[0], constants[:num_constants_f])
            individual_g_ = setConstants(individual[1], constants[num_constants_f:])
           
            return eval_func_NMSE_discrete((individual_f_, individual_g_), pset, X0, sparse_A, Y, Mask)

        def evaluateLM(constants):
           
            return evaluate(constants, individual)

       
        res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='Nelder-Mead')
       
        if res is not None:
            individual_f = setConstants(individual[0], res.x[:num_constants_f])
            individual_g = setConstants(individual[1], res.x[num_constants_f:])
        else:
            individual_f = setConstants(individual[0], np.array([sampling_const() for _ in range(num_constants_f)]))
            individual_g = setConstants(individual[1], np.array([sampling_const() for _ in range(num_constants_g)]))
       
        return (individual_f, individual_g)
    else:
        return individual


def tournamentSelection(P, num=12, prob=0.9):
    Q = random.sample(P, num)
    while len(Q) > 1:
        sort_index = np.argsort([ii[1] for ii in Q])
        E = Q[sort_index[0]]
        if random.uniform(0, 1) < prob:
            break
        Q.remove(E)
    return E


def eval_pop(pop):
    pop_eval = []
    for idx in tqdm(range(len(pop))):
        ind_i_1 = pop[idx][0][0]
        ind_i_2 = pop[idx][0][1]
        ind_i_fitness = eval_func((ind_i_1, ind_i_2), pset, X0, sparse_A, Y, Mask)
        pop_eval.append([(ind_i_1, ind_i_2), ind_i_fitness])
    return pop_eval


def multi_processing_eval_pop(pop, num_running_each_core=10):
    store_all = []
    store_all2 = []
    with multiprocessing.Pool() as p:
        for i in range(math.ceil(len(pop) / num_running_each_core)):
            pop_eval = p.apply_async(eval_pop, args=(
                pop[i * num_running_each_core:i * num_running_each_core + num_running_each_core],))
            store_all.append(pop_eval)
        p.close()
        p.join()
        # for i in tqdm(store_all):
        for i in store_all:
            store_all2.append(i.get())

    new_pop = []
    for store_i in store_all2:
        new_pop += store_i
    # print(len(new_pop))

    return new_pop

def evolve(pop, num_evo):
    pop_new = []
    for _ in tqdm(range(num_evo)):

        ind_one = tournamentSelection(pop, num=12)
        ind_one_new = gp.Mutations_f_g(ind_one[0], pset, sampling_const, converter, min_=min_, max_=max_)
        pop_new.append([ind_one_new, 1e30])
    return pop_new





def choose(pop, num_choose):
    Stat_Complex = stat_complex(pop)
    fitness_list = []
    for p_i in pop:
        complex_of_eq = len(p_i[0][0]) + len(p_i[0][1])
        fitness_list.append(p_i[1] * np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values())))))
    pop_choose = np.array(pop)[np.argsort(fitness_list)[:num_choose]].tolist()
    print([len(p_i[0][0])+len(p_i[0][1]) for p_i in pop_choose])
    return pop_choose

def choose_best(pop, num_choose):
    if num_choose == -1:
        pop_choose = np.array(pop)[np.argsort([p_i[1] for p_i in pop])].tolist()
    else:
        pop_choose = np.array(pop)[np.argsort([p_i[1] for p_i in pop])[:num_choose]].tolist()
    # print([len(p_i[0][0])+len(p_i[0][1]) for p_i in pop_choose])
    return pop_choose

def choose_diversity(pop, num_choose):
    pop_choose = get_return_pop(pop, filter=False)
    if num_choose is None:
        return pop_choose
    return pop_choose[:num_choose]


# multi processing
def opt_constant_one_core(pop, core_i, opt_const_dim=-1):
    pop_opt = []
    opt_flag_one_core = True
    # print(len(pop))
    for i in range(len(pop)):
        ind_i = copy.deepcopy(pop[i][0])
        if opt_flag_one_core:
            ind_i = optimizeConstants(ind_i, X0, sparse_A, Y, Mask, core_i, opt_const_dim=opt_const_dim)

        ind_i_fitness = eval_func(ind_i, pset, X0, sparse_A, Y, Mask)
        print('ind_i_fitness = %s'%ind_i_fitness)
        if ind_i_fitness <= termination_condition_err:
            opt_flag_one_core = False
            print("\n==== Found opt that meets the termination condition on core [%s]\n" % (core_i))
            print(ind_i_fitness)

        pop_opt.append([ind_i, ind_i_fitness])
    return pop_opt
def opt_constant_one_core_noliner(pop, core_i, opt_const_dim=-1):
    pop_opt = []
    opt_flag_one_core = True

    for i in tqdm(range(len(pop))):
        ind_i = copy.deepcopy(pop[i][0])
        if opt_flag_one_core:
            ind_i = optimizeConstants_noliner(ind_i, X0, sparse_A, Y, Mask, core_i, opt_const_dim=opt_const_dim)
          

        ind_i_fitness = eval_func_noliner(ind_i, pset1, X0, sparse_A, Y, Mask)
        print('ind_i_fitness = %s'%ind_i_fitness)
        if ind_i_fitness <= termination_condition_err:
            opt_flag_one_core = False
            print("\n==== Found opt that meets the termination condition on core [%s]\n" % (core_i))
            
        pop_opt.append([ind_i, ind_i_fitness])

    return pop_opt
def opt_constant_one_core_discrete(pop, core_i, opt_const_dim=-1):
    pop_opt = []
    opt_flag_one_core = True

    for i in tqdm(range(len(pop))):
        # print(' on core [%s] : %s/%s ... ' % (core_i, i + 1, len(pop)))
        ind_i = copy.deepcopy(pop[i][0])
        if opt_flag_one_core:
            ind_i = optimizeConstants_discrete(ind_i, X0, sparse_A, Y, Mask, core_i, opt_const_dim=opt_const_dim)

        ind_i_fitness = eval_func_NMSE_discrete(ind_i, pset, X0, sparse_A, Y, Mask)
        print('ind_i_fitness = %s'%ind_i_fitness)
        if ind_i_fitness <= termination_condition_err:
            opt_flag_one_core = False
            print("\n==== Found opt that meets the termination condition on core [%s]\n" % (core_i))
           
        pop_opt.append([ind_i, ind_i_fitness])
    return pop_opt

def opt_constant_one_core_SR(pop, core_i, opt_const_dim=-1):
    pop_opt = []
    opt_flag_one_core = True
   
    for i in tqdm(range(len(pop))):
        # print(' on core [%s] : %s/%s ... ' % (core_i, i + 1, len(pop)))
        ind_i = copy.deepcopy(pop[i][0])
        # print(ind_i)
        if opt_flag_one_core:
            ind_i = optimizeConstants_SR(ind_i, X0, Y, core_i, opt_const_dim=opt_const_dim)

        ind_i_fitness = eval_func_SR(ind_i, pset, X0, Y)
        # ind_i_fitness = eval_func(ind_i, pset, X0, sparse_A, Y, Mask)
        print('ind_i_fitness = %s'%ind_i_fitness)
        if ind_i_fitness <= termination_condition_err:
            opt_flag_one_core = False
            print("\n==== Found opt that meets the termination condition on core [%s]\n" % (core_i))
            
        pop_opt.append([ind_i, ind_i_fitness])

    return pop_opt
def process_individual_wrapper(i, pop, core_i, opt_const_dim, X0, sparse_A, Y, Mask, termination_condition_err):
    return process_individual(i, pop, core_i, opt_const_dim, X0, sparse_A, Y, Mask, termination_condition_err)


def process_individual(i, pop, core_i, opt_const_dim, X0, sparse_A, Y, Mask, termination_condition_err):
    ind_i = copy.deepcopy(pop[i][0])
    ind_i = optimizeConstants(ind_i, X0, sparse_A, Y, Mask, core_i, opt_const_dim=opt_const_dim)
    ind_i_fitness = eval_func(ind_i, pset, X0, sparse_A, Y, Mask)
    print(f'ind_i_fitness = {ind_i_fitness}')

    return ind_i, ind_i_fitness


def opt_constant_multi_core(pop, core_i, opt_const_dim=-1, num_cores=None):
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    pop_opt = []
    opt_flag_one_core = True

    process_partial = partial(
        process_individual,
        pop=pop,
        core_i=core_i,
        opt_const_dim=opt_const_dim,
        X0=X0,
        sparse_A=sparse_A,
        Y=Y,
        Mask=Mask,
        termination_condition_err=termination_condition_err
    )

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_partial, 
            range(len(pop))),
            total=len(pop),
            desc=f"Optimizing on core [{core_i}]"
        ))

    # 处理返回结果
    for i, (ind_i, ind_i_fitness) in enumerate(results):
        print(f'ind_i_fitness = {ind_i_fitness}')

        if ind_i_fitness <= termination_condition_err and opt_flag_one_core:
            opt_flag_one_core = False
            print(f"\n==== Found opt that meets the termination condition on core [{core_i}]\n")

        pop_opt.append([ind_i, ind_i_fitness])

    return pop_opt

def worker_process_wrapper(q, chunk, core_id, opt_const_dim):

    try:
        res = opt_constant_one_core(chunk, core_id, opt_const_dim)
        q.put(res)
    except Exception as e:
        q.put(e)
def worker_process_wrapper_SR(q, chunk, core_id, opt_const_dim):
    
    try:
        res = opt_constant_one_core_SR(chunk, core_id, opt_const_dim)
        q.put(res)
    except Exception as e:
        q.put(e)

def worker_process_wrapper_noliner(q, chunk, core_id, opt_const_dim):
   
    try:
        res = opt_constant_one_core_noliner(chunk, core_id, opt_const_dim)
        q.put(res)
    except Exception as e:
        q.put(e)
def multi_processing_opt_constant_SR(pop, num_running_each_core=10, opt_const_dim=-1):
    total_individuals = len(pop)
    if total_individuals == 0:
        return []

    MAX_CORES = 40
    num_cores = min(MAX_CORES, multiprocessing.cpu_count(), total_individuals)
    
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    chunks = [
        pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals
    ]


    pending_chunks = list(enumerate(chunks))
    pending_chunks.reverse()
    
    running_processes = {}
    collected_results = {}  
    
    CHUNK_TIMEOUT = 600      

    pbar = tqdm(total=total_individuals, desc="Optimizing", unit="ind")

    try:
        while len(pending_chunks) > 0 or len(running_processes) > 0:
            
            while len(running_processes) < num_cores and len(pending_chunks) > 0:
                c_idx, c_data = pending_chunks.pop()
                
                q = multiprocessing.Queue()
                p = multiprocessing.Process(
                    target=worker_process_wrapper_SR, 
                    args=(q, c_data, c_idx + 1, opt_const_dim)
                )
                p.daemon = True
                p.start()
                
                running_processes[c_idx] = {
                    'proc': p,
                    'queue': q,
                    'start_time': time.time(),
                    'data': c_data
                }

            active_indices = list(running_processes.keys())
            
            for c_idx in active_indices:
                info = running_processes[c_idx]
                proc = info['proc']
                q = info['queue']
                start_t = info['start_time']
                chunk_data = info['data']
                
                is_timeout = (time.time() - start_t) > CHUNK_TIMEOUT
                is_dead = not proc.is_alive()
                
                result_sublist = None
                error_msg = None

                if is_timeout:
                    tqdm.write(f"[WARN] Core-{c_idx+1}  ({CHUNK_TIMEOUT}s). ")
                    if proc.is_alive():
                        proc.terminate()
                        time.sleep(0.1)
                        if proc.is_alive():
                            proc.kill()
                    proc.join()
                    error_msg = "TIMEOUT"

                elif is_dead:
                    try:
                        res = q.get_nowait()
                        if isinstance(res, Exception):
                            tqdm.write(f"[ERROR] Core-{c_idx+1}  {res}")
                            error_msg = "EXCEPTION"
                        else:
                            result_sublist = res
                    except queue.Empty:
                        tqdm.write(f"[ERROR] Core-{c_idx+1}.")
                        error_msg = "CRASH"
                    proc.join()

                if result_sublist is not None:
                    collected_results[c_idx] = result_sublist
                    del running_processes[c_idx]
                    pbar.update(len(result_sublist))
                    
                elif error_msg is not None:

                    collected_results[c_idx] = chunk_data
                    del running_processes[c_idx]
                    pbar.update(len(chunk_data))
            time.sleep(0.1)

    finally:
        pbar.close()

    new_pop = []
    for i in range(len(chunks)):
        new_pop.extend(collected_results[i])

    return new_pop
def multi_processing_opt_constant_discrete(pop, num_running_each_core=10, opt_const_dim=-1):
    num_cores = multiprocessing.cpu_count()
    total_individuals = len(pop)
    
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    
    chunks = [
        pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals 
    ]

    with multiprocessing.Pool(num_cores) as p:
        args = [(chunk, i + 1, opt_const_dim) for i, chunk in enumerate(chunks)]
        results = p.starmap(opt_constant_one_core_discrete, args)
    new_pop = [item for sublist in results for item in sublist]
    return new_pop
def multi_processing_opt_constant_2026_2_27(pop, num_running_each_core=10, opt_const_dim=-1):
    num_cores = multiprocessing.cpu_count()
    total_individuals = len(pop)
    
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    
    chunks = [
        pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals  
    ]
    with multiprocessing.Pool(num_cores) as p:
        args = [(chunk, i + 1, opt_const_dim) for i, chunk in enumerate(chunks)]
        results = p.starmap(opt_constant_one_core, args)
    new_pop = [item for sublist in results for item in sublist]
    return new_pop
    



def multi_processing_opt_constant(pop, num_running_each_core=10, opt_const_dim=-1):
    total_individuals = len(pop)
    if total_individuals == 0:
        return []

    MAX_CORES = 40
    num_cores = min(MAX_CORES, multiprocessing.cpu_count(), total_individuals)
    
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    chunks = [
        pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals
    ]
    pending_chunks = list(enumerate(chunks))
    pending_chunks.reverse()
    
    running_processes = {}
    collected_results = {}  
    
    CHUNK_TIMEOUT = 600     

    pbar = tqdm(total=total_individuals, desc="Optimizing", unit="ind")

    try:
        while len(pending_chunks) > 0 or len(running_processes) > 0:
            
            while len(running_processes) < num_cores and len(pending_chunks) > 0:
                c_idx, c_data = pending_chunks.pop()
                q = multiprocessing.Queue()
                p = multiprocessing.Process(
                    target=worker_process_wrapper, 
                    args=(q, c_data, c_idx + 1, opt_const_dim)
                )
                p.daemon = True
                p.start()
                
                running_processes[c_idx] = {
                    'proc': p,
                    'queue': q,
                    'start_time': time.time(),
                    'data': c_data
                }

            active_indices = list(running_processes.keys())
            
            for c_idx in active_indices:
                info = running_processes[c_idx]
                proc = info['proc']
                q = info['queue']
                start_t = info['start_time']
                chunk_data = info['data']
                
                is_timeout = (time.time() - start_t) > CHUNK_TIMEOUT
                is_dead = not proc.is_alive()
                
                result_sublist = None
                error_msg = None

                if is_timeout:
                    tqdm.write(f"[WARN] Core-{c_idx+1} ({CHUNK_TIMEOUT}s).")
                    if proc.is_alive():
                        proc.terminate()
                        time.sleep(0.1)
                        if proc.is_alive():
                            proc.kill()
                    proc.join()
                    error_msg = "TIMEOUT"

                elif is_dead:
                    try:
                        res = q.get_nowait()
                        if isinstance(res, Exception):
                            tqdm.write(f"[ERROR] Core-{c_idx+1}  {res}")
                            error_msg = "EXCEPTION"
                        else:
                            result_sublist = res
                    except queue.Empty:
                        tqdm.write(f"[ERROR] Core-{c_idx+1} .")
                        error_msg = "CRASH"
                    proc.join()

                if result_sublist is not None:
                    collected_results[c_idx] = result_sublist
                    del running_processes[c_idx]
                    pbar.update(len(result_sublist))
                    
                elif error_msg is not None:
                    
                    collected_results[c_idx] = chunk_data
                    del running_processes[c_idx]
                    pbar.update(len(chunk_data))

            time.sleep(0.1)

    finally:
        pbar.close()

    new_pop = []
    for i in range(len(chunks)):
        new_pop.extend(collected_results[i])

    return new_pop
def multi_processing_opt_constant_noliner(pop, num_running_each_core=10, opt_const_dim=-1):
    total_individuals = len(pop)
    if total_individuals == 0:
        return []

    MAX_CORES = 40
    num_cores = min(MAX_CORES, multiprocessing.cpu_count(), total_individuals)
    
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    chunks = [
        pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals
    ]


    pending_chunks = list(enumerate(chunks))
    pending_chunks.reverse()
    
    running_processes = {}
    collected_results = {}  
    
    CHUNK_TIMEOUT = 600      

    pbar = tqdm(total=total_individuals, desc="Optimizing", unit="ind")

    try:
        while len(pending_chunks) > 0 or len(running_processes) > 0:
            
            while len(running_processes) < num_cores and len(pending_chunks) > 0:
                c_idx, c_data = pending_chunks.pop()
                
                q = multiprocessing.Queue()
                p = multiprocessing.Process(
                    target=worker_process_wrapper_noliner, 
                    args=(q, c_data, c_idx + 1, opt_const_dim)
                )
                p.daemon = True
                p.start()
                
                running_processes[c_idx] = {
                    'proc': p,
                    'queue': q,
                    'start_time': time.time(),
                    'data': c_data
                }

            active_indices = list(running_processes.keys())
            
            for c_idx in active_indices:
                info = running_processes[c_idx]
                proc = info['proc']
                q = info['queue']
                start_t = info['start_time']
                chunk_data = info['data']
                
                is_timeout = (time.time() - start_t) > CHUNK_TIMEOUT
                is_dead = not proc.is_alive()
                
                result_sublist = None
                error_msg = None

                if is_timeout:
                    tqdm.write(f"[WARN] Core-{c_idx+1} ({CHUNK_TIMEOUT}s). ")
                    if proc.is_alive():
                        proc.terminate()
                        time.sleep(0.1)
                        if proc.is_alive():
                            proc.kill()
                    proc.join()
                    error_msg = "TIMEOUT"

                elif is_dead:
                    try:
                        res = q.get_nowait()
                        if isinstance(res, Exception):
                            tqdm.write(f"[ERROR] Core-{c_idx+1}  {res}")
                            error_msg = "EXCEPTION"
                        else:
                            result_sublist = res
                    except queue.Empty:
                        tqdm.write(f"[ERROR] Core-{c_idx+1} ")
                        error_msg = "CRASH"
                    proc.join()

                if result_sublist is not None:
                    collected_results[c_idx] = result_sublist
                    del running_processes[c_idx]
                    pbar.update(len(result_sublist))
                    
                elif error_msg is not None:

                    collected_results[c_idx] = chunk_data
                    del running_processes[c_idx]
                    pbar.update(len(chunk_data))

            time.sleep(0.1)

    finally:
        pbar.close()

    new_pop = []
    for i in range(len(chunks)):
        new_pop.extend(collected_results[i])

    return new_pop
def get_return_pop(pop, filter=True):
    # compute complex for each one in pop
    complex_of_eq_list = []
    for idx in range(len(pop)):
        complex_of_eq = len(pop[idx][0][0]) + len(pop[idx][0][1])
        pop[idx].append(complex_of_eq)
        complex_of_eq_list.append(complex_of_eq)
    #
    complex_of_eq_list_sort = sorted(list(set(complex_of_eq_list)))
    best_ind_list = []
    for complex_i in complex_of_eq_list_sort:
        best_ind_complex_i = None
        for p_i in pop:
            if p_i[-1] == complex_i:
                if best_ind_complex_i is None or p_i[1] < best_ind_complex_i[1]:
                    best_ind_complex_i = p_i
        best_ind_list.append(best_ind_complex_i)

    #filter bad inds
    if filter:
        best_ind_list_filter = []
        for ind in best_ind_list:
            if ind[1] <= tol_err:
                best_ind_list_filter.append(ind)
    else:
        best_ind_list_filter = best_ind_list

    return best_ind_list_filter


def make_pop_from_file(init_pop_str):
    POP = []
    for ii in tqdm(range(len(init_pop_str))):
        f_eq_str = init_pop_str.iloc[ii]['f_eq_str']
        g_eq_str = init_pop_str.iloc[ii]['g_eq_str']
        b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str, pset_f)
        b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str, pset_g)
        POP.append([(b1, b2), 1e30])
    return POP
  

if __name__ == "__main__":
    b1 = gp.PrimitiveTree.from_string_sympy("(4.607899076892659*x0**2 - 2.955516151374038*x0 + 3.608233058407272)/(0.18819219046851232*x0 + 0.34473271763189295)", pset_f) #3.0413980223824524*log(239.66292803906651*x0 - 216.7035683894636)
    b2 = gp.PrimitiveTree.from_string_sympy("(3.9696344860868287*x0*x1 + 0.02339022406592109*x0 - 0.02949684171391441*x1**2)/(0.4297805626416753*x0 + 1.9302479981166982)", pset_g)
