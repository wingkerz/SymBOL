import torch
import numpy as np
import scipy.integrate as spi

from data_create.lib.Topo import Topo
from data_create.lib.DynamicsEquation import DynamicsEquation
from data_create.lib.InitCondition import InitCondition


class NetworkSystem:
    def __init__(self, N, dim, topo_type=None, f_eq_str=None, g_eq_str=None, init_lbs=None, init_ubs=None,
                 topo=None, network_dynamics=None, init_condition=None):
        self.N = N
        self.dim = dim
        if topo is None:
            self.topo = Topo(N, topo_type)
        else:
            self.topo = topo
        if network_dynamics is None:
            self.network_dynamics = DynamicsEquation(dim, f_eq_str, g_eq_str)
        else:
            self.network_dynamics = network_dynamics
        if init_condition is None:
            self.init_condition = InitCondition(N, dim, init_lbs, init_ubs)
        else:
            self.init_condition = init_condition
        #
        self.MAX_FLOAT_VALUE = 1e38
        self.MIN_FLOAT_VALUE = -1e38
        self.resimulating_max_num = 5
        self.resimulating_count_num = 0

        self.delta_error = 1e2

    # check validity of solutions.
    # if return True means the solution is not valid; otherwise return False
    def check_sol_1(self, X, is_print=False):
        flag = np.any(np.isnan(X)) or np.any(np.isinf(X)) or \
               np.any(X > self.MAX_FLOAT_VALUE) or np.any(X < self.MIN_FLOAT_VALUE)
        if is_print:
            print('check sol 1 -- %s' % flag)
        return flag

    # if return True means the solution is not valid; otherwise return False
    def check_sol_2(self, X_1, t_inc, X_2, is_print=False):
        _, fd_diff = self.finite_difference(X_1, t_inc)
        flag = np.linalg.norm(fd_diff.reshape(-1) - X_2[2:-2].reshape(-1), ord=np.inf) > self.delta_error
        if is_print:
            print('check sol 2 -- %s' % flag)
        return flag

    # five piont finite difference approximation
    def finite_difference(self, X, delta_t):
        # X: [t, n ,d]
        diff_X = (X[0:-4] - 8 * X[1:-3] + 8 * X[3:-1] - X[4:]) / (12.0 * delta_t)
        return X[2:-2], diff_X

    def simulating_data(self, t_start=0., t_inc=1e-4, t_end=0.01, resample_init_condition=False, is_print=False,
                        norm_state=True):
        self.resimulating_count_num += 1
        if self.resimulating_count_num > self.resimulating_max_num:
            self.resimulating_count_num = 0
            return None
        if is_print:
            print('re-simulating_data [%s] ...' % self.resimulating_count_num)
        if resample_init_condition:
            self.init_condition.resampling()
        init_state = self.init_condition.sampled_init_condition

        sparse_adj = self.topo.sparse_adj

        def diff_func(X, t):
            total_diff_signal_, _, _ = self.network_dynamics.generate_diff_data(
                torch.from_numpy(X).view(-1, self.dim),
                sparse_adj)
            return total_diff_signal_.view(-1).numpy()

        t_range = np.arange(t_start, t_end + t_inc, t_inc)
        New_X = spi.odeint(diff_func, init_state.view(-1).numpy(), t_range, rtol=1e-9, atol=1e-9)

        # check state
        if self.check_sol_1(New_X, is_print):
            return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True, is_print=is_print, norm_state=norm_state)

        data = {'t': torch.from_numpy(t_range.reshape(-1, 1)),  # [len(t_range),]
                'state_data': torch.from_numpy(New_X.reshape(len(t_range), self.N, self.dim)),  # [len(t_range), N, dim]
                'sparse_adj': sparse_adj,
                }

        # get total_diff_signal to check diff
        total_diff_signal = []
        for tt_idx in range(data['state_data'].size(0)):
            input_state_ = data['state_data'][tt_idx].view(-1, self.dim)
            total_diff_signal_at_tt_idx, _, _ = self.network_dynamics.generate_diff_data(
                input_state_,
                sparse_adj)
            total_diff_signal.append(total_diff_signal_at_tt_idx.view(1, self.N, self.dim))
        total_diff_signal = torch.cat(total_diff_signal, dim=0)  # [len(t_range), N, dim]
        # check diff
        if self.check_sol_1(total_diff_signal.numpy(), is_print) or \
                self.check_sol_2(data['state_data'].numpy(), t_inc, total_diff_signal.numpy(), is_print):
            return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True, is_print=is_print, norm_state=norm_state)

        # get diff signal including total diff signal, self dynamics signal, and interaction dynamics signal
        total_diff_signal = []
        self_diff_in = []
        self_diff_out = []
        interact_diff_in = []
        interact_diff_out = []
        mu = torch.mean(data['state_data'].view(-1, self.dim), dim=0)
        std = torch.std(data['state_data'].view(-1, self.dim), dim=0)        
        for tt_idx in range(data['state_data'].size(0)):
            input_state_ = data['state_data'][tt_idx].view(-1, self.dim)
            if norm_state:
                input_state_ = (input_state_ - mu.view(1,-1))/(std.view(1,-1) + 1e-5)
            total_diff_signal_at_tt_idx, self_diff_at_tt_idx, interact_diff_at_tt_idx = self.network_dynamics.generate_diff_data(
                input_state_,
                sparse_adj)   
            
            self_diff_in.append(self_diff_at_tt_idx[0].view(1, self.N, self.dim))
            self_diff_out.append(self_diff_at_tt_idx[1].view(1, self.N, self.dim))
            interact_diff_in.append(interact_diff_at_tt_idx[0].view(1, -1, self.dim * 2))
            interact_diff_out.append(interact_diff_at_tt_idx[1].view(1, -1, self.dim))
            total_diff_signal.append(total_diff_signal_at_tt_idx.view(1, self.N, self.dim))
            
            # check diff's nan or inf again after normalizing
            if self.check_sol_1(total_diff_signal_at_tt_idx.numpy(), is_print):
                return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
                                            is_print=is_print, norm_state=norm_state)
                                            

        total_diff_signal = torch.cat(total_diff_signal, dim=0)  # [len(t_range), N, dim]
        self_diff_in = torch.cat(self_diff_in, dim=0)  # [len(t_range), N, dim]
        self_diff_out = torch.cat(self_diff_out, dim=0)  # [len(t_range), N, dim]
        interact_diff_in = torch.cat(interact_diff_in, dim=0)  # [len(t_range), #edges, dim + dim]
        interact_diff_out = torch.cat(interact_diff_out, dim=0)  # [len(t_range), #edges, dim]

        data['total_diff'] = total_diff_signal
        data['self_diff'] = (self_diff_in, self_diff_out)
        data['interact_diff'] = (interact_diff_in, interact_diff_out)

        return data

if __name__ == "__main__":
    # build dynamics equation
    dim = 1
    # f_eq_str = '(x_1 - 1.567) / x_1'
    # g_eq_str = 'x_1 - x_2 * (x_1 + x_2) + x_2'
    f_eq_str = 'x_1 ** 2'
    g_eq_str = 'x_1 * x_2 + x_2 ** 5'
    # f_eq_str = None
    # g_eq_str = 'x_2 - x_1'
    de = DynamicsEquation(dim, f_eq_str, g_eq_str)

    # build topo
    N = 9
    topo_type = 'grid'
    topo = Topo(N, topo_type)

    # build init condition
    init_cond = InitCondition(N, dim, num_sampling=1)

    # build network system
    ns = NetworkSystem(N, dim,
                       topo=topo, network_dynamics=de, init_condition=init_cond)

    data = ns.simulating_data(t_start=0., t_inc=1e-4, t_end=1e-2, resample_init_condition=False)

    print(data)

    import matplotlib.pyplot as plt

    for nn in range(N):
        plt.plot(data['state_data'][:, nn].view(-1))
    plt.show()
