import torch
import numpy as np
import scipy.integrate as spi
import random

from torch_scatter import scatter_sum

from data_create.lib.Topo import Topo
from data_create.lib.DynamicsEquation_new import DynamicsEquation
from data_create.lib.InitCondition import InitCondition


class NetworkSystem:
    def __init__(self, N, dim, topo_type=None, f_eq_str=None, g_eq_str=None, init_lbs=None, init_ubs=None,
                 topo=None, network_dynamics=None, init_condition=None):
        
        self.obs_mask = None
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

        
        
        stop_value = self.MAX_FLOAT_VALUE #########
        def stop_event(t, y):
            return np.min(stop_value - abs(y))
        self.stop_event = stop_event
        self.stop_event.terminal = True

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

    # def _generate_obs_mask(self, num_time_steps,N, dim):
    #     """生成三维观测掩码 (时间步数, N, dim)"""
    #     # 假设时间步数为固定值（例如 100），需根据实际模拟过程调整
    #
    #     obs_ratio = 0.2  # 每个维度的观测比例
    #     obs_mask = torch.zeros(num_time_steps, N, dim, dtype=torch.bool)
    #
    #     # 每个时间步和每个状态维度独立生成观测掩码
    #     for t in range(num_time_steps):
    #         for d in range(dim):
    #             # 随机选择 20% 的节点作为观测节点
    #             obs_indices = torch.randperm(N)[:int(N * obs_ratio)]
    #             obs_mask[t, obs_indices, d] = 1
    #     return obs_mask  # 形状 (num_time_steps, N, dim)

    def _generate_obs_mask(self, num_time_steps, N, dim):
        """生成三维观测掩码 (时间步数, N, dim)
        规则：
        - 前1000个时间步：所有节点和维度掩码值为2
        - 1001-5000时间步：随机选200个时间步，所有节点和维度掩码值为1
        - 其余时间步：掩码值为0
        """
        obs_mask = torch.zeros(num_time_steps, N, dim, dtype=torch.long)  # 使用long类型以支持0,1,2值

        # 前1000个时间步的所有节点和维度掩码值为2
        first_part = min(1000, num_time_steps)
        obs_mask[:first_part, :, :] = 2

        # if num_time_steps > 1000:
        #     # 确定可选的第二部分时间范围
        #     start = 1000
        #     end = min(5000, num_time_steps)
        #     available_steps = end - start
        #
        #     # 从101-500时间步中随机选择200个时间步
        #     if available_steps > 200:
        #         selected_steps = torch.randperm(available_steps)[:200] + start
        #     else:
        #         selected_steps = torch.arange(start, end)
        #
        #     # 在选中的时间步中，所有节点和维度掩码值为1
        #     obs_mask[selected_steps, :, :] = 1

        return obs_mask  # 形状 (num_time_steps, N, dim)

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
        # sparse_adj =
        adj = self.topo.adj
        # obs_mask = self.obs_mask

        
        def diff_func(X, t):
            print('in diff_func',X)
            total_diff_signal_, _, _ = self.network_dynamics.generate_diff_data(
                torch.from_numpy(X).view(-1, self.dim),
                sparse_adj)
                
            #print(total_diff_signal_)
            #exit(1)
            
            return total_diff_signal_.view(-1).detach().numpy()

        t_range = np.arange(t_start, t_end, t_inc)
        # print(diff_func)
        # t_range = np.arange(t_start, t_end + t_inc, t_inc)
        # print(t_range)
        # print("111111111111")
        t_start = t_range[0]
        t_end = t_range[-1]

        # 2. 使用 solve_ivp 代替 odeint
        # 注意：method='RK45' 是单步法，能保证 0-2s 和 0-10s 的前段轨迹高度一致
        sol = spi.solve_ivp(
            fun=lambda t, y: diff_func(y, t),  # 适配参数顺序 (t, y)
            t_span=(t_start, t_end),  # 时间区间
            y0=init_state.view(-1).numpy(),  # 初始状态
            t_eval=t_range,  # 指定输出的时间点
            method='RK45',  # 使用 RK45 算法
            rtol=1e-12,  # 相对误差限制
            atol=1e-12  # 绝对误差限制
        )

        # 3. 提取结果
        # solve_ivp 返回的 sol.y 形状是 [Nodes, Time]，需要转置以匹配之前的 [Time, Nodes]
        New_X = sol.y.T
        # New_X = spi.odeint(diff_func, init_state.view(-1).numpy(), t_range, rtol=1e-12, atol=1e-12)
        # print(diff_func)
        # New_X = spi.odeint(diff_func, init_state.view(-1).numpy(), t_range, method='LSODA', rtol=1e-3, atol=1e-6)
        # New_X = spi.odeint(diff_func, init_state.view(-1).numpy(), t_range)


        
        ##########################
        
        """
        #print(init_state.view(-1).numpy(),)
        
        def diff_func(t, X):
            #print('in diff_func',X)
            total_diff_signal_, _, _ = self.network_dynamics.generate_diff_data(
                torch.from_numpy(X).view(-1, self.dim),
                sparse_adj)
                
            #print(total_diff_signal_)
            #exit(1)
            
            return total_diff_signal_.view(-1).numpy()
        t_range = np.arange(t_start, t_end + t_inc, t_inc)
        
        sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), init_state.view(-1).numpy(), t_eval=t_range, dense_output=True)#, method='RK45', rtol=1e-3, atol=1e-6)
        #print(New_X.status)
        #exit(1)
        #New_X = sol.y.T
        New_X = sol.sol(t_range).T
        
        #print(New_X.shape)
        
        #exit(1)
        #New_X
        """
        ##########################

        self.obs_mask = self._generate_obs_mask(len(t_range), self.N, self.dim)
        # check state
        if self.check_sol_1(New_X, is_print):
            return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
                                        is_print=is_print, norm_state=norm_state)
        state_data = torch.from_numpy(New_X.reshape(len(t_range), self.N, self.dim))
        # observed_states = torch.zeros_like(state_data)
        # for t in range(state_data.shape[0]-1):
        #     # 对每个时间步应用掩码
        #     observed_states[t] = state_data[t] * self.obs_mask[t].float()
        # observed_states = state_data * self.obs_mask.float()
        # data = {'t': torch.from_numpy(t_range.reshape(-1, 1)),  # [len(t_range),]
        #         't_target': torch.from_numpy(t_range.reshape(-1, 1)),  # [len(t_range),]
        #         'state_data': torch.from_numpy(New_X.reshape(len(t_range), self.N, self.dim)),  # [len(t_range), N, dim]
        #         'sparse_adj': sparse_adj,
        #         'adj': adj,
        #         'X0':torch.from_numpy(New_X.reshape(len(t_range), self.N, self.dim))[0],
        #         'obs_mask':self._generate_obs_mask(len(t_range),self.N, self.dim),
        #         'obs_state':state_data,
        #         }
        data = {
                't_target': torch.from_numpy(t_range.reshape(-1, 1)),  # [len(t_range),]
                'adj': sparse_adj,
                'X0': torch.from_numpy(New_X.reshape(len(t_range), self.N, self.dim))[0],
                'obs_mask': self._generate_obs_mask(len(t_range), self.N, self.dim),
                'obs_state': state_data, # [len(t_range), N, dim]
                }

        # get total_diff_signal to check diff
        # 179-251注释了
        # total_diff_signal = []
        # for tt_idx in range(data['state_data'].size(0)):
        #     input_state_ = data['state_data'][tt_idx].view(-1, self.dim)
        #     total_diff_signal_at_tt_idx, _, _ = self.network_dynamics.generate_diff_data(
        #         input_state_,
        #         sparse_adj)
        #     total_diff_signal.append(total_diff_signal_at_tt_idx.view(1, self.N, self.dim))
        # total_diff_signal = torch.cat(total_diff_signal, dim=0)  # [len(t_range), N, dim]
        # # check diff
        # if self.check_sol_1(total_diff_signal.detach().numpy(), is_print) or \
        #         self.check_sol_2(data['state_data'].detach().numpy(), t_inc, total_diff_signal.detach().numpy(), is_print):
        #     return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
        #                                 is_print=is_print, norm_state=norm_state)
        #
        # # get diff signal including total diff signal, self dynamics signal, and interaction dynamics signal
        # #low_bound = -5.
        # #up_bound = 5.
        # # sampling_density = 2500
        # #sampling_density = 400
        # #a = [torch.linspace(low_bound, up_bound, sampling_density) for _ in range(self.dim)]
        #
        # #input_state_ = torch.meshgrid(*a)
        # #input_state_ = [i.reshape(-1, 1) for i in input_state_]
        # #input_state_ = torch.cat(input_state_, dim=-1).view(-1, self.dim)
        #
        # #input_state_ = torch.randn(200*self.dim, self.dim)
        #
        # support_max = 10.
        # support_min = -10.
        #
        # input_state_ = torch.rand(500*self.dim, self.dim) * (support_max - support_min) + support_min
        #
        # # input_min, input_max = 0, 5
        # # input_state_ = torch.zeros(200*self.dim, self.dim)
        # # torch.nn.init.uniform_(input_state_,a=input_min, b=input_max)
        # self_diff = self.network_dynamics.generate_diff_data_1(
        #     input_state_)
        #
        # # sampling_density = 50
        # #sampling_density = 20
        # #a = [torch.linspace(low_bound, up_bound, sampling_density) for _ in range(self.dim + self.dim)]
        #
        # #input_state_2 = torch.meshgrid(*a)
        # #input_state_2 = [i.reshape(-1, 1) for i in input_state_2]
        # #input_state_2 = torch.cat(input_state_2, dim=-1).view(-1, self.dim + self.dim)
        #
        # #input_state_2 = torch.randn(200*self.dim, self.dim + self.dim)
        # input_state_2 = torch.rand(500*self.dim, self.dim + self.dim) * (support_max - support_min) + support_min
        # # input_state_2 = torch.zeros(200*self.dim, self.dim + self.dim)
        # # torch.nn.init.uniform_(input_state_2,a=input_min, b=input_max)
        # interact_diff = self.network_dynamics.generate_diff_data_2(
        #     input_state_2)
        #
        # self_diff_in = self_diff[0].view(-1, self.dim)
        # self_diff_out = self_diff[1].view(-1, self.dim)
        #
        # interact_diff_in = interact_diff[0].view(-1, self.dim + self.dim)
        # interact_diff_out = interact_diff[1].view(-1, self.dim)
        #
        # print('self_diff_out.size=', self_diff_out.size())
        # print('interact_diff_out.size=', interact_diff_out.size())
        #
        # total_diff_signal = self_diff_out + interact_diff_out
        # # total_diff_signal = interact_diff_out
        #
        # # check diff
        # if self.check_sol_1(self_diff_out.detach().numpy(), is_print) or self.check_sol_1(interact_diff_out.detach().numpy(), is_print):
        #     return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
        #                                 is_print=is_print, norm_state=norm_state)
        #
        # data['total_diff'] = total_diff_signal
        # data['self_diff'] = (self_diff_in, self_diff_out)
        # data['interact_diff'] = (interact_diff_in, interact_diff_out)

        return data




class HeteNetworkSystem:
    def __init__(self, N, dim,
                 topo=None, network_dynamics=None, init_condition=None, hete_node_ids=None, hete_id=None):
        self.N = N
        self.dim = dim
        self.topo = topo
        self.network_dynamics = network_dynamics
        self.init_condition = init_condition
        
        self.hete_num = len(network_dynamics)
        self.hete_node_ids = hete_node_ids
        self.hete_id = hete_id
        #
        self.MAX_FLOAT_VALUE = 1e38
        # self.MIN_FLOAT_VALUE = -1e38
        self.MIN_FLOAT_VALUE = 1e-5

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
        
        sparse_adj_hete_parts = []
        for ii in range(self.hete_num):
            sparse_adj_hete_ii = torch.zeros(sparse_adj.size(1),).view(-1)
            for n_id in self.hete_node_ids[ii]:
                sparse_adj_hete_ii[sparse_adj[1]==n_id] = 1
            sparse_adj_hete_parts.append(sparse_adj_hete_ii.long())

        def diff_func(X, t):
            total_diff_signal_ = torch.from_numpy(np.zeros_like(X)).view(-1, self.dim)
            for ii in range(self.hete_num):
                total_diff_signal_ii, _, _ = self.network_dynamics[ii].generate_diff_data(
                    torch.from_numpy(X).view(-1, self.dim),
                    sparse_adj[:, sparse_adj_hete_parts[ii]==1])
                total_diff_signal_[self.hete_node_ids[ii].long()] = total_diff_signal_ii[self.hete_node_ids[ii].long()]
            
            return total_diff_signal_.view(-1).numpy()

        t_range = np.arange(t_start, t_end + t_inc, t_inc)
        New_X = spi.odeint(diff_func, init_state.view(-1).numpy(), t_range, rtol=1e-9, atol=1e-9)

        # check state
        if self.check_sol_1(New_X, is_print):
            return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
                                        is_print=is_print, norm_state=norm_state)

        data = {'t': torch.from_numpy(t_range.reshape(-1, 1)),  # [len(t_range),]
                'state_data': torch.from_numpy(New_X.reshape(len(t_range), self.N, self.dim)),  # [len(t_range), N, dim]
                'sparse_adj': sparse_adj,
                }

        # get total_diff_signal to check diff
        total_diff_signal = []
        for tt_idx in range(data['state_data'].size(0)):
            input_state_ = data['state_data'][tt_idx].view(-1, self.dim)
            total_diff_signal_at_tt_idx = torch.zeros_like(input_state_)
            for ii in range(self.hete_num):
                total_diff_signal_at_tt_idx_ii, _, _ = self.network_dynamics[ii].generate_diff_data(
                    input_state_,
                    sparse_adj[:, sparse_adj_hete_parts[ii]==1])
                total_diff_signal_at_tt_idx[self.hete_node_ids[ii].long()] = total_diff_signal_at_tt_idx_ii[self.hete_node_ids[ii].long()]
                
            total_diff_signal.append(total_diff_signal_at_tt_idx.view(1, self.N, self.dim))
        total_diff_signal = torch.cat(total_diff_signal, dim=0)  # [len(t_range), N, dim]
        # check diff
        if self.check_sol_1(total_diff_signal.numpy(), is_print) or \
                self.check_sol_2(data['state_data'].numpy(), t_inc, total_diff_signal.numpy(), is_print):
            return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
                                        is_print=is_print, norm_state=norm_state)

        # get diff signal including total diff signal, self dynamics signal, and interaction dynamics signal
        #low_bound = -5.
        #up_bound = 5.
        # sampling_density = 2500
        #sampling_density = 400
        #a = [torch.linspace(low_bound, up_bound, sampling_density) for _ in range(self.dim)]

        #input_state_ = torch.meshgrid(*a)
        #input_state_ = [i.reshape(-1, 1) for i in input_state_]
        #input_state_ = torch.cat(input_state_, dim=-1).view(-1, self.dim)

        #input_state_ = torch.randn(200*self.dim, self.dim)
        
        support_max = 10.
        support_min = -10.

        input_state_ = torch.rand(500*self.dim, self.dim) * (support_max - support_min) + support_min

        # input_state_ = torch.randn(500*self.dim, self.dim)

        # input_min, input_max = 0, 5
        # input_state_ = torch.zeros(200*self.dim, self.dim)
        # torch.nn.init.uniform_(input_state_,a=input_min, b=input_max)
        self_diff = self.network_dynamics[self.hete_id].generate_diff_data_1(
            input_state_)

        # sampling_density = 50
        #sampling_density = 20
        #a = [torch.linspace(low_bound, up_bound, sampling_density) for _ in range(self.dim + self.dim)]

        #input_state_2 = torch.meshgrid(*a)
        #input_state_2 = [i.reshape(-1, 1) for i in input_state_2]
        #input_state_2 = torch.cat(input_state_2, dim=-1).view(-1, self.dim + self.dim)

        #input_state_2 = torch.randn(200*self.dim, self.dim + self.dim)
        input_state_2 = torch.rand(500*self.dim, self.dim + self.dim) * (support_max - support_min) + support_min
        # input_state_2 = torch.zeros(200*self.dim, self.dim + self.dim)
        # torch.nn.init.uniform_(input_state_2,a=input_min, b=input_max)
        interact_diff = self.network_dynamics[self.hete_id].generate_diff_data_2(
            input_state_2)

        self_diff_in = self_diff[0].view(-1, self.dim)
        self_diff_out = self_diff[1].view(-1, self.dim)

        interact_diff_in = interact_diff[0].view(-1, self.dim + self.dim)
        interact_diff_out = interact_diff[1].view(-1, self.dim)

        print('self_diff_out.size=', self_diff_out.size())
        print('interact_diff_out.size=', interact_diff_out.size())

        # total_diff_signal = self_diff_out + interact_diff_out
        total_diff_signal = interact_diff_out

        # check diff
        if self.check_sol_1(self_diff_out.numpy(), is_print) or self.check_sol_1(interact_diff_out.numpy(), is_print):
            return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
                                        is_print=is_print, norm_state=norm_state)

        data['total_diff'] = total_diff_signal
        data['self_diff'] = (self_diff_in, self_diff_out)
        data['interact_diff'] = (interact_diff_in, interact_diff_out)

        return data
        
## --------------------------------------------

class GenerateDataForNetworkDynamics:
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

    # check validity of solutions.
    # if return True means the solution is not valid; otherwise return False
    def check_sol_1(self, X, is_print=False):
        flag = np.any(np.isnan(X)) or np.any(np.isinf(X)) or \
               np.any(X > self.MAX_FLOAT_VALUE) or np.any(X < self.MIN_FLOAT_VALUE)
        if is_print:
            print('check sol 1 -- %s' % flag)
        return flag

    def simulating_data(self, t_start=0., t_inc=1e-4, t_end=0.01, resample_init_condition=False, is_print=False,
                        norm_state=True):
        self.resimulating_count_num += 1
        if self.resimulating_count_num > self.resimulating_max_num:
            self.resimulating_count_num = 0
            return None
        if is_print:
            print('re-simulating_data [%s] ...' % self.resimulating_count_num)
        # if resample_init_condition:
        #     self.init_condition.resampling()
        # init_state = self.init_condition.sampled_init_condition

        sparse_adj = self.topo.sparse_adj

        t_range = np.arange(t_start, t_end + t_inc, t_inc)
        New_X = torch.randn(len(t_range), self.N, self.dim)
        
        total_diff_signal = []
        for t_i in range(len(t_range)):
            total_diff_signal_t_i, _, _ = self.network_dynamics.generate_diff_data(New_X[t_i], sparse_adj)
            total_diff_signal.append(total_diff_signal_t_i.unsqueeze(0))
        total_diff_signal = torch.cat(total_diff_signal, dim=0)
        
        # check diff
        if self.check_sol_1(total_diff_signal.numpy(), is_print):
            return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
                                        is_print=is_print, norm_state=norm_state)

        data = {'t': torch.from_numpy(t_range.reshape(-1, 1)),  # [len(t_range),]
                'state_data': New_X,  # [len(t_range), N, dim]
                'sparse_adj': sparse_adj,
                'total_diff': total_diff_signal  # [len(t_range), N, dim]
                }

        # get diff signal including total diff signal, self dynamics signal, and interaction dynamics signal
        low_bound = -5.
        up_bound = 5.
        # sampling_density = 2500
        sampling_density = 400
        a = [torch.linspace(low_bound, up_bound, sampling_density) for _ in range(self.dim)]

        input_state_ = torch.meshgrid(*a)
        input_state_ = [i.reshape(-1, 1) for i in input_state_]
        input_state_ = torch.cat(input_state_, dim=-1).view(-1, self.dim)

        # input_state_ = torch.randn(500*self.dim, self.dim)

        # input_state_ = torch.randn(500*self.dim, self.dim)

        # input_min, input_max = 0, 5
        # input_state_ = torch.zeros(200*self.dim, self.dim)
        # torch.nn.init.uniform_(input_state_,a=input_min, b=input_max)
        self_diff = self.network_dynamics.generate_diff_data_1(
            input_state_)

        # sampling_density = 50
        sampling_density = 20
        a = [torch.linspace(low_bound, up_bound, sampling_density) for _ in range(self.dim + self.dim)]

        input_state_2 = torch.meshgrid(*a)
        input_state_2 = [i.reshape(-1, 1) for i in input_state_2]
        input_state_2 = torch.cat(input_state_2, dim=-1).view(-1, self.dim + self.dim)

        # input_state_2 = torch.randn(500*self.dim, self.dim + self.dim)
        # input_state_2 = torch.zeros(200*self.dim, self.dim + self.dim)
        # torch.nn.init.uniform_(input_state_2,a=input_min, b=input_max)
        interact_diff = self.network_dynamics.generate_diff_data_2(
            input_state_2)

        self_diff_in = self_diff[0].view(-1, self.dim)
        self_diff_out = self_diff[1].view(-1, self.dim)

        interact_diff_in = interact_diff[0].view(-1, self.dim + self.dim)
        interact_diff_out = interact_diff[1].view(-1, self.dim)

        #print('self_diff_out.size=', self_diff_out.size())
        #print('interact_diff_out.size=', interact_diff_out.size())

        # total_diff_signal = self_diff_out + interact_diff_out
        #total_diff_signal = interact_diff_out

        # check diff
        if self.check_sol_1(self_diff_out.numpy(), is_print) or self.check_sol_1(interact_diff_out.numpy(), is_print):
            return self.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end, resample_init_condition=True,
                                        is_print=is_print, norm_state=norm_state)

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
