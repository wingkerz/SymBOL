import torch

from data_create.lib.Topo import Topo
from data_create.lib.DynamicsEquation import DynamicsEquation
from data_create.lib.InitCondition import InitCondition
from data_create.lib.NetworkSystem import NetworkSystem


class GeneralDynamics:
    def __init__(self, N, dim, topo_type='grid', f_eq_str=None, g_eq_str='x_2 - x_1', topo=None, init_cond=None):
        self.f_eq_str = f_eq_str
        self.g_eq_str = g_eq_str

        # build dynamics equation
        de = DynamicsEquation(dim, self.f_eq_str, self.g_eq_str)

        # build topo
        if topo is None:
            topo = Topo(N, topo_type)
        
        if init_cond is None:
            # build init condition
            # init_cond = InitCondition(N, dim, lbs=torch.zeros(dim), ubs=torch.ones(dim) * 10., num_sampling=1)
            # init_cond = InitCondition(N, dim, lbs=torch.ones(dim)*(-5.), ubs=torch.ones(dim) * 5., num_sampling=1)
            init_cond = InitCondition(N, dim, lbs=torch.ones(dim) * (-3.), ubs=torch.ones(dim) * 3., num_sampling=1)

        # build network system
        self.ns = NetworkSystem(N, dim,
                                topo=topo, network_dynamics=de, init_condition=init_cond)
        self.dynamics_name = 'f:' + str(self.f_eq_str) + ', g:' + str(self.g_eq_str)

    def simulating_data(self, t_start=0., t_inc=1e-4, t_end=1e-2, resample_init_condition=False,norm_state=True):
        data = self.ns.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end,
                                       resample_init_condition=resample_init_condition,
                                       norm_state=norm_state)
        return data

    def display(self, data):
        import matplotlib.pyplot as plt
        for nn in range(self.ns.N):
            plt.plot(data['state_data'][:, nn].view(-1))
        title_name = self.dynamics_name
        plt.title(title_name)
        plt.show()
