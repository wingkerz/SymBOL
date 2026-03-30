import os
import sys
import contextlib

import torch

from data_create.lib.Topo import Topo
from data_create.lib.DynamicsEquation_new import DynamicsEquation
from data_create.lib.InitCondition import InitCondition
from data_create.lib.NetworkSystem_new import NetworkSystem,  GenerateDataForNetworkDynamics , HeteNetworkSystem



def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd
    
    

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
            
            

class GeneralDynamics:
    def __init__(self, N, dim, topo_type='grid', f_eq_str=None, g_eq_str='x_2 - x_1', topo=None, init_cond=None, netSR=False, flag_hete=False, hete_node_ids=None, hete_id=0):
        self.f_eq_str = f_eq_str
        self.g_eq_str = g_eq_str
        
        # build dynamics equation
        if flag_hete:
            de = [DynamicsEquation(dim, self.f_eq_str[ii], self.g_eq_str[ii]) for ii in range(len(self.f_eq_str))]
        else:
            de = DynamicsEquation(dim, self.f_eq_str, self.g_eq_str)

        # build topo
        if topo is None:
            topo = Topo(N, topo_type)
        
        if init_cond is None:
 
            init_cond = InitCondition(N, dim, lbs=None, ubs=None, num_sampling=1)

        # build network system
        if flag_hete:
            self.ns = HeteNetworkSystem(N, dim,
                                topo=topo, network_dynamics=de, init_condition=init_cond, hete_node_ids=hete_node_ids, hete_id=hete_id)
        else:
            if netSR:
                self.ns = GenerateDataForNetworkDynamics(N, dim,
                                    topo=topo, network_dynamics=de, init_condition=init_cond)
            else:
                self.ns = NetworkSystem(N, dim,
                                    topo=topo, network_dynamics=de, init_condition=init_cond)
        self.dynamics_name = 'f:' + str(self.f_eq_str) + ', g:' + str(self.g_eq_str)

    def simulating_data(self, t_start=0., t_inc=1e-5, t_end=1e-2, resample_init_condition=False,norm_state=True):
        with stdout_redirected():

        #if True:
            data = self.ns.simulating_data(t_start=t_start, t_inc=t_inc, t_end=t_end,
                                       resample_init_condition=resample_init_condition,
                                       norm_state=norm_state)
        return data

    def display(self, data):
        import matplotlib.pyplot as plt
        for dim_i in range(data['state_data'].size(-1)):
            for nn in range(self.ns.N):
                plt.plot(data['state_data'][:, nn, dim_i].view(-1))
        title_name = self.dynamics_name
        plt.title(title_name)
        plt.show()
