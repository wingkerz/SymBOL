import os
import time
from copy import copy
import hydra
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import pickle
import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from data_create.NetworkSystemInstances_new import GeneralDynamics
from data_create.lib.Topo import Topo
from data_create.lib.InitCondition import InitCondition
from string_create.Creation import Creation
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("using device : ", device)
class GenerateData:
    def __init__(
            self,
            cfg,
            eq_filename=None,
    ):
        self.max_dim = 5
        self.eq_filename = eq_filename
        self.state_dim = 1
        columns = ['no']
        columns += [f'f_eq_str_{i}' for i in range(self.state_dim)]
        columns += [f'g_eq_str_{i}' for i in range(self.state_dim)]
        columns += [f's_eq_str_{i}' for i in range(self.state_dim)]
        columns += ['state', 'bool']
        dtype_dict = {col: str for col in columns if "eq_str" in col}
        dtype_dict.update({"no": int, "bool": bool})  
        self.dataset = pd.read_csv(
            self.eq_filename,
            header=None,
            names=columns,
            dtype=dtype_dict  
        )
        self.repeat_num = 5
        self.dataset = self.dataset.loc[self.dataset.index.repeat(self.repeat_num)]
        self.dataset = shuffle(self.dataset)
        self.same_topo_num = 1
        self.generated_data = []
        self.cfg = cfg
    def generate_data(self, ):
        state_dim = self.state_dim
        lineno = 0
        simu_ok_num = 0
        start_time = time.time()
        while lineno < len(self.dataset):
            print(self.dataset.columns)  
            print(self.dataset.iloc[lineno])  
            row_data = self.dataset.iloc[lineno]
            f_eq_str = [row_data[f'f_eq_str_{i}'] for i in range(self.state_dim)]
            g_eq_str = [row_data[f'g_eq_str_{i}'] for i in range(self.state_dim)]
            s_eq_str = [row_data[f's_eq_str_{i}'] for i in range(self.state_dim)]
            state = self.dataset.iloc[lineno]['state']
            bool1 = self.dataset.iloc[lineno]['bool']
            self.norm_state = state
            lineno += 1
            if not bool(bool1):
                continue
            if simu_ok_num % self.same_topo_num == 0:
                print('**resample topo**[ succeed number = %s (dataset lineno %s)]' % (simu_ok_num,lineno))
                topo_index = (simu_ok_num // self.same_topo_num) % len(self.cfg.topo.type_list)
                topo_type_sampled = self.cfg.topo.type_list[topo_index]
                N_sampled = 100
                topo = Topo(N_sampled, topo_type_sampled, high_order = False)
                N_sampled = topo.N
            else:
                print('**keep same topo**[ succeed number = %s (dataset lineno %s)]' % (simu_ok_num,lineno))
            init_cond = InitCondition(N_sampled, state_dim, lbs=None, ubs=None, num_sampling=1)
            print(f_eq_str)
            print(s_eq_str)
            s_ns = GeneralDynamics(N_sampled, state_dim, topo_type_sampled,
                                   f_eq_str, g_eq_str, s_eq_str,  
                                   topo, init_cond)
            try:
                print(self.cfg.t.start)
                print(self.cfg.t.inc)
                print(self.cfg.t.end)
                s_data = s_ns.simulating_data(self.cfg.t.start,
                                              self.cfg.t.inc,
                                              self.cfg.t.end,
                                              self.cfg.resample_init_condition,
                                              norm_state=self.norm_state)
            except Exception as e:
                print(f"Simulation failed: {e}")
                print(f"Current x_1_0: {s_ns.ns.init_condition.X[:, 0]}")
                print(f"Current x_1_1: {s_ns.ns.init_condition.X[:, 1]}")
                raise
            print(f_eq_str, g_eq_str)
            if s_data is None:
                print('**Simulation failed!!!**')
                continue
            self.generated_data.append(s_data)
            simu_ok_num += 1
            print('**Simulation succeed!!!**[ succeed number = %s (dataset lineno %s)]**, cost=%s' % (simu_ok_num, lineno,time.time()-start_time))
    def saved_data_to_file(self, filename=None):
        if filename is None:
            print(self.norm_state)
            filename =  'data/new/test_data_on_%s.pickle' % str(self.norm_state)
        f = open(filename, 'wb')
        pickle.dump(self.generated_data, f)
        f.close()
    def load_data_from_file(self, filename):
        if os.path.isfile(filename):
            print('file exists, loading ...')
            with open(filename, 'rb') as f:
                self.generated_data = pickle.load(f)
                print('--ok')
        else:
            print('no file [%s] exists'%filename)
@hydra.main(config_name="Simulation_config", version_base='1.2', config_path='configs')
def main(cfg):
    generate_data = GenerateData(cfg, eq_filename='data/dataset_test_highorder.csv')
    generate_data.generate_data()
    generate_data.saved_data_to_file()
if __name__ == "__main__":
    main()
